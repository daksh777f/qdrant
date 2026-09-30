# ruff: noqa: E501
"""What does quantization buy on a small device?  (and what does it cost?)

    python benchmarks/edge_quantization.py [--n 20000] [--dim 384] [--queries 200]

For each configuration a shard is built once, then opened and queried in a FRESH process so the
resident-memory figure is what that configuration needs to serve searches, not an ingestion peak.

* recall@10: overlap with exact cosine search over the same vectors (100 queries);
* latency: dense search, p50 / p95 over the queries, warm cache;
* disk: bytes actually allocated under the shard directory (not apparent size: Edge preallocates
  sparse files, which would overstate usage about 4x);
* RSS: growth of the process' resident set from before the shard is opened to after the queries
  (Linux /proc). It includes file-backed pages the OS can evict, so it is an upper bound on what
  must stay in RAM, and it is the honest thing to compare across configurations.

Synthetic clustered vectors (50 centres + noise). Real embeddings behave differently: rerun on
your own vectors before choosing a mode. Writes benchmarks/results/edge_quantization.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CONFIGS = [
    ("baseline (float32)", {"quantization": None, "vectors_on_disk": False}),
    ("scalar int8", {"quantization": "scalar", "vectors_on_disk": False}),
    ("binary 1-bit", {"quantization": "binary", "vectors_on_disk": False}),
    ("binary + originals on disk", {"quantization": "binary", "vectors_on_disk": True}),
]


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def dir_bytes(path: Path) -> int:
    """Bytes actually allocated on disk. Edge preallocates sparse files (WAL, storage pages), so
    st_size overstates real usage several-fold; allocated blocks is what a device pays."""
    return sum(f.stat().st_blocks * 512 for f in path.rglob("*") if f.is_file())


def child(shard_dir: str, vec_file: str, queries: int) -> None:
    """Open a built shard in this fresh process and measure serving it."""
    from loci.edge import EdgeMemoryStore

    vecs = np.load(vec_file)
    rng = np.random.default_rng(1)
    qi = rng.integers(0, len(vecs), queries)
    qv = vecs[qi] + 0.3 * rng.normal(size=(queries, vecs.shape[1]))
    before = rss_mb()
    store = EdgeMemoryStore(shard_dir, vecs.shape[1], "bench")
    lat = []
    for q in qv:
        t = time.perf_counter()
        store.search(vector=q.tolist(), limit=10)
        lat.append((time.perf_counter() - t) * 1000)
    after = rss_mb()
    mat = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    hits = 0
    n_eval = min(100, queries)
    for q in qv[:n_eval]:
        exact = set(np.argsort(-(mat @ (q / np.linalg.norm(q))))[:10].tolist())
        got = {int(h.payload["key"][1:]) for h in store.search(vector=q.tolist(), limit=10)}
        hits += len(exact & got)
    store.close()
    print(
        json.dumps(
            {
                "rss_growth_mb": round(after - before, 1),
                "p50_ms": round(float(np.percentile(lat, 50)), 2),
                "p95_ms": round(float(np.percentile(lat, 95)), 2),
                "recall_at_10": round(hits / (n_eval * 10), 3),
            }
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument(
        "--out", default="edge_quantization.json", help="file name under benchmarks/results/"
    )
    ap.add_argument("--child", nargs=2, metavar=("SHARD_DIR", "VEC_FILE"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(args.child[0], args.child[1], args.queries)
        return

    from loci.edge import EdgeMemoryStore, Memory

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(50, args.dim))
    vecs = (
        centers[rng.integers(0, 50, args.n)] + 0.5 * rng.normal(size=(args.n, args.dim))
    ).astype(np.float32)
    work = Path(tempfile.mkdtemp(prefix="loci-quant-"))
    vec_file = work / "vecs.npy"
    np.save(vec_file, vecs)
    xs = rng.random((args.n, 3))

    rows = []
    for name, kw in CONFIGS:
        shard = work / name.replace(" ", "_").replace("+", "and").replace("(", "").replace(")", "")
        store = EdgeMemoryStore(shard, args.dim, "bench", **kw)
        t0 = time.perf_counter()
        for i in range(args.n):
            store.put(Memory(f"m{i}", vecs[i].tolist(), *xs[i], timestamp_ms=1_000 + i))
        ingest_s = time.perf_counter() - t0
        store.optimize()
        store.close()
        out = subprocess.run(
            [
                sys.executable,
                __file__,
                "--queries",
                str(args.queries),
                "--child",
                str(shard),
                str(vec_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        m = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append({"config": name, "disk_mb": round(dir_bytes(shard) / 1e6, 1),
                     "ingest_per_s": round(args.n / ingest_s), **m})  # fmt: skip
        print(rows[-1], flush=True)

    base = rows[0]
    print(
        f"\n{'config':30} {'disk MB':>8} {'RSS +MB':>8} {'p50 ms':>7} {'p95 ms':>7} {'recall@10':>10}"
    )
    for r in rows:
        print(
            f"{r['config']:30} {r['disk_mb']:8.1f} {r['rss_growth_mb']:8.1f} {r['p50_ms']:7.2f} "
            f"{r['p95_ms']:7.2f} {r['recall_at_10']:10.3f}"
        )
    result = {
        "n": args.n,
        "dim": args.dim,
        "rows": rows,
        "baseline_rss_growth_mb": base["rss_growth_mb"],
        "note": "synthetic clustered vectors; fresh-process RSS includes evictable file-backed pages",
    }
    out_path = Path(__file__).parent / "results" / args.out
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
