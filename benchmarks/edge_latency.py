"""Offline search latency and recall on the Qdrant Edge store.

    python benchmarks/edge_latency.py [--n 10000] [--dim 384] [--queries 200]

Synthetic clustered vectors + templated text (no model needed). Reports
p50/p95 latency for dense, hybrid (dense+BM25, RRF) and space-filtered search,
and recall@10 of the Edge index against exact brute-force cosine.
Writes benchmarks/results/edge_latency.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from loci.edge import EdgeMemoryStore, Memory

WORDS = [
    "red",
    "blue",
    "green",
    "toolbox",
    "pallet",
    "forklift",
    "dock",
    "aisle",
    "shelf",
    "crate",
    "charger",
    "door",
    "scanner",
]


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def timed(fn, queries) -> list[float]:
    out = []
    for q in queries:
        t = time.perf_counter()
        fn(q)
        out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    centers = rng.normal(size=(50, args.dim))
    vecs = centers[rng.integers(0, 50, args.n)] + 0.5 * rng.normal(size=(args.n, args.dim))
    xs = rng.random((args.n, 3))
    texts = [" ".join(rng.choice(WORDS, 4)) for _ in range(args.n)]

    tmp = tempfile.mkdtemp(prefix="loci-edge-bench-")
    store = EdgeMemoryStore(Path(tmp) / "shard", args.dim, "bench")
    t0 = time.perf_counter()
    for i in range(args.n):
        store.put(Memory(f"m{i}", vecs[i].tolist(), *xs[i], timestamp_ms=1_000 + i, text=texts[i]))
    ingest_s = time.perf_counter() - t0
    store.optimize()
    indexed = store.indexed_vectors()

    qi = rng.integers(0, args.n, args.queries)
    qv = (vecs[qi] + 0.3 * rng.normal(size=(args.queries, args.dim))).tolist()
    qt = [texts[i].split()[0] for i in qi]
    region = {"x_min": 0.0, "x_max": 0.25, "y_min": 0.0, "y_max": 0.25, "z_min": 0.0, "z_max": 1.0}

    dense = timed(lambda q: store.search(vector=q, limit=10), qv)
    hybrid = timed(
        lambda p: store.search(vector=p[0], text=p[1], limit=10), list(zip(qv, qt, strict=True))
    )
    spatial = timed(lambda q: store.search(vector=q, bounds=region, limit=10), qv)

    # recall@10 of the Edge index vs exact cosine over the same vectors
    mat = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    hits = 0
    for q in qv[:100]:
        qn = np.asarray(q) / np.linalg.norm(q)
        exact = set(np.argsort(-(mat @ qn))[:10].tolist())
        got = {int(h.payload["key"][1:]) for h in store.search(vector=q, limit=10)}
        hits += len(exact & got)
    recall = hits / (100 * 10)

    result = {
        "n": args.n,
        "dim": args.dim,
        "hnsw_indexed_vectors": indexed,
        "ingest_per_s": round(args.n / ingest_s, 1),
        "latency_ms": {
            name: {
                "p50": round(pct(v, 50), 2),
                "p95": round(pct(v, 95), 2),
                "mean": round(statistics.mean(v), 2),
            }
            for name, v in [("dense", dense), ("hybrid_rrf", hybrid), ("dense_spatial", spatial)]
        },
        "recall_at_10_vs_exact": round(recall, 3),
        "note": "synthetic clustered vectors, single process, warm cache, no network",
    }
    print(json.dumps(result, indent=2))
    out = Path(__file__).parent / "results" / "edge_latency.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    store.close()


if __name__ == "__main__":
    main()
