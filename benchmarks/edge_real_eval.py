# ruff: noqa: E501
"""Real-data evaluation: public, human-labelled datasets through the real Qdrant Edge engine.

    python benchmarks/edge_real_eval.py                         # embedder from LOCI_EMBEDDER (auto)
    python benchmarks/edge_real_eval.py --embedder fastembed    # all-MiniLM-L6-v2 via Qdrant's fastembed
    python benchmarks/edge_real_eval.py --embedder onnx:/models/minilm
    python benchmarks/edge_real_eval.py --embedder hash         # zero-download stand-in (labelled)

Nothing here is generated: the text and the labels are human-made public datasets, fetched from
pinned URLs and verified by SHA-256 before use (cached under benchmarks/data/, not committed):

* STS-Benchmark test (1,379 pairs, similarity 0-5 by crowd workers)          CC BY-SA 4.0
* MSRP / Microsoft Research Paraphrase Corpus test (1,725 pairs, same/different by two judges)
* SICK test (4,927 pairs, relatedness 1-5)                                    CC BY-NC-SA 3.0

Questions answered, each mapped to a platform decision:

1. Retrieval (does the device find the right memory?): each human-confirmed paraphrase pair is a
   query and its one correct answer, hidden among every other sentence in the three datasets.
   Run on an EdgeMemoryStore (real Qdrant Edge shard): BM25 only, dense only, hybrid RRF.
   Recall@1/@10, MRR@10, latency.
2. Duplicate detection (the DEDUPE rule): MSRP human same/different labels vs. embedding cosine.
   ROC-AUC, best-F1 threshold, and the false-merge rate at the policy's thresholds.
3. Abstention (the answer gate): half the queries have their answer removed from the corpus.
   Does gate confidence separate answerable from unanswerable? ROC-AUC, coverage, precision.
4. Embedder sanity: STS-B Spearman correlation (published for all-MiniLM-L6-v2: about 0.82).

Writes benchmarks/results/edge_real_eval.{json,md}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loci.edge.embed import make_embedder  # noqa: E402
from loci.edge.gate import AnswerGate, GateConfig  # noqa: E402
from loci.edge.policy import PolicyConfig  # noqa: E402
from loci.edge.store import EdgeMemoryStore, Memory  # noqa: E402

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
DATASETS = {
    "stsb": (
        "https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-test.csv",
        "11523b625219e94e9ca05d2816b5f02cac1614c5894fe657376fa0806378d053",
    ),
    "msrp": (
        "https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt",
        "0360d0d3427a0ae14882e2b3a799df0fab6cb5203d5e8ee481949283da62acb6",
    ),
    "sick": (
        "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_test_annotated.txt",
        "2b8aa806658d6fc23c6824c83776c2d4fee7556000817b5ec0f982861413b7d0",
    ),
}


# ------------------------------------------------------------------ data


def fetch(name: str) -> Path:
    url, sha = DATASETS[name]
    DATA.mkdir(exist_ok=True)
    path = DATA / Path(url).name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310  # nosec B310 - pinned https URL
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"{path.name}: SHA-256 {got} != pinned {sha}; refusing to evaluate on it")
    return path


def load() -> dict[str, list[tuple[str, str, float]]]:
    out: dict[str, list[tuple[str, str, float]]] = {}
    with fetch("stsb").open(encoding="utf-8") as f:
        out["stsb"] = [(a, b, float(s)) for a, b, s in csv.reader(f)]
    rows = fetch("msrp").read_text(encoding="utf-8-sig").splitlines()[1:]
    out["msrp"] = [
        (r.split("\t")[3], r.split("\t")[4], float(r.split("\t")[0]))
        for r in rows
        if r.count("\t") >= 4
    ]
    rows = fetch("sick").read_text(encoding="utf-8").splitlines()[1:]
    out["sick"] = [
        (r.split("\t")[1], r.split("\t")[2], float(r.split("\t")[3]))
        for r in rows
        if r.count("\t") >= 4
    ]
    return out


# ------------------------------------------------------------------ small stats helpers


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    r = ranks(s) + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cos_pairs(emb, pairs) -> np.ndarray:
    a = emb.embed_many([p[0] for p in pairs])
    b = emb.embed_many([p[1] for p in pairs])
    return (a * b).sum(axis=1)


# ------------------------------------------------------------------ the four evaluations


def eval_retrieval(emb, data, n_queries: int, seed: int) -> tuple[dict, dict]:
    positives = [(a, b) for a, b, s in data["stsb"] if s >= 4.0] + [
        (a, b) for a, b, lab in data["msrp"] if lab == 1
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(positives))[:n_queries]
    queries = [positives[i] for i in idx]
    answerable = set(range(0, len(queries), 2))  # the other half: answer removed (for the gate)
    q_texts = {q for q, _ in queries}
    removed = {queries[i][1] for i in range(len(queries)) if i not in answerable}
    corpus = sorted(
        {t for ds in data.values() for a, b, _ in ds for t in (a, b)} - q_texts - removed
    )
    # A sentence can appear in several pairs; keep only cleanly defined queries: answerable ones
    # whose answer is in the corpus, unanswerable ones whose answer is not, never query == answer.
    in_corpus = set(corpus)
    keep = [
        i
        for i, (q, gold) in enumerate(queries)
        if q != gold and ((i in answerable) == (gold in in_corpus))
    ]
    queries = [queries[i] for i in keep]
    answerable = {j for j, i in enumerate(keep) if i in answerable}
    tmp = Path(tempfile.mkdtemp(prefix="loci-real-"))
    store = EdgeMemoryStore(tmp / "s", emb.dim, "eval")
    t0 = time.perf_counter()
    vecs = emb.embed_many(corpus)
    embed_s = time.perf_counter() - t0
    key_of = {}
    t0 = time.perf_counter()
    for i, (text, v) in enumerate(zip(corpus, vecs, strict=True)):
        store.put(Memory(f"c{i}", v.tolist(), 0.5, 0.5, 0.0, 1_000 + i, text=text))
        key_of[text] = f"c{i}"
    ingest_s = time.perf_counter() - t0
    store.optimize()
    qv = emb.embed_many([q for q, _ in queries])

    modes = {
        "bm25 (Qdrant Edge sparse)": lambda i: store.search(text=queries[i][0], limit=10),
        "dense": lambda i: store.search(vector=qv[i].tolist(), limit=10),
        "hybrid RRF (dense + bm25)": lambda i: store.search(
            vector=qv[i].tolist(), text=queries[i][0], limit=10
        ),
    }
    res = {}
    ans = sorted(answerable)
    for name, fn in modes.items():
        r1 = r10 = mrr = 0.0
        lat = []
        for i in ans:
            t = time.perf_counter()
            hits = fn(i)
            lat.append((time.perf_counter() - t) * 1000)
            keys = [h.payload["key"] for h in hits]
            gold = key_of[queries[i][1]]
            if gold in keys:
                rank = keys.index(gold)
                r1 += rank == 0
                r10 += 1
                mrr += 1 / (rank + 1)
        n = len(ans)
        res[name] = {
            "recall@1": round(r1 / n, 3),
            "recall@10": round(r10 / n, 3),
            "mrr@10": round(mrr / n, 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
        }

    # ---- gate: answerable vs unanswerable
    gate = AnswerGate(store, emb, None, GateConfig())
    conf, is_ans, correct = [], [], []
    for i, (q, gold) in enumerate(queries):
        a = gate.ask(q, limit=5)
        conf.append(a.confidence)
        is_ans.append(i in answerable)
        correct.append(bool(a.hits) and a.hits[0].get("text") == gold)
    th = GateConfig().answer_threshold
    conf_a = np.asarray(conf)
    ia = np.asarray(is_ans)
    answered = conf_a >= th
    gate_res = {
        "queries": len(queries),
        "answerable": int(ia.sum()),
        "auroc_answerable_vs_not": round(auroc(conf, is_ans), 3),
        "threshold": th,
        "answered_when_answerable_pct": round(100 * answered[ia].mean(), 1),
        "answered_when_unanswerable_pct": round(100 * answered[~ia].mean(), 1),
        "precision_of_answers_pct": round(
            100 * float(np.mean([c for c, a_ in zip(correct, answered, strict=True) if a_]))
            if answered.any()
            else 0.0,
            1,
        ),
    }
    meta = {
        "corpus_sentences": len(corpus),
        "retrieval_queries": len(ans),
        "embed_per_s": round(len(corpus) / embed_s, 1),
        "ingest_per_s": round(len(corpus) / ingest_s, 1),
        "footprint": store.footprint(),
    }
    store.close()
    return {"modes": res, **meta}, gate_res


def eval_dedupe(emb, data) -> dict:
    pairs = data["msrp"]
    cos = cos_pairs(emb, pairs)
    same = np.asarray([lab == 1 for _, _, lab in pairs])
    best = (0.0, 0.0)
    for th in np.unique(np.round(cos, 3)):
        pred = cos >= th
        tp = (pred & same).sum()
        p = tp / max(pred.sum(), 1)
        r = tp / same.sum()
        f1 = 2 * p * r / max(p + r, 1e-9)
        best = max(best, (float(f1), float(th)))

    def at(th: float) -> dict:
        pred = cos >= th
        return {
            "threshold": th,
            "pairs_merged_pct": round(100 * pred.mean(), 1),
            "precision_same_pct": round(100 * (pred & same).sum() / max(pred.sum(), 1), 1),
            "false_merge_rate_pct": round(100 * (pred & ~same).sum() / max((~same).sum(), 1), 1),
            "recall_same_pct": round(100 * (pred & same).sum() / same.sum(), 1),
        }

    cfg = PolicyConfig()
    return {
        "pairs": len(pairs),
        "human_same_pct": round(100 * same.mean(), 1),
        "auroc": round(auroc(cos, same), 3),
        "best_f1": round(best[0], 3),
        "best_f1_threshold": best[1],
        "at_dedupe_threshold": at(cfg.dedupe_similarity),
        "at_ambiguous_threshold": at(cfg.ambiguous_similarity),
    }


def eval_sts(emb, data) -> dict:
    out = {}
    for name in ("stsb", "sick"):
        pairs = data[name]
        out[name] = round(spearman(cos_pairs(emb, pairs), [s for _, _, s in pairs]), 3)
    return out


# ------------------------------------------------------------------ report


def report(R: dict) -> str:
    rt, g, d, s = R["retrieval"], R["gate"], R["dedupe"], R["sts_spearman"]
    L = [
        "# Real-data evaluation",
        "",
        f"Embedder: **{R['embedder']}** ({'learned semantic model' if R['embedder_real'] else 'NOT a semantic model: lexical stand-in'}). {R['embedder_note']}.",
        "Datasets: STS-B test, MSRP test, SICK test (public, human-labelled, SHA-256 verified). Engine: Qdrant Edge on this machine.",
        "",
        f"## 1. Retrieval: {rt['retrieval_queries']} paraphrase queries, each with one correct answer among {rt['corpus_sentences']:,} real sentences",
        "",
        "| mode | recall@1 | recall@10 | MRR@10 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, v in rt["modes"].items():
        L.append(
            f"| {m} | {v['recall@1']} | {v['recall@10']} | {v['mrr@10']} | {v['p50_ms']} | {v['p95_ms']} |"
        )
    a, b = d["at_dedupe_threshold"], d["at_ambiguous_threshold"]
    L += [
        "",
        f"## 2. Duplicate detection vs. human judges (MSRP, {d['pairs']:,} pairs, {d['human_same_pct']}% judged same)",
        "",
        f"ROC-AUC **{d['auroc']}**; best F1 {d['best_f1']} at cosine {d['best_f1_threshold']}.",
        "",
        "| cosine threshold | pairs merged | precision (same) | false merges (of different pairs) | recall (same) |",
        "|---:|---:|---:|---:|---:|",
        f"| {a['threshold']} (DEDUPE) | {a['pairs_merged_pct']}% | {a['precision_same_pct']}% | {a['false_merge_rate_pct']}% | {a['recall_same_pct']}% |",
        f"| {b['threshold']} (look-alike band) | {b['pairs_merged_pct']}% | {b['precision_same_pct']}% | {b['false_merge_rate_pct']}% | {b['recall_same_pct']}% |",
        "",
        "Text similarity alone confuses related-but-different statements; on the device a merge additionally requires the same place (and the cloud the same time window), which this text-only test cannot credit.",
        "",
        f"## 3. Abstention: {g['queries']} queries, {g['answerable']} answerable, the rest with their answer removed",
        "",
        f"Gate confidence separates answerable from unanswerable with ROC-AUC **{g['auroc_answerable_vs_not']}**. At threshold {g['threshold']}: answers {g['answered_when_answerable_pct']}% of answerable and {g['answered_when_unanswerable_pct']}% of unanswerable queries; {g['precision_of_answers_pct']}% of given answers are the correct sentence.",
        "",
        "## 4. Embedder sanity (Spearman vs. human similarity)",
        "",
        f"STS-B {s['stsb']}, SICK {s['sick']}. "
        + (
            "Published STS-B for all-MiniLM-L6-v2 is about 0.82; a large gap would mean the model or pipeline is wrong."
            if R["embedder_real"]
            else "This is a lexical stand-in, so low correlation is expected: it only matches shared words. Run with a real model (--embedder fastembed) for semantic results."
        ),
        "",
        f"Device footprint of the evaluation shard: {rt['footprint']['points_local']:,} memories, {rt['footprint']['disk_local_bytes'] / 1e6:.1f} MB on disk. Embedding {rt['embed_per_s']} sentences/s, ingest {rt['ingest_per_s']}/s.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--embedder", default=None, help="auto | fastembed[:MODEL] | onnx:DIR | hash[:DIM]"
    )
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    emb, note = make_embedder(args.embedder, hash_dim=384)
    print(f"embedder: {emb.name}  ({note})", flush=True)
    data = load()
    print({k: len(v) for k, v in data.items()}, flush=True)
    retrieval, gate = eval_retrieval(emb, data, args.queries, args.seed)
    R = {
        "embedder": emb.name,
        "embedder_real": emb.real,
        "embedder_note": note,
        "retrieval": retrieval,
        "gate": gate,
        "dedupe": eval_dedupe(emb, data),
        "sts_spearman": eval_sts(emb, data),
    }
    md = report(R)
    print(md)
    if not args.no_write:
        RESULTS.mkdir(exist_ok=True)
        tag = "" if emb.real else "_standin"
        (RESULTS / f"edge_real_eval{tag}.json").write_text(
            json.dumps(R, indent=2, default=float) + "\n"
        )
        (RESULTS / f"edge_real_eval{tag}.md").write_text(md)


if __name__ == "__main__":
    main()
