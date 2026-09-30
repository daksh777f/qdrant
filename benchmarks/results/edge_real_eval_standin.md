# Real-data evaluation

Embedder: **hash-bow-384 (stand-in, not a semantic model)** (NOT a semantic model: lexical stand-in). hash stand-in requested.
Datasets: STS-B test, MSRP test, SICK test (public, human-labelled, SHA-256 verified). Engine: Qdrant Edge on this machine.

## 1. Retrieval: 495 paraphrase queries, each with one correct answer among 9,225 real sentences

| mode | recall@1 | recall@10 | MRR@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| bm25 (Qdrant Edge sparse) | 0.899 | 0.976 | 0.926 | 0.64 | 1.26 |
| dense | 0.782 | 0.857 | 0.808 | 0.79 | 1.24 |
| hybrid RRF (dense + bm25) | 0.788 | 0.988 | 0.866 | 1.68 | 2.29 |

## 2. Duplicate detection vs. human judges (MSRP, 1,725 pairs, 66.5% judged same)

ROC-AUC **0.746**; best F1 0.821 at cosine 0.498.

| cosine threshold | pairs merged | precision (same) | false merges (of different pairs) | recall (same) |
|---:|---:|---:|---:|---:|
| 0.95 (DEDUPE) | 0.8% | 100.0% | 0.0% | 1.2% |
| 0.85 (look-alike band) | 14.2% | 90.6% | 4.0% | 19.4% |

Text similarity alone confuses related-but-different statements; on the device a merge additionally requires the same place (and the cloud the same time window), which this text-only test cannot credit.

## 3. Abstention: 995 queries, 495 answerable, the rest with their answer removed

Gate confidence separates answerable from unanswerable with ROC-AUC **0.894**. At threshold 0.5: answers 85.1% of answerable and 11.8% of unanswerable queries; 79.0% of given answers are the correct sentence.

## 4. Embedder sanity (Spearman vs. human similarity)

STS-B 0.488, SICK 0.525. This is a lexical stand-in, so low correlation is expected: it only matches shared words. Run with a real model (--embedder fastembed) for semantic results.

Device footprint of the evaluation shard: 9,225 memories, 68.9 MB on disk. Embedding 17136.2 sentences/s, ingest 3586.1/s.
