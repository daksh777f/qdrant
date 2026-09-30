# LOCI Edge: offline-first memory and intelligence on Qdrant Edge

> **A robot's memory is a place and a moment, so deciding what to send, what is a duplicate,
> and what is a conflict are geometric questions. Only surprise crosses the wire.**

This is the answer to *Problem Statement 03: AI-Powered Edge Memory & Intelligence Platform*,
built on the LOCI spatiotemporal memory engine. Each device keeps a searchable Qdrant Edge
shard, works fully offline, decides what is worth syncing, reconciles conflicting sightings
by place and time, and gets fleet-level intelligence pushed back down from the cloud.

![Bytes sent and events kept, place-aware vs place-blind dedupe](assets/edge-hero.svg)

<sub>Synthetic, seeded stream (500 observations, 40 real events). Reproduce with `make verify`.
Place-blind dedupe is what a plain similarity threshold does. Both send about a tenth of the
bytes at the default setting, but only place-aware dedupe keeps every event.</sub>

## Quickstart (no Docker, no Qdrant Server, no API key)

```bash
git clone https://github.com/daksh777f/qdrant.git && cd qdrant
make setup      # pip install -e ".[dev,edge-ui]"
make verify     # about 40 s: re-measures every claim below, non-zero exit if one fails
make ui         # mission control at http://127.0.0.1:8765
make demo       # two terminal demos
```

Walkthrough: [DEMO_SCRIPT.md](DEMO_SCRIPT.md). A scripted recording (about 90 s) is at
[assets/edge-demo.webm](assets/edge-demo.webm); `make record` regenerates it and doubles as an
end-to-end check of the UI story.

## The eight goals, and where each is met

| PS03 goal | What implements it | Evidence |
|---|---|---|
| Searchable semantic memory on the device | `EdgeMemoryStore`: a Qdrant Edge shard with dense + BM25 vectors, Hilbert-bucket integer indexes, provenance on every record | `tests/test_edge_p0.py`; the whole existing suite also passes on Edge (`make test BACKEND=edge`) |
| Low-latency vector and hybrid search offline | dense + BM25 with RRF over the writable shard and the fleet mirror | p95 2.1 ms hybrid, 1.3 ms dense at 5,000 x 384-d; recall@10 = 1.0 vs exact search |
| Decide what stays local and what syncs | `SyncPolicy`: private / urgent / duplicate / moved / new / ambiguous, each verdict stored with its evidence | decision feed in the UI; `tests/test_edge_p2.py` |
| Sync with the server when connectivity returns | durable SQLite `Outbox` (backoff + jitter, survives restarts), idempotent push, delta pull, to a local stand-in **or a Qdrant Server** (`QdrantServerCloud`) | two robots converge in about 0.3 s after a 150-observation outage; identical results on both clouds |
| Intermittent connectivity | every read and write is local; pushes queue and retry; `Link` switch simulates cuts | "Cut the network" button; netsplit tests |
| Evolving and conflicting memory | place-and-time reconciler (merge, moved, needs-review), audit log, operator inbox, versioned records | 80/80 labelled pairs correct; `tests/test_edge_conflicts.py` |
| User-facing inspection | mission-control web UI: map, decisions, sync diff, search with confidence, conflict inbox, activity | `make ui`; browser-tested |
| A real edge-to-cloud AI workflow | cloud briefings pushed into every robot's mirror (optional LLM); confidence gate that escalates to the cloud and caches the answer | `tests/test_edge_ai.py` |

## How the decisions are made

**Sync policy** (first match wins; every verdict is stored with similarity, distance, thresholds):

| Situation | Verdict |
|---|---|
| marked private | `KEEP_LOCAL`, never leaves the device (enforced again at push) |
| marked urgent | `SYNC_NOW` |
| near-identical to something known, same place | `DEDUPE` (only a sighting counter changes) |
| near-identical but somewhere else | `SYNC_NOW` ("moved") |
| nothing known is similar (cosine < 0.85) | `SYNC_NOW` (new; an absolute rule, see below) |
| look-alike (0.85 to 0.95): different place, another device's memory, or the default | `SYNC_NOW` (identity is ambiguous; the cloud adjudicates) |
| opt-in `hold_back_familiar` mode: familiar re-view of this device's own memory | `SUMMARIZE_SYNC` or `KEEP_LOCAL` |

The novelty score is measured against the local shard **and** the fleet mirror, so what robot A
saw is not new to robot B. The "new" rule is absolute because calibrated novelty is relative to
recent history: in a stream where everything is new, "new" looks average. The verification
harness caught exactly that bug.

**Conflicts** (cloud side, order-independent, audited): duplicates in the same place and time
window merge (higher confidence wins, then newer); the same thing somewhere else keeps both and
the newest is `current`, so "where is it now?" is one filter; look-alikes between 0.85 and 0.95
are never merged automatically and wait in a review inbox.

**Answer gate**: confidence = 0.5 similarity + 0.3 wording overlap + 0.2 margin (scaled by
similarity) minus a staleness penalty, then one of `ANSWER_LOCAL`, `ESCALATE_CLOUD` (answer from
the cloud and cache it into the mirror), `LOW_CONFIDENCE_OFFLINE` (flagged guess) or `ABSTAIN`.

## Measured results (`make verify`, full run)

| Check | Result |
|---|---|
| Recall@10 of the edge index vs exact search | 1.0 |
| Offline hybrid search latency | p50 1.6 ms, p95 2.1 ms |
| Bytes sent at the default setting | 14.7% of sending everything (floor for this stream: 8.0%) |
| Real events still retrievable at that setting | 100% (place-blind dedupe: 88%) |
| Most aggressive setting tested (cosine 0.70) | place-aware 90% vs place-blind 55% of events |
| Genuinely new random memories synced (negative control) | 100% |
| Repeated views kept off the wire | 99.7% |
| Conflict engine on labelled pairs (duplicate, moved, distinct, ambiguous) | 80/80 |
| Junk questions answered / relevant questions answered | 0% / 98.3% |
| Private memories that reached the cloud when force-queued | 0 of 15 |

Full tables: [`benchmarks/results/edge_verify.md`](../benchmarks/results/edge_verify.md).

**The trade-off, quantified.** At the default threshold the opt-in `hold_back_familiar` mode
sends 12.5% of bytes and keeps 95% of events, against 14.7% and 100% by default. Two percentage
points of bandwidth are not worth silently losing real events, so recall-first is the default.

## Running against a real Qdrant Server

```bash
make qdrant-up                  # docker run qdrant/qdrant on :6333
make verify-server              # the whole harness with Qdrant Server as the cloud
make ui-server                  # mission control on it; the "cloud:" chip shows which cloud is live
make test-server                # the cloud contract suite against the live server
```

or set `LOCI_QDRANT_URL` (plus `LOCI_QDRANT_API_KEY` for a secured server or Qdrant Cloud, and
`LOCI_QDRANT_COLLECTION` to name the collection) for any of the commands above. If a URL is set but
the server is unreachable, the demo falls back to the local stand-in **and says so** in the UI.

`QdrantServerCloud` (`loci/edge/cloud_server.py`) implements the same cloud contract as the local
stand-in on one collection, using `qdrant-client` >= 1.10. A refused connection, timeout or
502/503/504/429 is raised as `LinkDown`, so a server outage is indistinguishable from a network cut
to the robot: the outbox keeps everything and retries. Other errors (a wrong API key, say) are
not disguised as outages.

**What has and has not been verified.** No Docker daemon or Qdrant binary was available while this
was built, so it has *not* been run against a live server here. What has been done:

* one contract suite (`tests/test_edge_cloud_contract.py`) runs against both cloud backends, and
  they share a single implementation of the write rules (`plan_upsert`), so they cannot drift;
* the whole platform runs on the client path: the 12-check harness and the full UI story pass with
  `LOCI_QDRANT_URL=":memory:"` (qdrant-client's in-process engine, same API surface) with results
  identical to the local stand-in;
* the real HTTP error paths are tested (connection refused, and a fake server answering 503 and 401);
* the same contract suite runs against a live server when `LOCI_TEST_QDRANT_URL` is set. **That is
  the check still owed:** run `make qdrant-up test-server verify-server` once on a machine with Docker.

Not implemented: Qdrant Edge's snapshot-based shard sync (`update_from_snapshot`). Devices pull a
version-based delta instead, which works with any cloud but moves per-point data rather than
segment files.

## Configuration

* `PolicyConfig`: `dedupe_similarity` (0.95), `same_place_radius` (0.05), `ambiguous_similarity`
  (0.85), `hold_back_familiar` (off). `GateConfig.answer_threshold` (0.5).
* Optional LLM for cloud briefings (cloud side only; private memories never reach it):
  set `GROQ_API_KEY`, `CEREBRAS_API_KEY` or `GEMINI_API_KEY` (all have free tiers), or
  `LOCI_LLM_BASE_URL` + `LOCI_LLM_API_KEY` (+ `LOCI_LLM_MODEL`). Without a key, or on any error, a
  deterministic writer produces the briefing and the UI labels which one was used.
* `python -m loci.edge.ui --no-autosync` syncs only when "Sync now" is pressed (deterministic demos).

## What is real and what is simulated

Real: the Qdrant Edge shards, hybrid search, Hilbert indexes, the SQLite outbox and retries, the
sync policy, the reconciler, the gate, the UI, the tests and the measurements.

Simulated: the robots' "cameras" (seeded noisy vectors), the network (a `Link` switch), and the
cloud by default (`LocalCloud`, a second Edge shard behind the `CloudStore` interface; a Qdrant Server
is one environment variable away). The default text
embedder is a hashed bag-of-words stand-in so the demo needs no model download.

## Known limitations

* **Not yet run against a live Qdrant Server.** The client exists and is tested as described above, but
  the HTTP path against a real server (and snapshot-based sync) is unverified. See the section above.
* **LLM path verified only against a local fake server**, not a live provider. Model names change;
  override with `LOCI_LLM_MODEL`.
* **Same-place look-alikes are intrinsically ambiguous.** The default sends them so the cloud can
  ask a human; that costs bytes.
* Thresholds are tuned on synthetic data with a stand-in embedder. With a real model, re-run
  `make verify` and adjust `similarity_ref` and the dedupe threshold.
* The reconciler scans the whole cloud on every pass (fine for demos, not for large fleets), and the
  UI/API have no authentication (bind to localhost).
* Everything runs in one process; devices are not separate machines.

## Where things live

`loci/edge/` (store, outbox, cloud, sync, policy, conflicts, cloud_ai, gate, sim, ui) ·
`loci/backends/edge.py` (Edge backend for the existing client) · `benchmarks/edge_verify.py` ·
`tests/test_edge_*.py` · `scripts/record_demo.py` · [plan and decisions](PS03_GAP_ANALYSIS_AND_PLAN.md).
