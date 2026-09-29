# PS03 Gap Analysis and Implementation Plan

Problem Statement 03: **AI-Powered Edge Memory & Intelligence Platform** (Qdrant).
Written 2026-09-29 from a full read of this repo (baseline: 655 tests pass, 10 skipped),
the PS PDF, and a scan of public competing entries.

## 1. What the PS demands (8 goals) vs. what we have today

| # | PS goal | Status today | Evidence |
|---|---------|--------------|----------|
| 1 | Searchable semantic memory **on an edge device** | **Missing.** No Qdrant Edge anywhere. Local mode is a numpy `MemoryStore`; production mode is `qdrant-client` against a server. | `loci/backends/` only has `memory.py`; `grep -i edge` finds nothing in `loci/` |
| 2 | Low-latency vector **and hybrid** search offline | **Partial.** Vector + Hilbert/time filters work. No sparse/BM25, no fusion, no text search. | `LocalLociClient.query_scored` is dense-only |
| 3 | Decide what stays local vs. syncs | **Partial (raw material only).** `NoveltyCalibrator` and `predict_and_retrieve` give a calibrated novelty score, but nothing consumes it for a sync decision. No privacy/sensitivity flag. | `loci/retrieval/novelty.py` |
| 4 | Sync edge ↔ Qdrant Server on reconnect | **Missing.** `cloud_transport.py` is a synchronous HTTP call for `insert`/`query` only. No queue, no retry-across-restarts, no pull direction. | `loci/cloud_transport.py` |
| 5 | Intermittent connectivity | **Missing.** `retry.py` retries transient errors in-process; nothing survives a restart or outage. | `loci/retry.py` |
| 6 | Evolving / conflicting memory | **Partial.** Consolidation + retention age memory. No update semantics, no conflict detection across devices, no audit trail. | `temporal/consolidation.py`, `retention.py` |
| 7 | UI: memory, search results, sync status, activity | **Missing.** `demo_spatial` is a camera/VLM/voice demo and `demo/` is a robot simulation. Neither shows sync state, outbox, or decisions. | `demo_spatial/app/main.py` endpoints |
| 8 | Meaningful edge-to-cloud **AI workflow** | **Missing.** No cloud-side intelligence and no cloud→edge flow. | `cloud/api/server.py` is insert/query/admin only |

**Bottom line:** we have a strong, unique *memory core* (goals 3 and 6 raw material), and none of the
edge/sync/UI/cloud-loop layers that the PS actually grades. We currently would not satisfy 5 of the 8 goals.

### Correction to the earlier hand-off notes
The note says the storage layer needs "about eight operations". `MemoryStore` actually exposes more
(`delete_points_in_time_range`, `payload_value_range`, `collection_count`, `total_points`, etc.). Budget for
the full surface. The note's Edge adapter (`edge.py`) and "159/163 pass" result were **not** in the
attachments, so that number is unverified. We rebuild and re-measure it ourselves.

## 2. Verified facts about Qdrant Edge (spike, qdrant-edge-py 0.8.0)

Confirmed working in this environment:
- `EdgeShard.create/load`, `update(UpdateOperation.upsert_points)`, `query(QueryRequest)`.
- Named dense + sparse vectors; `Bm25` embedder built in (no model download).
- Hybrid: `Prefetch` (dense) + `Prefetch` (BM25) + `Fusion.Rrf(k=60)` returned correct fused order.
- `create_field_index("hb", PayloadSchemaType.Integer)` and `MatchAny` filtering (our Hilbert bucket path).
- `snapshot_manifest()`, `unpack_snapshot(path, target)`, `update_from_snapshot(path)` exist.
- Decay/`Formula` expressions and `DecayKind` exist (for stale-memory scoring).

**Not yet verified (must be Day-1 gates):** partial-snapshot pull from a real Qdrant Server; two shards under
write load; and whether a Qdrant Server is reachable in the demo environment (assume it is *not*).

## 3. Competitive landscape (public repos, ideas only)

| Entry | What it does well | What it lacks (our opening) |
|-------|-------------------|-----------------------------|
| qdrant-labs/memory-fleet | Two-shard pattern, opt-in push, fleet-sleep decay merge, ops console, 30 s mirror refresh | Conflict policy is just "local wins"; no novelty-driven sync; webcam objects only |
| EdgeQ (rohanjain1648) | 4-way policy (KEEP_LOCAL / DEDUPE / SUMMARIZE_SYNC / SYNC_NOW), SQLite outbox, uuid5 deterministic IDs, sensitivity levels, LLM summaries | Novelty is plain similarity, not calibrated or spatial; the "fleet utility" score is heuristic |
| Smriti (CTAGRAM) | Field-level 3-way merge, PII redaction, priority sync for urgent notes, p50/p95 numbers on 10k notes | Domain-specific (health); no spatial/temporal reasoning |
| EDGE.MEM (hemv-857) | Dense+BM25 RRF, residency policy, peer federation, 124 automated checks incl. Playwright | No novelty scoring; note-centric |
| EdgeMind (Diptanshu-215) | Writable + per-site mirror shards, adaptive full/delta/partial sync, CLIP photos, 0.3 s propagation | No principled "what to sync" model |
| Memora-Edge | Chat + voice + local-only flag | Thin sync logic |

Everyone ticks "outbox + privacy flag + hybrid search + console". **Parity on those does not win.**
What nobody has: a *calibrated, spatiotemporal* notion of surprise driving sync, and *geometric* conflict
resolution ("same place + same time + same thing = same observation"). That is our differentiator.

Licensing: several of these repos have no visible license. Take **ideas and system design only**; do not copy code.

## 4. Council critique (five advisors, run on this plan)

Convergent findings (high confidence, all five agreed):
1. **Goal 8 was absent from our plan.** Novelty-scored policy is local scoring, not a cloud AI workflow.
   Fix: cloud consolidates fleet memories (reuse `consolidation.py`), optionally LLM-summarizes, and pushes
   summaries/"priors" **down** into each device's read-only mirror shard.
2. **One memorable claim, proven live with numbers.** Thesis: *"A robot's memory is a place and a moment, so
   novelty, dedup and conflict are geometric operations. Only surprise crosses the wire."*
   Hero result: bytes-synced vs. novelty threshold with recall@k staying flat vs. naive sync.
3. **Over-scoped originals:** field-level 3-way merge, rollback, decay sleep, priority lanes, redaction tiers,
   quantization tuning, chaos suite, device roster, docker polish. Smriti/memory-fleet/EDGE.MEM already own
   those. Cut to stubs or a single boolean `private` flag.
4. **No-server fallback is mandatory.** Put a `CloudStore` interface in from day 1 with `QdrantServerCloud`
   and `LocalCloud` (second Edge shard / qdrant-client local path). Demo must run with no Docker.
5. **Riskiest assumption:** that Edge two-shard + sync integrates cleanly under 655 tests. Mitigation: freeze
   the store interface, add Edge as a **third backend**, no refactor of existing clients.
6. **Story clarity:** pick ONE concrete scenario (warehouse robot loses Wi-Fi, keeps remembering, reconciles
   "the pallet moved" on reconnect) and make every UI element serve it. Add one crisp non-robot variant
   (field inspector notes) using the same primitives so judges don't see "robotics tool bolted on".
7. **Demo insurance:** pre-record a video by Day 8; rehearse the offline→reconnect moment most.

Where advisors disagreed: whether to lead with novelty (Expansionist, First-Principles) or with space-time
conflict resolution (Outsider). **Decision: lead with the surprise-driven-sync thesis; show conflict resolution
as the second act** ("and when two robots disagree, geometry resolves it").

## 5. Feature set (final, cut down)

**Must-have (grades goals 1-8)**
- F1 `EdgeStore`: Qdrant Edge backend behind the existing store interface. Named vectors `dense` + `bm25`,
  integer payload indexes on Hilbert bucket fields, writable shard + read-only mirror shard searched together.
- F2 Hybrid search API: dense + BM25 with RRF, combined with space/time filters, latency recorded per query.
- F3 Durable SQLite outbox: idempotent deterministic point IDs, exponential backoff with jitter, survives
  restart, batched push, `netsplit` toggle for the demo.
- F4 **Surprise-driven sync policy** with explainable **decision records** (KEEP_LOCAL / DEDUPE /
  SUMMARIZE_SYNC / SYNC_NOW) using calibrated novelty (computed against local shard **and fleet mirror**, so
  what device A saw is not novel to device B). One boolean `private` flag pins items local.
- F5 `CloudStore` interface + `QdrantServerCloud` + `LocalCloud`; push and pull (delta via version/updated_at;
  partial snapshot when a real server is available).
- F6 Space-time conflict engine, two rules only: (a) near-duplicate vector + same Hilbert cell + time window
  from different devices = merge (higher confidence wins, sightings unioned); (b) same object in a different
  cell = "moved": keep both, newest wins for "where is it now?". Every resolution written to an audit log.
- F7 Cloud AI loop: cloud consolidation across devices (+ optional LLM summary, deterministic fallback when no
  API key), pushed down as read-only summaries into mirrors.
- F8 Mission-control web UI (4 panels): Hilbert-cell map with novelty heatmap; outbox/sync status with
  **Cut network** button and bytes counter; decision explainer; conflict inbox. Plus memory browser and search box.
- F9 One benchmark: p50/p95 offline hybrid-search latency, bytes synced vs. naive sync, recall@10 retained.

**Stretch (only after F1-F9 are demoed end-to-end)**
- MCP tools over edge memory (`loci-mcp` already exists; point it at `EdgeStore`) so an LLM agent queries
  offline memory live.
- Decay-based archive of stale memories using Edge `Formula`/`DecayKind`.
- Binary quantization on the Edge shard with measured recall.

**Explicitly cut:** field-level 3-way merge, rollback, priority lanes beyond one urgent lane, redaction tiers,
device roster, chaos suite (keep one convergence test + one netsplit test), docker polish beyond a compose file.

## 6. Phase plan (Days counted from today; online round starts 3 Oct, offline finale 11 Oct)

| Phase | Days | Deliverable | Exit gate (must pass to continue) |
|-------|------|-------------|-----------------------------------|
| **P0 Vertical slice** | 1-2 | `EdgeStore` minimal + outbox + `LocalCloud`; script: write offline → queue → push → print bytes | Script runs end to end; existing 655 tests untouched and green |
| **P1 Edge backend, full** | 2-4 | F1+F2: full store surface, parametrized run of `tests/test_local_client.py` against Edge, hybrid search, Hilbert integer indexes | Local-client suite passes on Edge (document any principled diffs, e.g. scroll order) |
| **P2 Sync brain** | 4-6 | F3+F4+F5: policy + decision records + backoff + push/pull + netsplit | Netsplit test: writes during outage, reconnect, both sides converge; decision log explains every item |
| **P3 UI thin** | 5-7 | F8 (parallel to P2 once APIs exist) | Judge can run the full story from the browser alone |
| **P4 Conflicts + cloud loop** | 6-8 | F6+F7 | Two simulated devices disagree; inbox shows resolution; cloud summary appears in the other device's mirror |
| **P5 Numbers + demo** | 8-9 | F9 hero chart, README rewrite around the one thesis, 3-min script, **recorded video (insurance)** | Video recorded by Day 8 |
| **P6 Freeze / stretch** | 9-10 | Feature freeze Day 8; only bugfix + stretch (MCP, decay) | CI green, fresh-clone quickstart works with **no Docker** |

Schedule rule: UI must show *something real* by Day 6; polish is not left to the last day.

## 7. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Edge two-shard/snapshot behaves differently under write load | Day-1 gate; keep delta-pull path independent of snapshots |
| No Qdrant Server at demo | `LocalCloud` fallback behind same interface; recorded video |
| Hilbert reads as "robotics only" | One non-robot scenario in README/demo using text embeddings + place |
| Judges know memory-fleet | Do not clone it; differentiate on calibrated surprise + geometric conflicts; cite it honestly |
| Refactor breaks 655 tests | Third backend only; no changes to existing clients beyond an injectable store |

## 8. Open questions for the team
1. Which single hero scenario: **warehouse patrol robot** (recommended, matches LOCI) or field-inspector notes?
2. Is an Anthropic/other LLM API key allowed in the demo, or must the cloud summarizer be fully deterministic?
3. Is Docker/Qdrant Server guaranteed at the offline finale (11 Oct), or must everything run bare?
