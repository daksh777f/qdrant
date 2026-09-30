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
- F7 Cloud AI loop: cloud consolidation across devices plus an LLM step (free-tier Groq / Cerebras / Gemini via
  their OpenAI-compatible endpoints, key from an env var, never committed, no paid services) used for
  (a) fleet summaries pushed down into mirrors and (b) answering `ESCALATE_CLOUD` queries. The LLM runs only on
  the cloud side, so the edge stays fully offline-capable; `private` memories are never sent to it. Without a
  key, or when rate-limited, a deterministic summarizer runs and the UI labels which one was used.
- F8 Mission-control web UI: Hilbert-cell map with novelty heatmap; outbox/sync status with **Cut network**
  button and bytes counter; decision explainer (sync and abstain/escalate decisions); conflict inbox;
  **sync-diff screen** (F12); memory browser with provenance badges (F11); search box showing confidence and route.
- F9 Measured numbers, one command (`make verify`): p50/p95 offline hybrid-search latency, recall@10 of the
  edge index vs. the full cloud index, bytes synced vs. naive sync at each novelty threshold (hero chart),
  sync convergence time after an outage, and a **negative control** (random memories must score as novel and
  must not be suppressed). Numbers are printed and written to `benchmarks/results/`.
- F10 **Abstain / escalate gate** (query-side local-vs-cloud decision): every answer carries a confidence
  computed from top score, margin over runner-up, calibrated novelty and staleness of the matched memory.
  Outcomes: `ANSWER_LOCAL` (confident), `ESCALATE_CLOUD` (low confidence and online: ask the cloud/LLM over
  fleet memory, then cache the answer back), `LOW_CONFIDENCE_OFFLINE` (low confidence and offline: answer is
  returned but explicitly flagged, never presented as certain). Thresholds come from the benchmark, not guesses.
- F11 **Provenance and versioning on every record**: `device_id`, `observed_at`, `confidence`, `sync_state`
  (`local_only` | `queued` | `synced` | `conflict`), monotonic `version`, and a `content_hash`. A per-shard
  **digest** (hash over sorted `id:version`) detects edge/server divergence cheaply.
- F12 **Sync-diff view**: local shard vs. cloud collection, computed from digests + versions, listing exactly
  what will be pushed, pulled, or reconciled. It is both the engine's plan and a UI screen.

**Stretch (only after F1-F9 are demoed end-to-end)**
- MCP tools over edge memory (`loci-mcp` already exists; point it at `EdgeStore`) so an LLM agent queries
  offline memory live.
- Decay-based archive of stale memories using Edge `Formula`/`DecayKind`.
- Binary quantization on the Edge shard with measured recall.

**Packaging (adopted, cheap):** one-command setup, a scripted and clearly labelled *synthetic* demo
("go offline, add data, reconnect, watch the sync"), and a README Limitations section stating what is
simulated (robot sensor stream, `LocalCloud`) and what is real (Edge shards, outbox, sync, benchmarks).

**Explicitly cut:** field-level 3-way merge, rollback, priority lanes beyond one urgent lane, redaction tiers,
device roster, chaos suite (keep one convergence test + one netsplit test), docker polish beyond a compose file.

## 6. Phase plan (Days counted from today; online round starts 3 Oct, offline finale 11 Oct)

| Phase | Days | Deliverable | Exit gate (must pass to continue) |
|-------|------|-------------|-----------------------------------|
| **P0 Vertical slice** | 1-2 | `EdgeStore` minimal + outbox + `LocalCloud`; script: write offline → queue → push → print bytes | Script runs end to end; existing 655 tests untouched and green |
| **P1 Edge backend, full** | 2-4 | F1+F2: full store surface, parametrized run of `tests/test_local_client.py` against Edge, hybrid search, Hilbert integer indexes | Local-client suite passes on Edge (document any principled diffs, e.g. scroll order) |
| **P2 Sync brain** | 4-6 | F3+F4+F5+F11+F12 (provenance, digests, diff): policy + decision records + backoff + push/pull + netsplit | Netsplit test: writes during outage, reconnect, both sides converge; decision log explains every item |
| **P3 UI thin** | 5-7 | F8 (parallel to P2 once APIs exist) | Judge can run the full story from the browser alone |
| **P4 Conflicts + cloud loop + gate** | 6-8 | F6+F7+F10 | Two simulated devices disagree; inbox shows resolution; cloud summary appears in the other device's mirror |
| **P5 Numbers + demo** | 8-9 | F9 (`make verify`, hero chart, negative control), README rewrite around the one thesis, 3-min script, **recorded video (insurance)** | Video recorded by Day 8 |
| **P6 Freeze / stretch** | 9-10 | Feature freeze Day 8; only bugfix + stretch (MCP, decay) | CI green, fresh-clone quickstart works with **no Docker** |

Schedule rule: UI must show *something real* by Day 6; polish is not left to the last day.

### Progress log
- **P0 done** (`loci/edge/`): Edge store, SQLite outbox, `LocalCloud`, idempotent push.
- **P1 done** (`loci/backends/edge.py`, `EdgeMemoryStore` mirror shard): the whole test suite passes on both the
  numpy and the Edge backend (`LOCI_TEST_BACKEND=edge pytest tests`), 713 passed / 10 skipped each. Hybrid search
  now spans writable shard + fleet mirror; delta pull lets robot B find what robot A saw. Measured (synthetic,
  50k x 384-d, HNSW built): dense p50 2.3 ms / p95 3.3 ms, hybrid p50 2.0 ms / p95 2.8 ms, recall@10 = 1.0 vs
  exact; space-filtered dense p50 7.3 ms (slower than unfiltered: known cost of the Hilbert MatchAny + exact
  range filter, to optimize later). Real Qdrant Server and partial-snapshot pull remain unverified.
- **P2 done** (`policy.py`, `decisions.py`, `sync.py`): surprise-driven policy (private / urgent / dedupe /
  moved / novelty / summarize / keep-local), judged against local shard **and** fleet mirror; every verdict is a
  persisted `Decision` with evidence and thresholds; summary lane (N observations -> <=2 centroid points);
  `SyncEngine.diff()` (push / pull / in-sync / reconcile / held-local); netsplit convergence test. Demo
  `examples/edge_p2_patrol.py` (synthetic, deterministic): 201 observations, 6 raw + 46 summarized, ~5.3 KB vs
  ~101 KB naive estimate (94.8% saved), robot-b finds robot-a's spill offline, private memory never leaves.
  Suite: 724 passed / 10 skipped on both backends. Not yet done: recall-vs-threshold sweep (P5 hero chart),
  real-server pull, conflict rules (P4).
- **P3 done** (`loci/edge/ui/`, `sim.py`, `embed.py`): mission-control web UI, run with
  `pip install -e ".[edge-ui]" && python -m loci.edge.ui` (http://127.0.0.1:8765). Panels: Hilbert-cell warehouse
  map shaded by mean novelty; per-robot **cut/restore network** button; controls (patrol, urgent spill, moved
  toolbox, private note, sync); decision feed with reasons and evidence; sync-diff (push / pull / in-sync /
  held-local / reconcile); hybrid search with source + provenance badges; fleet activity; bytes ledger (sent vs
  naive estimate, "saved" only shown once something is sent). Driven end to end in headless Chromium (offline ->
  restore -> converged; robot-b finds robot-a's spill offline in ~1 ms; no console errors; no horizontal scroll at
  390 px). Screenshot: `docs/assets/edge-mission-control.png`. The text embedder is a hashed bag-of-words stand-in
  (no model download); swap in a real model for real data. Not yet in the UI: conflict inbox and abstain/escalate
  route (P4).
- **P4 done** (`conflicts.py`, `cloud_ai.py`, `gate.py`, UI panels):
  * *Conflicts (F6)*: pure `resolve_entity` (order-independent; merged / previous / current) applied by a cloud
    `Reconciler`; duplicates from different devices in the same place + time window are merged (higher confidence,
    then newer, wins), moved objects keep both with the newest `current`; look-alikes in the 0.85-0.95 similarity
    band are **never** auto-merged and go to a human review inbox (approve / keep separate, remembered as
    operator links). Every change is audited. Verdicts propagate to devices via a role revision (`rrev`); own
    points keep their content/version. `search(current_only=True)` answers "where is it now?".
  * *Cloud AI loop (F7)*: `CloudBrain` writes per-area fleet briefings into the cloud as `kind="insight"` memories
    that the normal pull carries into every robot's mirror (searchable offline). Optional LLM through any
    OpenAI-compatible endpoint (GROQ_API_KEY / CEREBRAS_API_KEY / GEMINI_API_KEY or LOCI_LLM_*), cloud-side only,
    memory text passed as quoted data, output capped, deterministic fallback labelled in the UI.
    **The LLM path is verified only against a local fake OpenAI-compatible server**, not a live provider.
  * *Gate (F10)*: confidence = 0.5 similarity + 0.3 wording + 0.2 margin(x similarity) - staleness; routes
    ANSWER_LOCAL / ESCALATE_CLOUD (answer from cloud and cache into the mirror) / LOW_CONFIDENCE_OFFLINE (flagged
    guess) / ABSTAIN. Threshold 0.5 chosen from a measured sweep (stand-in embedder): 59 relevant queries p5 0.63,
    60 junk queries max 0.34, 0% junk answered for any threshold 0.4-0.6. Re-measure when swapping embedders.
  * *Policy*: a same-place look-alike of another device's mirrored memory is synced so the cloud can adjudicate.
  * Suite: 765 passed / 10 skipped on both backends. Screenshot: `docs/assets/edge-conflicts-and-gate.png`.
- **P5 done** (`benchmarks/edge_verify.py`, `Makefile`, `docs/EDGE_PLATFORM.md`, `docs/DEMO_SCRIPT.md`,
  `scripts/record_demo.py`, README section, CI `edge` job): `make verify` re-measures every claim (12 checks,
  about 40 s, non-zero exit on failure) and writes `benchmarks/results/edge_verify.{json,md}` plus the hero chart
  (two stacked panels, validated palette, light/dark). `make record` re-records the captioned 90 s walkthrough and
  doubles as an end-to-end UI check. Verified from a fresh non-editable install in a new venv (demo, verify,
  UI all work with no Docker/server/key), and in a CI-equivalent env (mypy clean, tests pass, edge suites skip).
  **The harness found three real defects that no unit test had**, all fixed with regression tests:
  (1) relative-only novelty made an all-new stream look "average" and suppressed 86% of genuinely new memories
  (negative control); (2) during calibrator warm-up a same-place look-alike was judged "familiar", kept local, and
  later views deduped against that never-synced item, so the cloud never learned the event (recall 70%);
  (3) `SyncPolicy` discarded a caller-supplied empty calibrator (`or` on an object with `__len__`).
  Design consequence, measured: the default now sends ambiguous look-alikes (14.7% of bytes, 100% of events);
  the old hold-back behaviour is the opt-in `hold_back_familiar` mode (12.5%, 95%). Place-aware dedupe vs a
  place-blind similarity dedupe at the default threshold: 100% vs 88% of events at 14.7% vs 10.0% of bytes.
  Also fixed: 25 mypy errors that would have failed CI's typecheck job, and a race in the demo story (auto-sync
  vs. narration) via `--no-autosync`.
- **Biggest remaining gap vs. the PS wording:** sync targets `LocalCloud`, not a Qdrant Server. Next (P6): a
  `QdrantServerCloud` implementing `CloudStore` with `qdrant-client` (testable here against
  `QdrantClient(":memory:")`), then partial-snapshot pull if a server is available.
- Fixed on the way: consolidation depended on store scroll order; it is now canonically sorted.

## 7. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Edge two-shard/snapshot behaves differently under write load | Day-1 gate; keep delta-pull path independent of snapshots |
| No Qdrant Server at demo | `LocalCloud` fallback behind same interface; recorded video |
| Hilbert reads as "robotics only" | One non-robot scenario in README/demo using text embeddings + place |
| Judges know memory-fleet | Do not clone it; differentiate on calibrated surprise + geometric conflicts; cite it honestly |
| Refactor breaks 655 tests | Third backend only; no changes to existing clients beyond an injectable store |

## 8. Decisions (confirmed by the team)
1. Hero scenario: **warehouse patrol robot**. A short non-robot variant stays in the README only.
2. LLM: free-tier keys only (Groq, Cerebras or Gemini); no paid services. Optional, with deterministic fallback.
3. Qdrant Server is **not guaranteed**: `LocalCloud` is the default demo path; `QdrantServerCloud` is optional
   and only exercised when a server is reachable.

## 9. Ideas adopted from the abstain-gate reference
| Idea | Decision |
|------|----------|
| Fail-closed / abstain gate | Adopted as F10: the principled query-side local-vs-cloud rule |
| Measured numbers, negative control, one command | Adopted into F9 |
| Provenance on every record | Adopted as F11 |
| Snapshot IDs / staleness fingerprints | Adopted as per-shard digest + per-point version/hash (F11) |
| Diff of two arms | Adopted as the sync-diff view (F12) |
| Multiple surfaces on one core | UI is required; MCP stays a cheap stretch |
| Honest packaging | Adopted (Packaging paragraph, section 5) |
