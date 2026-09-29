"""Surprise-driven sync policy.

An observation is judged against everything the device already knows: its own
shard *and* the fleet mirror. So what robot A saw is not novel to robot B.

Order of rules (first match wins), each recorded with its evidence:

1. ``private``                         -> KEEP_LOCAL   (never leaves the device)
2. ``metadata["urgent"]``              -> SYNC_NOW     (safety-critical jumps the queue)
3. near-identical *and* same place     -> DEDUPE       (already known; just count the sighting)
4. near-identical but a new place      -> SYNC_NOW     ("moved": the object changed location)
4b. look-alike of another device's memory, same place -> SYNC_NOW (identity is ambiguous; the
                                          cloud holds both sightings and can adjudicate)
5. calibrated novelty >= sync_novelty  -> SYNC_NOW
6. novelty >= summarize_novelty        -> SUMMARIZE_SYNC (batched into a summary, not raw)
7. otherwise                           -> KEEP_LOCAL   (familiar; not worth the bandwidth)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loci.edge.decisions import DEDUPE, KEEP_LOCAL, SUMMARIZE_SYNC, SYNC_NOW, Decision
from loci.edge.ids import memory_id
from loci.edge.store import EdgeMemoryStore, Memory
from loci.retrieval.novelty import NoveltyCalibrator


@dataclass
class PolicyConfig:
    dedupe_similarity: float = 0.95  # cosine at/above which two observations are "the same thing"
    same_place_radius: float = 0.05  # normalised distance below which two sightings share a place
    ambiguous_similarity: float = 0.85  # look-alike of another device's memory, same place
    sync_novelty: float = 0.6
    summarize_novelty: float = 0.3

    def as_dict(self) -> dict[str, float]:
        return {
            "dedupe_similarity": self.dedupe_similarity,
            "same_place_radius": self.same_place_radius,
            "ambiguous_similarity": self.ambiguous_similarity,
            "sync_novelty": self.sync_novelty,
            "summarize_novelty": self.summarize_novelty,
        }


class SyncPolicy:
    def __init__(
        self, config: PolicyConfig | None = None, calibrator: NoveltyCalibrator | None = None
    ) -> None:
        self.config = config or PolicyConfig()
        self.calibrator = calibrator or NoveltyCalibrator(window_size=200, min_samples=10)

    def decide(self, store: EdgeMemoryStore, mem: Memory) -> Decision:
        """Judge *mem* against the device's knowledge. Call **before** storing it."""
        cfg = self.config
        own_id = memory_id(mem.device_id or store.device_id, mem.key)
        base = Decision(
            action=KEEP_LOCAL,
            reason="",
            key=mem.key,
            point_id=own_id,
            thresholds=cfg.as_dict(),
            x=mem.x,
            y=mem.y,
            z=mem.z,
        )

        # Nearest known neighbour across local shard + fleet mirror (excluding itself).
        hits = [h for h in store.search(vector=mem.vector, limit=3) if h.id != own_id]
        best = hits[0] if hits else None
        if best is not None:
            base.best_similarity = round(best.score, 4)
            base.neighbor_id = best.id
            base.neighbor_source = best.source
            p = best.payload
            base.displacement = round(
                math.dist(
                    (mem.x, mem.y, mem.z), (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))
                ),
                4,
            )
            base.novelty = round(self.calibrator.calibrated_novelty(best.score), 4)
            self.calibrator.observe(best.score)
        else:
            base.novelty = 1.0

        if mem.private:
            base.action, base.reason = KEEP_LOCAL, "private: never leaves the device"
        elif mem.metadata.get("urgent"):
            base.action, base.reason = SYNC_NOW, "urgent: safety-critical, jumps the queue"
        elif best is None:
            base.action, base.reason = SYNC_NOW, "nothing similar known yet: first sighting"
        elif best.score >= cfg.dedupe_similarity:
            if base.displacement is not None and base.displacement <= cfg.same_place_radius:
                base.action = DEDUPE
                base.reason = (
                    f"same thing, same place (sim {best.score:.2f}, moved {base.displacement:.3f}) "
                    f"already known via {best.source}"
                )
            else:
                base.action = SYNC_NOW
                base.reason = (
                    f"moved: seen before (sim {best.score:.2f}) but {base.displacement:.2f} "
                    "away from where it was"
                )
        elif (
            best.source == "mirror"
            and best.score >= cfg.ambiguous_similarity
            and base.displacement is not None
            and base.displacement <= cfg.same_place_radius
        ):
            base.action = SYNC_NOW
            base.reason = (
                f"might be the same object as another device's sighting (sim {best.score:.2f}, "
                "same place): only the cloud can adjudicate"
            )
        elif base.novelty >= cfg.sync_novelty:
            base.action, base.reason = SYNC_NOW, f"novel (novelty {base.novelty:.2f})"
        elif base.novelty >= cfg.summarize_novelty:
            base.action = SUMMARIZE_SYNC
            base.reason = f"somewhat familiar (novelty {base.novelty:.2f}): summarize before sync"
        else:
            base.action = KEEP_LOCAL
            base.reason = f"familiar (novelty {base.novelty:.2f}): not worth the bandwidth"
        return base
