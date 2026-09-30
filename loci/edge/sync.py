"""Sync engine: surprise-driven push, summary lane, delta pull, and the sync-diff."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loci.edge.cloud import CloudStore, LinkDown, wire_bytes
from loci.edge.decisions import (
    DEDUPE,
    KEEP_LOCAL,
    SUMMARIZE_SYNC,
    SYNC_NOW,
    Decision,
    DecisionLog,
)
from loci.edge.ids import content_hash, memory_id
from loci.edge.outbox import Outbox
from loci.edge.policy import SyncPolicy
from loci.edge.store import ROLE_FIELDS, EdgeMemoryStore, Memory
from loci.schema import WorldState
from loci.temporal.consolidation import ConsolidationPolicy, consolidate_states


@dataclass
class PullReport:
    pulled: int = 0
    roles_updated: int = 0  # cloud verdicts (merged / previous / current) applied to our own points
    bytes_received: int = 0
    link_down: bool = False


@dataclass
class PushReport:
    sent: int = 0
    failed: int = 0
    skipped_private: int = 0
    bytes_sent: int = 0
    link_down: bool = False


@dataclass
class SummaryReport:
    groups: int = 0
    covered: int = 0  # raw observations replaced by summaries
    bytes_sent: int = 0
    link_down: bool = False


@dataclass
class SyncDiff:
    """What a sync would do right now: the engine's plan and the UI's diff screen."""

    cloud_reachable: bool = True
    to_push: list[str] = field(default_factory=list)
    to_pull: list[str] = field(default_factory=list)
    in_sync: list[str] = field(default_factory=list)
    reconcile: list[str] = field(default_factory=list)  # cloud copy is newer than ours
    role_updates: list[str] = field(default_factory=list)  # cloud verdicts we have not applied
    held_local: dict[str, str] = field(default_factory=dict)  # id -> why it stays here
    awaiting_summary: list[str] = field(default_factory=list)
    local_digest: str = ""

    @property
    def converged(self) -> bool:
        return self.cloud_reachable and not (
            self.to_push or self.to_pull or self.reconcile or self.role_updates
        )


class SyncEngine:
    """Moves memories between a device and a :class:`CloudStore`, deciding what is worth sending."""

    def __init__(
        self,
        store: EdgeMemoryStore,
        outbox: Outbox,
        cloud: CloudStore,
        *,
        policy: SyncPolicy | None = None,
        log: DecisionLog | None = None,
        batch_size: int = 64,
    ) -> None:
        self.store = store
        self.outbox = outbox
        self.cloud = cloud
        self.policy = policy or SyncPolicy()
        self.log = log
        self._batch = batch_size

    # -- decide -------------------------------------------------------------

    def observe(self, mem: Memory, now_ms: int | None = None) -> Decision:
        """The main entry point: judge an observation, then store/queue/merge accordingly."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        d = self.policy.decide(self.store, mem)
        if d.action == DEDUPE:
            if d.neighbor_source == "local" and d.neighbor_id:
                self.store.bump_seen(d.neighbor_id, now)
        else:
            stored = self.store.put(mem)
            d.point_id = stored.id
            if d.action == SYNC_NOW:
                self.outbox.enqueue(stored.id, SYNC_NOW, now)
                self.store.set_sync_state([stored.id], "queued")
            elif d.action == SUMMARIZE_SYNC:
                self.outbox.enqueue(stored.id, SUMMARIZE_SYNC, now)
                self.store.set_sync_state([stored.id], "queued_summary")
        if self.log is not None:
            d.ts_ms = now
            self.log.add(d)
        return d

    def submit(self, point_id: str, private: bool = False) -> str:
        """Low-level: enqueue an already-stored memory for sync (private ones are refused)."""
        if private:
            return KEEP_LOCAL
        self.outbox.enqueue(point_id, SYNC_NOW)
        self.store.set_sync_state([point_id], "queued")
        return SYNC_NOW

    # -- push ---------------------------------------------------------------

    def push(self, now_ms: int | None = None) -> PushReport:
        """Send every due SYNC_NOW row. Safe to call repeatedly, online or not."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        report = PushReport()
        while True:
            rows = self.outbox.due(now, limit=self._batch, decisions=(SYNC_NOW,))
            if not rows:
                return report
            ids = [r.point_id for r in rows]
            points = self.store.read_for_push(ids)
            found = {p["id"] for p in points}
            # Rows whose point vanished locally are dropped, not retried forever.
            self.outbox.mark_sent([i for i in ids if i not in found])
            points = [p for p in points if not p["payload"].get("private")]
            report.skipped_private += len(found) - len(points)
            try:
                if points:
                    self.cloud.upsert(points)
            except LinkDown:
                self.outbox.mark_failed(ids, now)
                report.failed += len(ids)
                report.link_down = True
                return report
            except Exception:
                self.outbox.mark_failed(ids, now)
                report.failed += len(ids)
                return report
            sent_ids = [p["id"] for p in points]
            self.outbox.mark_sent(ids)
            self.store.set_sync_state(sent_ids, "synced")
            report.sent += len(sent_ids)
            report.bytes_sent += sum(wire_bytes(p) for p in points)

    def summarize_pending(
        self, *, max_states_per_group: int = 2, now_ms: int | None = None
    ) -> SummaryReport:
        """Replace queued SUMMARIZE_SYNC observations with a few centroid summaries.

        This is where bandwidth is saved: N somewhat-familiar observations of a
        scene cross the wire as at most ``max_states_per_group`` summary points.
        The raw observations stay on the device (``sync_state="summarized"``).
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        report = SummaryReport()
        rows = self.outbox.due(now, limit=10_000, decisions=(SUMMARIZE_SYNC,))
        if not rows:
            return report
        ids = [r.point_id for r in rows]
        points = [p for p in self.store.read_for_push(ids) if not p["payload"].get("private")]
        groups: dict[str, list[dict]] = {}
        for p in points:
            groups.setdefault(str(p["payload"].get("metadata", {}).get("scene", "")), []).append(p)

        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=max_states_per_group)
        summary_points: list[dict] = []
        for scene, members in sorted(groups.items()):
            members.sort(key=lambda p: (p["payload"]["timestamp_ms"], p["id"]))
            states = [
                WorldState(
                    x=p["payload"]["x"],
                    y=p["payload"]["y"],
                    z=p["payload"]["z"],
                    timestamp_ms=p["payload"]["timestamp_ms"],
                    vector=p["vector"],
                    scene_id=scene,
                    confidence=p["payload"].get("confidence", 1.0),
                )
                for p in members
            ]
            summaries = consolidate_states(states, policy, seed=0)
            member_ids = [p["id"] for p in members]
            for n, s in enumerate(summaries):
                key = f"summary:{scene}:{n}:{content_hash(','.join(member_ids), s.vector)}"
                sid = memory_id(self.store.device_id, key)
                summary_points.append(
                    {
                        "id": sid,
                        "vector": s.vector,
                        "payload": {
                            "key": key,
                            "text": f"summary of {s.metadata.get('source_count', 0)} observations"
                            + (f" in {scene}" if scene else ""),
                            "x": s.x,
                            "y": s.y,
                            "z": s.z,
                            "timestamp_ms": s.timestamp_ms,
                            "device_id": self.store.device_id,
                            "confidence": s.confidence,
                            "private": False,
                            "summary": True,
                            "source_count": s.metadata.get("source_count", 0),
                            # Provenance stays compact: a count, a hash of the members, a sample.
                            "members_hash": content_hash(",".join(member_ids), [])[:12],
                            "member_sample": member_ids[:5],
                            "version": 1,
                            "content_hash": content_hash(key, s.vector),
                        },
                    }
                )
        try:
            if summary_points:
                self.cloud.upsert(summary_points)
        except LinkDown:
            self.outbox.mark_failed(ids, now)
            report.link_down = True
            return report
        self.outbox.mark_sent(ids)
        self.store.set_sync_state([p["id"] for p in points], "summarized")
        report.groups = len(groups)
        report.covered = len(points)
        report.bytes_sent = sum(wire_bytes(p) for p in summary_points)
        return report

    # -- pull ---------------------------------------------------------------

    def pull(self) -> PullReport:
        """Delta-pull the cloud's changes.

        * Other devices' memories (and cloud insights) go into the fleet mirror when
          missing or when the role revision is newer.
        * For our own points, only the cloud's role verdict (merged / previous /
          current) is applied; our content and version are never overwritten.

        Offline, it reports ``link_down`` and changes nothing.
        """
        report = PullReport()
        try:
            remote = self.cloud.index()
            have = self.store.mirror_state()
            mine = self.store.states()
            me = self.store.device_id
            want, role_ids = [], []
            for pid, (ver, dev, rrev) in remote.items():
                if dev == me:
                    if pid in mine and rrev > mine[pid]["rrev"]:
                        role_ids.append(pid)
                elif have.get(pid, (0, 0)) < (ver, rrev):
                    want.append(pid)
            for start in range(0, len(want), self._batch):
                points = self.cloud.get(want[start : start + self._batch])
                self.store.mirror_upsert(points)
                report.pulled += len(points)
                report.bytes_received += sum(wire_bytes(p) for p in points)
            if role_ids:
                updates = {}
                for p in self.cloud.get(role_ids):
                    fields = {k: p["payload"].get(k) for k in ROLE_FIELDS}  # None clears a verdict
                    fields["rrev"] = p["payload"].get("rrev", 0)
                    updates[p["id"]] = fields
                self.store.apply_cloud_roles(updates)
                report.roles_updated = len(updates)
        except LinkDown:
            report.link_down = True
        return report

    # -- diff ---------------------------------------------------------------

    def diff(self) -> SyncDiff:
        """Compare local shard, fleet mirror and cloud, without changing anything."""
        out = SyncDiff(local_digest=self.store.digest())
        states = self.store.states()
        try:
            remote = self.cloud.index()
        except LinkDown:
            remote = None
            out.cloud_reachable = False
        for pid, st in states.items():
            if st["private"]:
                out.held_local[pid] = "private"
            elif st["sync_state"] == "local_only":
                out.held_local[pid] = "familiar: kept local by policy"
            elif st["sync_state"] == "summarized":
                out.held_local[pid] = "covered by a synced summary"
            elif st["sync_state"] == "queued_summary":
                out.awaiting_summary.append(pid)
            elif remote is not None:
                rv = remote.get(pid, (0, "", 0))[0]
                if rv < st["version"]:
                    out.to_push.append(pid)
                elif rv == st["version"]:
                    out.in_sync.append(pid)
                else:
                    out.reconcile.append(pid)
            else:
                out.to_push.append(pid)  # cannot verify offline: still pending
        if remote is not None:
            have = self.store.mirror_state()
            me = self.store.device_id
            for pid, (ver, dev, rrev) in remote.items():
                if dev != me and have.get(pid, (0, 0)) < (ver, rrev):
                    out.to_pull.append(pid)
                elif dev == me and pid in states and rrev > states[pid]["rrev"]:
                    out.role_updates.append(pid)
        return out
