"""Push sync: drain the outbox to the cloud, idempotently, surviving outages."""

from __future__ import annotations

import time
from dataclasses import dataclass

from loci.edge.cloud import CloudStore, LinkDown, wire_bytes
from loci.edge.outbox import Outbox
from loci.edge.store import EdgeMemoryStore


@dataclass
class PullReport:
    pulled: int = 0
    bytes_received: int = 0
    link_down: bool = False


@dataclass
class PushReport:
    sent: int = 0
    failed: int = 0
    skipped_private: int = 0
    bytes_sent: int = 0
    link_down: bool = False


class SyncEngine:
    """Moves queued memories from the edge shard to a :class:`CloudStore`.

    P0 policy is deliberately trivial (everything not ``private`` is queued);
    the novelty-driven policy replaces :meth:`submit`'s decision in P2.
    """

    def __init__(
        self,
        store: EdgeMemoryStore,
        outbox: Outbox,
        cloud: CloudStore,
        *,
        batch_size: int = 64,
    ) -> None:
        self.store = store
        self.outbox = outbox
        self.cloud = cloud
        self._batch = batch_size

    def submit(self, point_id: str, private: bool = False) -> str:
        """Decide + enqueue one stored memory. Private memories never leave the device."""
        if private:
            return "KEEP_LOCAL"
        self.outbox.enqueue(point_id, "SYNC_NOW")
        self.store.set_sync_state([point_id], "queued")
        return "SYNC_NOW"

    def push(self, now_ms: int | None = None) -> PushReport:
        """Send every due outbox row. Safe to call repeatedly, online or not."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        report = PushReport()
        while True:
            rows = self.outbox.due(now, limit=self._batch)
            if not rows:
                return report
            ids = [r.point_id for r in rows]
            points = self.store.read_for_push(ids)
            found = {p["id"] for p in points}
            # Rows whose point vanished locally (deleted) are dropped, not retried forever.
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

    def pull(self) -> PullReport:
        """Delta-pull other devices' memories into the fleet mirror.

        Compares the cloud's ``{id: version}`` map with the mirror's, skips this
        device's own points (already local), and fetches only what is missing
        or newer. Offline, it reports ``link_down`` and changes nothing.
        """
        report = PullReport()
        try:
            remote = self.cloud.versions()
            have = self.store.mirror_versions()
            own = set(self.store.versions())
            want = [i for i, v in remote.items() if i not in own and have.get(i, 0) < v]
            for start in range(0, len(want), self._batch):
                points = self.cloud.get(want[start : start + self._batch])
                self.store.mirror_upsert(points)
                report.pulled += len(points)
                report.bytes_received += sum(wire_bytes(p) for p in points)
        except LinkDown:
            report.link_down = True
        return report
