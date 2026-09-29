"""Durable SQLite outbox: what still has to reach the cloud.

Survives restarts. One row per point ID (a newer write coalesces onto the
pending row), so the queue holds IDs only; the push reads the latest content
from the local shard.
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutboxRow:
    point_id: str
    attempts: int
    next_attempt_ms: int
    decision: str


class Outbox:
    def __init__(
        self,
        path: str | Path,
        *,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 30.0,
        rng: random.Random | None = None,
    ) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS outbox ("
            " point_id TEXT PRIMARY KEY,"
            " decision TEXT NOT NULL DEFAULT 'SYNC_NOW',"
            " attempts INTEGER NOT NULL DEFAULT 0,"
            " next_attempt_ms INTEGER NOT NULL DEFAULT 0,"
            " enqueued_ms INTEGER NOT NULL)"
        )
        self._db.commit()
        self._base = backoff_base_s
        self._cap = backoff_cap_s
        self._rng = rng or random.Random()  # noqa: S311 - retry jitter, not security

    def enqueue(self, point_id: str, decision: str = "SYNC_NOW", now_ms: int | None = None) -> None:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._db.execute(
            "INSERT INTO outbox(point_id, decision, enqueued_ms) VALUES (?,?,?) "
            "ON CONFLICT(point_id) DO UPDATE SET decision=excluded.decision",
            (point_id, decision, now),
        )
        self._db.commit()

    def due(
        self,
        now_ms: int | None = None,
        limit: int = 100,
        decisions: tuple[str, ...] = ("SYNC_NOW",),
    ) -> list[OutboxRow]:
        """Rows ready to send, urgent first. ``decisions`` selects the lane."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        rows: list[tuple] = []
        for decision in decisions:  # one parameterised query per lane
            cur = self._db.execute(
                "SELECT point_id, attempts, next_attempt_ms, decision, enqueued_ms FROM outbox "
                "WHERE next_attempt_ms <= ? AND decision = ? ORDER BY enqueued_ms LIMIT ?",
                (now, decision, limit),
            )
            rows.extend(cur.fetchall())
        rows.sort(key=lambda r: r[4])
        return [OutboxRow(*r[:4]) for r in rows[:limit]]

    def count_by_decision(self) -> dict[str, int]:
        rows = self._db.execute("SELECT decision, COUNT(*) FROM outbox GROUP BY decision")
        return {d: n for d, n in rows.fetchall()}

    def mark_sent(self, point_ids: list[str]) -> None:
        self._db.executemany("DELETE FROM outbox WHERE point_id=?", [(p,) for p in point_ids])
        self._db.commit()

    def mark_failed(self, point_ids: list[str], now_ms: int | None = None) -> None:
        """Bump attempts and schedule the retry with exponential backoff + jitter."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        for pid in point_ids:
            row = self._db.execute(
                "SELECT attempts FROM outbox WHERE point_id=?", (pid,)
            ).fetchone()
            if row is None:
                continue
            attempts = row[0] + 1
            delay = min(self._cap, self._base * (2 ** (attempts - 1)))
            delay *= 0.5 + self._rng.random() * 0.5  # jitter in [50%, 100%]
            self._db.execute(
                "UPDATE outbox SET attempts=?, next_attempt_ms=? WHERE point_id=?",
                (attempts, now + int(delay * 1000), pid),
            )
        self._db.commit()

    def retry_now(self) -> None:
        """Clear all backoff timers, e.g. when connectivity is restored."""
        self._db.execute("UPDATE outbox SET next_attempt_ms=0")
        self._db.commit()

    def pending(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    def close(self) -> None:
        self._db.close()
