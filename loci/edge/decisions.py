"""Explainable decision records: why each observation stayed local, synced, or merged."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KEEP_LOCAL = "KEEP_LOCAL"
DEDUPE = "DEDUPE"
SUMMARIZE_SYNC = "SUMMARIZE_SYNC"
SYNC_NOW = "SYNC_NOW"
ACTIONS = (KEEP_LOCAL, DEDUPE, SUMMARIZE_SYNC, SYNC_NOW)


@dataclass
class Decision:
    """The verdict on one observation, with the evidence behind it."""

    action: str
    reason: str
    key: str = ""
    point_id: str = ""
    novelty: float = 1.0
    best_similarity: float | None = None
    neighbor_id: str | None = None
    neighbor_source: str | None = None  # "local" or "mirror"
    displacement: float | None = None  # distance to the neighbour's position, if matched
    thresholds: dict[str, float] = field(default_factory=dict)
    ts_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionLog:
    """Append-only SQLite log of :class:`Decision` records (the UI's activity feed)."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL,"
            " point_id TEXT, action TEXT NOT NULL, body TEXT NOT NULL)"
        )
        self._db.commit()

    def add(self, d: Decision) -> None:
        d.ts_ms = d.ts_ms or int(time.time() * 1000)
        self._db.execute(
            "INSERT INTO decisions(ts_ms, point_id, action, body) VALUES (?,?,?,?)",
            (d.ts_ms, d.point_id, d.action, json.dumps(d.to_dict())),
        )
        self._db.commit()

    def recent(self, limit: int = 50, action: str | None = None) -> list[Decision]:
        q = "SELECT body FROM decisions"
        args: tuple = ()
        if action:
            q += " WHERE action=?"
            args = (action,)
        q += " ORDER BY id DESC LIMIT ?"
        rows = self._db.execute(q, (*args, limit)).fetchall()
        return [Decision(**json.loads(r[0])) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self._db.execute("SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall()
        out = dict.fromkeys(ACTIONS, 0)
        out.update({a: n for a, n in rows})
        return out

    def close(self) -> None:
        self._db.close()
