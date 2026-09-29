"""Space-time conflict resolution: which sightings are the same thing, and where is it now?

Two devices that saw the same object offline both report it. Instead of picking a
winner by wall-clock alone, the cloud reasons geometrically:

* **same thing** -- embeddings are near-identical (cosine >= ``same_similarity``);
* **same sighting** -- same thing, in the same place (``same_place_radius``) and time
  window (``merge_window_ms``): the duplicates are *merged*, the higher-confidence
  (then newer) one wins and the sightings are unioned;
* **moved** -- same thing, different place (or much later): *both are kept*, the
  newest is ``current`` ("where is it now?") and the rest are ``previous``;
* **maybe the same** -- similar but not near-identical (``review_similarity``), close in
  space and time: never merged automatically; it goes to a human review inbox.

:func:`resolve_entity` is a pure function of the sightings, so results do not depend on
arrival order. :class:`Reconciler` applies it to a cloud store and records every change
in a :class:`ConflictLog` (the audit trail and the UI's conflict inbox).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from loci.edge.store import ROLE_FIELDS


@dataclass
class ConflictConfig:
    same_similarity: float = 0.95
    review_similarity: float = 0.85
    same_place_radius: float = 0.05
    merge_window_ms: int = 300_000
    neighbors: int = 8


@dataclass(frozen=True)
class Sighting:
    id: str
    device: str
    timestamp_ms: int
    x: float
    y: float
    z: float
    confidence: float = 1.0


@dataclass(frozen=True)
class Verdict:
    entity_id: str
    role: str  # "current" | "previous" | "merged"
    merged_into: str | None
    entity_devices: tuple[str, ...]
    entity_size: int


def _same_sighting(a: Sighting, b: Sighting, cfg: ConflictConfig) -> bool:
    place = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) <= cfg.same_place_radius
    return place and abs(a.timestamp_ms - b.timestamp_ms) <= cfg.merge_window_ms


def resolve_entity(members: list[Sighting], cfg: ConflictConfig) -> dict[str, Verdict]:
    """Decide the role of every sighting of one entity (a set of "same thing" sightings)."""
    if not members:
        return {}
    entity_id = min(m.id for m in members)
    devices = tuple(sorted({m.device for m in members}))
    # Highest confidence first, then newest, then id: the head of each duplicate cluster.
    priority = sorted(members, key=lambda m: (-m.confidence, -m.timestamp_ms, m.id))
    heads: list[Sighting] = []
    merged_into: dict[str, str] = {}
    for m in priority:
        for h in heads:
            if _same_sighting(m, h, cfg):
                merged_into[m.id] = h.id
                break
        else:
            heads.append(m)
    newest = min(heads, key=lambda m: (-m.timestamp_ms, -m.confidence, m.id))
    out: dict[str, Verdict] = {}
    for m in members:
        if m.id in merged_into:
            role, into = "merged", merged_into[m.id]
        else:
            role, into = ("current" if m.id == newest.id else "previous"), None
        out[m.id] = Verdict(entity_id, role, into, devices, len(members))
    return out


class ConflictLog:
    """SQLite audit trail + human-review inbox + operator overrides ("links")."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
                ts_ms INTEGER NOT NULL, rule TEXT NOT NULL, status TEXT NOT NULL,
                a_id TEXT, b_id TEXT, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS links (
                a TEXT NOT NULL, b TEXT NOT NULL, verdict TEXT NOT NULL, PRIMARY KEY (a, b));
            """
        )
        self._db.commit()

    @staticmethod
    def _pair(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def add(
        self, key: str, rule: str, status: str, a_id: str, b_id: str, evidence: dict[str, Any]
    ) -> bool:
        """Insert unless *key* was already recorded; returns True if new."""
        cur = self._db.execute(
            "INSERT OR IGNORE INTO conflicts(key, ts_ms, rule, status, a_id, b_id, evidence)"
            " VALUES (?,?,?,?,?,?,?)",
            (key, int(time.time() * 1000), rule, status, a_id, b_id, json.dumps(evidence)),
        )
        self._db.commit()
        return cur.rowcount > 0

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q = "SELECT id, ts_ms, rule, status, a_id, b_id, evidence FROM conflicts"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        rows = self._db.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [
            {
                "id": r[0], "ts_ms": r[1], "rule": r[2], "status": r[3],
                "a_id": r[4], "b_id": r[5], "evidence": json.loads(r[6]),
            }
            for r in rows
        ]  # fmt: skip

    def counts(self) -> dict[str, int]:
        rows = self._db.execute("SELECT status, COUNT(*) FROM conflicts GROUP BY status")
        return {s: n for s, n in rows.fetchall()}

    def resolve(self, conflict_id: int, approve: bool) -> dict[str, Any] | None:
        """Operator verdict on a pending review: approve => same thing, reject => different."""
        row = self._db.execute(
            "SELECT a_id, b_id, status FROM conflicts WHERE id=?", (conflict_id,)
        ).fetchone()
        if row is None or row[2] != "pending_review":
            return None
        a, b = self._pair(row[0], row[1])
        self._db.execute(
            "INSERT OR REPLACE INTO links(a, b, verdict) VALUES (?,?,?)",
            (a, b, "same" if approve else "different"),
        )
        self._db.execute(
            "UPDATE conflicts SET status=? WHERE id=?",
            ("approved" if approve else "rejected", conflict_id),
        )
        self._db.commit()
        return {"id": conflict_id, "status": "approved" if approve else "rejected"}

    def link(self, a: str, b: str) -> str | None:
        x, y = self._pair(a, b)
        row = self._db.execute("SELECT verdict FROM links WHERE a=? AND b=?", (x, y)).fetchone()
        return row[0] if row else None

    def fingerprint(self) -> str:
        n = self._db.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        return f"links:{n}"

    def close(self) -> None:
        self._db.close()


@dataclass
class ReconcileReport:
    ran: bool = False
    points_updated: int = 0
    merged: int = 0
    moved: int = 0
    reviews_opened: int = 0
    entities: int = 0
    notes: list[str] = field(default_factory=list)


def _cos(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else 0.0


class Reconciler:
    """Applies :func:`resolve_entity` to a cloud store and audits every change.

    Runs where the cloud runs. A pass is skipped when nothing changed since the last one.
    """

    def __init__(self, cloud: Any, log: ConflictLog, cfg: ConflictConfig | None = None) -> None:
        self.cloud = cloud
        self.log = log
        self.cfg = cfg or ConflictConfig()
        self._fingerprint = ""

    @staticmethod
    def _eligible(p: dict) -> bool:
        pl = p["payload"]
        return not pl.get("summary") and pl.get("kind") != "insight" and "x" in pl

    def run(self, *, force: bool = False) -> ReconcileReport:
        rep = ReconcileReport()
        pts = [p for p in self.cloud.scan() if self._eligible(p)]
        fp = hashlib.sha256(
            (
                ",".join(sorted(f"{p['id']}:{p['payload'].get('version', 0)}" for p in pts))
                + self.log.fingerprint()
            ).encode()
        ).hexdigest()
        if fp == self._fingerprint and not force:
            return rep
        rep.ran = True
        by_id = {p["id"]: p for p in pts}
        parent = {i: i for i in by_id}

        def find(i: str) -> str:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        cfg = self.cfg
        for p in pts:
            for hit in self.cloud.search(p["vector"], cfg.neighbors + 1):
                q = by_id.get(hit["id"])
                if q is None or q["id"] == p["id"]:
                    continue
                sim = hit["score"]
                verdict = self.log.link(p["id"], q["id"])
                if verdict == "different":
                    continue
                if verdict == "same" or sim >= cfg.same_similarity:
                    parent[find(p["id"])] = find(q["id"])
                elif sim >= cfg.review_similarity and self._close(p, q):
                    a, b = sorted((p["id"], q["id"]))
                    key = f"review:{a}:{b}"
                    if self.log.add(
                        key, "needs_review", "pending_review", a, b, self._evidence(p, q, sim)
                    ):
                        rep.reviews_opened += 1

        groups: dict[str, list[str]] = {}
        for i in by_id:
            groups.setdefault(find(i), []).append(i)
        updates: dict[str, dict] = {}
        for ids in groups.values():
            sightings = [self._sighting(by_id[i]) for i in ids]
            verdicts = resolve_entity(sightings, cfg) if len(ids) > 1 else {}
            if len(ids) > 1:
                rep.entities += 1
            for i in ids:
                new = self._fields(verdicts.get(i))
                old = {k: by_id[i]["payload"].get(k) for k in ROLE_FIELDS}
                if _norm(new) != _norm(old):
                    updates[i] = new
                    self._audit(rep, by_id, i, old, new, verdicts)
        if updates:
            rep.points_updated = self.cloud.apply_roles(updates)
        self._fingerprint = fp if not updates else ""  # re-run once to confirm a fixed point
        return rep

    # -- helpers ---------------------------------------------------------------

    def _close(self, p: dict, q: dict) -> bool:
        return _same_sighting(self._sighting(p), self._sighting(q), self.cfg)

    @staticmethod
    def _sighting(p: dict) -> Sighting:
        pl = p["payload"]
        return Sighting(
            p["id"], str(pl.get("device_id", "")), int(pl.get("timestamp_ms", 0)),
            float(pl["x"]), float(pl["y"]), float(pl.get("z", 0.0)),
            float(pl.get("confidence", 1.0)),
        )  # fmt: skip

    @staticmethod
    def _fields(v: Verdict | None) -> dict[str, Any]:
        if v is None:  # singleton: no verdict needed
            return dict.fromkeys(ROLE_FIELDS)
        return {
            "entity_id": v.entity_id, "role": v.role, "merged_into": v.merged_into,
            "entity_devices": list(v.entity_devices), "entity_size": v.entity_size,
        }  # fmt: skip

    def _evidence(self, p: dict, q: dict, sim: float) -> dict[str, Any]:
        a, b = p["payload"], q["payload"]
        return {
            "similarity": round(sim, 4),
            "displacement": round(
                math.dist((a["x"], a["y"], a.get("z", 0)), (b["x"], b["y"], b.get("z", 0))), 4
            ),
            "dt_ms": abs(int(a.get("timestamp_ms", 0)) - int(b.get("timestamp_ms", 0))),
            "a": {"id": p["id"], "device": a.get("device_id"), "text": a.get("text", ""),
                  "x": a["x"], "y": a["y"], "confidence": a.get("confidence", 1.0)},
            "b": {"id": q["id"], "device": b.get("device_id"), "text": b.get("text", ""),
                  "x": b["x"], "y": b["y"], "confidence": b.get("confidence", 1.0)},
        }  # fmt: skip

    def _audit(self, rep, by_id, pid, old, new, verdicts) -> None:
        role_old, role_new = old.get("role"), new.get("role")
        if role_new == role_old:
            return
        p = by_id[pid]
        ver = p["payload"].get("version", 0)
        if role_new == "merged":
            other = by_id[new["merged_into"]]
            ev = self._evidence(p, other, _cos(p["vector"], other["vector"]))
            ev["winner"] = new["merged_into"]
            ev["rule"] = "same thing, same place, same time window: duplicate sighting merged"
            if self.log.add(
                f"merge:{pid}:{ver}:{new['merged_into']}",
                "merge_duplicate",
                "auto",
                pid,
                other["id"],
                ev,
            ):
                rep.merged += 1
        elif role_new == "previous":
            cur = next(i for i, v in verdicts.items() if v.role == "current")
            other = by_id[cur]
            ev = self._evidence(p, other, _cos(p["vector"], other["vector"]))
            ev["current"] = cur
            ev["rule"] = "same thing seen elsewhere/later: both kept, newest is current"
            if self.log.add(f"moved:{pid}:{ver}:{cur}", "moved", "auto", pid, cur, ev):
                rep.moved += 1


def _norm(d: dict) -> dict:
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()}
