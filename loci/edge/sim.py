"""A two-robot warehouse fleet for the demo UI.

SYNTHETIC: the robot's "camera" is simulated (landmark text + a little noise
run through :class:`HashEmbedder`) and the cloud is a local stand-in, or a Qdrant Server
when ``LOCI_QDRANT_URL`` is set.
The Edge shards, sync policy, outbox, sync and diff are the real engine.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from loci.edge.cloud import Link, LinkedCloud
from loci.edge.cloud_ai import CloudBrain, LLMClient
from loci.edge.cloud_server import QdrantServerCloud, open_cloud
from loci.edge.conflicts import ConflictLog, Reconciler
from loci.edge.decisions import Decision, DecisionLog
from loci.edge.embed import HashEmbedder
from loci.edge.gate import AnswerGate
from loci.edge.outbox import Outbox
from loci.edge.store import EdgeMemoryStore, Memory
from loci.edge.sync import SummaryReport, SyncEngine
from loci.spatial.hilbert import HilbertIndex

# name, text, (x, y)
LANDMARKS = [
    ("dock", "loading dock 4 with pallet racks", (0.10, 0.15)),
    ("aisle", "aisle 3 shelving units", (0.30, 0.55)),
    ("charger", "battery charging station", (0.55, 0.85)),
    ("packing", "packing table and label printer", (0.75, 0.35)),
    ("exit", "exit door and badge scanner", (0.92, 0.70)),
]
ROUTES = {"robot-a": [0, 1, 2, 3], "robot-b": [2, 3, 4, 0]}
TOOLBOX_SPOTS = [(0.72, 0.40), (0.28, 0.58), (0.12, 0.18)]
SUMMARY_BATCH = 10  # auto-sync summarizes once this many observations are waiting
GRID = 16  # map resolution == Hilbert order-4 side


@dataclass
class Node:
    name: str
    link: Link
    store: EdgeMemoryStore
    outbox: Outbox
    log: DecisionLog
    engine: SyncEngine
    gate: AnswerGate | None = None
    step: int = 0
    last_sync: dict[str, Any] = field(default_factory=dict)


class Fleet:
    """Owns the robots, the cloud, a bytes ledger and an activity feed."""

    def __init__(self, data_dir: str | Path | None = None, dim: int = 64, seed: int = 7) -> None:
        self._tmp = None
        if data_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="loci-fleet-")
            data_dir = self._tmp.name
        self.dir = Path(data_dir)
        self.dim = dim
        self.embedder = HashEmbedder(dim)
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        env = dict(os.environ)
        self._own_collection = "LOCI_QDRANT_COLLECTION" not in env
        # A server keeps its data across runs, but the robots here are fresh: use a fresh collection
        # (dropped on close) unless the operator named one.
        env.setdefault("LOCI_QDRANT_COLLECTION", f"loci_edge_demo_{int(time.time() * 1000)}")
        self.cloud, self.cloud_kind = open_cloud(self.dir / "cloud", dim, env)
        # Cloud-side processes: conflict reconciler and the fleet-briefing brain.
        self.conflicts = ConflictLog(self.dir / "conflicts.db")
        self.reconciler = Reconciler(self.cloud, self.conflicts)
        self.llm = LLMClient.from_env()
        self.brain = CloudBrain(self.cloud, self.embedder, self.llm)
        self.cloud_last: dict[str, Any] = {}
        self.events: deque[dict] = deque(maxlen=200)
        self.ledger = {"observations": 0, "naive_bytes": 0, "sent_bytes": 0, "summary_bytes": 0}
        self.nodes: dict[str, Node] = {}
        self._toolbox = -1
        self._cells: dict[tuple[int, int], int] | None = None
        for name in ROUTES:
            link = Link(True)
            store = EdgeMemoryStore(
                self.dir / name, dim, name, mirror_path=self.dir / f"{name}-mirror"
            )
            outbox = Outbox(self.dir / f"{name}-outbox.db")
            log = DecisionLog(self.dir / f"{name}-decisions.db")
            linked = LinkedCloud(self.cloud, link)
            engine = SyncEngine(store, outbox, linked, log=log)
            gate = AnswerGate(store, self.embedder, linked)
            self.nodes[name] = Node(name, link, store, outbox, log, engine, gate)
        self._auto = False
        self._thread: threading.Thread | None = None

    # -- helpers ------------------------------------------------------------

    def node(self, name: str) -> Node:
        if name not in self.nodes:
            raise KeyError(name)
        return self.nodes[name]

    def _event(self, robot: str, kind: str, msg: str) -> None:
        self.events.appendleft(
            {"ts_ms": int(time.time() * 1000), "robot": robot, "kind": kind, "msg": msg}
        )

    def _observe(
        self,
        n: Node,
        key: str,
        text: str,
        x: float,
        y: float,
        *,
        tilt: float | None = None,
        **kw: Any,
    ) -> Decision:
        base = self.embedder.embed(text)
        if tilt is not None:  # a blurry view: only ``tilt`` cosine similar to the clean one
            u = self.rng.normal(size=self.dim)
            u -= (u @ base) * base
            u /= np.linalg.norm(u)
            base = tilt * base + np.sqrt(1 - tilt**2) * u
        vec = base + 0.01 * self.rng.normal(size=self.dim)
        vec /= np.linalg.norm(vec)
        mem = Memory(
            key=key,
            vector=vec.tolist(),
            x=min(1.0, max(0.0, x)),
            y=min(1.0, max(0.0, y)),
            z=0.0,
            timestamp_ms=int(time.time() * 1000),
            device_id=n.name,
            text=text,
            **kw,
        )
        d = n.engine.observe(mem)
        self.ledger["observations"] += 1
        self.ledger["naive_bytes"] += 4 * self.dim + 250  # estimate: vector + ~250 B payload
        return d

    # -- actions -------------------------------------------------------------

    def set_link(self, name: str, up: bool) -> None:
        with self.lock:
            n = self.node(name)
            if n.link.up == up:
                return
            n.link.set(up)
            if up:
                n.outbox.retry_now()
            self._event(
                name, "network", "network restored" if up else "network CUT: working offline"
            )

    def patrol(self, name: str, steps: int = 10) -> list[dict]:
        out = []
        with self.lock:
            n = self.node(name)
            route = ROUTES[name]
            for _ in range(max(1, min(steps, 200))):
                _, text, (lx, ly) = LANDMARKS[route[n.step % len(route)]]
                n.step += 1
                d = self._observe(
                    n,
                    f"{name}-p{n.step}",
                    text,
                    lx + self.rng.normal(0, 0.004),
                    ly + self.rng.normal(0, 0.004),
                )
                out.append(d.to_dict())
            counts = n.log.counts()
            self._event(name, "patrol", f"patrolled {len(out)} steps")
            n.last_sync.setdefault("counts", counts)
        return out

    def spill(self, name: str) -> dict:
        with self.lock:
            n = self.node(name)
            d = self._observe(
                n,
                f"{name}-spill-{int(time.time() * 1000)}",
                "oil spill near dock 9",
                0.14 + self.rng.normal(0, 0.01),
                0.22,
                metadata={"urgent": True},
            )
            self._event(name, "urgent", "reported an oil spill (urgent)")
            return d.to_dict()

    def move_toolbox(self, name: str) -> dict:
        with self.lock:
            n = self.node(name)
            self._toolbox = (self._toolbox + 1) % len(TOOLBOX_SPOTS)
            x, y = TOOLBOX_SPOTS[self._toolbox]
            d = self._observe(n, f"{name}-toolbox-{time.time_ns()}", "red toolbox", x, y)
            self._event(name, "observe", f"saw the red toolbox at ({x:.2f}, {y:.2f})")
            return d.to_dict()

    def both_see_toolbox(self) -> dict:
        """Both robots independently report the toolbox at (almost) the same spot."""
        with self.lock:
            spot = TOOLBOX_SPOTS[(self._toolbox + 1) % len(TOOLBOX_SPOTS)]
            out = {}
            for i, (name, n) in enumerate(self.nodes.items()):
                d = self._observe(
                    n,
                    f"{name}-toolbox-{time.time_ns()}",
                    "red toolbox",
                    spot[0] + 0.004 * i,
                    spot[1],
                    confidence=0.7 + 0.2 * i,
                )
                out[name] = d.to_dict()
                self._event(
                    name, "observe", f"saw the red toolbox at ({spot[0]:.2f}, {spot[1]:.2f})"
                )
            return out

    def blurry_toolbox(self, name: str) -> dict:
        """A poor view of the toolbox: similar, but not similar enough to merge automatically."""
        with self.lock:
            n = self.node(name)
            x, y = TOOLBOX_SPOTS[max(self._toolbox, 0)]
            d = self._observe(
                n,
                f"{name}-blurry-{time.time_ns()}",
                "red toolbox",
                x + 0.004,
                y,
                tilt=0.90,
                confidence=0.5,
            )
            self._event(name, "observe", "got a blurry view of the red toolbox")
            return d.to_dict()

    def private_note(self, name: str) -> dict:
        with self.lock:
            n = self.node(name)
            d = self._observe(
                n,
                f"{name}-private-{time.time_ns()}",
                "operator badge left on shelf",
                0.31,
                0.52,
                private=True,
            )
            self._event(name, "private", "stored a private note (stays on device)")
            return d.to_dict()

    def sync(self, name: str, *, flush_summaries: bool = True) -> dict:
        """Push raw, send summaries, then pull. Offline it reports link_down and loses nothing.

        ``flush_summaries=False`` (auto-sync) waits until ``SUMMARY_BATCH`` observations
        are queued, so summaries cover a worthwhile pile instead of one or two points.
        """
        with self.lock:
            n = self.node(name)
            push = n.engine.push()
            if not push.link_down:
                self._cloud_cycle()
            queued = n.outbox.count_by_decision().get("SUMMARIZE_SYNC", 0)
            if flush_summaries or queued >= SUMMARY_BATCH:
                summ = n.engine.summarize_pending()
            else:
                summ = SummaryReport()
            pull = n.engine.pull()
            self.ledger["sent_bytes"] += push.bytes_sent
            self.ledger["summary_bytes"] += summ.bytes_sent
            res = {
                "sent": push.sent,
                "bytes_sent": push.bytes_sent,
                "summarized": summ.covered,
                "summary_bytes": summ.bytes_sent,
                "pulled": pull.pulled,
                "link_down": push.link_down or summ.link_down or pull.link_down,
            }
            n.last_sync = {**res, "ts_ms": int(time.time() * 1000)}
            if res["link_down"]:
                self._event(name, "sync", "sync attempted: offline, nothing lost, will retry")
            elif push.sent or summ.covered or pull.pulled:
                self._event(
                    name,
                    "sync",
                    f"sent {push.sent} raw + {summ.covered} summarized, pulled {pull.pulled}",
                )
            return res

    def _cloud_cycle(self) -> None:
        """What the cloud does after data arrives: reconcile conflicts, then brief the fleet."""
        rec = self.reconciler.run()
        brief = self.brain.publish()
        if rec.ran and (rec.merged or rec.moved or rec.reviews_opened):
            self._event(
                "cloud",
                "reconcile",
                f"reconciled: {rec.merged} merged, {rec.moved} moved, "
                f"{rec.reviews_opened} need review",
            )
        if brief.insights:
            self._event(
                "cloud",
                "insight",
                f"published {brief.insights} fleet briefing(s) [{brief.generator}]",
            )
        self.cloud_last = {
            "reconcile": {"merged": rec.merged, "moved": rec.moved, "reviews": rec.reviews_opened},
            "briefing": {
                "insights": brief.insights,
                "generator": brief.generator,
                "llm_errors": brief.llm_errors,
            },
        }

    def resolve_conflict(self, conflict_id: int, approve: bool) -> dict | None:
        with self.lock:
            res = self.conflicts.resolve(conflict_id, approve)
            if res is not None:
                self._event(
                    "cloud",
                    "review",
                    f"operator {'approved' if approve else 'rejected'} match #{conflict_id}",
                )
                self._cloud_cycle()
            return res

    def auto_tick(self) -> None:
        """Background sync for every online robot."""
        with self.lock:
            for name, n in self.nodes.items():
                if n.link.up:
                    self.sync(name, flush_summaries=False)

    def start_auto(self, interval: float = 1.5) -> None:
        if self._thread is not None:
            return
        self._auto = True

        def loop() -> None:
            while self._auto:
                time.sleep(interval)
                try:
                    self.auto_tick()
                except Exception:  # keep the demo alive; the error shows up as missing sync
                    self._event("fleet", "error", "auto-sync tick failed")

        self._thread = threading.Thread(target=loop, daemon=True, name="fleet-autosync")
        self._thread.start()

    def stop_auto(self) -> None:
        self._auto = False
        self._thread = None

    # -- read models ------------------------------------------------------------

    def _bytes_view(self) -> dict:
        """Bytes ledger. ``saved_pct`` is None until something has actually been sent,
        so an offline robot with everything queued is not reported as "100% saved"."""
        sent = self.ledger["sent_bytes"] + self.ledger["summary_bytes"]
        naive = self.ledger["naive_bytes"]
        saved = round(100 * (1 - sent / naive), 1) if sent and naive and sent < naive else None
        return {**self.ledger, "sent_total": sent, "saved_pct": saved}

    def state(self) -> dict:
        with self.lock:
            robots = []
            for name, n in self.nodes.items():
                robots.append(
                    {
                        "name": name,
                        "online": n.link.up,
                        "local": n.store.count(),
                        "mirror": n.store.count(include_mirror=True) - n.store.count(),
                        "outbox": n.outbox.count_by_decision(),
                        "decisions": n.log.counts(),
                        "digest": n.store.digest(),
                        "last_sync": n.last_sync,
                    }
                )
            try:
                cloud_count = self.cloud.count()
            except Exception:
                cloud_count = -1
            return {
                "robots": robots,
                "cloud_memories": cloud_count,
                "bytes": self._bytes_view(),
                "events": list(self.events)[:30],
                "auto_sync": self._auto,
            }

    def decisions(self, name: str, limit: int = 40) -> list[dict]:
        with self.lock:
            return [d.to_dict() for d in self.node(name).log.recent(limit=limit)]

    def diff(self, name: str) -> dict:
        with self.lock:
            n = self.node(name)
            d = n.engine.diff()
            text = {m["id"]: m.get("text", "") for m in n.store.list_memories(limit=2000)}
            for m in n.store.list_memories(limit=2000):
                text.setdefault(m["id"], m.get("text", ""))

            def item(i: str, why: str = "") -> dict:
                return {"id": i, "text": text.get(i, ""), "why": why}

            return {
                "cloud_reachable": d.cloud_reachable,
                "converged": d.converged,
                "local_digest": d.local_digest,
                "to_push": [item(i) for i in d.to_push[:25]],
                "to_pull": [{"id": i, "text": "", "why": ""} for i in d.to_pull[:25]],
                "in_sync": len(d.in_sync),
                "reconcile": [item(i) for i in d.reconcile[:25]],
                "held_local": [item(i, why) for i, why in list(d.held_local.items())[:25]],
                "held_local_total": len(d.held_local),
                "awaiting_summary": len(d.awaiting_summary),
                "counts": {
                    "to_push": len(d.to_push),
                    "to_pull": len(d.to_pull),
                    "in_sync": len(d.in_sync),
                    "reconcile": len(d.reconcile),
                    "held_local": len(d.held_local),
                    "awaiting_summary": len(d.awaiting_summary),
                },
            }

    def memories(self, name: str, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.node(name).store.list_memories(limit=limit)
            keep = (
                "id",
                "source",
                "text",
                "x",
                "y",
                "timestamp_ms",
                "device_id",
                "confidence",
                "private",
                "version",
                "content_hash",
                "sync_state",
                "seen_count",
                "summary",
                "source_count",
                "role",
                "kind",
                "generator",
                "entity_id",
            )
            return [{k: r[k] for k in keep if k in r} for r in rows]

    def search(self, name: str, text: str, limit: int = 8, *, current_only: bool = False) -> dict:
        with self.lock:
            n = self.node(name)
            t = time.perf_counter()
            hits = n.store.search(
                vector=self.embedder(text), text=text, limit=limit, current_only=current_only
            )
            ms = (time.perf_counter() - t) * 1000
            return {
                "query": text,
                "latency_ms": round(ms, 2),
                "online": n.link.up,
                "results": [
                    {
                        "id": h.id,
                        "score": round(h.score, 4),
                        "source": h.source,
                        **{
                            k: h.payload.get(k)
                            for k in (
                                "text",
                                "x",
                                "y",
                                "device_id",
                                "version",
                                "sync_state",
                                "content_hash",
                                "timestamp_ms",
                                "private",
                                "seen_count",
                                "role",
                                "kind",
                                "generator",
                            )
                        },
                    }
                    for h in hits
                ],
            }

    def ask(self, name: str, text: str, *, history: bool = False) -> dict:
        """Gated question answering: local, escalate to cloud, low-confidence, or abstain."""
        with self.lock:
            n = self.node(name)
            assert n.gate is not None  # every node is built with a gate
            ans = n.gate.ask(text, current_only=not history)
            d = ans.to_dict()
            d["online"] = n.link.up
            if d["route"] == "ESCALATE_CLOUD":
                self._event(
                    name,
                    "escalate",
                    f"asked the cloud about '{text}' and cached {ans.cached} hit(s)",
                )
            return d

    def cloud_view(self) -> dict:
        with self.lock:
            pts = self.cloud.scan()
            insights = [
                {
                    k: p["payload"].get(k)
                    for k in ("text", "generator", "x", "y", "version", "timestamp_ms")
                }
                for p in pts
                if p["payload"].get("kind") == "insight"
            ]
            roles: dict[str, int] = {}
            for p in pts:
                r = p["payload"].get("role") or "single"
                roles[r] = roles.get(r, 0) + 1
            return {
                "conflicts": self.conflicts.list(limit=40),
                "conflict_counts": self.conflicts.counts(),
                "insights": insights,
                "roles": roles,
                "cloud_kind": self.cloud_kind,
                "llm": (
                    {"enabled": True, "provider": self.llm.provider, "model": self.llm.model}
                    if self.llm
                    else {"enabled": False, "provider": None, "model": None}
                ),
                "last": self.cloud_last,
            }

    def _cell_hilbert(self) -> dict[tuple[int, int], int]:
        if self._cells is None:
            h = HilbertIndex([4])
            self._cells = {
                (ix, iy): h.encode((ix + 0.5) / GRID, (iy + 0.5) / GRID, 0.0, 0.0)["hilbert_r4"]
                for ix in range(GRID)
                for iy in range(GRID)
            }
        return self._cells

    def map(self, name: str) -> dict:
        with self.lock:
            n = self.node(name)
            cells: dict[tuple[int, int], dict] = {}
            points = []
            for d in n.log.recent(limit=400):
                if d.x is None or d.y is None:
                    continue
                ix, iy = min(GRID - 1, int(d.x * GRID)), min(GRID - 1, int(d.y * GRID))
                c = cells.setdefault((ix, iy), {"n": 0, "novelty": 0.0})
                c["n"] += 1
                c["novelty"] += d.novelty
                points.append({"x": d.x, "y": d.y, "action": d.action, "novelty": d.novelty})
            hil = self._cell_hilbert()
            mirror = [
                {"x": m["x"], "y": m["y"], "text": m.get("text", ""), "device": m.get("device_id")}
                for m in n.store.list_memories(limit=500)
                if m["source"] == "mirror" and "x" in m
            ]
            return {
                "grid": GRID,
                "cells": [
                    {
                        "ix": ix,
                        "iy": iy,
                        "hilbert": hil[(ix, iy)],
                        "count": c["n"],
                        "novelty": round(c["novelty"] / c["n"], 3),
                    }
                    for (ix, iy), c in cells.items()
                ],
                "points": points,
                "mirror": mirror,
                "landmarks": [
                    {"name": nm, "text": t, "x": x, "y": y} for nm, t, (x, y) in LANDMARKS
                ],
            }

    def close(self) -> None:
        self.stop_auto()
        with self.lock:
            for n in self.nodes.values():
                n.store.close()
                n.outbox.close()
                n.log.close()
            self.conflicts.close()
            if self._own_collection and isinstance(self.cloud, QdrantServerCloud):
                with contextlib.suppress(Exception):  # the server may already be gone
                    self.cloud.drop()
            self.cloud.close()
        if self._tmp is not None:
            self._tmp.cleanup()
