"""An edge device as its own OS process: ``python -m loci.edge.node --name robot-c``.

Each node owns a data directory with its Qdrant Edge shards (writable + fleet mirror), its
SQLite outbox and decision log, and runs its own patrol loop: observe, decide on-device, store,
and sync with the cloud over HTTP whenever its uplink is up. Kill it (even ``kill -9``) and
start it again: memories, the outbox and the decision history survive, and syncing resumes.

It reports telemetry in a heartbeat (process id, resident memory, CPU, disk, Qdrant Edge call
latencies, queue sizes, position, recent decisions) and takes commands from the cloud in the
reply: ``outage`` (drop the uplink for N seconds: the node keeps working and its heartbeats
really stop), ``event`` (spill / toolbox), ``pause``.

The patrol is SIMULATED (landmark descriptions plus sensor noise); everything else is the real
engine running in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from loci.edge.cloud import Link, LinkDown
from loci.edge.cloud_http import HttpCloud
from loci.edge.decisions import DecisionLog
from loci.edge.embed import make_embedder
from loci.edge.outbox import Outbox
from loci.edge.sim import LANDMARKS, TOOLBOX_SPOTS
from loci.edge.store import EdgeMemoryStore, Memory
from loci.edge.sync import SyncEngine


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    import resource

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024


class _GatedCloud:
    """The HTTP cloud behind this node's own uplink switch (for commanded outages)."""

    def __init__(self, cloud: HttpCloud, link: Link) -> None:
        self._cloud, self.link = cloud, link

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._cloud, name)
        if not callable(target):
            return target

        def call(*a: Any, **k: Any) -> Any:
            self.link.require()
            return target(*a, **k)

        return call


class Node:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.name
        self.interval = args.interval
        self.exit_with_parent = bool(getattr(args, "exit_with_parent", False))
        self.data = Path(args.data or f"./edge-nodes/{args.name}")
        self.data.mkdir(parents=True, exist_ok=True)
        self.embedder, self.embed_note = make_embedder(args.embedder)
        self.link = Link(True)
        self.http = HttpCloud(args.cloud, token=args.token or os.environ.get("LOCI_CLOUD_TOKEN"))
        self.cloud = _GatedCloud(self.http, self.link)
        self.store = EdgeMemoryStore(
            self.data / "shard", self.embedder.dim, self.name, mirror_path=self.data / "mirror"
        )
        self.outbox = Outbox(self.data / "outbox.db")
        self.log = DecisionLog(self.data / "decisions.db")
        self.engine = SyncEngine(self.store, self.outbox, self.cloud, log=self.log)
        state_file = self.data / "node_state.json"
        self._state_file = state_file
        st = json.loads(state_file.read_text()) if state_file.exists() else {}
        self.step = int(st.get("step", 0))
        self.restarts = int(st.get("restarts", -1)) + 1
        self.rng = np.random.default_rng(zlib.crc32(self.name.encode()) + self.step)
        self.pos = (0.5, 0.5)
        self.outage_until = 0.0
        self.paused = False
        self.started = time.time()
        self.cpu0 = (time.process_time(), time.time())
        self.last_sync: dict[str, Any] = {}
        self.bytes_sent = 0
        self.cloud_seen = False
        self._save()

    def _save(self) -> None:
        self._state_file.write_text(json.dumps({"step": self.step, "restarts": self.restarts}))

    # -- world ---------------------------------------------------------------------------

    def _observe(self, text: str, x: float, y: float, **kw: Any) -> None:
        vec = self.embedder.embed(text) + 0.01 * self.rng.normal(size=self.embedder.dim)
        vec = vec / np.linalg.norm(vec)
        self.pos = (float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1)))
        self.engine.observe(
            Memory(
                f"{self.name}-{self.step}-{time.time_ns()}",
                vec.tolist(),
                self.pos[0],
                self.pos[1],
                0.0,
                int(time.time() * 1000),
                device_id=self.name,
                text=text,
                **kw,
            )
        )

    def patrol_step(self) -> None:
        self.step += 1
        _, text, (lx, ly) = LANDMARKS[int(self.rng.integers(0, len(LANDMARKS)))]
        self._observe(text, lx + self.rng.normal(0, 0.004), ly + self.rng.normal(0, 0.004))
        self._save()

    def event(self, kind: str) -> None:
        if kind == "spill":
            self._observe("oil spill near dock 9", 0.14, 0.22, metadata={"urgent": True})
        elif kind == "toolbox":
            x, y = TOOLBOX_SPOTS[int(self.rng.integers(0, len(TOOLBOX_SPOTS)))]
            self._observe("red toolbox", x, y)

    # -- network -------------------------------------------------------------------------

    def sync(self) -> None:
        push = self.engine.push()
        summ = (
            self.engine.summarize_pending()
            if self.outbox.count_by_decision().get("SUMMARIZE_SYNC", 0) >= 10
            else None
        )
        pull = self.engine.pull()
        self.bytes_sent += push.bytes_sent + (summ.bytes_sent if summ else 0)
        self.last_sync = {
            "ts_ms": int(time.time() * 1000),
            "sent": push.sent,
            "pulled": pull.pulled,
            "roles": pull.roles_updated,
            "link_down": push.link_down or pull.link_down,
        }

    def telemetry(self) -> dict[str, Any]:
        now_cpu, now = time.process_time(), time.time()
        cpu = 100 * (now_cpu - self.cpu0[0]) / max(now - self.cpu0[1], 1e-6)
        self.cpu0 = (now_cpu, now)
        fp = self.store.footprint()
        return {
            "name": self.name,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "platform": (
                f"{platform.system()} {platform.machine()} / Python {platform.python_version()}"
            ),
            "uptime_s": round(now - self.started, 1),
            "restarts": self.restarts,
            "rss_mb": round(rss_mb(), 1),
            "cpu_pct": round(cpu, 1),
            "footprint": fp,
            "outbox": self.outbox.count_by_decision(),
            "decisions": self.log.counts(),
            "recent_decisions": [d.to_dict() for d in self.log.recent(limit=8)],
            "qdrant_ops": self.store.ops.stats()[:8],
            "qdrant_calls": self.store.ops.total_calls(),
            "pos": self.pos,
            "step": self.step,
            "embedder": self.embedder.name,
            "uplink": "down (commanded outage)" if not self.link.up else "up",
            "last_sync": self.last_sync,
            "bytes_sent": self.bytes_sent,
            "digest": self.store.digest(),
        }

    def apply(self, reply: dict) -> None:
        cfg = reply.get("config") or {}
        dim = cfg.get("dim")
        if dim is not None and int(dim) != self.embedder.dim:
            print(
                f"[{self.name}] embedder dim {self.embedder.dim} != fleet dim {dim}; "
                "start the node with the fleet's embedder",
                flush=True,
            )
        for cmd in reply.get("commands") or []:
            if "outage_s" in cmd:
                self.outage_until = time.time() + float(cmd["outage_s"])
                self.link.set(False)
                print(f"[{self.name}] uplink down for {cmd['outage_s']} s", flush=True)
            if "event" in cmd:
                self.event(str(cmd["event"]))
            if "pause" in cmd:
                self.paused = bool(cmd["pause"])

    def tick(self) -> None:
        if not self.link.up and time.time() >= self.outage_until:
            self.link.set(True)
            self.outbox.retry_now()
            print(f"[{self.name}] uplink restored", flush=True)
        if not self.paused:
            self.patrol_step()
        if self.link.up:
            try:
                self.sync()
                self.apply(self.http.heartbeat(self.telemetry()))
                self.cloud_seen = True
            except LinkDown as exc:
                self.last_sync = {
                    "ts_ms": int(time.time() * 1000),
                    "link_down": True,
                    "error": str(exc)[:120],
                }

    def run(self) -> None:
        stop = {"flag": False}

        def handle(*_: Any) -> None:
            stop["flag"] = True

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)
        print(
            f"[{self.name}] pid {os.getpid()} data={self.data} embedder={self.embedder.name} "
            f"restarts={self.restarts} memories={self.store.count()} "
            f"queued={self.outbox.pending()}",
            flush=True,
        )
        parent = os.getppid()
        while not stop["flag"]:
            if self.exit_with_parent and os.getppid() != parent:
                print(f"[{self.name}] launcher exited; stopping", flush=True)
                break
            t0 = time.time()
            self.tick()
            time.sleep(max(0.0, self.interval - (time.time() - t0)))
        self.close()

    def close(self) -> None:
        self._save()
        self.store.close()
        self.outbox.close()
        self.log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cloud", default="http://127.0.0.1:8765", help="mission control / cloud URL")
    ap.add_argument("--data", default=None, help="data directory (default ./edge-nodes/NAME)")
    ap.add_argument("--token", default=None, help="cloud token (or LOCI_CLOUD_TOKEN)")
    ap.add_argument(
        "--embedder",
        default=os.environ.get("LOCI_EMBEDDER", "hash:64"),
        help="must match the fleet's (default: LOCI_EMBEDDER or hash:64)",
    )
    ap.add_argument("--interval", type=float, default=1.0, help="seconds per patrol step")
    ap.add_argument("--steps", type=int, default=0, help="stop after N ticks (0 = run forever)")
    ap.add_argument(
        "--exit-with-parent", action="store_true", help="stop when the launching process exits"
    )
    args = ap.parse_args()
    node = Node(args)
    if args.steps:
        for _ in range(args.steps):
            node.tick()
        node.close()
        return
    node.run()


if __name__ == "__main__":
    main()
