"""P2 demo: two patrol robots, a network split, and only surprise crossing the wire.

    pip install -e ".[edge]" && python examples/edge_p2_patrol.py

SYNTHETIC: random vectors stand in for camera embeddings; LocalCloud stands in
for a Qdrant Server. The Edge shards, policy, outbox, sync and diff are real.
"""

from __future__ import annotations

import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from loci.edge import (
    DecisionLog,
    EdgeMemoryStore,
    Link,
    LinkedCloud,
    LocalCloud,
    Memory,
    Outbox,
    SyncEngine,
)

DIM = 64


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


class Robot:
    def __init__(self, work: Path, name: str, cloud: LocalCloud) -> None:
        self.name = name
        self.link = Link(True)
        self.store = EdgeMemoryStore(work / name, DIM, name, mirror_path=work / f"{name}-mirror")
        self.log = DecisionLog(work / f"{name}-decisions.db")
        self.engine = SyncEngine(
            self.store, Outbox(work / f"{name}.db"), LinkedCloud(cloud, self.link), log=self.log
        )

    def see(self, key: str, base: np.ndarray, x: float, t: int, text: str = "", **kw) -> None:
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        vec = base + 0.02 * rng.normal(size=DIM)
        self.engine.observe(Memory(key, vec.tolist(), x, 0.5, 0.0, t, text=text, **kw))


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="loci-p2-"))
    cloud = LocalCloud(work / "cloud", DIM)
    a, b = Robot(work, "robot-a", cloud), Robot(work, "robot-b", cloud)
    landmarks = [(unit(i), 0.1 + 0.2 * i, f"landmark {i}") for i in range(5)]
    t0 = int(time.time() * 1000)

    print("1) NETWORK SPLIT: both robots patrol offline and keep deciding locally")
    a.link.set(False)
    b.link.set(False)
    naive_bytes = 0
    for k in range(200):
        base, x, name = landmarks[k % 5]
        a.see(f"a{k}", base, x, t0 + k, name)
        naive_bytes += 4 * DIM + 250
    a.see("spill", unit(77), 0.95, t0 + 300, "oil spill near dock 9", metadata={"urgent": True})
    b.see("secret", unit(88), 0.5, t0 + 301, "operator badge left on shelf", private=True)
    counts = a.log.counts()
    print(f"   robot-a: 201 observations -> {counts}")
    print(f"   robot-a stored {a.store.count()} memories, queued {a.engine.outbox.pending()}")
    print("   why the last repeat stayed home:", a.log.recent(limit=1, action="DEDUPE")[0].reason)

    print("2) RECONNECT: push, then pull, then compare")
    a.link.set(True)
    b.link.set(True)
    rep = a.engine.push(now_ms=t0 + 10**9)
    b.engine.push(now_ms=t0 + 10**9)
    b.engine.pull()
    a.engine.pull()
    summ = a.engine.summarize_pending(now_ms=t0 + 10**9)
    total = rep.bytes_sent + summ.bytes_sent
    print(f"   robot-a sent {rep.sent} raw memories ({rep.bytes_sent} B) + summaries of")
    print(f"   {summ.covered} observations ({summ.bytes_sent} B) = {total} B")
    print(f"   naive sync would ship every observation: ~{naive_bytes} B (estimate)")
    print(f"   bytes saved: {100 * (1 - total / naive_bytes):.1f}%")
    d = a.engine.diff()
    print(
        f"   diff: converged={d.converged} in_sync={len(d.in_sync)} held_local={len(d.held_local)}"
    )

    print("3) robot-b goes offline again and still knows about the spill robot-a saw")
    b.link.set(False)
    hit = b.store.search(vector=unit(77).tolist(), text="spill", limit=1)[0]
    print(f"   robot-b finds '{hit.payload['text']}' from the fleet mirror ({hit.source})")
    on_cloud = cloud.get(list(cloud.versions()))
    leaked = any("badge" in p["payload"].get("text", "") for p in on_cloud)
    print(f"   robot-b's private memory reached the cloud: {leaked}")


if __name__ == "__main__":
    main()
