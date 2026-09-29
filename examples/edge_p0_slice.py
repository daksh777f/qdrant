"""P0 vertical slice: a patrol robot remembers offline, then syncs on reconnect.

    pip install -e ".[edge]" && python examples/edge_p0_slice.py

Everything here is SYNTHETIC: random vectors stand in for camera embeddings and
LocalCloud stands in for a Qdrant Server. The Edge shard, the SQLite outbox,
the retry/backoff and the sync are real.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from loci.edge import EdgeMemoryStore, Link, LocalCloud, Memory, Outbox, SyncEngine

DIM = 64


def main() -> None:
    rng = np.random.default_rng(0)
    work = Path(tempfile.mkdtemp(prefix="loci-edge-"))
    link = Link(up=True)
    store = EdgeMemoryStore(work / "robot-a", DIM, "robot-a")
    outbox = Outbox(work / "outbox.db", backoff_base_s=0.2)
    cloud = LocalCloud(work / "cloud", DIM, link)
    engine = SyncEngine(store, outbox, cloud)

    print("1) NETWORK CUT. The robot patrols and keeps remembering.")
    link.set(False)
    t0 = int(time.time() * 1000)
    tool = rng.normal(size=DIM)
    for step in range(30):
        vec = tool if step == 12 else rng.normal(size=DIM)
        text = "red toolbox near dock 4" if step == 12 else f"corridor frame {step}"
        m = store.put(
            Memory(
                f"frame-{step}",
                vec.tolist(),
                x=step / 30,
                y=0.5,
                z=0.0,
                timestamp_ms=t0 + step * 100,
                text=text,
                private=(step == 20),
            )
        )
        engine.submit(m.id, private=m.private)
    print(f"   stored={store.count()}  queued={outbox.pending()}  (1 memory is private)")

    t = time.perf_counter()
    hits = store.search(vector=tool.tolist(), text="toolbox", limit=1)
    ms = (time.perf_counter() - t) * 1000
    print(f"   offline hybrid search: '{hits[0].payload['text']}' in {ms:.1f} ms")
    rep = engine.push()
    print(f"   push while offline: sent={rep.sent} failed={rep.failed} (nothing lost)")

    print("2) NETWORK BACK. Backoff elapses, outbox drains.")
    link.set(True)
    rep = engine.push(now_ms=int(time.time() * 1000) + 60_000)
    print(f"   sent={rep.sent} memories, {rep.bytes_sent} bytes on the wire")
    records = store.get(list(store.versions()))
    private_ids = [str(r.id) for r in records if r.payload["private"]]
    print(f"   private memories kept on device: {len(private_ids)}")
    synced = cloud.versions()
    converged = all(synced.get(i) == v for i, v in store.versions().items() if i not in private_ids)
    print(f"   cloud holds {cloud.count()} memories; converged with edge: {converged}")
    print(f"   edge digest={store.digest()}")


if __name__ == "__main__":
    main()
