"""P0 vertical slice: Edge store -> durable outbox -> LocalCloud, across an outage."""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("qdrant_edge")

from loci.edge import EdgeMemoryStore, Link, LocalCloud, Memory, Outbox, SyncEngine  # noqa: E402

DIM = 16


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).normal(size=DIM).tolist()


def _mem(key: str, seed: int, x=0.5, y=0.5, z=0.0, t=1_000, text="", **kw) -> Memory:
    return Memory(key=key, vector=_vec(seed), x=x, y=y, z=z, timestamp_ms=t, text=text, **kw)


@pytest.fixture
def rig(tmp_path):
    link = Link(True)
    store = EdgeMemoryStore(tmp_path / "edge", DIM, "robot-a")
    outbox = Outbox(tmp_path / "outbox.db", backoff_base_s=1.0, rng=random.Random(0))
    cloud = LocalCloud(tmp_path / "cloud", DIM, link)
    engine = SyncEngine(store, outbox, cloud)
    yield store, outbox, cloud, engine, link
    outbox.close()
    store.close()
    cloud.close()


def test_dense_search_finds_exact_memory(rig):
    store, *_ = rig
    for i in range(20):
        store.put(_mem(f"m{i}", i, x=i / 20))
    hit = store.search(vector=_vec(7), limit=1)[0]
    assert hit.payload["key"] == "m7"


def test_hybrid_search_uses_text_when_vector_is_uninformative(rig):
    store, *_ = rig
    store.put(_mem("a", 1, text="red toolbox near loading dock"))
    store.put(_mem("b", 2, text="blue pallet in aisle three"))
    store.put(_mem("c", 3, text="forklift charging station"))
    hits = store.search(vector=_vec(99), text="toolbox", limit=3)
    assert hits[0].payload["key"] == "a"


def test_spatial_and_time_filters_are_exact(rig):
    store, *_ = rig
    store.put(_mem("left", 1, x=0.1, t=1_000))
    store.put(_mem("right", 1, x=0.9, t=1_000))
    store.put(_mem("late", 1, x=0.1, t=9_000))
    region = {"x_min": 0.0, "x_max": 0.3, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}
    keys = {h.payload["key"] for h in store.search(vector=_vec(1), bounds=region, limit=10)}
    assert keys == {"left", "late"}
    keys = {
        h.payload["key"]
        for h in store.search(vector=_vec(1), bounds=region, time_window_ms=(0, 5_000), limit=10)
    }
    assert keys == {"left"}


def test_same_key_same_id_and_version_increments(rig):
    store, *_ = rig
    a = store.put(_mem("obj", 1))
    b = store.put(_mem("obj", 2))
    assert a.id == b.id and (a.version, b.version) == (1, 2)
    assert store.count() == 1


def test_offline_writes_survive_outage_and_converge(rig):
    store, outbox, cloud, engine, link = rig
    link.set(False)
    for i in range(10):
        m = store.put(_mem(f"m{i}", i))
        engine.submit(m.id)
    # Offline: search still works, push fails but loses nothing.
    assert store.search(vector=_vec(3), limit=1)[0].payload["key"] == "m3"
    rep = engine.push(now_ms=0)
    assert rep.link_down and rep.sent == 0 and outbox.pending() == 10
    # Backoff: an immediate retry is not due yet.
    assert engine.push(now_ms=0).sent == 0 and outbox.due(now_ms=0) == []
    link.set(True)
    rep = engine.push(now_ms=10**9)
    assert rep.sent == 10 and rep.bytes_sent > 0 and outbox.pending() == 0
    assert cloud.versions() == store.versions()
    assert {r.payload["sync_state"] for r in store.get(list(store.versions()))} == {"synced"}


def test_repeated_push_is_idempotent(rig):
    store, outbox, cloud, engine, _ = rig
    m = store.put(_mem("obj", 1))
    engine.submit(m.id)
    engine.push()
    engine.submit(m.id)
    engine.push()
    assert cloud.count() == 1


def test_private_memory_never_leaves_device(rig):
    store, outbox, cloud, engine, _ = rig
    m = store.put(_mem("secret", 1, private=True))
    assert engine.submit(m.id, private=True) == "KEEP_LOCAL"
    # Even if it were forced into the outbox, the push filters it.
    outbox.enqueue(m.id)
    rep = engine.push()
    assert rep.sent == 0 and rep.skipped_private == 1 and cloud.count() == 0


def test_update_after_queue_sends_latest_version(rig):
    store, _, cloud, engine, _ = rig
    m = store.put(_mem("obj", 1))
    engine.submit(m.id)
    store.put(_mem("obj", 2))  # newer version before the push
    engine.push()
    assert cloud.versions()[m.id] == 2


def test_outbox_and_store_survive_restart(tmp_path):
    store = EdgeMemoryStore(tmp_path / "edge", DIM, "robot-a")
    outbox = Outbox(tmp_path / "ob.db")
    m = store.put(_mem("obj", 1))
    outbox.enqueue(m.id)
    store.close()
    outbox.close()
    store2 = EdgeMemoryStore(tmp_path / "edge", DIM, "robot-a")
    outbox2 = Outbox(tmp_path / "ob.db")
    assert store2.count() == 1 and outbox2.pending() == 1
    assert store2.search(vector=_vec(1), limit=1)[0].payload["key"] == "obj"
    store2.close()
    outbox2.close()


def test_digest_detects_divergence(rig):
    store, _, cloud, engine, _ = rig
    m = store.put(_mem("obj", 1))
    d1 = store.digest()
    store.put(_mem("obj", 2))
    assert store.digest() != d1
    engine.submit(m.id)
    engine.push()
    assert cloud.versions() == store.versions()


# --------------------------------------------------------------------------
# P1: fleet mirror + delta pull
# --------------------------------------------------------------------------


@pytest.fixture
def fleet(tmp_path):
    """Two robots sharing one LocalCloud; each with a writable shard + mirror."""
    link_a, link_b = Link(True), Link(True)
    cloud_a = LocalCloud(tmp_path / "cloud", DIM, link_a)
    a = EdgeMemoryStore(tmp_path / "a", DIM, "robot-a", mirror_path=tmp_path / "a-mirror")
    b = EdgeMemoryStore(tmp_path / "b", DIM, "robot-b", mirror_path=tmp_path / "b-mirror")
    # Robot B reaches the same cloud shard through its own link.
    cloud_b = cloud_a
    cloud_b._link = link_b
    ea = SyncEngine(a, Outbox(tmp_path / "a.db"), cloud_a)
    eb = SyncEngine(b, Outbox(tmp_path / "b.db"), cloud_b)
    yield a, b, ea, eb, link_a, link_b
    a.close()
    b.close()
    cloud_a.close()


def test_robot_b_finds_what_robot_a_saw_while_offline(fleet):
    a, b, ea, eb, _, link_b = fleet
    m = a.put(_mem("toolbox", 5, text="red toolbox near dock 4"))
    ea.submit(m.id)
    assert ea.push().sent == 1
    assert b.search(vector=_vec(5), limit=1, include_mirror=True) == []  # not pulled yet
    rep = eb.pull()
    assert rep.pulled == 1 and rep.bytes_received > 0
    link_b.set(False)  # B goes offline; the knowledge is already local
    hit = b.search(vector=_vec(5), text="toolbox", limit=1)[0]
    assert hit.id == m.id and hit.source == "mirror"


def test_pull_is_delta_and_skips_own_points(fleet):
    a, b, ea, eb, *_ = fleet
    m1 = a.put(_mem("one", 1))
    mine = b.put(_mem("mine", 9))
    ea.submit(m1.id)
    eb.submit(mine.id)
    ea.push()
    eb.push()
    assert eb.pull().pulled == 1  # A's point only; B's own point is not mirrored
    assert eb.pull().pulled == 0  # nothing new -> nothing fetched
    m1 = a.put(_mem("one", 2))  # A updates the memory -> version 2
    ea.submit(m1.id)
    ea.push()
    assert eb.pull().pulled == 1 and b.mirror_versions()[m1.id] == 2


def test_pull_offline_changes_nothing(fleet):
    a, b, ea, eb, _, link_b = fleet
    ea.submit(a.put(_mem("x", 1)).id)
    ea.push()
    link_b.set(False)
    rep = eb.pull()
    assert rep.link_down and rep.pulled == 0 and b.mirror_versions() == {}


def test_local_copy_wins_over_mirror_on_same_id(tmp_path):
    s = EdgeMemoryStore(tmp_path / "s", DIM, "robot-a", mirror_path=tmp_path / "m")
    m = s.put(_mem("obj", 1, text="local wins"))
    s.mirror_upsert(
        [{"id": m.id, "vector": _vec(1), "payload": {"text": "stale mirror", "version": 1}}]
    )
    hits = s.search(vector=_vec(1), limit=5)
    assert [h.source for h in hits if h.id == m.id] == ["local"]
    s.close()


def test_hybrid_search_spans_local_and_mirror(fleet):
    a, b, ea, eb, *_ = fleet
    ma = a.put(_mem("far", 1, text="blue pallet aisle nine"))
    ea.submit(ma.id)
    ea.push()
    eb.pull()
    b.put(_mem("near", 2, text="forklift charging station"))
    top = b.search(vector=_vec(77), text="pallet", limit=2)
    assert top[0].id == ma.id and {h.source for h in top} == {"mirror", "local"}
