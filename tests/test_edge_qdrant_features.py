"""Qdrant Edge features the platform relies on: on-device recency decay, MMR, facets, and the
instrumented call log behind the UI's inspector."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_edge")

from loci.edge import EdgeMemoryStore, Memory  # noqa: E402

DIM = 16


def unit(v):
    v = np.asarray(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def store(tmp_path):
    s = EdgeMemoryStore(tmp_path / "s", DIM, "robot-a", mirror_path=tmp_path / "m")
    yield s
    s.close()


def test_recency_decay_prefers_the_recent_sighting_of_the_same_thing(store):
    base = np.random.default_rng(0).normal(size=DIM)
    old = store.put(Memory("old", unit(base), 0.1, 0.5, 0, 1_000_000, text="red toolbox"))
    new = store.put(Memory("new", unit(base + 0.3 * np.random.default_rng(1).normal(size=DIM)),
                           0.9, 0.5, 0, 9_000_000, text="red toolbox"))  # fmt: skip
    q = unit(base)
    assert store.search(vector=q, limit=1)[0].id == old.id  # pure similarity: the old view
    hits = store.search(vector=q, limit=2, recency_half_life_ms=1_000_000, now_ms=9_000_000)
    assert hits[0].id == new.id  # decayed on-device: the recent view wins
    assert hits[1].score < 0.01  # 8 half-lives old: its score is scaled by 2^-8


def test_recency_half_life_semantics(store):
    v = unit(np.ones(DIM))
    store.put(Memory("a", v, 0.5, 0.5, 0, 1_000, text="x"))
    s0 = store.search(vector=v, limit=1)[0].score
    s1 = store.search(vector=v, limit=1, recency_half_life_ms=500, now_ms=1_500)[0].score
    assert s1 == pytest.approx(s0 * 0.5, rel=1e-3)  # one half-life old => half the score


def test_mmr_trades_duplicates_for_coverage(store):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=DIM), rng.normal(size=DIM)
    for i in range(6):  # six near-copies of A
        store.put(Memory(f"a{i}", unit(a + 0.02 * rng.normal(size=DIM)), 0.5, 0.5, 0, 1_000 + i))
    store.put(Memory("b", unit(0.7 * a + 0.7 * b), 0.5, 0.5, 0, 2_000))
    q = unit(a)
    plain = [h.payload["key"] for h in store.search(vector=q, limit=4)]
    diverse = [
        h.payload["key"] for h in store.search(vector=q, limit=4, diverse=True, diversity=0.3)
    ]
    assert "b" not in plain and "b" in diverse


def test_facets_count_on_device(store):
    for i in range(5):
        m = store.put(Memory(f"k{i}", unit(np.eye(DIM)[i]), 0.5, 0.5, 0, 1_000 + i))
        if i < 2:
            store.set_sync_state([m.id], "synced")
    assert store.facets("sync_state") == {"local_only": 3, "synced": 2}


def test_every_engine_call_is_recorded_with_its_real_shape(store):
    store.put(Memory("k", unit(np.ones(DIM)), 0.5, 0.5, 0, 1_000, text="oil spill"))
    store.search(vector=unit(np.ones(DIM)), text="spill", limit=3)
    store.search(vector=unit(np.ones(DIM)), limit=3, recency_half_life_ms=1000, now_ms=2000)
    ops = store.ops.recent(50)
    kinds = [o["op"] for o in ops]
    assert "update" in kinds and "query" in kinds and "retrieve" in kinds
    details = " | ".join(o["detail"] for o in ops if o["op"] == "query")
    assert "nearest(dense)" in details and "nearest(bm25)" in details
    assert "formula(decay)(nearest(dense))" in details
    assert {o["shard"] for o in ops if o["op"] == "query"} == {"local", "mirror"}
    assert all(o["us"] >= 0 and o["device"] == "robot-a" for o in ops)
    stats = {s["kind"]: s for s in store.ops.stats()}
    assert stats["update"]["calls"] >= 1 and stats["update"]["p95_ms"] >= stats["update"]["p50_ms"]


def test_footprint_reports_real_disk_and_points(store):
    for i in range(20):
        store.put(Memory(f"k{i}", unit(np.random.default_rng(i).normal(size=DIM)), 0.5, 0.5, 0, i))
    fp = store.footprint()
    assert fp["points_local"] == 20 and fp["disk_local_bytes"] > 0 and fp["vector_dim"] == DIM
