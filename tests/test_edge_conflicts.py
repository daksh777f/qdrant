"""Space-time conflict resolution: pure resolver + cloud reconciler + audit/inbox."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("qdrant_edge")

from loci.edge import (  # noqa: E402
    DecisionLog,
    EdgeMemoryStore,
    Link,
    LinkedCloud,
    LocalCloud,
    Memory,
    Outbox,
    SyncEngine,
)
from loci.edge.conflicts import (  # noqa: E402
    ConflictConfig,
    ConflictLog,
    Reconciler,
    Sighting,
    resolve_entity,
)

DIM = 32
CFG = ConflictConfig()


def S(id, dev="a", t=1_000, x=0.5, y=0.5, conf=1.0):
    return Sighting(id, dev, t, x, y, 0.0, conf)


# ------------------------------------------------------------------ pure resolver


def test_duplicate_sighting_merges_higher_confidence_wins():
    v = resolve_entity([S("a1", "a", conf=0.6), S("b1", "b", conf=0.9, t=1_500)], CFG)
    assert v["b1"].role == "current" and v["a1"].role == "merged" and v["a1"].merged_into == "b1"
    assert v["a1"].entity_devices == ("a", "b") and v["a1"].entity_size == 2


def test_moved_object_keeps_both_newest_is_current():
    v = resolve_entity([S("old", "a", t=1_000, x=0.1), S("new", "b", t=9_000, x=0.8)], CFG)
    assert v["new"].role == "current" and v["old"].role == "previous"
    assert v["old"].merged_into is None


def test_same_place_but_outside_time_window_is_a_resighting_not_a_merge():
    late = 1_000 + CFG.merge_window_ms + 1
    v = resolve_entity([S("early", t=1_000), S("late", t=late)], CFG)
    assert {v["early"].role, v["late"].role} == {"previous", "current"}


def test_chain_of_moves_has_exactly_one_current():
    ms = [S("p1", t=1_000, x=0.1), S("p2", t=100_000, x=0.4), S("p3", t=900_000, x=0.9)]
    v = resolve_entity(ms, CFG)
    assert [v[i].role for i in ("p1", "p2", "p3")] == ["previous", "previous", "current"]


def test_resolution_is_independent_of_arrival_order():
    ms = [
        S("a", "a", t=1_000, x=0.1, conf=0.7),
        S("b", "b", t=1_200, x=0.1, conf=0.9),
        S("c", "a", t=50_000, x=0.6),
        S("d", "b", t=60_000, x=0.6, conf=0.5),
    ]
    expected = resolve_entity(ms, CFG)
    for perm in itertools.permutations(ms):
        assert resolve_entity(list(perm), CFG) == expected


# ------------------------------------------------------------------ integration rig


def unit(seed):
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def tilted(base, cos, seed=99):
    """A unit vector with cosine ``cos`` to ``base``."""
    u = np.random.default_rng(seed).normal(size=DIM)
    u -= (u @ base) * base
    u /= np.linalg.norm(u)
    return cos * base + np.sqrt(1 - cos**2) * u


class Rig:
    def __init__(self, tmp):
        self.cloud = LocalCloud(tmp / "cloud", DIM)
        self.log = ConflictLog(tmp / "conflicts.db")
        self.rec = Reconciler(self.cloud, self.log)
        self.robots = {}
        for name in ("robot-a", "robot-b"):
            link = Link(True)
            store = EdgeMemoryStore(tmp / name, DIM, name, mirror_path=tmp / f"{name}-m")
            self.robots[name] = (
                link,
                store,
                SyncEngine(
                    store,
                    Outbox(tmp / f"{name}.db"),
                    LinkedCloud(self.cloud, link),
                    log=DecisionLog(tmp / f"{name}-d.db"),
                ),
            )

    def see(self, robot, key, vec, x, t, text="red toolbox", conf=1.0, private=False):
        eng = self.robots[robot][2]
        return eng.observe(
            Memory(key, list(vec), x, 0.5, 0.0, t, text=text, confidence=conf, private=private)
        )

    def sync_all(self):
        for _, _, eng in self.robots.values():
            eng.push()
        rep = self.rec.run()
        for _, _, eng in self.robots.values():
            eng.pull()
        return rep

    def close(self):
        for _, store, eng in self.robots.values():
            store.close()
            eng.outbox.close()
            eng.log.close()
        self.log.close()
        self.cloud.close()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def roles(rig):
    return {p["id"]: p["payload"].get("role") for p in rig.cloud.scan()}


def test_two_robots_same_object_same_place_are_merged_not_duplicated(rig):
    tb = unit(1)
    a = rig.see("robot-a", "tb", tb + 0.01 * unit(2), 0.30, 1_000, conf=0.7)
    b = rig.see("robot-b", "tb", tb + 0.01 * unit(3), 0.31, 2_000, conf=0.9)
    rep = rig.sync_all()
    r = roles(rig)
    assert rep.merged == 1 and r[b.point_id] == "current" and r[a.point_id] == "merged"
    row = rig.log.list()[0]
    assert row["rule"] == "merge_duplicate" and row["status"] == "auto"
    assert row["evidence"]["similarity"] > 0.95 and row["evidence"]["winner"] == b.point_id
    assert set(rig.cloud.scan()[0]["payload"]["entity_devices"]) == {"robot-a", "robot-b"}


def test_moved_object_keeps_both_and_where_is_it_now_returns_newest(rig):
    tb = unit(4)
    a = rig.see("robot-a", "tb", tb, 0.10, 1_000)
    b = rig.see("robot-b", "tb", tb, 0.85, 9_000)
    rep = rig.sync_all()
    r = roles(rig)
    assert rep.moved == 1 and r[a.point_id] == "previous" and r[b.point_id] == "current"
    assert rig.log.list()[0]["rule"] == "moved"
    # Robot A learns its own sighting is now historical, and finds only the current spot.
    store_a = rig.robots["robot-a"][1]
    assert store_a.get([a.point_id])[0].payload["role"] == "previous"  # verdict applied locally
    assert store_a.states()[a.point_id]["version"] == 1  # content/version untouched
    now = store_a.search(vector=list(tb), limit=5, current_only=True)
    assert [h.id for h in now] == [b.point_id] and now[0].source == "mirror"
    both = store_a.search(vector=list(tb), limit=5, current_only=False)
    assert {h.id for h in both} == {a.point_id, b.point_id}


def test_ambiguous_match_goes_to_inbox_and_is_not_auto_merged(rig):
    tb = unit(5)
    a = rig.see("robot-a", "x", tb, 0.30, 1_000)
    b = rig.see("robot-b", "x", tilted(tb, 0.90), 0.31, 2_000, text="toolbox?")
    rep = rig.sync_all()
    assert rep.reviews_opened == 1 and rep.merged == 0
    assert set(roles(rig).values()) == {None}
    item = rig.log.list(status="pending_review")[0]
    assert 0.85 < item["evidence"]["similarity"] < 0.95
    assert {item["a_id"], item["b_id"]} == {a.point_id, b.point_id}


def test_operator_approval_merges_and_rejection_never_merges(rig):
    tb = unit(6)
    rig.see("robot-a", "x", tb, 0.30, 1_000)
    rig.see("robot-b", "x", tilted(tb, 0.90), 0.31, 2_000)
    rig.sync_all()
    cid = rig.log.list(status="pending_review")[0]["id"]
    assert rig.log.resolve(cid, approve=True)["status"] == "approved"
    assert rig.log.resolve(cid, approve=True) is None  # already decided
    rig.sync_all()
    assert sorted(v for v in roles(rig).values()) == ["current", "merged"]

    # A separate pair, rejected: stays separate on every later pass.
    rig.see("robot-a", "y", unit(7), 0.7, 3_000)
    rig.see("robot-b", "y", tilted(unit(7), 0.90, seed=3), 0.71, 4_000)
    rig.sync_all()
    cid2 = rig.log.list(status="pending_review")[0]["id"]
    rig.log.resolve(cid2, approve=False)
    rig.sync_all()
    rig.sync_all()
    assert list(roles(rig).values()).count("merged") == 1


def test_reconcile_is_idempotent_and_audit_is_not_duplicated(rig):
    tb = unit(8)
    rig.see("robot-a", "tb", tb, 0.2, 1_000)
    rig.see("robot-b", "tb", tb, 0.9, 5_000)
    rig.sync_all()
    n = len(rig.log.list())
    for _ in range(3):
        rep = rig.sync_all()
        assert rep.points_updated == 0 and rep.merged == 0 and rep.moved == 0
    assert len(rig.log.list()) == n == 1


def test_unrelated_objects_are_untouched(rig):
    rig.see("robot-a", "one", unit(10), 0.2, 1_000, text="dock")
    rig.see("robot-b", "two", unit(11), 0.2, 1_000, text="aisle")
    rep = rig.sync_all()
    assert rep.entities == 0 and set(roles(rig).values()) == {None} and rig.log.list() == []


def test_roles_reach_both_devices_and_they_converge(rig):
    tb = unit(12)
    rig.see("robot-a", "tb", tb, 0.1, 1_000)
    rig.see("robot-b", "tb", tb, 0.8, 9_000)
    rig.sync_all()
    for name in ("robot-a", "robot-b"):
        assert rig.robots[name][2].diff().converged, name


def test_private_observation_never_becomes_a_cloud_conflict(rig):
    tb = unit(13)
    rig.see("robot-a", "tb", tb, 0.3, 1_000, private=True)
    rig.see("robot-b", "tb", tb, 0.3, 1_100)
    rep = rig.sync_all()
    assert rig.cloud.count() == 1 and rep.entities == 0 and rig.log.list() == []
