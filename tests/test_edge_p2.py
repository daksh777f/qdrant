"""P2: surprise-driven sync policy, decision records, summary lane, sync-diff, netsplit."""

from __future__ import annotations

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
    PolicyConfig,
    SyncEngine,
    SyncPolicy,
)
from loci.retrieval.novelty import NoveltyCalibrator  # noqa: E402

DIM = 32


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def obs(key, base, x=0.5, y=0.5, t=1_000, noise=0.03, seed=0, **kw) -> Memory:
    rng = np.random.default_rng(seed)
    vec = base + noise * rng.normal(size=DIM)
    return Memory(key=key, vector=vec.tolist(), x=x, y=y, z=0.0, timestamp_ms=t, **kw)


class Robot:
    def __init__(self, tmp, name, shared_cloud, config=None):
        self.link = Link(True)
        self.store = EdgeMemoryStore(tmp / name, DIM, name, mirror_path=tmp / f"{name}-mirror")
        self.outbox = Outbox(tmp / f"{name}.db")
        self.log = DecisionLog(tmp / f"{name}-decisions.db")
        self.engine = SyncEngine(
            self.store,
            self.outbox,
            LinkedCloud(shared_cloud, self.link),
            # A config means a summary-lane test: let the calibrator be ready immediately.
            policy=SyncPolicy(config, NoveltyCalibrator(min_samples=1) if config else None),
            log=self.log,
        )

    def close(self):
        self.store.close()
        self.outbox.close()
        self.log.close()


@pytest.fixture
def world(tmp_path):
    cloud = LocalCloud(tmp_path / "cloud", DIM)
    robots: list[Robot] = []

    def make(name, config=None):
        r = Robot(tmp_path, name, cloud, config)
        robots.append(r)
        return r

    yield make
    for r in robots:
        r.close()
    cloud.close()


# ---------------------------------------------------------------- policy rules


def test_first_sighting_syncs_and_repeat_at_same_place_is_deduped(world):
    r = world("a")
    dock = unit(1)
    d1 = r.engine.observe(obs("s1", dock, seed=1))
    d2 = r.engine.observe(obs("s2", dock, seed=2))
    assert d1.action == "SYNC_NOW" and "first sighting" in d1.reason
    assert d2.action == "DEDUPE" and d2.neighbor_source == "local"
    assert r.store.count() == 1  # the repeat is not even stored
    assert r.store.states()[d1.point_id]["seen_count"] == 2


def test_moved_object_is_not_deduped(world):
    r = world("a")
    box = unit(2)
    r.engine.observe(obs("s1", box, x=0.1, seed=1))
    d = r.engine.observe(obs("s2", box, x=0.8, seed=2))
    assert d.action == "SYNC_NOW" and "moved" in d.reason and d.displacement > 0.5
    assert r.store.count() == 2  # both places are remembered


def test_private_stays_local_and_urgent_jumps_the_queue(world):
    r = world("a")
    base = unit(3)
    r.engine.observe(obs("s1", base, seed=1))
    priv = r.engine.observe(obs("s2", unit(4), seed=2, private=True))
    urgent = r.engine.observe(obs("s3", base, seed=3, metadata={"urgent": True}))
    assert priv.action == "KEEP_LOCAL" and "private" in priv.reason
    assert urgent.action == "SYNC_NOW" and "urgent" in urgent.reason  # even as a duplicate
    r.engine.push()
    assert r.engine.diff().held_local[priv.point_id] == "private"


def test_novel_observation_after_warmup_syncs(world):
    r = world("a")
    lm = [unit(10 + i) for i in range(4)]
    for i in range(20):
        r.engine.observe(obs(f"w{i}", lm[i % 4], x=0.1 + 0.2 * (i % 4), seed=i))
    d = r.engine.observe(obs("odd", unit(999), x=0.9, seed=99))
    assert d.action == "SYNC_NOW" and d.novelty > 0.6


def test_decision_records_carry_evidence(world):
    r = world("a")
    r.engine.observe(obs("s1", unit(5), seed=1))
    r.engine.observe(obs("s2", unit(5), seed=2))
    rec = r.log.recent(limit=1)[0]
    assert rec.action == "DEDUPE" and rec.best_similarity > 0.95
    assert rec.neighbor_id and rec.thresholds["dedupe_similarity"] == 0.95 and rec.ts_ms > 0
    assert r.log.counts()["SYNC_NOW"] == 1 and r.log.counts()["DEDUPE"] == 1


# ------------------------------------------------------------ fleet awareness


def test_what_robot_a_saw_is_not_novel_to_robot_b(world):
    a, b = world("robot-a"), world("robot-b")
    door = unit(6)
    a.engine.observe(obs("door", door, x=0.3, seed=1))
    a.engine.push()
    b.engine.pull()
    d = b.engine.observe(obs("door-b", door, x=0.3, seed=2))
    assert d.action == "DEDUPE" and d.neighbor_source == "mirror"
    assert b.outbox.pending() == 0 and b.store.count() == 0  # nothing stored, nothing sent


# ------------------------------------------------------------- summary lane


def test_summary_lane_sends_far_fewer_bytes_than_raw(world):
    cfg = PolicyConfig(
        dedupe_similarity=0.9999,
        sync_novelty=2.0,
        summarize_novelty=0.0,
        ambiguous_similarity=0.0,
        hold_back_familiar=True,
    )
    r = world("a", cfg)
    r.engine.observe(obs("first", unit(7), seed=0, metadata={"urgent": True}))  # goes raw
    base = unit(8)
    for i in range(40):
        r.engine.observe(obs(f"f{i}", base, x=0.5 + 0.001 * i, t=2_000 + i, seed=i, noise=0.2))
    assert r.outbox.count_by_decision()["SUMMARIZE_SYNC"] == 40
    raw_bytes = sum(
        len(str(p["payload"])) + 4 * DIM
        for p in r.store.read_for_push(
            [x for x, s in r.store.states().items() if s["sync_state"] == "queued_summary"]
        )
    )
    rep = r.engine.summarize_pending(max_states_per_group=2)
    assert rep.covered == 40 and 0 < rep.bytes_sent < raw_bytes / 5
    assert r.outbox.count_by_decision().get("SUMMARIZE_SYNC", 0) == 0
    assert {s["sync_state"] for s in r.store.states().values()} >= {"summarized"}


def test_summary_waits_out_an_outage(world):
    cfg = PolicyConfig(
        dedupe_similarity=0.9999,
        sync_novelty=2.0,
        summarize_novelty=0.0,
        ambiguous_similarity=0.0,
        hold_back_familiar=True,
    )
    r = world("a", cfg)
    r.engine.observe(obs("first", unit(1), seed=0, metadata={"urgent": True}))
    for i in range(5):
        r.engine.observe(obs(f"f{i}", unit(2), seed=i, noise=0.2))
    r.link.set(False)
    assert r.engine.summarize_pending().link_down
    assert r.outbox.count_by_decision()["SUMMARIZE_SYNC"] == 5  # nothing lost
    r.link.set(True)
    assert r.engine.summarize_pending(now_ms=10**14).covered == 5


# ------------------------------------------------------------- sync-diff / netsplit


def test_diff_shows_plan_then_converges(world):
    r = world("a")
    r.engine.observe(obs("s1", unit(1), seed=1))
    r.engine.observe(obs("s2", unit(2), seed=2, private=True))
    d = r.engine.diff()
    assert len(d.to_push) == 1 and len(d.held_local) == 1 and not d.converged
    r.engine.push()
    d = r.engine.diff()
    assert d.in_sync and not d.to_push and d.converged
    r.link.set(False)
    assert r.engine.diff().cloud_reachable is False


def test_netsplit_both_robots_converge_after_reconnect(world):
    a, b = world("robot-a"), world("robot-b")
    a.link.set(False)
    b.link.set(False)
    ta, tb = unit(21), unit(22)
    for i in range(6):
        a.engine.observe(obs(f"a{i}", unit(100 + i), x=0.1, seed=i))
        b.engine.observe(obs(f"b{i}", unit(200 + i), x=0.9, seed=i))
    # Split: each robot still answers from its own memory, and nothing is lost.
    assert a.store.search(vector=unit(102).tolist(), limit=1)[0].payload["key"] == "a2"
    assert a.engine.push().link_down and b.engine.push().link_down
    assert a.outbox.pending() == 6 and b.outbox.pending() == 6
    a.link.set(True)
    b.link.set(True)
    now = 10**14
    assert a.engine.push(now_ms=now).sent == 6 and b.engine.push(now_ms=now).sent == 6
    a.engine.pull()
    b.engine.pull()
    assert a.engine.diff().converged and b.engine.diff().converged
    # Each robot now finds the other's memories with no network.
    a.link.set(False)
    hit = a.store.search(vector=unit(203).tolist(), limit=1)[0]
    assert hit.payload["key"] == "b3" and hit.source == "mirror"
    assert ta is not tb  # (vectors above only document intent)


# ------------------------------------------------------------- the headline claim


def test_policy_saves_most_bytes_without_losing_landmarks(world):
    """A patrol revisits 5 landmarks 60 times: only surprise crosses the wire."""
    r = world("a")
    lms = [(unit(300 + i), 0.15 + 0.17 * i) for i in range(5)]
    naive = 0
    for k in range(300):
        base, x = lms[k % 5]
        m = obs(f"p{k}", base, x=x, t=1_000 + k, seed=k)
        naive += 4 * DIM + 250  # a naive sync ships every observation (vector + ~250 B payload)
        r.engine.observe(m)
    sent = r.engine.push()
    assert sent.sent <= 8  # ~one per landmark
    assert sent.bytes_sent < 0.05 * naive
    # Recall retained: every landmark is still retrievable from the cloud copy.
    cloud_ids = set(r.engine.cloud.versions())
    for base, _x in lms:
        top = r.store.search(vector=base.tolist(), limit=1)[0]
        assert top.id in cloud_ids


def test_lookalike_of_another_devices_memory_is_sent_for_the_cloud_to_adjudicate(world):
    a, b = world("robot-a"), world("robot-b")
    base = unit(40)
    a.engine.observe(obs("box", base, x=0.3, seed=1))
    a.engine.push()
    b.engine.pull()
    # ~0.90 cosine: not the same thing for sure, not different for sure.
    u = np.random.default_rng(5).normal(size=DIM)
    u -= (u @ base) * base
    u /= np.linalg.norm(u)
    blurry = 0.90 * base + np.sqrt(1 - 0.90**2) * u
    m = Memory("blurry", blurry.tolist(), 0.3, 0.5, 0.0, 2_000)
    d = b.engine.observe(m)
    assert d.action == "SYNC_NOW" and "adjudicate" in d.reason and d.neighbor_source == "mirror"
    # The same look-alike seen by the robot that owns the original stays a local matter.
    d2 = a.engine.observe(Memory("blurry-a", blurry.tolist(), 0.3, 0.5, 0.0, 3_000))
    assert "adjudicate" not in d2.reason


def test_policy_honours_a_caller_supplied_empty_calibrator():
    """Regression: an empty NoveltyCalibrator is falsy (__len__), so `or` used to discard it."""
    mine = NoveltyCalibrator(min_samples=1)
    assert SyncPolicy(None, mine).calibrator is mine


def test_look_alike_at_same_place_is_not_hidden_while_the_device_is_still_learning(world):
    """Regression (found by benchmarks/edge_verify.py): during warm-up the calibrator fell back
    to novelty = 1 - similarity, so a same-place twin (cos ~0.88) was judged 'familiar', kept
    local, and later views deduped against that never-synced item: the cloud never learned it."""
    r = world("a")
    base = unit(50)
    r.engine.observe(obs("original", base, x=0.4, seed=1))
    u = np.random.default_rng(6).normal(size=DIM)
    u -= (u @ base) * base
    u /= np.linalg.norm(u)
    twin = 0.88 * base + np.sqrt(1 - 0.88**2) * u
    d = r.engine.observe(Memory("twin", twin.tolist(), 0.41, 0.5, 0.0, 2_000))
    assert d.action == "SYNC_NOW" and "same spot" in d.reason
    r.engine.push()
    assert r.engine.cloud.count() == 2  # the cloud knows both


def test_genuinely_new_stream_is_never_suppressed_by_relative_novelty(world):
    """Regression (negative control): calibration is relative to history, so a stream in which
    everything is new used to look 'average' and get summarized or kept local."""
    r = world("a")
    rng = np.random.default_rng(3)
    actions = []
    for i in range(60):
        v = rng.normal(size=DIM)
        m = Memory(f"n{i}", (v / np.linalg.norm(v)).tolist(), *rng.random(2), 0.0, 1_000 + i)
        actions.append(r.engine.observe(m).action)
    assert actions.count("SYNC_NOW") >= 57


def test_hold_back_mode_still_sends_until_the_device_has_learned_its_noise(world):
    """In opt-in bandwidth mode, look-alikes are only held back once the calibrator is ready."""
    r = world("a")
    r.engine.policy = SyncPolicy(PolicyConfig(hold_back_familiar=True))  # default calibrator (10)
    base = unit(60)
    r.engine.observe(obs("original", base, x=0.4, seed=1))
    d = r.engine.observe(obs("view", base, x=0.4, seed=2, noise=0.12))  # ~0.9 cosine to original
    assert d.best_similarity is not None and d.best_similarity < 0.95
    assert d.action == "SYNC_NOW" and "not yet learned" in d.reason
