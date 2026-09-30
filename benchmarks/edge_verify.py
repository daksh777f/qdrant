# ruff: noqa: E501
"""One command that re-measures every headline claim about the edge platform.

    python benchmarks/edge_verify.py            # full run (about 2 minutes)
    python benchmarks/edge_verify.py --quick    # smaller streams (about 30 seconds)

Everything is synthetic and seeded, so runs are repeatable. Each check prints PASS or
FAIL and the process exits non-zero if any fails. Outputs:

    benchmarks/results/edge_verify.json   raw numbers
    benchmarks/results/edge_verify.md     tables (the "table view" of the chart)
    benchmarks/results/edge_hero.svg      bytes vs. recall, place-aware vs. vector-only

What is measured, and how it can mislead (read this before quoting a number):

* Synthetic stream: 32 objects (12 single, 6 pairs of look-alikes in different places,
  4 pairs of look-alikes in the SAME place), 8 of which move once. Observations are noisy
  views (cosine to the object about 0.97). Real data has messier structure.
* "Event recall": an event is (object, position). It counts as retrievable when the cloud
  holds a point that is cosine >= 0.9 to the object and within 0.06 of its position. Summary
  centroids count, so recall is slightly generous when look-alikes are merged into one.
* "Bytes": measured wire bytes (JSON payload + float32 vector) actually pushed, not estimates.
  The naive arm ships every observation.
* The text embedder is a hashed bag-of-words stand-in; thresholds must be re-measured for a
  real model (the script is how).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_latency  # noqa: E402

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
from loci.edge.conflicts import ConflictLog, Reconciler  # noqa: E402
from loci.edge.embed import HashEmbedder  # noqa: E402
from loci.edge.gate import ANSWER_LOCAL, AnswerGate, GateConfig  # noqa: E402

DIM = 64
RESULTS = Path(__file__).parent / "results"
THRESHOLDS_FULL = [0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70]
THRESHOLDS_QUICK = [0.97, 0.95, 0.90, 0.80]
DEFAULT_THRESHOLD = 0.95


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def tilted(base: np.ndarray, cos: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.normal(size=base.shape)
    u -= (u @ base) * base
    return cos * base + math.sqrt(1 - cos**2) * unit(u)


# ------------------------------------------------------------------ labelled stream


def make_world(seed: int, n_obs: int):
    """Objects with homes (and a few moves), plus a noisy observation stream."""
    rng = np.random.default_rng(seed)
    objs: list[dict] = []  # each: proto, homes [(epoch0 pos), (epoch1 pos) or None]

    def add(proto, pos, pos2=None):
        objs.append({"proto": proto, "pos": [pos, pos2]})

    def spot():
        return (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)))

    for _ in range(12):
        add(unit(rng.normal(size=DIM)), spot())
    for _ in range(6):  # look-alikes in different places: space can tell them apart
        a = unit(rng.normal(size=DIM))
        pa, pb = spot(), spot()
        while math.dist(pa, pb) < 0.3:
            pb = spot()
        add(a, pa)
        add(tilted(a, 0.88, rng), pb)
    for _ in range(4):  # look-alikes in the SAME place: only the vectors differ
        a = unit(rng.normal(size=DIM))
        p = spot()
        add(a, p)
        add(tilted(a, 0.88, rng), (p[0] + 0.01, p[1]))
    for i in rng.choice(len(objs), 8, replace=False):
        new = spot()
        while math.dist(new, objs[i]["pos"][0]) < 0.25:
            new = spot()
        objs[i]["pos"][1] = new

    events = {}
    for oi, o in enumerate(objs):
        for ep in (0, 1):
            if o["pos"][ep] is not None:
                events[(oi, ep)] = (o["proto"], o["pos"][ep])

    obs = []
    for t in range(n_obs):
        oi = int(rng.integers(0, len(objs)))
        ep = 1 if (t >= n_obs // 2 and objs[oi]["pos"][1] is not None) else 0
        base, pos = objs[oi]["proto"], objs[oi]["pos"][ep]
        vec = unit(base + 0.025 * rng.normal(size=DIM))
        xy = (pos[0] + rng.normal(0, 0.004), pos[1] + rng.normal(0, 0.004))
        obs.append(
            {
                "key": f"o{t}",
                "vec": vec,
                "x": float(np.clip(xy[0], 0, 1)),
                "y": float(np.clip(xy[1], 0, 1)),
                "ts": 1_000_000 + t * 1_000,
                "event": (oi, ep),
            }
        )
    # Ground truth is events that actually occurred in the stream: an event nobody saw cannot
    # be retrieved by anyone, including the send-everything baseline.
    seen = {o["event"] for o in obs}
    events = {k: v for k, v in events.items() if k in seen}
    return events, obs


def event_recall(cloud_points: list[dict], events: dict) -> float:
    pts = [
        (np.asarray(p["vector"]), (p["payload"]["x"], p["payload"]["y"]))
        for p in cloud_points
        if p["payload"].get("kind") != "insight" and "x" in p["payload"]
    ]
    found = 0
    for proto, pos in events.values():
        for v, xy in pts:
            cos = float(v @ proto / (np.linalg.norm(v) * np.linalg.norm(proto)))
            if cos >= 0.9 and math.dist(xy, pos) <= 0.06:
                found += 1
                break
    return found / len(events)


class Rig:
    """One robot + a LocalCloud in a temp dir."""

    def __init__(self, policy: SyncPolicy | None = None, name: str = "robot"):
        self.tmp = Path(tempfile.mkdtemp(prefix="loci-verify-"))
        self.link = Link(True)
        self.cloud = LocalCloud(self.tmp / "cloud", DIM)
        self.store = EdgeMemoryStore(self.tmp / name, DIM, name, mirror_path=self.tmp / f"{name}-m")
        self.outbox = Outbox(self.tmp / f"{name}.db")
        self.log = DecisionLog(self.tmp / f"{name}-d.db")
        self.engine = SyncEngine(
            self.store, self.outbox, LinkedCloud(self.cloud, self.link), policy=policy, log=self.log
        )

    def close(self):
        self.store.close()
        self.outbox.close()
        self.log.close()
        self.cloud.close()


def run_arm(kind: str, threshold: float, events, obs, **cfg_extra) -> dict:
    """kind: 'naive' | 'place_aware' | 'vector_only'."""
    if kind == "naive":
        rig = Rig()
    else:
        radius = 0.05 if kind == "place_aware" else 99.0
        rig = Rig(
            SyncPolicy(
                PolicyConfig(dedupe_similarity=threshold, same_place_radius=radius, **cfg_extra)
            )
        )
    counts: dict[str, int] = {}
    for o in obs:
        mem = Memory(o["key"], o["vec"].tolist(), o["x"], o["y"], 0.0, o["ts"])
        if kind == "naive":
            rig.engine.submit(rig.store.put(mem).id)
        else:
            d = rig.engine.observe(mem, now_ms=o["ts"])
            counts[d.action] = counts.get(d.action, 0) + 1
    far = 10**15
    push = rig.engine.push(now_ms=far)
    summ = rig.engine.summarize_pending(now_ms=far)
    res = {
        "bytes": push.bytes_sent + summ.bytes_sent,
        "raw_sent": push.sent,
        "summarized": summ.covered,
        "recall": event_recall(rig.cloud.scan(), events),
        "decisions": counts,
    }
    rig.close()
    return res


def sweep(events, obs, thresholds) -> dict:
    naive = run_arm("naive", 0.0, events, obs)
    rows = []
    for th in thresholds:
        pa = run_arm("place_aware", th, events, obs)
        vo = run_arm("vector_only", th, events, obs)
        rows.append(
            {
                "threshold": th,
                "place_aware": {**pa, "bytes_pct": round(100 * pa["bytes"] / naive["bytes"], 2)},
                "vector_only": {**vo, "bytes_pct": round(100 * vo["bytes"] / naive["bytes"], 2)},
            }
        )
    hb = run_arm("place_aware", DEFAULT_THRESHOLD, events, obs, hold_back_familiar=True)
    hb["bytes_pct"] = round(100 * hb["bytes"] / naive["bytes"], 2)
    return {
        "naive": naive,
        "rows": rows,
        "hold_back_mode": hb,
        "n_observations": len(obs),
        "n_events": len(events),
        # Floor: every event sent exactly once. Bytes % cannot go lower than this, and it depends
        # on how repetitive the stream is, which is why the check below is a plain "saves most".
        "ideal_bytes_pct": round(100 * len(events) / len(obs), 1),
    }


# ------------------------------------------------------------------ controls & checks


def negative_controls(n: int = 300) -> dict:
    rng = np.random.default_rng(11)
    rig = Rig()
    synced = 0
    for i in range(n):  # every memory is genuinely new: nothing may be suppressed
        d = rig.engine.observe(
            Memory(f"r{i}", unit(rng.normal(size=DIM)).tolist(), *rng.random(2), 0.0, 1_000 + i),
            now_ms=1_000 + i,
        )
        synced += d.action == "SYNC_NOW"
    rig.close()
    rig = Rig()
    base = unit(rng.normal(size=DIM))
    kept_back = 0
    for i in range(n):  # the same thing over and over: almost all should stay home
        d = rig.engine.observe(
            Memory(
                f"s{i}", unit(base + 0.02 * rng.normal(size=DIM)).tolist(), 0.4, 0.4, 0.0, 1_000 + i
            ),
            now_ms=1_000 + i,
        )
        kept_back += d.action != "SYNC_NOW"
    rig.close()
    return {
        "n": n,
        "random_memories_synced_pct": round(100 * synced / n, 1),
        "repeated_memory_suppressed_pct": round(100 * kept_back / n, 1),
    }


def convergence(m: int = 150) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="loci-conv-"))
    cloud = LocalCloud(tmp / "cloud", DIM)
    rec = Reconciler(cloud, ConflictLog(tmp / "c.db"))
    robots = {}
    rng = np.random.default_rng(5)
    shared = [unit(rng.normal(size=DIM)) for _ in range(10)]
    for name in ("robot-a", "robot-b"):
        link = Link(False)
        store = EdgeMemoryStore(tmp / name, DIM, name, mirror_path=tmp / f"{name}-m")
        eng = SyncEngine(store, Outbox(tmp / f"{name}.db"), LinkedCloud(cloud, link))
        robots[name] = (link, store, eng)
    for name, (_, _, eng) in robots.items():
        for i in range(m):
            base = shared[i % 10] if i % 3 == 0 else unit(rng.normal(size=DIM))
            eng.observe(
                Memory(
                    f"{name}-{i}", unit(base + 0.02 * rng.normal(size=DIM)).tolist(),
                    float(rng.random()), float(rng.random()), 0.0, 1_000 + i,
                ),
                now_ms=1_000 + i,
            )  # fmt: skip
    for link, _, eng in robots.values():
        link.set(True)
        eng.outbox.retry_now()
    t0 = time.perf_counter()
    rounds = 0
    while rounds < 6:
        rounds += 1
        for _, _, eng in robots.values():
            eng.push()
            eng.summarize_pending()
        rec.run()
        for _, _, eng in robots.values():
            eng.pull()
        if all(eng.diff().converged for _, _, eng in robots.values()):
            break
    ms = (time.perf_counter() - t0) * 1000
    ok = all(eng.diff().converged for _, _, eng in robots.values())
    out = {
        "offline_observations_per_robot": m,
        "converged": ok,
        "rounds": rounds,
        "reconnect_to_converged_ms": round(ms, 1),
        "cloud_memories": cloud.count(),
    }
    for _, store, eng in robots.values():
        store.close()
        eng.outbox.close()
    cloud.close()
    return out


def conflict_accuracy(n_each: int = 20) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="loci-conf-"))
    cloud = LocalCloud(tmp / "cloud", DIM)
    log = ConflictLog(tmp / "c.db")
    rng = np.random.default_rng(3)
    ids: dict[str, list[tuple[str, str]]] = {
        "dup": [],
        "moved": [],
        "distinct": [],
        "ambiguous": [],
    }
    pts = []

    def point(pid, dev, vec, x, y, t, conf=1.0):
        pts.append(
            {
                "id": pid,
                "vector": vec.tolist(),
                "payload": {
                    "text": pid, "x": x, "y": y, "z": 0.0, "timestamp_ms": t, "device_id": dev,
                    "confidence": conf, "version": 1, "private": False,
                },
            }
        )  # fmt: skip

    def uid(tag, i, k):
        return str(__import__("uuid").uuid5(__import__("uuid").NAMESPACE_URL, f"{tag}{i}{k}"))

    for i in range(n_each):
        v, x, y = unit(rng.normal(size=DIM)), float(rng.uniform(0.05, 0.4)), float(rng.random())
        a, b = uid("dup", i, "a"), uid("dup", i, "b")
        point(a, "a", unit(v + 0.01 * rng.normal(size=DIM)), x, y, 1_000, 0.6)
        point(b, "b", unit(v + 0.01 * rng.normal(size=DIM)), x + 0.004, y, 2_000, 0.9)
        ids["dup"].append((a, b))
        v2, y2 = unit(rng.normal(size=DIM)), float(rng.random())
        a, b = uid("mv", i, "a"), uid("mv", i, "b")
        point(a, "a", v2, 0.55, y2, 1_000)
        point(b, "b", v2, 0.95, y2, 9_000)
        ids["moved"].append((a, b))
        a, b = uid("ds", i, "a"), uid("ds", i, "b")
        point(a, "a", unit(rng.normal(size=DIM)), 0.6, float(rng.random()), 1_000)
        point(b, "b", unit(rng.normal(size=DIM)), 0.62, float(rng.random()), 1_500)
        ids["distinct"].append((a, b))
        v3, y3 = unit(rng.normal(size=DIM)), float(rng.random())
        a, b = uid("am", i, "a"), uid("am", i, "b")
        point(a, "a", v3, 0.8, y3, 1_000)
        point(b, "b", tilted(v3, 0.90, rng), 0.804, y3, 1_500)
        ids["ambiguous"].append((a, b))
    cloud.upsert(pts)
    Reconciler(cloud, log).run()
    role = {p["id"]: p["payload"].get("role") for p in cloud.scan()}
    pending = {(c["a_id"], c["b_id"]) for c in log.list(status="pending_review", limit=1000)}
    pending |= {(b, a) for a, b in pending}

    def ok(kind, a, b):
        ra, rb = role.get(a), role.get(b)
        if kind == "dup":
            return sorted([str(ra), str(rb)]) == ["current", "merged"] and rb == "current"
        if kind == "moved":
            return ra == "previous" and rb == "current"
        if kind == "distinct":
            return ra is None and rb is None and (a, b) not in pending
        return ra is None and rb is None and (a, b) in pending  # ambiguous: review, not merged

    res = {}
    for kind, pairs in ids.items():
        good = sum(ok(kind, a, b) for a, b in pairs)
        res[kind] = {"correct": good, "total": len(pairs)}
    res["errors"] = sum(v["total"] - v["correct"] for v in res.values() if isinstance(v, dict))
    log.close()
    cloud.close()
    return res


def gate_separation() -> dict:
    emb = HashEmbedder(DIM)
    tmp = Path(tempfile.mkdtemp(prefix="loci-gate-"))
    store = EdgeMemoryStore(tmp / "s", DIM, "g")
    mems = [
        "oil spill near dock 9", "battery charging station", "red toolbox",
        "packing table and label printer", "loading dock 4 with pallet racks",
        "aisle 3 shelving units", "exit door and badge scanner", "operator badge left on shelf",
    ]  # fmt: skip
    now = int(time.time() * 1000)
    for i, t in enumerate(mems):
        store.put(Memory(f"k{i}", emb(t), 0.05 + 0.11 * i, 0.5, 0.0, now, text=t))
    gate = AnswerGate(store, emb, None, GateConfig())
    stop = {"and", "with", "near"}
    rel = set()
    for m in mems:
        ws = [w for w in m.split() if len(w) > 2 and w not in stop]
        rel.update(ws)
        rel.update(f"{a} {b}" for i, a in enumerate(ws) for b in ws[i + 1 :])
    rel = sorted(rel)
    rng = np.random.default_rng(1)
    words = ["banana", "submarine", "quantum", "weather", "tomorrow", "invoice", "payment", "zebra", "lunch", "menu", "guitar", "volcano", "ocean", "marathon", "lawyer", "sandwich", "galaxy", "mirror", "pencil", "holiday", "football", "keyboard", "umbrella", "coffee", "river", "painting"]  # fmt: skip
    junk = [" ".join(rng.choice(words, int(rng.integers(1, 3)), replace=False)) for _ in range(60)]
    rc = [gate.ask(q).confidence for q in rel]
    jc = [gate.ask(q).confidence for q in junk]
    routes_j = [gate.ask(q).route for q in junk]
    routes_r = [gate.ask(q).route for q in rel]
    store.close()
    return {
        "threshold": GateConfig().answer_threshold,
        "relevant_queries": len(rel),
        "junk_queries": len(junk),
        "relevant_confidence_p5": round(float(np.percentile(rc, 5)), 3),
        "junk_confidence_max": round(max(jc), 3),
        "relevant_answered_pct": round(
            100 * sum(r == ANSWER_LOCAL for r in routes_r) / len(rel), 1
        ),
        "junk_answered_pct": round(100 * sum(r == ANSWER_LOCAL for r in routes_j) / len(junk), 1),
    }


def privacy_check(n: int = 60) -> dict:
    rng = np.random.default_rng(9)
    rig = Rig()
    private = 0
    for i in range(n):
        p = i % 4 == 0
        private += p
        rig.engine.observe(
            Memory(
                f"p{i}", unit(rng.normal(size=DIM)).tolist(), *rng.random(2), 0.0, 1_000 + i,
                text=("PRIVATE-SECRET " if p else "public ") + str(i), private=p,
            ),
            now_ms=1_000 + i,
        )  # fmt: skip
    for pid in rig.store.states():  # even if something forces them into the queue...
        rig.outbox.enqueue(pid)
    rig.engine.push(now_ms=10**15)
    leaks = sum("PRIVATE-SECRET" in p["payload"].get("text", "") for p in rig.cloud.scan())
    out = {"private_memories": private, "reached_cloud": leaks}
    rig.close()
    return out


# ------------------------------------------------------------------ chart & report


def hero_svg(rows: list[dict], default_th: float) -> str:
    W, H = 760, 600
    left, right, top = 64, 130, 112
    pw = W - left - right
    ph = 170
    gap = 70
    n = len(rows)
    xs = [left + pw * i / (n - 1) for i in range(n)]

    def y(panel: int, pct: float) -> float:
        base = top + panel * (ph + gap) + ph
        return base - ph * pct / 100

    def series(panel: int, key: str, metric: str) -> tuple[str, list[tuple[float, float, str]]]:
        pts = []
        for r, x in zip(rows, xs, strict=True):
            v = r[key]["bytes_pct"] if metric == "bytes" else 100 * r[key]["recall"]
            arm = "LOCI place-aware" if key == "place_aware" else "vector-only dedupe"
            what = "bytes sent" if metric == "bytes" else "events retrievable"
            pts.append((x, y(panel, v), f"{arm}, threshold {r['threshold']}: {v:.0f}% {what}"))
        return "M" + " L".join(f"{x:.1f} {yy:.1f}" for x, yy, _ in pts), pts

    o = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="t d">'
    )
    o.append(
        "<title id='t'>Bandwidth saved without losing events: place-aware vs vector-only dedupe</title>"
    )
    o.append(
        "<desc id='d'>Two stacked panels over the same dedupe-similarity axis. Top: bytes sent as a "
        "percent of sending everything. Bottom: percent of real-world events still retrievable "
        "from the cloud copy.</desc>"
    )
    o.append("""<style>
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--ref:#8a8985}
@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#333331;--s1:#3987e5;--s2:#d95926;--ref:#8a8985}}
text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;fill:var(--ink2);font-size:12px}
.h{fill:var(--ink);font-size:14px;font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}
.ln{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
</style>""")
    o.append(f'<rect width="{W}" height="{H}" fill="var(--surface)"/>')
    o.append(
        f'<text class="h" x="{left}" y="24">Place-aware dedupe keeps the events that place-blind dedupe loses</text>'
    )
    # legend
    lx = left
    for col, label in (
        ("var(--s1)", "LOCI place-aware dedupe"),
        ("var(--s2)", "vector-only dedupe (place-blind)"),
    ):
        o.append(
            f'<line x1="{lx}" y1="52" x2="{lx + 18}" y2="52" stroke="{col}" stroke-width="2" stroke-linecap="round"/>'
        )
        o.append(
            f'<circle cx="{lx + 9}" cy="52" r="4" fill="{col}" stroke="var(--surface)" stroke-width="2"/>'
        )
        o.append(f'<text x="{lx + 26}" y="56">{label}</text>')
        lx += 250
    titles = [
        "Bytes sent, % of sending everything (lower is better)",
        "Real-world events still retrievable from the cloud, % (higher is better)",
    ]
    for panel in (0, 1):
        for pct in (0, 25, 50, 75, 100):
            yy = y(panel, pct)
            o.append(
                f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + pw}" y2="{yy:.1f}"/>'
            )
            o.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end">{pct}</text>')
        o.append(f'<text x="{left}" y="{top + panel * (ph + gap) - 12}">{titles[panel]}</text>')
        for r, x in zip(rows, xs, strict=True):
            if abs(r["threshold"] - default_th) < 1e-9:
                o.append(
                    f'<line class="grid" x1="{x:.1f}" y1="{top + panel * (ph + gap)}" x2="{x:.1f}" y2="{top + panel * (ph + gap) + ph}" style="stroke:var(--ref)"/>'
                )
                if panel == 1:
                    o.append(
                        f'<text x="{x:.1f}" y="{top + panel * (ph + gap) + ph + 32}" text-anchor="middle">default</text>'
                    )
        ref = y(panel, 100)
        o.append(
            f'<text x="{left + pw + 8}" y="{ref + 4:.1f}">send everything</text>'
        ) if panel == 0 else None
        labels = []
        for key, col in (("place_aware", "var(--s1)"), ("vector_only", "var(--s2)")):
            d, pts = series(panel, key, "bytes" if panel == 0 else "recall")
            o.append(f'<path class="ln" d="{d}" stroke="{col}"/>')
            for px, py, tip in pts:
                o.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}" stroke="var(--surface)" stroke-width="2"><title>{tip}</title></circle>'
                )
            labels.append((pts[-1][1], col, key))
        # direct end-labels only where the two ends are far enough apart to read
        if abs(labels[0][0] - labels[1][0]) >= 16:
            for yy, col, key in labels:
                name = "place-aware" if key == "place_aware" else "vector-only"
                o.append(f'<circle cx="{left + pw + 14}" cy="{yy:.1f}" r="4" fill="{col}"/>')
                o.append(f'<text x="{left + pw + 22}" y="{yy + 4:.1f}">{name}</text>')
    for r, x in zip(rows, xs, strict=True):
        yy = top + 1 * (ph + gap) + ph + 16
        o.append(f'<text x="{x:.1f}" y="{yy}" text-anchor="middle">{r["threshold"]:.2f}</text>')
    o.append(
        f'<text x="{left + pw / 2}" y="{H - 12}" text-anchor="middle">dedupe similarity threshold — further right = more aggressive suppression</text>'
    )
    o.append("</svg>")
    return "\n".join(o)


def report_md(R: dict) -> str:
    sw = R["sweep"]
    L = [
        "# Edge platform: measured results",
        "",
        f"Generated by `benchmarks/edge_verify.py` ({'quick' if R['quick'] else 'full'} run). "
        "Synthetic, seeded data; see the script's docstring for what each number means and how it can mislead.",
        "",
        f"## Bandwidth vs. retrievable events ({sw['n_observations']} observations, {sw['n_events']} events)",
        "",
        "| dedupe similarity | place-aware bytes % | place-aware events % | vector-only bytes % | vector-only events % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in sw["rows"]:
        pa, vo = r["place_aware"], r["vector_only"]
        L.append(
            f"| {r['threshold']:.2f} | {pa['bytes_pct']} | {100 * pa['recall']:.0f} | "
            f"{vo['bytes_pct']} | {100 * vo['recall']:.0f} |"
        )
    L += [
        "",
        f"Naive (send everything): {sw['naive']['bytes']} bytes, {100 * sw['naive']['recall']:.0f}% events. "
        f"Floor (each event sent exactly once): {sw['ideal_bytes_pct']}% of bytes.",
        "",
        "Opt-in `hold_back_familiar` mode at the default threshold: "
        f"{sw['hold_back_mode']['bytes_pct']}% of bytes, {100 * sw['hold_back_mode']['recall']:.0f}% of events "
        "(it lets same-place look-alikes be batched or kept local; the default sends them so no "
        "event is silently lost).",
        "",
    ]
    nc, cv, ca, gs, pv, lat = (
        R[k]
        for k in ("negative_controls", "convergence", "conflicts", "gate", "privacy", "latency")
    )
    L += [
        "## Negative controls",
        "",
        f"- {nc['n']} genuinely new random memories: **{nc['random_memories_synced_pct']}%** synced (must not be suppressed).",
        f"- {nc['n']} views of the same thing: **{nc['repeated_memory_suppressed_pct']}%** kept off the wire.",
        "",
        "## Reconnect",
        "",
        f"- {cv['offline_observations_per_robot']} offline observations per robot, two robots: converged={cv['converged']} "
        f"in {cv['rounds']} round(s), **{cv['reconnect_to_converged_ms']} ms** (push, reconcile, pull), {cv['cloud_memories']} cloud memories.",
        "",
        "## Conflict engine (labelled pairs)",
        "",
        "| case | correct |",
        "|---|---:|",
    ]
    for k in ("dup", "moved", "distinct", "ambiguous"):
        L.append(f"| {k} | {ca[k]['correct']}/{ca[k]['total']} |")
    L += [
        "",
        "## Abstain gate",
        "",
        f"- threshold {gs['threshold']}: relevant answered **{gs['relevant_answered_pct']}%** ({gs['relevant_queries']} queries, p5 confidence {gs['relevant_confidence_p5']}); "
        f"junk answered **{gs['junk_answered_pct']}%** ({gs['junk_queries']} queries, max confidence {gs['junk_confidence_max']}).",
        "",
        "## Privacy",
        "",
        f"- {pv['private_memories']} private memories, force-enqueued: **{pv['reached_cloud']}** reached the cloud.",
        "",
        f"## Offline search latency (n={lat['n']}, dim={lat['dim']}, HNSW built)",
        "",
        "| search | p50 ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for k, v in lat["latency_ms"].items():
        L.append(f"| {k} | {v['p50']} | {v['p95']} |")
    L += [
        "",
        f"recall@10 vs exact search: **{lat['recall_at_10_vs_exact']}**",
        "",
        "## Checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for c in R["checks"]:
        L.append(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} ({c['detail']}) |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller streams, fewer thresholds")
    ap.add_argument("--no-write", action="store_true", help="do not write files under results/")
    args = ap.parse_args()
    q = args.quick
    t_start = time.perf_counter()

    def step(msg):
        print(f"[{time.perf_counter() - t_start:6.1f}s] {msg}", flush=True)

    step("latency & recall vs exact search")
    latency = edge_latency.run(n=2000 if q else 5000, dim=384, queries=60 if q else 150)
    step("bandwidth vs retrievable events sweep")
    events, obs = make_world(seed=42, n_obs=250 if q else 500)
    sw = sweep(events, obs, THRESHOLDS_QUICK if q else THRESHOLDS_FULL)
    step("negative controls")
    nc = negative_controls(120 if q else 300)
    step("reconnect convergence")
    cv = convergence(60 if q else 150)
    step("conflict engine on labelled pairs")
    ca = conflict_accuracy(10 if q else 20)
    step("abstain gate separation")
    gs = gate_separation()
    step("privacy")
    pv = privacy_check()

    at = next(r for r in sw["rows"] if abs(r["threshold"] - DEFAULT_THRESHOLD) < 1e-9)
    loosest = sw["rows"][-1]
    checks = [
        (
            "edge recall@10 vs exact >= 0.95",
            latency["recall_at_10_vs_exact"] >= 0.95,
            f"{latency['recall_at_10_vs_exact']}",
        ),
        (
            "offline hybrid search p95 < 50 ms",
            latency["latency_ms"]["hybrid_rrf"]["p95"] < 50,
            f"{latency['latency_ms']['hybrid_rrf']['p95']} ms",
        ),
        (
            "default policy saves >= 70% of naive bytes",
            at["place_aware"]["bytes_pct"] <= 30,
            f"sends {at['place_aware']['bytes_pct']}% (floor {sw['ideal_bytes_pct']}%)",
        ),
        (
            "default policy keeps >= 95% of events retrievable",
            at["place_aware"]["recall"] >= 0.95,
            f"{100 * at['place_aware']['recall']:.0f}%",
        ),
        (
            "place-aware recall >= vector-only recall at every threshold",
            all(
                r["place_aware"]["recall"] >= r["vector_only"]["recall"] - 1e-9 for r in sw["rows"]
            ),
            f"loosest: {100 * loosest['place_aware']['recall']:.0f}% vs {100 * loosest['vector_only']['recall']:.0f}%",
        ),
        (
            "negative control: >= 95% of new random memories synced",
            nc["random_memories_synced_pct"] >= 95,
            f"{nc['random_memories_synced_pct']}%",
        ),
        (
            "negative control: >= 90% of repeats suppressed",
            nc["repeated_memory_suppressed_pct"] >= 90,
            f"{nc['repeated_memory_suppressed_pct']}%",
        ),
        (
            "two robots converge after an outage",
            cv["converged"],
            f"{cv['reconnect_to_converged_ms']} ms, {cv['rounds']} round(s)",
        ),
        (
            "conflict engine: zero errors on labelled pairs",
            ca["errors"] == 0,
            f"{ca['errors']} errors",
        ),
        (
            "gate: no junk query answered",
            gs["junk_answered_pct"] == 0,
            f"{gs['junk_answered_pct']}%",
        ),
        (
            "gate: >= 90% of relevant queries answered",
            gs["relevant_answered_pct"] >= 90,
            f"{gs['relevant_answered_pct']}%",
        ),
        (
            "privacy: no private memory reaches the cloud",
            pv["reached_cloud"] == 0,
            f"{pv['reached_cloud']} leaked",
        ),
    ]
    R = {
        "quick": q, "latency": latency, "sweep": sw, "negative_controls": nc, "convergence": cv,
        "conflicts": ca, "gate": gs, "privacy": pv,
        "checks": [{"name": n, "ok": bool(ok), "detail": d} for n, ok, d in checks],
    }  # fmt: skip
    print()
    for c in R["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    failed = [c for c in R["checks"] if not c["ok"]]
    print(
        f"\n{len(R['checks']) - len(failed)}/{len(R['checks'])} checks passed in {time.perf_counter() - t_start:.0f}s"
    )
    if not args.no_write:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "edge_verify.json").write_text(json.dumps(R, indent=2, default=float) + "\n")
        (RESULTS / "edge_verify.md").write_text(report_md(R))
        (RESULTS / "edge_hero.svg").write_text(hero_svg(sw["rows"], DEFAULT_THRESHOLD))
        print(f"wrote {RESULTS}/edge_verify.json, edge_verify.md, edge_hero.svg")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
