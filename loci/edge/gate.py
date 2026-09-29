"""Abstain / escalate gate: answer locally, ask the cloud, or refuse.

Every answer carries a confidence built from named, inspectable components:

* ``similarity`` -- cosine of the best match, scaled by ``similarity_ref``;
* ``lexical``    -- share of the query's words found in the best match's text;
* ``margin``     -- how far the best match is ahead of the runner-up (scaled by similarity);
* ``stale``      -- penalty when the best match is older than ``stale_after_ms``.

    confidence = 0.5*similarity + 0.3*lexical + 0.2*margin - stale_penalty

Routes:

* ``ANSWER_LOCAL``          confident, answered from this device (works offline);
* ``ESCALATE_CLOUD``        not confident, online, and the cloud is: answer from the cloud
                            and cache the hits into the fleet mirror for next time;
* ``LOW_CONFIDENCE_OFFLINE`` not confident, no cloud, but above ``guess_floor``: best local
                            hits, clearly flagged as a guess;
* ``ABSTAIN``               nobody is confident (or nothing matches): no answer is given.

``answer_threshold`` was set from a measured sweep on the demo corpus with the hashed-BoW
stand-in embedder (59 relevant queries: p5 0.63, min 0.30; 60 junk queries: max 0.34, none answered
at any threshold in 0.4-0.6). A different embedder changes the cosine scale, so re-measure it
(``similarity_ref`` is the knob) when you swap models.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loci.edge.cloud import LinkDown
from loci.edge.store import EdgeMemoryStore

ANSWER_LOCAL = "ANSWER_LOCAL"
ESCALATE_CLOUD = "ESCALATE_CLOUD"
LOW_CONFIDENCE_OFFLINE = "LOW_CONFIDENCE_OFFLINE"
ABSTAIN = "ABSTAIN"

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class GateConfig:
    answer_threshold: float = 0.5  # measured: junk queries peak at 0.34, relevant p5 is 0.63
    similarity_ref: float = 0.6  # cosine at which the similarity component saturates
    margin_ref: float = 0.2
    stale_after_ms: int = 24 * 3600 * 1000
    stale_penalty: float = 0.2
    guess_floor: float = 0.2  # below this, even an offline device refuses instead of guessing


@dataclass
class Answer:
    route: str
    confidence: float
    components: dict[str, float]
    hits: list[dict[str, Any]]
    reason: str
    latency_ms: float = 0.0
    cached: int = 0
    cloud_confidence: float | None = None
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def score(
    query: str, cosines: list[float], top_text: str, top_ts: int, now_ms: int, cfg: GateConfig
) -> tuple[float, dict[str, float]]:
    """Confidence in [0, 1] plus its components, from ranked cosines and the top hit."""
    if not cosines:
        return 0.0, {"similarity": 0.0, "lexical": 0.0, "margin": 0.0, "stale": 0.0}
    top = cosines[0]
    second = cosines[1] if len(cosines) > 1 else 0.0
    q = set(_TOKEN.findall(query.lower()))
    t = set(_TOKEN.findall(top_text.lower()))
    sim = _clamp(top / cfg.similarity_ref)
    comp = {
        "similarity": sim,
        "lexical": len(q & t) / len(q) if q else 0.0,
        # A wide gap only means something if the best match is itself decent, so it is
        # scaled by similarity (a weak match far ahead of an even weaker one is still weak).
        "margin": _clamp((top - second) / cfg.margin_ref) * sim if top > 0 else 0.0,
        "stale": cfg.stale_penalty if top_ts and now_ms - top_ts > cfg.stale_after_ms else 0.0,
    }
    conf = 0.5 * comp["similarity"] + 0.3 * comp["lexical"] + 0.2 * comp["margin"] - comp["stale"]
    return round(_clamp(conf), 4), {k: round(v, 4) for k, v in comp.items()}


class AnswerGate:
    def __init__(
        self,
        store: EdgeMemoryStore,
        embed: Callable[[str], list[float]],
        cloud: Any | None = None,
        cfg: GateConfig | None = None,
    ) -> None:
        self.store = store
        self.embed = embed
        self.cloud = cloud
        self.cfg = cfg or GateConfig()

    def ask(
        self, text: str, limit: int = 5, now_ms: int | None = None, *, current_only: bool = True
    ) -> Answer:
        t0 = time.perf_counter()
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        cfg = self.cfg
        vec = self.embed(text)
        # Dense-only search gives comparable cosines; the hybrid ranking is used for the answer.
        dense = self.store.search(vector=vec, limit=limit, current_only=current_only)
        conf, comp = score(
            text,
            [h.score for h in dense],
            dense[0].payload.get("text", "") if dense else "",
            int(dense[0].payload.get("timestamp_ms", 0)) if dense else 0,
            now,
            cfg,
        )
        thresholds = {"answer_threshold": cfg.answer_threshold}
        hybrid = self.store.search(vector=vec, text=text, limit=limit, current_only=current_only)
        local_hits = [_hit(h.id, h.score, h.payload, h.source) for h in hybrid]

        def done(route, hits, reason, **kw) -> Answer:
            return Answer(
                route, conf, comp, hits, reason,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                thresholds=thresholds, **kw,
            )  # fmt: skip

        if conf >= cfg.answer_threshold:
            return done(ANSWER_LOCAL, local_hits, "confident local answer")

        if self.cloud is not None:
            try:
                cloud_hits = self.cloud.search(vec, limit, current_only=current_only)
            except LinkDown:
                cloud_hits = None
            if cloud_hits is not None:
                c_conf, _ = score(
                    text,
                    [h["score"] for h in cloud_hits],
                    cloud_hits[0]["payload"].get("text", "") if cloud_hits else "",
                    int(cloud_hits[0]["payload"].get("timestamp_ms", 0)) if cloud_hits else 0,
                    now,
                    cfg,
                )
                if c_conf >= cfg.answer_threshold:
                    me = self.store.device_id
                    cache = [
                        {"id": h["id"], "vector": h["vector"], "payload": h["payload"]}
                        for h in cloud_hits
                        if h["payload"].get("device_id") != me
                    ]
                    if self.store.has_mirror and cache:
                        self.store.mirror_upsert(cache)
                    hits = [_hit(h["id"], h["score"], h["payload"], "cloud") for h in cloud_hits]
                    return done(
                        ESCALATE_CLOUD, hits,
                        f"low local confidence ({conf:.2f}); the cloud is confident ({c_conf:.2f})",
                        cached=len(cache) if self.store.has_mirror else 0,
                        cloud_confidence=c_conf,
                    )  # fmt: skip
                return done(  # the cloud was reachable and is not confident either: fail closed
                    ABSTAIN,
                    [],
                    f"neither this device ({conf:.2f}) nor the cloud ({c_conf:.2f}) is sure",
                    cloud_confidence=c_conf,
                )

        if local_hits and conf >= cfg.guess_floor:
            return done(
                LOW_CONFIDENCE_OFFLINE, local_hits,
                f"low confidence ({conf:.2f}) and no cloud available: treat as a guess",
            )  # fmt: skip
        return done(ABSTAIN, [], "nothing relevant is known on this device")


def _hit(pid: str, score: float, payload: dict, source: str) -> dict[str, Any]:
    return {
        "id": pid, "score": round(float(score), 4), "source": source,
        **{k: payload.get(k) for k in (
            "text", "x", "y", "device_id", "version", "sync_state", "content_hash",
            "timestamp_ms", "private", "seen_count", "role", "kind", "generator",
        )},
    }  # fmt: skip
