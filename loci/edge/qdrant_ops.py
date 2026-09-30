"""Every Qdrant Edge call, timed and recorded: the data behind the UI's "Qdrant Edge inspector".

:class:`InstrumentedShard` is a transparent proxy around ``qdrant_edge.EdgeShard``. It forwards
each call unchanged and appends one :class:`Op` to a shared :class:`OpsLog` (a bounded ring
buffer plus per-kind counters and latency samples). Nothing is simulated or summarised by hand:
the op kind, the query shape and the result count are read from the real request and response.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_TRACKED = {"query", "update", "scroll", "retrieve", "count", "facet", "optimize"}


@dataclass
class Op:
    ts_ms: int
    device: str
    shard: str  # "local" | "mirror" | "cloud"
    op: str
    detail: str
    us: int
    n: int  # points returned / written


def describe_query(req: Any) -> str:
    """A short, faithful description of a QueryRequest ("RRF(dense, bm25) + filter[3]")."""

    def q_name(q: Any) -> str:
        if q is None:
            return "?"
        text = repr(q)
        head = text.split("(", 1)[0]  # e.g. "Query.Nearest", "Mmr", "Fusion.Rrf", "Formula"
        kind = head.rsplit(".", 1)[-1].lower()
        using = getattr(q, "using", None)
        if using is None and 'using="' in text:
            using = text.split('using="', 1)[1].split('"', 1)[0]
        if kind in {"rrf", "dbsf"}:
            return kind.upper()
        if kind == "formula":
            return "formula(decay)" if "Decay" in text or type(q).__name__ == "Formula" else kind
        if kind == "mmr":
            return f"MMR({using or 'dense'})"
        return f"{kind}({using})" if using else kind

    parts = []
    pre = getattr(req, "prefetches", None) or []
    if pre:
        parts.append(f"{q_name(req.query)}(" + ", ".join(q_name(p.query) for p in pre) + ")")
    else:
        parts.append(q_name(getattr(req, "query", None)))
    flt = getattr(req, "filter", None)
    if flt is None and pre:
        flt = getattr(pre[0], "filter", None)
    if flt is not None:
        n = len(getattr(flt, "must", None) or []) + len(getattr(flt, "must_not", None) or [])
        parts.append(f"filter[{n}]")
    params = getattr(req, "params", None)
    if params is not None and getattr(params, "quantization", None) is not None:
        parts.append("quantized+rescore")
    return " + ".join(parts)


_LABELS: dict[int, tuple[str, int]] = {}


class _LabelledUpdateOperation:
    """Drop-in for ``qdrant_edge.UpdateOperation`` that remembers what each operation is.

    Edge's update objects are opaque (no useful repr), so the kind and size are recorded when
    the operation is built and read back when the instrumented shard applies it.
    """

    def __getattr__(self, name: str) -> Any:
        import qdrant_edge as qe

        factory = getattr(qe.UpdateOperation, name)

        def make(*args: Any, **kwargs: Any) -> Any:
            op = factory(*args, **kwargs)
            n = len(args[0]) if args and isinstance(args[0], list) else 1
            if len(_LABELS) > 10_000:  # never grows without bound if an op is built but not applied
                _LABELS.clear()
            _LABELS[id(op)] = (name, n)
            return op

        return make


UpdateOps = _LabelledUpdateOperation()


def describe_update(op: Any) -> tuple[str, int]:
    return _LABELS.pop(id(op), ("update", 1))


def _features(op: Op) -> list[str]:
    """Which Qdrant capabilities a recorded call exercised (read from the call itself)."""
    d, out = op.detail, []
    if op.op == "query":
        if "nearest(dense)" in d:
            out.append("dense_hnsw")
        if "nearest(bm25)" in d:
            out.append("sparse_bm25")
        if "filter[" in d:
            out.append("payload_filter")
        if "formula" in d:
            out.append("decay_formula")
        if "MMR" in d:
            out.append("mmr")
        if "quantized" in d:
            out.append("quantization")
    elif op.op == "facet":
        out.append("facets")
    elif op.op == "update":
        out.append("upsert" if d == "upsert_points" else "payload_update")
    elif op.op in {"scroll", "retrieve", "count"}:
        out.append(op.op)
    elif op.shard == "qdrant-server":
        out.append("qdrant_server")
    return out


class OpsLog:
    """Thread-safe ring buffer of :class:`Op` plus per-kind latency samples."""

    def __init__(self, capacity: int = 400, samples_per_kind: int = 2000) -> None:
        self._ops: deque[Op] = deque(maxlen=capacity)
        self._lat: dict[str, deque[int]] = {}
        self._count: dict[str, int] = {}
        self.features: dict[str, int] = {}
        self._samples = samples_per_kind
        self._lock = threading.Lock()

    def add(self, op: Op) -> None:
        key = f"{op.op}:{op.detail.split(' + ')[0]}" if op.op == "query" else op.op
        with self._lock:
            self._ops.append(op)
            self._count[key] = self._count.get(key, 0) + 1
            self._lat.setdefault(key, deque(maxlen=self._samples)).append(op.us)
            for feature in _features(op):
                self.features[feature] = self.features.get(feature, 0) + 1

    def recent(self, limit: int = 50, device: str | None = None) -> list[dict]:
        with self._lock:
            ops = [o for o in reversed(self._ops) if device is None or o.device == device]
        return [asdict(o) for o in ops[:limit]]

    def stats(self) -> list[dict]:
        with self._lock:
            items = [(k, list(v), self._count[k]) for k, v in self._lat.items()]
        out = []
        for key, lat, n in items:
            arr = np.asarray(lat, dtype=float) / 1000.0
            out.append(
                {
                    "kind": key,
                    "calls": n,
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                }
            )
        return sorted(out, key=lambda r: -r["calls"])

    def total_calls(self) -> int:
        with self._lock:
            return sum(self._count.values())


class InstrumentedShard:
    """Forwards to a real ``EdgeShard``; records each tracked call in an :class:`OpsLog`."""

    def __init__(self, shard: Any, log: OpsLog | None, device: str, name: str) -> None:
        self._shard = shard
        self._log = log
        self._device = device
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        target = getattr(self._shard, attr)
        if attr not in _TRACKED or self._log is None or not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter_ns()
            result = target(*args, **kwargs)
            us = (time.perf_counter_ns() - t0) // 1000
            detail, n = "", 0
            if attr == "query":
                detail, n = describe_query(args[0] if args else None), len(result)
            elif attr == "update":
                detail, n = describe_update(args[0] if args else None)
            elif attr == "scroll":
                n = len(result[0]) if isinstance(result, tuple) else 0
            elif attr == "retrieve":
                n = len(result)
            elif attr == "facet":
                detail = f"facet({getattr(args[0], 'key', '?')})"
                n = len(getattr(result, "hits", []) or [])
            self._log.add(  # type: ignore[union-attr]
                Op(int(time.time() * 1000), self._device, self._name, attr, detail, int(us), n)
            )
            return result

        return call
