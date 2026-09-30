"""Cloud side of sync: an interface plus a Qdrant-Server-free stand-in.

Qdrant Server is not guaranteed at the demo, so :class:`LocalCloud` keeps the
"cloud" collection in a second Edge shard on disk behind the same
:class:`CloudStore` interface a real server client would implement.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol

import qdrant_edge as qe


class LinkDown(ConnectionError):
    """Raised by a cloud call while the (simulated) link is down."""


class Link:
    """A switchable connectivity flag: the demo's "cut the network" button."""

    def __init__(self, up: bool = True) -> None:
        self._up = up
        self._lock = threading.Lock()

    @property
    def up(self) -> bool:
        with self._lock:
            return self._up

    def set(self, up: bool) -> None:
        with self._lock:
            self._up = up

    def require(self) -> None:
        if not self.up:
            raise LinkDown("network link is down")


def _ids(ids: list[str]) -> list[Any]:
    """Edge takes ``list[int | UUID | str]``; ``list`` is invariant, so widen explicitly."""
    return list(ids)


def _dense(vec: Any) -> list[float]:
    """The dense vector from an Edge record (named-vector dict or bare list)."""
    return list(vec["dense"]) if isinstance(vec, dict) else list(vec)


def wire_bytes(point: dict) -> int:
    """Bytes a point costs on the wire: JSON payload + float32 vector."""
    body = json.dumps(point["payload"], sort_keys=True, separators=(",", ":"), default=str)
    return len(body.encode()) + 4 * len(point["vector"])


class CloudStore(Protocol):
    """What the sync engine needs from the cloud."""

    def upsert(self, points: list[dict]) -> int:
        """Store points (idempotent by ID); return bytes received."""
        ...

    def versions(self) -> dict[str, int]:
        """Return ``{point_id: version}`` for everything the cloud holds."""
        ...

    def index(self) -> dict[str, tuple[int, str, int]]:
        """Return ``{point_id: (version, device_id, role_revision)}`` for everything held.

        ``role_revision`` counts changes to the cloud's role verdict on a point
        (merged / previous / current), so devices can tell their copy is stale even
        when the device-owned ``version`` is unchanged. Ordinary first writes leave it 0.
        """
        ...

    def count(self) -> int: ...

    def get(self, ids: list[str]) -> list[dict]:
        """Fetch ``{id, vector, payload}`` for the given IDs."""
        ...

    def search(self, vector: list[float], limit: int = 8, *, current_only: bool = False) -> Any:
        """Nearest neighbours as ``[{id, score, vector, payload}]``."""
        ...


class LinkedCloud:
    """One device's view of a shared cloud: every call goes through *its own* link.

    Lets several simulated devices share one cloud store while each has an
    independent network state (the demo's per-robot "cut the network" switch).
    """

    def __init__(self, cloud: CloudStore, link: Link) -> None:
        self._cloud = cloud
        self.link = link

    def upsert(self, points: list[dict]) -> int:
        self.link.require()
        return self._cloud.upsert(points)

    def versions(self) -> dict[str, int]:
        self.link.require()
        return self._cloud.versions()

    def index(self) -> dict[str, tuple[int, str, int]]:
        self.link.require()
        return self._cloud.index()

    def search(self, vector: list[float], limit: int = 8, *, current_only: bool = False):
        self.link.require()
        return self._cloud.search(vector, limit, current_only=current_only)

    def count(self) -> int:
        self.link.require()
        return self._cloud.count()

    def get(self, ids: list[str]) -> list[dict]:
        self.link.require()
        return self._cloud.get(ids)


class LocalCloud:
    """Cloud stand-in backed by an Edge shard; needs no server or Docker."""

    def __init__(self, path: str | Path, vector_size: int, link: Link | None = None) -> None:
        self._link = link or Link(True)
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        cfg = qe.EdgeConfig(
            vectors={"dense": qe.EdgeVectorParams(size=vector_size, distance=qe.Distance.Cosine)},
        )
        if any(self._path.iterdir()):
            self._shard = qe.EdgeShard.load(str(self._path))
        else:
            self._shard = qe.EdgeShard.create(str(self._path), cfg)
        self.bytes_received = 0

    def upsert(self, points: list[dict]) -> int:
        """Store points; a re-push of an already-held version is a no-op (idempotent).

        A newer version of a point that already carried a role verdict drops that
        verdict (the reconciler will re-judge it) and bumps ``rrev`` so devices notice.
        """
        self._link.require()
        existing = {
            str(r.id): r.payload or {}
            for r in self._shard.retrieve(
                _ids([p["id"] for p in points]), with_payload=True, with_vector=False
            )
        }
        ops = []
        received = 0
        for p in points:
            old = existing.get(p["id"])
            if old is not None and int(old.get("version", 0)) >= int(
                p["payload"].get("version", 0)
            ):
                continue  # already have this (or a newer) version
            old_rrev = int(old.get("rrev", 0)) if old is not None else 0
            rrev = old_rrev + 1 if old is not None and old.get("role") else old_rrev
            ops.append(
                qe.Point(
                    p["id"],
                    {"dense": p["vector"]},
                    dict(p["payload"], sync_state="synced", rrev=rrev),
                )
            )
            received += wire_bytes(p)
        if ops:
            self._shard.update(qe.UpdateOperation.upsert_points(ops))
        self.bytes_received += received
        return received

    def index(self) -> dict[str, tuple[int, str, int]]:
        self._link.require()
        out: dict[str, tuple[int, str, int]] = {}
        for rec in self._scan_payloads():
            pl = rec.payload or {}
            out[str(rec.id)] = (
                int(pl.get("version", 0)),
                str(pl.get("device_id", "")),
                int(pl.get("rrev", 0)),
            )
        return out

    def _scan_payloads(self, *, with_vector: bool = False):
        offset = None
        while True:
            recs, offset = self._shard.scroll(
                qe.ScrollRequest(
                    limit=256, offset=offset, with_payload=True, with_vector=with_vector
                )
            )
            yield from recs
            if offset is None:
                return

    def scan(self) -> list[dict]:
        """Every point as ``{id, vector, payload}`` (used by the reconciler)."""
        self._link.require()
        out = []
        for r in self._scan_payloads(with_vector=True):
            out.append(
                {"id": str(r.id), "vector": _dense(r.vector), "payload": dict(r.payload or {})}
            )
        return out

    def search(self, vector: list[float], limit: int = 8, *, current_only: bool = False):
        """Dense nearest-neighbour search over the cloud copy (cosine)."""
        self._link.require()
        flt = None
        if current_only:
            flt = qe.Filter(
                must_not=[qe.FieldCondition("role", match=qe.MatchAny(["merged", "previous"]))]
            )
        req = qe.QueryRequest(
            query=qe.Query.Nearest(list(vector), using="dense"),
            filter=flt,
            limit=limit,
            with_payload=True,
            with_vector=True,
        )
        out = []
        for h in self._shard.query(req):
            out.append(
                {
                    "id": str(h.id),
                    "score": float(h.score),
                    "vector": _dense(h.vector),
                    "payload": dict(h.payload or {}),
                }
            )
        return out

    def apply_roles(self, updates: dict[str, dict]) -> int:
        """Write reconciler output (role fields) and bump each point's role revision."""
        current = {
            str(r.id): r.payload or {}
            for r in self._shard.retrieve(_ids(list(updates)), with_payload=True, with_vector=False)
        }
        n = 0
        for pid, fields in updates.items():
            if pid not in current:
                continue
            rrev = int(current[pid].get("rrev", 0)) + 1
            self._shard.update(qe.UpdateOperation.set_payload([pid], {**fields, "rrev": rrev}))
            n += 1
        return n

    def versions(self) -> dict[str, int]:
        return {i: v[0] for i, v in self.index().items()}

    def count(self) -> int:
        return int(self._shard.count(qe.CountRequest()))

    def get(self, ids: list[str]) -> list[dict]:
        self._link.require()
        out = []
        for r in self._shard.retrieve(_ids(ids), with_payload=True, with_vector=True):
            out.append(
                {"id": str(r.id), "vector": _dense(r.vector), "payload": dict(r.payload or {})}
            )
        return out

    def close(self) -> None:
        self._shard.flush()
        self._shard.close()
