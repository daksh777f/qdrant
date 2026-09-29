"""Cloud side of sync: an interface plus a Qdrant-Server-free stand-in.

Qdrant Server is not guaranteed at the demo, so :class:`LocalCloud` keeps the
"cloud" collection in a second Edge shard on disk behind the same
:class:`CloudStore` interface a real server client would implement.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

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

    def count(self) -> int: ...

    def get(self, ids: list[str]) -> list[dict]:
        """Fetch ``{id, vector, payload}`` for the given IDs."""
        ...


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
        self._link.require()
        ops = [
            qe.Point(p["id"], {"dense": p["vector"]}, dict(p["payload"], sync_state="synced"))
            for p in points
        ]
        if ops:
            self._shard.update(qe.UpdateOperation.upsert_points(ops))
        received = sum(wire_bytes(p) for p in points)
        self.bytes_received += received
        return received

    def versions(self) -> dict[str, int]:
        self._link.require()
        out: dict[str, int] = {}
        offset = None
        while True:
            recs, offset = self._shard.scroll(
                qe.ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
            )
            for r in recs:
                out[str(r.id)] = int((r.payload or {}).get("version", 0))
            if offset is None:
                return out

    def count(self) -> int:
        return int(self._shard.count(qe.CountRequest()))

    def get(self, ids: list[str]) -> list[dict]:
        self._link.require()
        out = []
        for r in self._shard.retrieve(ids, with_payload=True, with_vector=True):
            vec = r.vector["dense"] if isinstance(r.vector, dict) else r.vector
            out.append({"id": str(r.id), "vector": list(vec), "payload": dict(r.payload or {})})
        return out

    def close(self) -> None:
        self._shard.flush()
        self._shard.close()
