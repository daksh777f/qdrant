"""On-device memory: a Qdrant Edge shard with dense + BM25 vectors.

Each memory carries LOCI's Hilbert bucket fields (integer payload indexes) so
space/time-scoped search stays a single indexed filter, plus provenance
(device, version, content hash, sync state).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import qdrant_edge as qe

from loci.edge.ids import content_hash, memory_id
from loci.spatial.hilbert import HilbertIndex

_INT_FIELDS = ("timestamp_ms",)
_FLOAT_FIELDS = ("x", "y", "z")
_KEYWORD_FIELDS = ("sync_state", "device_id")
_RRF_K = 60


@dataclass
class Memory:
    """One observation: what (vector + text), where (x,y,z), when, and who saw it."""

    key: str
    vector: list[float]
    x: float
    y: float
    z: float
    timestamp_ms: int
    device_id: str = ""
    text: str = ""
    confidence: float = 1.0
    private: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Assigned by the store:
    id: str = ""
    version: int = 0
    sync_state: str = "local_only"


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class EdgeMemoryStore:
    """A device's writable Qdrant Edge shard."""

    def __init__(
        self,
        path: str | Path,
        vector_size: int,
        device_id: str,
        *,
        epoch_size_ms: int = 5000,
        resolutions: list[int] | None = None,
    ) -> None:
        self.device_id = device_id
        self.vector_size = vector_size
        self._epoch_size_ms = epoch_size_ms
        self._hilbert = HilbertIndex(resolutions or [4, 8, 12])
        self._bm25 = qe.Bm25(qe.Bm25Config())
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = qe.EdgeConfig(
            vectors={"dense": qe.EdgeVectorParams(size=vector_size, distance=qe.Distance.Cosine)},
            sparse_vectors={"bm25": qe.EdgeSparseVectorParams(modifier=qe.Modifier.Idf)},
        )
        if any(p.iterdir()):
            self._shard = qe.EdgeShard.load(str(p))
        else:
            self._shard = qe.EdgeShard.create(str(p), cfg)
            self._create_indexes()

    # -- setup ---------------------------------------------------------

    def _create_indexes(self) -> None:
        upd = qe.UpdateOperation
        for r in self._hilbert.resolutions:
            self._shard.update(
                upd.create_field_index(f"hilbert_r{r}", qe.PayloadSchemaType.Integer)
            )
        for f in _INT_FIELDS:
            self._shard.update(upd.create_field_index(f, qe.PayloadSchemaType.Integer))
        for f in _FLOAT_FIELDS:
            self._shard.update(upd.create_field_index(f, qe.PayloadSchemaType.Float))
        for f in _KEYWORD_FIELDS:
            self._shard.update(upd.create_field_index(f, qe.PayloadSchemaType.Keyword))

    # -- write ---------------------------------------------------------

    def put(self, mem: Memory) -> Memory:
        """Store (or update) a memory. Same ``key`` on this device => same ID, version+1."""
        if len(mem.vector) != self.vector_size:
            raise ValueError(f"vector has dim {len(mem.vector)}, expected {self.vector_size}")
        device = mem.device_id or self.device_id
        mem_id = memory_id(device, mem.key)
        existing = self._shard.retrieve([mem_id], with_payload=True, with_vector=False)
        version = (int((existing[0].payload or {}).get("version", 0)) + 1) if existing else 1

        t_norm = (mem.timestamp_ms % self._epoch_size_ms) / self._epoch_size_ms
        payload: dict[str, Any] = {
            "key": mem.key,
            "text": mem.text,
            "x": mem.x,
            "y": mem.y,
            "z": mem.z,
            "timestamp_ms": mem.timestamp_ms,
            "device_id": device,
            "confidence": mem.confidence,
            "private": mem.private,
            "metadata": mem.metadata,
            "version": version,
            "content_hash": content_hash(mem.text, mem.vector),
            "sync_state": "local_only",
            "updated_ms": int(time.time() * 1000),
        }
        payload.update(self._hilbert.encode(mem.x, mem.y, mem.z, t_norm))
        vectors: dict[str, Any] = {"dense": mem.vector}
        if mem.text:
            vectors["bm25"] = self._bm25.embed_document(mem.text)
        self._shard.update(qe.UpdateOperation.upsert_points([qe.Point(mem_id, vectors, payload)]))
        mem.id, mem.version, mem.sync_state, mem.device_id = mem_id, version, "local_only", device
        return mem

    def set_sync_state(self, ids: list[str], state: str) -> None:
        if ids:
            self._shard.update(qe.UpdateOperation.set_payload(ids, {"sync_state": state}))

    # -- read ----------------------------------------------------------

    def search(
        self,
        vector: list[float] | None = None,
        text: str | None = None,
        *,
        limit: int = 10,
        bounds: dict[str, float] | None = None,
        time_window_ms: tuple[int, int] | None = None,
    ) -> list[SearchHit]:
        """Dense, BM25, or hybrid (RRF) search with optional space/time filters."""
        if vector is None and not text:
            raise ValueError("search needs a vector, text, or both")
        flt = self._filter(bounds, time_window_ms)
        branches = []
        if vector is not None:
            branches.append(("dense", qe.Query.Nearest(vector, using="dense")))
        if text:
            branches.append(("bm25", qe.Query.Nearest(self._bm25.embed_query(text), using="bm25")))
        if len(branches) == 1:
            req = qe.QueryRequest(query=branches[0][1], filter=flt, limit=limit, with_payload=True)
        else:
            req = qe.QueryRequest(
                prefetches=[
                    qe.Prefetch(query=q, filter=flt, limit=max(limit * 3, 20)) for _, q in branches
                ],
                query=qe.Fusion.Rrf(k=_RRF_K),
                limit=limit,
                with_payload=True,
            )
        return [
            SearchHit(str(h.id), float(h.score), dict(h.payload or {}))
            for h in self._shard.query(req)
        ]

    def _filter(
        self, bounds: dict[str, float] | None, time_window_ms: tuple[int, int] | None
    ) -> Any:
        must = []
        if bounds is not None:
            buckets = self._hilbert.query_buckets(bounds)
            field_name = self._hilbert.payload_field()
            must.append(qe.FieldCondition(field_name, match=qe.MatchAny([int(b) for b in buckets])))
            for axis in "xyz":
                lo, hi = bounds.get(f"{axis}_min", 0.0), bounds.get(f"{axis}_max", 1.0)
                must.append(qe.FieldCondition(axis, range=qe.RangeFloat(gte=lo, lte=hi)))
        if time_window_ms is not None:
            lo, hi = time_window_ms
            must.append(qe.FieldCondition("timestamp_ms", range=qe.RangeFloat(gte=lo, lte=hi)))
        return qe.Filter(must=must) if must else None

    def get(self, ids: list[str], *, with_vector: bool = False) -> list[Any]:
        return list(self._shard.retrieve(ids, with_payload=True, with_vector=with_vector))

    def count(self) -> int:
        return int(self._shard.count(qe.CountRequest()))

    def versions(self) -> dict[str, int]:
        """``{point_id: version}`` for every stored memory."""
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

    def digest(self) -> str:
        """Order-independent fingerprint of ``id:version`` pairs; equal digests => equal state."""
        h = hashlib.sha256()
        for pid, ver in sorted(self.versions().items()):
            h.update(f"{pid}:{ver};".encode())
        return h.hexdigest()[:16]

    def read_for_push(self, ids: list[str]) -> list[dict]:
        """Latest content of *ids* as ``{id, vector, payload}`` dicts (dense vector only)."""
        out = []
        for r in self._shard.retrieve(ids, with_payload=True, with_vector=True):
            vec = r.vector["dense"] if isinstance(r.vector, dict) else r.vector
            out.append({"id": str(r.id), "vector": list(vec), "payload": dict(r.payload or {})})
        return out

    def close(self) -> None:
        self._shard.flush()
        self._shard.close()
