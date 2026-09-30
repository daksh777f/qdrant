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
from loci.edge.qdrant_ops import InstrumentedShard, OpsLog, UpdateOps
from loci.spatial.hilbert import HilbertIndex

_INT_FIELDS = ("timestamp_ms",)
_FLOAT_FIELDS = ("x", "y", "z")
_KEYWORD_FIELDS = ("sync_state", "device_id", "role", "kind")
_RRF_K = 60
_HIDDEN_ROLES = ("merged", "previous")
ROLE_FIELDS = ("entity_id", "role", "merged_into", "entity_devices", "entity_size")


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
    source: str = "local"  # "local" (writable shard) or "mirror" (fleet mirror)


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
        mirror_path: str | Path | None = None,
        quantization: str | None = None,
        vectors_on_disk: bool = False,
        indexing_threshold_kb: int | None = None,
        ops_log: OpsLog | None = None,
    ) -> None:
        self.device_id = device_id
        self.ops = ops_log if ops_log is not None else OpsLog()
        self._paths = {"local": Path(path), "mirror": Path(mirror_path) if mirror_path else None}
        self.vector_size = vector_size
        self._epoch_size_ms = epoch_size_ms
        self._hilbert = HilbertIndex(resolutions or [4, 8, 12])
        self._bm25 = qe.Bm25(qe.Bm25Config())
        self.quantization = quantization
        self._search_params = _search_params(quantization)
        self._cfg = qe.EdgeConfig(
            vectors={
                "dense": qe.EdgeVectorParams(
                    size=vector_size,
                    distance=qe.Distance.Cosine,
                    on_disk=vectors_on_disk or None,
                    quantization_config=_quantization_config(quantization),
                )
            },
            sparse_vectors={"bm25": qe.EdgeSparseVectorParams(modifier=qe.Modifier.Idf)},
            # Segments smaller than this (in KB of vectors) stay unindexed and unquantized.
            optimizers=(
                qe.EdgeOptimizersConfig(indexing_threshold=indexing_threshold_kb)
                if indexing_threshold_kb is not None
                else None
            ),
        )
        self._shard = self._open_shard(Path(path), "local")
        # The mirror holds other devices' memories pulled from the cloud. Local
        # writes never touch it; it is rewritten only by :meth:`mirror_upsert`.
        self._mirror = self._open_shard(Path(mirror_path), "mirror") if mirror_path else None

    # -- setup ---------------------------------------------------------

    def _open_shard(self, path: Path, name: str) -> Any:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            shard = qe.EdgeShard.load(str(path))
            self._create_indexes(shard, only_missing=True)  # older shards: add new indexes
        else:
            shard = qe.EdgeShard.create(str(path), self._cfg)
            self._create_indexes(shard)
        return InstrumentedShard(shard, self.ops, self.device_id, name)

    def _create_indexes(self, shard: Any, *, only_missing: bool = False) -> None:
        have = set(str(k) for k in shard.info().payload_schema) if only_missing else set()
        wanted = [
            (f"hilbert_r{r}", qe.PayloadSchemaType.Integer) for r in self._hilbert.resolutions
        ]
        wanted += [(f, qe.PayloadSchemaType.Integer) for f in _INT_FIELDS]
        wanted += [(f, qe.PayloadSchemaType.Float) for f in _FLOAT_FIELDS]
        wanted += [(f, qe.PayloadSchemaType.Keyword) for f in _KEYWORD_FIELDS]
        for field_name, schema in wanted:
            if field_name not in have:
                shard.update(UpdateOps.create_field_index(field_name, schema))

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
        self._shard.update(UpdateOps.upsert_points([qe.Point(mem_id, vectors, payload)]))
        mem.id, mem.version, mem.sync_state, mem.device_id = mem_id, version, "local_only", device
        return mem

    def set_sync_state(self, ids: list[str], state: str) -> None:
        if ids:
            self._shard.update(UpdateOps.set_payload(list[Any](ids), {"sync_state": state}))

    # -- read ----------------------------------------------------------

    def search(
        self,
        vector: list[float] | None = None,
        text: str | None = None,
        *,
        limit: int = 10,
        bounds: dict[str, float] | None = None,
        time_window_ms: tuple[int, int] | None = None,
        include_mirror: bool = True,
        current_only: bool = False,
        recency_half_life_ms: int | None = None,
        now_ms: int | None = None,
        diverse: bool = False,
        diversity: float = 0.5,
    ) -> list[SearchHit]:
        """Dense, BM25, or hybrid (RRF) search over the local shard and the fleet mirror.

        Both shards are queried and merged. Where the same point ID is in both,
        the local copy wins (local edits override the mirror).

        Scoring runs inside Qdrant Edge, on the device:

        * ``recency_half_life_ms``: the dense score is multiplied by an exponential time decay
          (a Qdrant ``Formula`` with ``Decay``): a memory ``half_life`` old counts half.
        * ``diverse``: the dense branch uses Qdrant's MMR, so near-duplicate hits do not crowd
          out the rest (``diversity`` is MMR's lambda: 1 = pure relevance, 0 = pure diversity).
        """
        if vector is None and not text:
            raise ValueError("search needs a vector, text, or both")
        flt = self._filter(bounds, time_window_ms, current_only=current_only)
        shards = [("local", self._shard)]
        if include_mirror and self._mirror is not None:
            shards.append(("mirror", self._mirror))
        branches: list[tuple[str, Any]] = []
        if vector is not None:
            nearest = qe.Query.Nearest(vector, using="dense")
            if diverse:
                branches.append(("dense", qe.Mmr(vector, float(diversity), 100, using="dense")))
            elif recency_half_life_ms:
                now = now_ms if now_ms is not None else int(time.time() * 1000)
                branches.append(("dense", _Recency(nearest, now, recency_half_life_ms)))
            else:
                branches.append(("dense", nearest))
        if text:
            branches.append(("bm25", qe.Query.Nearest(self._bm25.embed_query(text), using="bm25")))
        # Per branch, merge the shards by raw score (local wins on duplicate IDs),
        # then fuse the branches with RRF. Fusing per shard first would tie each
        # shard's top hit regardless of how well it actually matched.
        per_branch = [self._branch_hits(q, flt, limit, shards) for _, q in branches]
        if len(per_branch) == 1:
            return per_branch[0][:limit]
        return _rrf(per_branch, limit)

    def _branch_hits(self, query: Any, flt: Any, limit: int, shards: list) -> list[SearchHit]:
        pool = max(limit * 3, 20)
        merged: dict[str, SearchHit] = {}
        for source, shard in shards:  # local first, so it wins duplicate IDs
            if isinstance(query, _Recency):
                req = qe.QueryRequest(
                    prefetches=[
                        qe.Prefetch(
                            query=query.nearest, filter=flt, limit=pool, params=self._search_params
                        )
                    ],
                    query=query.formula(),
                    limit=pool,
                    with_payload=True,
                )
            else:
                req = qe.QueryRequest(
                    query=query,
                    filter=flt,
                    limit=pool,
                    with_payload=True,
                    params=self._search_params,
                )
            ordered = isinstance(query, qe.Mmr)  # MMR's order is the answer; its scores are not
            for rank, h in enumerate(shard.query(req)):
                score = 1.0 / (rank + 1) if ordered else float(h.score)
                merged.setdefault(
                    str(h.id), SearchHit(str(h.id), score, dict(h.payload or {}), source)
                )
        return sorted(merged.values(), key=lambda h: -h.score)

    # -- fleet mirror ----------------------------------------------------

    def mirror_upsert(self, points: list[dict]) -> None:
        """Write points pulled from the cloud into the read-only fleet mirror."""
        if self._mirror is None:
            raise RuntimeError("store was created without a mirror_path")
        ops = []
        for p in points:
            payload = dict(p["payload"], sync_state="mirror")
            vectors: dict[str, Any] = {"dense": p["vector"]}
            if payload.get("text"):
                vectors["bm25"] = self._bm25.embed_document(payload["text"])
            ops.append(qe.Point(p["id"], vectors, payload))
        if ops:
            self._mirror.update(UpdateOps.upsert_points(ops))

    @property
    def has_mirror(self) -> bool:
        return self._mirror is not None

    def mirror_state(self) -> dict[str, tuple[int, int]]:
        """``{id: (version, role_revision)}`` for mirrored memories."""
        if self._mirror is None:
            return {}
        return {
            str(r.id): (
                int((r.payload or {}).get("version", 0)),
                int((r.payload or {}).get("rrev", 0)),
            )
            for r in self._iter(self._mirror)
        }

    def apply_cloud_roles(self, updates: dict[str, dict]) -> None:
        """Record the cloud's role verdict (merged / previous / current) on our own points.

        Roles are cloud metadata, not new content, so the device-owned ``version`` is untouched.
        """
        for pid, fields in updates.items():
            self._shard.update(UpdateOps.set_payload([pid], fields))

    @staticmethod
    def _iter(shard: Any):
        offset = None
        while True:
            recs, offset = shard.scroll(
                qe.ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
            )
            yield from recs
            if offset is None:
                return

    def mirror_versions(self) -> dict[str, int]:
        return self._versions(self._mirror) if self._mirror is not None else {}

    def _filter(
        self,
        bounds: dict[str, float] | None,
        time_window_ms: tuple[int, int] | None,
        *,
        current_only: bool = False,
    ) -> Any:
        must: list[Any] = []
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
        must_not: list[Any] = []
        if current_only:  # hide duplicates and superseded positions ("where is it now?")
            must_not.append(qe.FieldCondition("role", match=qe.MatchAny(list(_HIDDEN_ROLES))))
        if not must and not must_not:
            return None
        return qe.Filter(must=must or None, must_not=must_not or None)

    def get(self, ids: list[str], *, with_vector: bool = False) -> list[Any]:
        return list(self._shard.retrieve(ids, with_payload=True, with_vector=with_vector))

    def count(self, *, include_mirror: bool = False) -> int:
        n = int(self._shard.count(qe.CountRequest()))
        if include_mirror and self._mirror is not None:
            n += int(self._mirror.count(qe.CountRequest()))
        return n

    def versions(self) -> dict[str, int]:
        """``{point_id: version}`` for every locally written memory (mirror excluded)."""
        return self._versions(self._shard)

    @staticmethod
    def _versions(shard: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        offset = None
        while True:
            recs, offset = shard.scroll(
                qe.ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
            )
            for r in recs:
                out[str(r.id)] = int((r.payload or {}).get("version", 0))
            if offset is None:
                return out

    def states(self) -> dict[str, dict[str, Any]]:
        """``{id: {version, sync_state, private, seen_count}}`` for local memories."""
        out: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            recs, offset = self._shard.scroll(
                qe.ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
            )
            for r in recs:
                pl = r.payload or {}
                out[str(r.id)] = {
                    "version": int(pl.get("version", 0)),
                    "sync_state": pl.get("sync_state", "local_only"),
                    "private": bool(pl.get("private", False)),
                    "seen_count": int(pl.get("seen_count", 1)),
                    "rrev": int(pl.get("rrev", 0)),
                }
            if offset is None:
                return out

    def list_memories(self, *, include_mirror: bool = True, limit: int = 500) -> list[dict]:
        """Payload-only listing (no vectors) of local and mirrored memories, newest first."""
        rows: list[dict] = []
        for source, shard in (("local", self._shard), ("mirror", self._mirror)):
            if shard is None or (source == "mirror" and not include_mirror):
                continue
            offset = None
            while True:
                recs, offset = shard.scroll(
                    qe.ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
                )
                rows.extend({"id": str(r.id), "source": source, **(r.payload or {})} for r in recs)
                if offset is None:
                    break
        rows.sort(key=lambda r: r.get("timestamp_ms", 0), reverse=True)
        return rows[:limit]

    def bump_seen(self, point_id: str, now_ms: int) -> bool:
        """Count another sighting of an existing local memory (no version change)."""
        recs = self._shard.retrieve([point_id], with_payload=True, with_vector=False)
        if not recs:
            return False
        n = int((recs[0].payload or {}).get("seen_count", 1)) + 1
        self._shard.update(
            UpdateOps.set_payload([point_id], {"seen_count": n, "last_seen_ms": now_ms})
        )
        return True

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

    def facets(self, key: str, *, include_mirror: bool = False, limit: int = 20) -> dict[str, int]:
        """Value counts for a payload field, computed by Qdrant's facet API on the device."""
        out: dict[str, int] = {}
        shards = [self._shard] + ([self._mirror] if include_mirror and self._mirror else [])
        for shard in shards:
            for h in shard.facet(qe.FacetRequest(key, limit=limit, exact=True)).hits:
                out[str(h.value)] = out.get(str(h.value), 0) + int(h.count)
        return out

    def footprint(self) -> dict[str, Any]:
        """What this memory costs the device: points, indexed vectors, allocated disk."""

        def allocated(p: Path | None) -> int:
            if p is None or not p.exists():
                return 0
            return sum(f.stat().st_blocks * 512 for f in p.rglob("*") if f.is_file())

        info = self._shard.info()
        return {
            "points_local": int(info.points_count),
            "points_mirror": int(self._mirror.info().points_count) if self._mirror else 0,
            "indexed_vectors": int(info.indexed_vectors_count),
            "segments": int(info.segments_count),
            "disk_local_bytes": allocated(self._paths["local"]),
            "disk_mirror_bytes": allocated(self._paths["mirror"]),
            "vector_dim": self.vector_size,
            "quantization": self.quantization or "none",
        }

    def optimize(self) -> None:
        """Run Edge's optimizers (builds HNSW past the indexing threshold)."""
        self._shard.optimize()

    def indexed_vectors(self) -> int:
        """Vectors covered by an HNSW index (0 => searches are exact scans)."""
        return int(self._shard.info().indexed_vectors_count)

    def close(self) -> None:
        for shard in (self._shard, self._mirror):
            if shard is not None:
                shard.flush()
                shard.close()


@dataclass
class _Recency:
    """Dense nearest-neighbour re-scored on-device by ``score * exp_decay(age)``."""

    nearest: Any
    now_ms: int
    half_life_ms: int

    def formula(self) -> Any:
        x = qe.Expression
        decay = x.Decay(
            qe.DecayKind.Exp,
            x.Variable("timestamp_ms"),
            x.Constant(float(self.now_ms)),
            0.5,  # the factor at distance == scale: one half-life
            float(self.half_life_ms),
        )
        return qe.Formula(x.Mult([x.Variable("$score"), decay]))


def _quantization_config(mode: str | None) -> Any:
    """Quantized copies of the dense vectors (originals are kept for rescoring).

    * ``"scalar"``: int8, 4x smaller vectors in RAM;
    * ``"binary"``: 1 bit per dimension, 32x smaller vectors in RAM (best with high dimensions).
    """
    if mode is None:
        return None
    if mode == "scalar":
        return qe.ScalarQuantizationConfig(qe.ScalarType.Int8, quantile=0.99, always_ram=True)
    if mode == "binary":
        return qe.BinaryQuantizationConfig(always_ram=True)
    raise ValueError(f"unknown quantization {mode!r}; use None, 'scalar' or 'binary'")


def _search_params(mode: str | None) -> Any:
    """Oversample and rescore with the original vectors so quantization costs little recall."""
    if mode is None:
        return None
    oversampling = 3.0 if mode == "binary" else 2.0
    return qe.SearchParams(
        quantization=qe.QuantizationSearchParams(rescore=True, oversampling=oversampling)
    )


def _rrf(ranked_lists: list[list[SearchHit]], limit: int) -> list[SearchHit]:
    """Reciprocal-rank fusion over already-ranked hit lists."""
    fused: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranked in ranked_lists:
        for rank, h in enumerate(ranked):
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            hits.setdefault(h.id, h)
    top = sorted(fused, key=lambda i: -fused[i])[:limit]
    return [SearchHit(i, fused[i], hits[i].payload, hits[i].source) for i in top]
