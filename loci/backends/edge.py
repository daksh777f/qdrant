"""Qdrant Edge storage backend: a drop-in for :class:`~loci.backends.memory.MemoryStore`.

One on-disk Edge shard per collection, so LOCI's two-collection layout
(``loci_data`` + ``loci_summary``) runs on the embedded engine with the same
method signatures and result shapes as the numpy store. Requires the optional
``qdrant-edge-py`` package (``pip install loci-stdb[edge]``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import qdrant_edge as qe

_DISTANCE = {
    "cosine": qe.Distance.Cosine,
    "dot": qe.Distance.Dot,
    "euclidean": qe.Distance.Euclid,
}
_PAGE = 512


class EdgeStore:
    """Multi-collection store on Qdrant Edge with the ``MemoryStore`` interface.

    Args:
        path: Directory holding one sub-directory per collection. ``None``
            uses a private temp directory removed on :meth:`close`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._tmp: tempfile.TemporaryDirectory | None = None
        if path is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="loci-edge-")
            path = self._tmp.name
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._shards: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    # -- collection lifecycle -------------------------------------------

    def _dir(self, name: str) -> Path:
        return self._root / name

    def _open(self, name: str) -> Any | None:
        shard = self._shards.get(name)
        if shard is not None:
            return shard
        meta_file = self._dir(name) / "loci_meta.json"
        if not meta_file.exists():
            return None
        self._meta[name] = json.loads(meta_file.read_text())
        shard = qe.EdgeShard.load(str(self._dir(name) / "shard"))
        self._shards[name] = shard
        return shard

    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        if self._open(name) is not None:
            return
        if distance not in _DISTANCE:
            raise ValueError(f"unsupported distance {distance!r}")
        shard_dir = self._dir(name) / "shard"
        shard_dir.mkdir(parents=True, exist_ok=True)
        cfg = qe.EdgeConfig(
            vectors={"dense": qe.EdgeVectorParams(size=vector_size, distance=_DISTANCE[distance])}
        )
        self._shards[name] = qe.EdgeShard.create(str(shard_dir), cfg)
        self._meta[name] = {"vector_size": vector_size, "distance": distance}
        (self._dir(name) / "loci_meta.json").write_text(json.dumps(self._meta[name]))

    def collection_exists(self, name: str) -> bool:
        return self._open(name) is not None

    def delete_collection(self, name: str) -> None:
        shard = self._shards.pop(name, None)
        if shard is not None:
            shard.close()
        self._meta.pop(name, None)
        shutil.rmtree(self._dir(name), ignore_errors=True)

    def create_payload_index(self, collection: str, field_name: str) -> None:
        shard = self._open(collection)
        if shard is None:
            return
        is_int = field_name.startswith("hilbert") or field_name.endswith("_ms")
        schema = qe.PayloadSchemaType.Integer if is_int else qe.PayloadSchemaType.Keyword
        shard.update(qe.UpdateOperation.create_field_index(field_name, schema))

    # -- write -----------------------------------------------------------

    def upsert(self, collection: str, points: list[dict]) -> None:
        shard = self._shards[collection] if collection in self._shards else self._open(collection)
        if shard is None:
            raise KeyError(collection)
        size = self._meta[collection]["vector_size"]
        ops = []
        for p in points:
            vector = list(p["vector"])
            if len(vector) != size:
                raise ValueError(
                    f"vector for point {p['id']!r} has dimension {len(vector)}, "
                    f"expected {size} for collection {collection!r}"
                )
            ops.append(qe.Point(str(p["id"]), {"dense": vector}, dict(p["payload"])))
        if ops:
            shard.update(qe.UpdateOperation.upsert_points(ops))

    def set_payload(self, collection: str, point_id: str, payload: dict) -> None:
        shard = self._open(collection)
        if shard is not None and self._exists(shard, point_id):
            shard.update(qe.UpdateOperation.set_payload([point_id], payload))

    def delete_points(self, collection: str, ids: list[str]) -> int:
        shard = self._open(collection)
        if shard is None or not ids:
            return 0
        present = [str(r.id) for r in shard.retrieve(ids, with_payload=False, with_vector=False)]
        if present:
            shard.update(qe.UpdateOperation.delete_points(present))
        return len(present)

    def delete_points_in_time_range(
        self, collection: str, start_ms: int, end_ms_exclusive: int, *, field: str = "timestamp_ms"
    ) -> int:
        shard = self._open(collection)
        if shard is None:
            return 0
        flt = qe.Filter(
            must=[qe.FieldCondition(field, range=qe.RangeFloat(gte=start_ms, lt=end_ms_exclusive))]
        )
        n = int(shard.count(qe.CountRequest(exact=True, filter=flt)))
        if n:
            shard.update(qe.UpdateOperation.delete_points_by_filter(flt))
        return n

    # -- read ------------------------------------------------------------

    @staticmethod
    def _exists(shard: Any, point_id: str) -> bool:
        return bool(shard.retrieve([point_id], with_payload=False, with_vector=False))

    @staticmethod
    def _to_dict(rec: Any, score: float | None = None) -> dict:
        vec = rec.vector["dense"] if isinstance(rec.vector, dict) else rec.vector
        out: dict[str, Any] = {
            "id": str(rec.id),
            "vector": list(vec) if vec is not None else [],
            "payload": dict(rec.payload or {}),
        }
        if score is not None:
            out["score"] = score
        return out

    def retrieve(self, collection: str, ids: list[str]) -> list[dict]:
        shard = self._open(collection)
        if shard is None or not ids:
            return []
        by_id = {str(r.id): r for r in shard.retrieve(ids, with_payload=True, with_vector=True)}
        return [self._to_dict(by_id[i]) for i in ids if i in by_id]

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: dict | None = None,
    ) -> list[dict]:
        shard = self._open(collection)
        if shard is None:
            return []
        req = qe.QueryRequest(
            query=qe.Query.Nearest(list(query_vector), using="dense"),
            filter=_to_filter(payload_filter),
            limit=limit,
            with_payload=True,
            with_vector=True,
        )
        sign = -1.0 if self._meta[collection]["distance"] == "euclidean" else 1.0
        return [self._to_dict(h, score=sign * float(h.score)) for h in shard.query(req)]

    def _scroll_pages(self, shard: Any, flt: Any):
        offset = None
        while True:
            recs, offset = shard.scroll(
                qe.ScrollRequest(
                    limit=_PAGE, offset=offset, filter=flt, with_payload=True, with_vector=True
                )
            )
            yield from recs
            if offset is None:
                return

    def scroll(
        self,
        collection: str,
        payload_filter: dict | None = None,
        limit: int = 10,
        order_by: str | None = None,
    ) -> list[dict]:
        shard = self._open(collection)
        if shard is None:
            return []
        flt = _to_filter(payload_filter)
        if order_by is None:
            out = []
            for rec in self._scroll_pages(shard, flt):
                out.append(self._to_dict(rec))
                if len(out) >= limit:
                    break
            return out
        rows = [self._to_dict(r) for r in self._scroll_pages(shard, flt)]
        rows.sort(key=lambda r: r["payload"].get(order_by, 0))
        return rows[:limit]

    @property
    def total_points(self) -> int:
        return sum(self.collection_count(n) for n in self.collection_names())

    def collection_names(self) -> list[str]:
        """Names of all collections, sorted."""
        names = {p.name for p in self._root.iterdir() if (p / "loci_meta.json").exists()}
        return sorted(names | set(self._shards))

    def collection_count(self, name: str) -> int:
        shard = self._open(name)
        return int(shard.count(qe.CountRequest(exact=True))) if shard is not None else 0

    def payload_value_range(self, collection: str, field: str) -> tuple[Any, Any] | None:
        shard = self._open(collection)
        if shard is None:
            return None
        values = [
            v
            for rec in self._scroll_pages(shard, None)
            if (v := (rec.payload or {}).get(field)) is not None
        ]
        return (min(values), max(values)) if values else None

    def close(self) -> None:
        for shard in self._shards.values():
            shard.flush()
            shard.close()
        self._shards.clear()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def _to_filter(payload_filter: dict | None) -> Any:
    """Translate the ``MemoryStore`` filter dict into an Edge ``Filter``."""
    if not payload_filter:
        return None
    must = []
    for key, cond in payload_filter.items():
        if isinstance(cond, dict):
            if "any" in cond:
                must.append(qe.FieldCondition(key, match=qe.MatchAny(list(cond["any"]))))
                continue
            bounds = {k: cond[k] for k in ("gte", "lte", "gt", "lt") if k in cond}
            must.append(qe.FieldCondition(key, range=qe.RangeFloat(**bounds)))
        else:
            must.append(qe.FieldCondition(key, match=qe.MatchValue(cond)))
    return qe.Filter(must=must)
