"""The cloud side on a real Qdrant Server (or any qdrant-client endpoint).

:class:`QdrantServerCloud` implements the same :class:`~loci.edge.cloud.CloudStore` contract as
the local stand-in, on one Qdrant collection, using ``qdrant-client`` (>= 1.10). Everything above
it (sync engine, reconciler, briefings, answer gate, UI) is unchanged; only the cloud differs.

    cloud = QdrantServerCloud("http://localhost:6333", vector_size=384)
    engine = SyncEngine(store, outbox, LinkedCloud(cloud, link))

Connection problems (refused, timeout, 502/503/504/429) are raised as
:class:`~loci.edge.cloud.LinkDown`, so a server outage looks exactly like a network cut to the
edge: the outbox keeps everything and retries with backoff. Other errors (bad API key, bad
request) are raised as they are.

Verification status: the code paths are exercised against ``qdrant-client``'s in-process engine
(``url=":memory:"``, the same API surface) by ``tests/test_edge_cloud_contract.py``, which also
runs against a live server when ``LOCI_TEST_QDRANT_URL`` is set. The HTTP path itself has not
been run in CI here.

Concurrency note: an upsert reads the existing versions, then writes. Two writers pushing the
same point ID at the same instant can race; point IDs are per-device (device id is hashed into
them), so in normal operation each ID has a single writer.
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from qdrant_client import QdrantClient, models

from loci.edge.cloud import CloudAdmin, LinkDown, LocalCloud, plan_upsert
from loci.retry import _is_transient

T = TypeVar("T")

DEFAULT_COLLECTION = "loci_edge_cloud"
_HIDDEN_ROLES = ["merged", "previous"]
_PAGE = 256


def _dense(vec: Any) -> list[float]:
    return list(vec["dense"]) if isinstance(vec, dict) else list(vec)


class QdrantServerCloud:
    """A :class:`CloudStore` backed by a Qdrant collection.

    Args:
        url: Server URL, or ``":memory:"`` for qdrant-client's in-process engine (tests, demos).
        vector_size: Embedding dimension; an existing collection must match it.
        api_key: Optional API key (Qdrant Cloud / secured servers).
        collection: Collection name. Created if missing.
        client: An existing ``QdrantClient`` (takes precedence over *url*).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        vector_size: int,
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        client: QdrantClient | None = None,
        timeout: float = 10.0,
        ops: Any = None,
    ) -> None:
        if client is None and not url:
            raise ValueError("give a url (or ':memory:') or a client")
        self.collection = collection
        self.vector_size = vector_size
        self.bytes_received = 0
        self._ops = ops
        self.kind = f"qdrant-server @ {url or 'client'}"
        self._client = client or (
            QdrantClient(":memory:")
            if url == ":memory:"
            else QdrantClient(url=url, api_key=api_key, timeout=int(timeout))
        )
        self._call(self._ensure_collection)

    # -- plumbing -----------------------------------------------------------------

    def _call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a client call, turning connectivity failures into ``LinkDown``."""
        t0 = time.perf_counter_ns()
        try:
            result = fn(*args, **kwargs)
            if self._ops is not None:
                from loci.edge.qdrant_ops import Op

                name = getattr(fn, "__name__", "call")
                n = len(result) if isinstance(result, (list, dict)) else 0
                us = (time.perf_counter_ns() - t0) // 1000
                now = int(time.time() * 1000)
                self._ops.add(
                    Op(now, "cloud", "qdrant-server", name.strip("_"), self.collection, us, n)
                )
            return result
        except LinkDown:
            raise
        except Exception as exc:
            # Not OSError: PermissionError (a bad API key, say) is an OSError but not an outage.
            if isinstance(exc, (ConnectionError, TimeoutError)) or _is_transient(exc):
                raise LinkDown(f"qdrant server unavailable: {type(exc).__name__}: {exc}") from exc
            raise

    def _ensure_collection(self) -> None:
        c = self._client
        if c.collection_exists(self.collection):
            params = c.get_collection(self.collection).config.params.vectors
            size = params.size if isinstance(params, models.VectorParams) else None
            if size is not None and size != self.vector_size:
                raise ValueError(
                    f"collection {self.collection!r} has dimension {size}, "
                    f"expected {self.vector_size}"
                )
            return
        c.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )
        with warnings.catch_warnings():  # the in-process engine warns that indexes are no-ops
            warnings.simplefilter("ignore", UserWarning)
            for field in ("role", "device_id"):
                c.create_payload_index(
                    self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def _existing(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        recs = self._client.retrieve(
            self.collection, ids=list(ids), with_payload=True, with_vectors=False
        )
        return {str(r.id): dict(r.payload or {}) for r in recs}

    def _scroll(self, *, with_vectors: bool) -> list[Any]:
        out: list[Any] = []
        offset = None
        while True:
            recs, offset = self._client.scroll(
                self.collection,
                limit=_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            out.extend(recs)
            if offset is None:
                return out

    # -- CloudStore ---------------------------------------------------------------

    def upsert(self, points: list[dict]) -> int:
        def run() -> int:
            to_write, received = plan_upsert(points, self._existing([p["id"] for p in points]))
            if to_write:
                self._client.upsert(
                    self.collection,
                    points=[
                        models.PointStruct(id=w["id"], vector=w["vector"], payload=w["payload"])
                        for w in to_write
                    ],
                    wait=True,  # read-after-write: the reconciler searches straight after a push
                )
            return received

        received = self._call(run)
        self.bytes_received += received
        return received

    def index(self) -> dict[str, tuple[int, str, int]]:
        def run() -> dict[str, tuple[int, str, int]]:
            out = {}
            for r in self._scroll(with_vectors=False):
                pl = r.payload or {}
                out[str(r.id)] = (
                    int(pl.get("version", 0)),
                    str(pl.get("device_id", "")),
                    int(pl.get("rrev", 0)),
                )
            return out

        return self._call(run)

    def versions(self) -> dict[str, int]:
        return {i: v[0] for i, v in self.index().items()}

    def count(self) -> int:
        return int(self._call(self._client.count, self.collection, exact=True).count)

    def get(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        recs = self._call(
            self._client.retrieve,
            self.collection,
            ids=list(ids),
            with_payload=True,
            with_vectors=True,
        )
        by_id = {str(r.id): r for r in recs}
        return [
            {"id": i, "vector": _dense(by_id[i].vector), "payload": dict(by_id[i].payload or {})}
            for i in ids
            if i in by_id
        ]

    def scan(self) -> list[dict]:
        recs = self._call(self._scroll, with_vectors=True)
        return [
            {"id": str(r.id), "vector": _dense(r.vector), "payload": dict(r.payload or {})}
            for r in recs
        ]

    def search(
        self, vector: list[float], limit: int = 8, *, current_only: bool = False
    ) -> list[dict]:
        flt = None
        if current_only:  # hide duplicates and superseded positions ("where is it now?")
            flt = models.Filter(
                must_not=[
                    models.FieldCondition(key="role", match=models.MatchAny(any=_HIDDEN_ROLES))
                ]
            )
        res = self._call(
            self._client.query_points,
            self.collection,
            query=list(vector),
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=True,
        )
        return [
            {
                "id": str(h.id),
                "score": float(h.score),
                "vector": _dense(h.vector),
                "payload": dict(h.payload or {}),
            }
            for h in res.points
        ]

    def apply_roles(self, updates: dict[str, dict]) -> int:
        def run() -> int:
            current = self._existing(list(updates))
            n = 0
            for pid, fields in updates.items():
                if pid not in current:
                    continue
                rrev = int(current[pid].get("rrev", 0)) + 1
                self._client.set_payload(
                    self.collection, payload={**fields, "rrev": rrev}, points=[pid], wait=True
                )
                n += 1
            return n

        return self._call(run)

    def drop(self) -> None:
        """Delete this collection (demos and tests that created it)."""
        self._call(self._client.delete_collection, self.collection)

    def close(self) -> None:
        self._client.close()


def open_cloud(
    path: Any,
    vector_size: int,
    env: Mapping[str, str] | None = None,
    ops: Any = None,
) -> tuple[CloudAdmin, str]:
    """The configured cloud: Qdrant Server if ``LOCI_QDRANT_URL`` is set, else the local stand-in.

    Returns ``(cloud, description)``. If a URL is configured but the server is unreachable, falls
    back to the stand-in and says so in the description (never silently).

    Environment: ``LOCI_QDRANT_URL`` (or ``:memory:``), ``LOCI_QDRANT_API_KEY``,
    ``LOCI_QDRANT_COLLECTION``.
    """
    env = os.environ if env is None else env
    url = env.get("LOCI_QDRANT_URL")
    if url:
        try:
            cloud = QdrantServerCloud(
                url,
                vector_size=vector_size,
                api_key=env.get("LOCI_QDRANT_API_KEY"),
                collection=env.get("LOCI_QDRANT_COLLECTION", DEFAULT_COLLECTION),
                ops=ops,
            )
            return cloud, cloud.kind
        except LinkDown as exc:
            local = LocalCloud(path, vector_size, ops=ops)
            return local, f"local stand-in (Qdrant Server at {url} unreachable: {exc})"
    return LocalCloud(path, vector_size, ops=ops), "local stand-in (no Qdrant Server configured)"
