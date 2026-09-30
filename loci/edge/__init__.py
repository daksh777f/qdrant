"""Qdrant Edge layer for LOCI: offline-first memory that syncs to a cloud store.

P0 vertical slice: :class:`EdgeMemoryStore` (on-device Qdrant Edge shard),
:class:`Outbox` (durable SQLite queue), :class:`LocalCloud` (a cloud stand-in
that needs no Qdrant Server) and :class:`SyncEngine` (idempotent push with
backoff). Requires the optional ``qdrant-edge-py`` package.
"""

from loci.edge.cloud import CloudStore, Link, LinkDown, LinkedCloud, LocalCloud
from loci.edge.cloud_server import QdrantServerCloud, open_cloud
from loci.edge.decisions import Decision, DecisionLog
from loci.edge.ids import content_hash, memory_id
from loci.edge.outbox import Outbox
from loci.edge.policy import PolicyConfig, SyncPolicy
from loci.edge.store import EdgeMemoryStore, Memory, SearchHit
from loci.edge.sync import PullReport, PushReport, SummaryReport, SyncDiff, SyncEngine

__all__ = [
    "CloudStore",
    "Decision",
    "DecisionLog",
    "EdgeMemoryStore",
    "Link",
    "LinkDown",
    "LinkedCloud",
    "LocalCloud",
    "Memory",
    "Outbox",
    "PolicyConfig",
    "PullReport",
    "QdrantServerCloud",
    "PushReport",
    "SearchHit",
    "SummaryReport",
    "SyncDiff",
    "SyncPolicy",
    "SyncEngine",
    "content_hash",
    "open_cloud",
    "memory_id",
]
