"""Deterministic identifiers and hashes.

The same observation always maps to the same point ID, so a retried or
duplicated push overwrites instead of duplicating (idempotent sync).
"""

from __future__ import annotations

import hashlib
import uuid

_NAMESPACE = uuid.UUID("6f1f3c52-51c0-4a7e-9d3e-0c1c1f0a10c1")


def memory_id(device_id: str, key: str) -> str:
    """Return a stable UUID5 for *key* observed by *device_id*."""
    return str(uuid.uuid5(_NAMESPACE, f"{device_id}\x1f{key}"))


def content_hash(text: str, vector: list[float]) -> str:
    """Short hash of a memory's content, used to detect divergence."""
    h = hashlib.sha256(text.encode())
    h.update(",".join(f"{v:.5f}" for v in vector).encode())
    return h.hexdigest()[:16]
