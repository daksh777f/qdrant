"""Test bootstrap helpers.

Keep the repository root importable so pytest can load ``loci`` directly from the
working tree without requiring ``pip install -e .`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _optional_edge_backend(request, monkeypatch, tmp_path):
    """``LOCI_TEST_BACKEND=edge`` runs every LocalLociClient test on Qdrant Edge.

    Tests that exercise the numpy ``MemoryStore`` class directly
    (``test_memory_store.py``) and the Edge suites themselves are left alone.
    """
    import os

    if os.environ.get("LOCI_TEST_BACKEND") != "edge":
        return
    if request.module.__name__ in {"test_memory_store", "test_edge_p0", "test_local_client_edge"}:
        return
    from loci.backends.edge import EdgeStore

    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return EdgeStore(tmp_path / f"edge{counter['n']}")

    monkeypatch.setattr("loci.local_client.MemoryStore", factory)
