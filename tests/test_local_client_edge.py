"""Run the whole LocalLociClient suite against the Qdrant Edge backend.

Every ``LocalLociClient(...)`` built by ``tests/test_local_client.py`` gets an
:class:`~loci.backends.edge.EdgeStore` instead of the numpy store, so passing
here means the Edge backend is a behavioural drop-in.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_edge")

from loci.backends.edge import EdgeStore  # noqa: E402
from tests.test_local_client import *  # noqa: E402,F401,F403


@pytest.fixture(autouse=True)
def _edge_backend(monkeypatch, tmp_path):
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return EdgeStore(tmp_path / f"store{counter['n']}")

    monkeypatch.setattr("loci.local_client.MemoryStore", factory)


def test_client_really_runs_on_edge_store():
    """Guard: the monkeypatch must actually route the client onto Edge."""
    from loci.local_client import LocalLociClient

    client = LocalLociClient(vector_size=4)
    assert isinstance(client.store, EdgeStore)
