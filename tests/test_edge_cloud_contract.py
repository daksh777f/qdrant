"""One contract, every cloud: LocalCloud and QdrantServerCloud must behave identically.

The Qdrant Server variant runs against qdrant-client's in-process engine (``:memory:``). Set
``LOCI_TEST_QDRANT_URL=http://localhost:6333`` to run the same suite against a live server
(each test gets its own throw-away collection).
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

pytest.importorskip("qdrant_edge")
pytest.importorskip("qdrant_client")

from loci.edge import (  # noqa: E402
    EdgeMemoryStore,
    Link,
    LinkDown,
    LinkedCloud,
    LocalCloud,
    Memory,
    Outbox,
    SyncEngine,
)
from loci.edge.cloud_ai import CloudBrain  # noqa: E402
from loci.edge.cloud_server import QdrantServerCloud, open_cloud  # noqa: E402
from loci.edge.conflicts import ConflictLog, Reconciler  # noqa: E402
from loci.edge.embed import HashEmbedder  # noqa: E402

DIM = 16
LIVE = os.environ.get("LOCI_TEST_QDRANT_URL")
PARAMS = ["local", "server-memory", "http", *(["server-live"] if LIVE else [])]


def unit(seed: int) -> list[float]:
    v = np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def pt(n: int, *, dev="a", version=1, x=0.5, t=1_000, seed=None, text="") -> dict:
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"contract-{n}"))
    return {
        "id": pid,
        "vector": unit(n if seed is None else seed),
        "payload": {
            "text": text or f"item {n}", "x": x, "y": 0.5, "z": 0.0, "timestamp_ms": t,
            "device_id": dev, "confidence": 1.0, "version": version, "private": False,
        },
    }  # fmt: skip


@pytest.fixture(params=PARAMS)
def cloud(request, tmp_path):
    if request.param == "local":
        c = LocalCloud(tmp_path / "cloud", DIM)
    elif request.param == "server-memory":
        c = QdrantServerCloud(":memory:", vector_size=DIM)
    elif request.param == "http":
        pytest.importorskip("fastapi")
        c = _HttpUnderTest(LocalCloud(tmp_path / "cloud", DIM))
    else:
        c = QdrantServerCloud(
            LIVE, vector_size=DIM, collection=f"loci_test_{uuid.uuid4().hex[:10]}"
        )
    yield c
    if request.param == "server-live":
        c._client.delete_collection(c.collection)
    c.close()


class _HttpUnderTest:
    """Device-facing calls go over a real HTTP socket (HttpCloud -> cloud_router); the
    cloud-side admin calls (scan, apply_roles) go to the backing store, as in production."""

    def __init__(self, backing):
        import socket
        import threading
        import time

        import uvicorn
        from fastapi import FastAPI

        from loci.edge.cloud_http import HttpCloud, cloud_router

        self.backing = backing
        app = FastAPI()
        app.include_router(cloud_router(lambda: backing, token="t0k"))
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
        threading.Thread(target=self.server.run, daemon=True).start()
        while not self.server.started:
            time.sleep(0.02)
        self.http = HttpCloud(f"http://127.0.0.1:{port}", token="t0k")

    def __getattr__(self, name):
        if name in {"upsert", "index", "versions", "count", "get", "search"}:
            return getattr(self.http, name)
        return getattr(self.backing, name)

    def close(self):
        self.server.should_exit = True
        self.backing.close()


# ------------------------------------------------------------------ the contract


def test_upsert_is_idempotent_and_ignores_older_versions(cloud):
    assert cloud.upsert([pt(1, version=2)]) > 0
    assert cloud.upsert([pt(1, version=2)]) == 0  # same version again: no-op
    assert cloud.upsert([pt(1, version=1)]) == 0  # stale version: ignored
    assert cloud.count() == 1 and cloud.versions()[pt(1)["id"]] == 2
    assert cloud.upsert([pt(1, version=3)]) > 0  # newer wins
    assert cloud.versions()[pt(1)["id"]] == 3 and cloud.count() == 1


def test_index_reports_version_device_and_role_revision(cloud):
    cloud.upsert([pt(1, dev="robot-a"), pt(2, dev="robot-b")])
    idx = cloud.index()
    assert idx[pt(1)["id"]] == (1, "robot-a", 0) and idx[pt(2)["id"]] == (1, "robot-b", 0)


def test_apply_roles_bumps_revision_and_ignores_unknown_ids(cloud):
    cloud.upsert([pt(1)])
    n = cloud.apply_roles(
        {pt(1)["id"]: {"role": "current", "entity_id": "e"}, str(uuid.uuid4()): {"role": "x"}}
    )
    assert n == 1
    assert cloud.index()[pt(1)["id"]][2] == 1
    assert cloud.get([pt(1)["id"]])[0]["payload"]["role"] == "current"
    cloud.apply_roles({pt(1)["id"]: {"role": "previous"}})
    assert cloud.index()[pt(1)["id"]][2] == 2


def test_newer_version_drops_a_stale_verdict_and_bumps_revision(cloud):
    cloud.upsert([pt(1)])
    cloud.apply_roles({pt(1)["id"]: {"role": "merged", "entity_id": "e", "merged_into": "z"}})
    cloud.upsert([pt(1, version=2)])
    rec = cloud.get([pt(1)["id"]])[0]["payload"]
    assert "role" not in rec and rec["version"] == 2
    assert cloud.index()[pt(1)["id"]][2] == 2  # devices can tell their copy is stale


def test_search_is_cosine_ranked_and_current_only_hides_merged_and_previous(cloud):
    cloud.upsert([pt(1), pt(2, seed=1), pt(3), pt(4, seed=99)])  # 1, 2, 3 are the same vector...
    cloud.apply_roles(
        {
            pt(2)["id"]: {"role": "merged"},
            pt(3)["id"]: {"role": "previous"},
            pt(1)["id"]: {"role": "current"},
        }
    )
    everything = cloud.search(unit(1), limit=5)
    assert everything[0]["score"] >= everything[-1]["score"] and len(everything) == 4
    current = {h["id"] for h in cloud.search(unit(1), limit=5, current_only=True)}
    assert pt(2)["id"] not in current and pt(3)["id"] not in current
    assert pt(1)["id"] in current and pt(4)["id"] in current  # untouched points stay visible
    assert len(everything[0]["vector"]) == DIM


def test_scan_and_get_return_vectors_and_payloads(cloud):
    cloud.upsert([pt(i) for i in range(1, 6)])
    scanned = {p["id"]: p for p in cloud.scan()}
    assert len(scanned) == 5 and len(scanned[pt(3)["id"]]["vector"]) == DIM
    got = cloud.get([pt(2)["id"], str(uuid.uuid4()), pt(4)["id"]])
    assert [g["id"] for g in got] == [pt(2)["id"], pt(4)["id"]]  # unknown ids are skipped
    assert cloud.get([]) == []


def test_scan_pages_through_more_than_one_page(cloud):
    cloud.upsert([pt(i) for i in range(1, 301)])
    assert cloud.count() == 300 and len(cloud.scan()) == 300 and len(cloud.index()) == 300


# ------------------------------------------------------------------ whole platform on this cloud


def test_two_robots_reconcile_brief_and_converge_on_any_cloud(cloud, tmp_path):
    emb = HashEmbedder(DIM)
    rec = Reconciler(cloud, ConflictLog(tmp_path / "c.db"))
    robots = {}
    for name in ("robot-a", "robot-b"):
        link = Link(True)
        store = EdgeMemoryStore(tmp_path / name, DIM, name, mirror_path=tmp_path / f"{name}-m")
        robots[name] = (
            store,
            SyncEngine(store, Outbox(tmp_path / f"{name}.db"), LinkedCloud(cloud, link)),
        )
    vec = emb.embed("red toolbox")
    robots["robot-a"][1].observe(
        Memory("tb", vec.tolist(), 0.1, 0.5, 0.0, 1_000, text="red toolbox")
    )
    robots["robot-b"][1].observe(
        Memory("tb", vec.tolist(), 0.85, 0.5, 0.0, 9_000, text="red toolbox")
    )
    for _ in range(2):
        for _, eng in robots.values():
            eng.push()
        rec.run()
        CloudBrain(cloud, emb).publish()
        for _, eng in robots.values():
            eng.pull()
    roles = sorted(
        str(p["payload"].get("role"))
        for p in cloud.scan()
        if p["payload"].get("text") == "red toolbox"
    )
    assert roles == ["current", "previous"]  # moved: both kept, newest is current
    for _store, eng in robots.values():
        assert eng.diff().converged
    store_a, _ = robots["robot-a"]
    assert any(
        h.payload.get("kind") == "insight"
        for h in store_a.search(vector=emb("toolbox moved"), text="moved")
    )
    for store, eng in robots.values():
        store.close()
        eng.outbox.close()


# ------------------------------------------------------------------ Qdrant Server specifics


class _Down:
    """A client whose every call fails like an unreachable server."""

    def __init__(self, exc):
        self._exc = exc

    def __getattr__(self, name):
        def boom(*a, **k):
            raise self._exc

        return boom


def _server_with(client):
    c = QdrantServerCloud.__new__(QdrantServerCloud)
    c.collection, c.vector_size, c.bytes_received, c.kind, c._client = "x", DIM, 0, "test", client
    return c


def test_connectivity_failures_become_linkdown_so_the_outbox_keeps_everything(tmp_path):
    from qdrant_client.http.exceptions import ResponseHandlingException

    down = _server_with(_Down(ResponseHandlingException(ConnectionError("refused"))))
    with pytest.raises(LinkDown):
        down.upsert([pt(1)])
    with pytest.raises(LinkDown):
        down.index()
    # ...and through the engine it is indistinguishable from a network cut: nothing is lost.
    store = EdgeMemoryStore(tmp_path / "s", DIM, "robot-a")
    outbox = Outbox(tmp_path / "o.db")
    eng = SyncEngine(store, outbox, down)
    eng.observe(Memory("k", unit(3), 0.2, 0.2, 0.0, 1_000))
    rep = eng.push()
    assert rep.link_down and rep.sent == 0 and outbox.pending() == 1
    store.close()
    outbox.close()


def test_other_errors_are_not_disguised_as_a_network_cut():
    bad = _server_with(_Down(PermissionError("bad api key")))
    with pytest.raises(PermissionError):
        bad.count()


def test_existing_collection_with_wrong_dimension_is_rejected():
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=99, distance=models.Distance.COSINE)
    )
    with pytest.raises(ValueError, match="dimension"):
        QdrantServerCloud(client=client, collection="c", vector_size=DIM)


def test_open_cloud_picks_the_server_when_configured_and_says_so_when_it_cannot(tmp_path):
    c, why = open_cloud(tmp_path / "a", DIM, {})
    assert isinstance(c, LocalCloud) and "no Qdrant Server configured" in why
    c.close()
    c, why = open_cloud(tmp_path / "b", DIM, {"LOCI_QDRANT_URL": ":memory:"})
    assert isinstance(c, QdrantServerCloud) and "qdrant-server" in why
    c.close()
    c, why = open_cloud(
        tmp_path / "c", DIM, {"LOCI_QDRANT_URL": "http://127.0.0.1:1"}
    )  # nothing there
    assert isinstance(c, LocalCloud) and "unreachable" in why  # falls back visibly, never silently
    c.close()


# ------------------------------------------------------------------ real HTTP error paths


def _fake_http(status: int):
    """A server that answers every request with *status* and a JSON error body."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def _reply(self):
            body = json.dumps({"status": {"error": "nope"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = _reply

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_http_503_from_the_server_is_treated_as_an_outage():
    srv = _fake_http(503)
    try:
        with pytest.raises(LinkDown):
            QdrantServerCloud(f"http://127.0.0.1:{srv.server_port}", vector_size=DIM)
    finally:
        srv.shutdown()


def test_http_401_is_a_real_error_not_an_outage():
    srv = _fake_http(401)
    try:
        with pytest.raises(Exception) as ei:  # noqa: PT011 - the exact type is qdrant-client's
            QdrantServerCloud(f"http://127.0.0.1:{srv.server_port}", vector_size=DIM)
        assert not isinstance(ei.value, LinkDown)
    finally:
        srv.shutdown()


def test_http_cloud_rejects_a_bad_token_and_reports_outage_as_linkdown(tmp_path):
    pytest.importorskip("fastapi")
    from loci.edge.cloud_http import HttpCloud

    c = _HttpUnderTest(LocalCloud(tmp_path / "cloud", DIM))
    bad = HttpCloud(c.http.base.rsplit("/cloud", 1)[0], token="wrong")
    with pytest.raises(RuntimeError, match="401"):
        bad.count()
    c.close()
    import time

    time.sleep(0.3)
    with pytest.raises(LinkDown):  # server gone: an outage, not an error
        c.http.count()
