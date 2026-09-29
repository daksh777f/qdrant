"""Mission-control UI: HTTP API over the simulated two-robot fleet."""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_edge")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from loci.edge.sim import Fleet  # noqa: E402
from loci.edge.ui import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    fleet = Fleet(tmp_path / "fleet")
    with TestClient(create_app(fleet, autosync=False)) as c:
        yield c
    fleet.close()


def post(c, path, **body):
    r = c.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "LOCI Edge" in r.text and "Sync diff" in r.text


def test_state_lists_both_robots_online(client):
    s = client.get("/api/state").json()
    assert [r["name"] for r in s["robots"]] == ["robot-a", "robot-b"]
    assert all(r["online"] for r in s["robots"]) and s["cloud_memories"] == 0
    assert s["bytes"]["saved_pct"] is None  # nothing sent yet: not "100% saved"


def test_unknown_robot_is_404(client):
    assert client.get("/api/robots/nope/dashboard").status_code == 404
    assert client.post("/api/robots/nope/link", json={"up": False}).status_code == 404


def test_cut_network_then_patrol_then_restore_and_sync(client):
    post(client, "/api/robots/robot-a/link", up=False)
    assert client.get("/api/state").json()["robots"][0]["online"] is False
    p = post(client, "/api/robots/robot-a/patrol", steps=40)
    assert p["steps"] == 40 and p["decisions"].get("DEDUPE", 0) > 0
    off = post(client, "/api/robots/robot-a/sync")
    assert off["link_down"] and off["sent"] == 0
    assert client.get("/api/state").json()["bytes"]["saved_pct"] is None  # still nothing sent
    dash = client.get("/api/robots/robot-a/dashboard").json()
    assert dash["diff"]["cloud_reachable"] is False and dash["diff"]["counts"]["to_push"] > 0
    post(client, "/api/robots/robot-a/link", up=True)
    ok = post(client, "/api/robots/robot-a/sync")  # backoff was cleared on reconnect
    assert not ok["link_down"] and ok["sent"] > 0
    s = client.get("/api/state").json()
    assert s["cloud_memories"] > 0 and s["bytes"]["sent_total"] > 0
    assert s["bytes"]["saved_pct"] > 50
    d = client.get("/api/robots/robot-a/dashboard").json()["diff"]
    assert d["converged"] and d["counts"]["to_push"] == 0


def test_private_never_reaches_cloud_and_is_held_local(client):
    post(client, "/api/robots/robot-a/private")
    post(client, "/api/robots/robot-a/sync")
    diff = client.get("/api/robots/robot-a/dashboard").json()["diff"]
    assert any(h["why"] == "private" for h in diff["held_local"])
    fleet = client.app.state.fleet
    texts = [p["payload"].get("text", "") for p in fleet.cloud.get(list(fleet.cloud.versions()))]
    assert not any("badge" in t for t in texts)


def test_robot_b_finds_robot_a_spill_offline(client):
    post(client, "/api/robots/robot-a/spill")
    post(client, "/api/robots/robot-a/sync")
    post(client, "/api/robots/robot-b/sync")  # pulls into b's mirror
    post(client, "/api/robots/robot-b/link", up=False)
    r = client.get("/api/robots/robot-b/search", params={"q": "oil spill"}).json()
    assert r["online"] is False and r["latency_ms"] < 250
    top = r["results"][0]
    assert (
        "oil spill" in top["text"] and top["source"] == "mirror" and top["device_id"] == "robot-a"
    )


def test_toolbox_moved_is_synced_not_deduped(client):
    a = post(client, "/api/robots/robot-a/toolbox")
    b = post(client, "/api/robots/robot-a/toolbox")  # a different spot
    assert a["action"] == "SYNC_NOW" and b["action"] == "SYNC_NOW" and "moved" in b["reason"]


def test_dashboard_map_and_decisions_shapes(client):
    post(client, "/api/robots/robot-a/patrol", steps=20)
    dash = client.get("/api/robots/robot-a/dashboard").json()
    assert dash["map"]["grid"] == 16 and dash["map"]["cells"] and dash["map"]["points"]
    assert {"hilbert", "novelty", "count"} <= set(dash["map"]["cells"][0])
    d = dash["decisions"][0]
    assert d["action"] in {"SYNC_NOW", "DEDUPE", "SUMMARIZE_SYNC", "KEEP_LOCAL"} and d["reason"]
    m = client.get("/api/robots/robot-a/memories").json()
    assert m and {"version", "content_hash", "sync_state"} <= set(m[0])
    assert "vector" not in m[0]


def test_search_validation(client):
    assert client.get("/api/robots/robot-a/search").status_code == 422
    assert client.get("/api/robots/robot-a/search", params={"q": ""}).status_code == 422
