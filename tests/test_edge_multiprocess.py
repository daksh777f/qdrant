"""Edge devices as real OS processes: outage, kill -9 mid-outage, restart, backlog delivered.

Starts mission control (the cloud) and one edge node as separate processes talking HTTP.
Slow-ish (about 30 s), so it is marked ``slow``; run with ``pytest -m slow`` or the edge CI job.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("qdrant_edge")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310
        return json.loads(r.read())


def _post(url: str, body: dict):
    req = urllib.request.Request(  # noqa: S310
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
        return json.loads(r.read())


def _wait(pred, timeout: float, what: str):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = pred()
            if v:
                return v
        except Exception:  # noqa: S110 - the server may still be starting
            pass
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for: {what}")


def _device(base: str, name: str):
    return next((d for d in _get(f"{base}/api/devices") if d["name"] == name), None)


def test_node_survives_outage_and_kill_minus_9_and_its_backlog_reaches_the_cloud(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("LOCI_QDRANT_URL", None)
    port = _port()
    base = f"http://127.0.0.1:{port}"
    ui = subprocess.Popen(
        [sys.executable, "-m", "loci.edge.ui", "--port", str(port), "--no-autosync"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    node_cmd = [
        sys.executable, "-m", "loci.edge.node", "--name", "robot-c", "--cloud", base,
        "--data", str(tmp_path / "robot-c"), "--interval", "0.2",
    ]  # fmt: skip
    node = None
    try:
        _wait(lambda: _get(f"{base}/api/state"), 30, "mission control up")
        node = subprocess.Popen(node_cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        d = _wait(
            lambda: (x := _device(base, "robot-c")) and x["online"] and x["bytes_sent"] > 0 and x,
            30,
            "node online and syncing",
        )
        assert d["pid"] == node.pid and d["pid"] != ui.pid  # really a separate process
        assert d["qdrant_calls"] > 0 and d["footprint"]["points_local"] > 0 and d["rss_mb"] > 0

        step_before = d["step"]
        # Uplink outage: the node keeps working but its heartbeats really stop.
        _post(f"{base}/api/devices/robot-c/command", {"outage_s": 60})
        _wait(lambda: not _device(base, "robot-c")["online"], 20, "node seen as offline")
        time.sleep(2)  # observations pile up locally, unsent
        node.send_signal(signal.SIGKILL)  # crash in the middle of the outage
        node.wait(timeout=10)

        node = subprocess.Popen(node_cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        d = _wait(
            lambda: (x := _device(base, "robot-c")) and x["online"] and x["restarts"] >= 1 and x,
            30,
            "node back after restart",
        )
        assert d["pid"] == node.pid
        assert d["step"] >= step_before + 10  # it kept patrolling (and persisting) while offline
        _wait(
            lambda: sum((_device(base, "robot-c")["outbox"] or {}).values()) == 0,
            30,
            "backlog drained",
        )
    finally:
        if node is not None and node.poll() is None:
            node.send_signal(signal.SIGTERM)
            node.wait(timeout=15)
        ui.send_signal(signal.SIGTERM)
        ui.wait(timeout=15)

    # With both processes stopped, check the node's own shard against what the cloud received.
    from loci.edge import EdgeMemoryStore

    store = EdgeMemoryStore(tmp_path / "robot-c" / "shard", 64, "robot-c")
    states = store.states()
    store.close()
    queued = [i for i, s in states.items() if s["sync_state"] in {"queued", "queued_summary"}]
    synced = [i for i, s in states.items() if s["sync_state"] == "synced"]
    assert not queued, f"{len(queued)} memories were never delivered"
    assert synced, "nothing was synced"
