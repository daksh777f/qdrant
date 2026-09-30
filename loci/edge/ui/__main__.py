"""Run mission control (the cloud + UI), optionally with edge devices as separate processes.

    python -m loci.edge.ui                     # mission control, two simulator robots
    python -m loci.edge.ui --nodes 2           # ... plus robot-c and robot-d as their own processes
    LOCI_QDRANT_URL=http://localhost:6333 python -m loci.edge.ui   # cloud on a Qdrant Server

Device processes keep their data in ./edge-nodes/<name> (survives restarts) and are stopped
when mission control exits. Start more on other machines with ``python -m loci.edge.node``
(bind with ``--host 0.0.0.0`` and set ``LOCI_CLOUD_TOKEN`` on both sides).
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def _spawn_nodes(n: int, base: str, interval: float) -> list[subprocess.Popen]:
    procs = []
    for i in range(n):
        name = f"robot-{chr(ord('c') + i)}"
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "loci.edge.node",
                    "--name",
                    name,
                    "--cloud",
                    base,
                    "--data",
                    os.path.join(os.environ.get("LOCI_NODES_DIR", "edge-nodes"), name),
                    "--interval",
                    str(interval),
                    "--exit-with-parent",
                ],  # fmt: skip
                env=os.environ.copy(),
            )
        )
    return procs


def main() -> None:
    import uvicorn

    from loci.edge.ui.app import create_app

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--no-autosync",
        action="store_true",
        help="sync only when 'Sync now' is pressed (deterministic demos and recordings)",
    )
    ap.add_argument("--nodes", type=int, default=0, help="also start N edge-device processes")
    ap.add_argument("--node-interval", type=float, default=1.0, help="seconds per device step")
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost"} and not os.environ.get("LOCI_CLOUD_TOKEN"):
        print(
            "WARNING: listening beyond localhost without LOCI_CLOUD_TOKEN: the cloud API is open."
        )
    base = f"http://127.0.0.1:{args.port}"
    print(f"LOCI Edge Mission Control on http://{args.host}:{args.port}  (synthetic demo data)")

    procs: list[subprocess.Popen] = []

    def start_nodes() -> None:
        for _ in range(100):  # wait for the server before the devices start calling it
            try:
                urllib.request.urlopen(base + "/api/state", timeout=1)  # noqa: S310
                break
            except OSError:
                time.sleep(0.2)
        procs.extend(_spawn_nodes(args.nodes, base, args.node_interval))

    def stop_nodes() -> None:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    if args.nodes:
        threading.Thread(target=start_nodes, daemon=True).start()
        atexit.register(stop_nodes)
    try:
        uvicorn.run(
            create_app(autosync=not args.no_autosync),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    finally:
        stop_nodes()


if __name__ == "__main__":
    main()
