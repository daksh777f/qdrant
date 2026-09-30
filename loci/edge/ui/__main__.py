"""Run the mission-control UI:  python -m loci.edge.ui [--host 127.0.0.1] [--port 8765]"""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    from loci.edge.ui.app import create_app

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--no-autosync",
        action="store_true",
        help="sync only when 'Sync now' is pressed (deterministic demos and recordings)",
    )
    args = ap.parse_args()
    print(f"LOCI Edge Mission Control on http://{args.host}:{args.port}  (synthetic demo data)")
    uvicorn.run(
        create_app(autosync=not args.no_autosync),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
