"""Mission-control web UI for the LOCI edge fleet (FastAPI + a single static page)."""

from loci.edge.ui.app import create_app

__all__ = ["create_app"]
