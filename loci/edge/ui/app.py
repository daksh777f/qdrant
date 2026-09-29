"""HTTP API + static page for the edge mission-control UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from loci.edge.sim import Fleet

STATIC = Path(__file__).parent / "static"


class LinkBody(BaseModel):
    up: bool


class PatrolBody(BaseModel):
    steps: int = 10


def create_app(fleet: Fleet | None = None, *, autosync: bool = True) -> FastAPI:
    owned = fleet is None
    fleet = fleet or Fleet()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autosync:
            fleet.start_auto()
        yield
        if owned:
            fleet.close()
        else:
            fleet.stop_auto()

    app = FastAPI(title="LOCI Edge Mission Control", lifespan=lifespan)
    app.state.fleet = fleet

    def robot(name: str) -> str:
        if name not in fleet.nodes:
            raise HTTPException(404, f"unknown robot {name!r}")
        return name

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> dict:
        return fleet.state()

    @app.post("/api/robots/{name}/link")
    def link(name: str, body: LinkBody) -> dict:
        fleet.set_link(robot(name), body.up)
        return {"robot": name, "online": body.up}

    @app.post("/api/robots/{name}/patrol")
    def patrol(name: str, body: PatrolBody) -> dict:
        d = fleet.patrol(robot(name), body.steps)
        counts: dict[str, int] = {}
        for x in d:
            counts[x["action"]] = counts.get(x["action"], 0) + 1
        return {"steps": len(d), "decisions": counts}

    @app.post("/api/robots/{name}/spill")
    def spill(name: str) -> dict:
        return fleet.spill(robot(name))

    @app.post("/api/robots/{name}/toolbox")
    def toolbox(name: str) -> dict:
        return fleet.move_toolbox(robot(name))

    @app.post("/api/robots/{name}/private")
    def private(name: str) -> dict:
        return fleet.private_note(robot(name))

    @app.post("/api/robots/{name}/sync")
    def sync(name: str) -> dict:
        return fleet.sync(robot(name))

    @app.get("/api/robots/{name}/dashboard")
    def dashboard(name: str) -> dict:
        robot(name)
        return {
            "decisions": fleet.decisions(name, 40),
            "diff": fleet.diff(name),
            "map": fleet.map(name),
        }

    @app.get("/api/robots/{name}/memories")
    def memories(name: str, limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        return fleet.memories(robot(name), limit)

    @app.get("/api/robots/{name}/search")
    def search(name: str, q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=50)):
        return fleet.search(robot(name), q, limit)

    return app
