"""The cloud over HTTP, so edge devices can be separate processes (or separate machines).

* :func:`cloud_router` exposes any :class:`~loci.edge.cloud.CloudStore` as a small JSON API,
  plus ``/heartbeat`` (device telemetry in, pending commands out).
* :class:`HttpCloud` is the device-side client. It implements ``CloudStore``; a refused
  connection, timeout or 5xx is raised as :class:`~loci.edge.cloud.LinkDown`, so an unreachable
  cloud is handled exactly like a cut network (the outbox keeps everything).

Authentication: if the server is given a ``token`` every request must carry
``Authorization: Bearer <token>`` (use this whenever the port is reachable by others).
"""

from __future__ import annotations

import hmac
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Annotated, Any

from loci.edge.cloud import LinkDown, wire_bytes

try:  # the server side needs FastAPI (the `edge-ui` extra); the client does not
    from fastapi import Body, Header

    JsonBody = Annotated[dict, Body()]
    AuthHeader = Annotated[str | None, Header()]
except ImportError:  # pragma: no cover - device-only install
    JsonBody = dict  # type: ignore[misc]
    AuthHeader = str | None  # type: ignore[misc,assignment]


class HttpCloud:
    """Device-side ``CloudStore`` over HTTP (stdlib only, so it runs on small devices)."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 5.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be http(s)")
        self.base = base_url.rstrip("/") + "/cloud"
        self.token = token
        self.timeout = timeout
        self.kind = f"http cloud @ {base_url}"
        self.bytes_sent = 0
        self.bytes_received = 0

    def _req(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)  # noqa: S310
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310  # nosec B310
                raw = r.read()
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 or exc.code == 429:
                raise LinkDown(f"cloud answered {exc.code}") from exc
            raise RuntimeError(f"cloud rejected request: {exc.code} {exc.read()[:200]!r}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LinkDown(f"cloud unreachable: {exc}") from exc
        self.bytes_sent += len(data or b"")
        self.bytes_received += len(raw)
        return json.loads(raw) if raw else None

    def upsert(self, points: list[dict]) -> int:
        return int(self._req("POST", "/upsert", {"points": points})["received"])

    def index(self) -> dict[str, tuple[int, str, int]]:
        return {k: (int(v[0]), str(v[1]), int(v[2])) for k, v in self._req("GET", "/index").items()}

    def versions(self) -> dict[str, int]:
        return {i: v[0] for i, v in self.index().items()}

    def count(self) -> int:
        return int(self._req("GET", "/count")["count"])

    def get(self, ids: list[str]) -> list[dict]:
        return list(self._req("POST", "/get", {"ids": ids})) if ids else []

    def search(self, vector: list[float], limit: int = 8, *, current_only: bool = False) -> Any:
        body = {"vector": vector, "limit": limit, "current_only": current_only}
        return self._req("POST", "/search", body)

    def heartbeat(self, telemetry: dict) -> dict:
        return dict(self._req("POST", "/heartbeat", telemetry) or {})


def cloud_router(
    get_cloud: Callable[[], Any],
    *,
    on_upsert: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[dict], dict] | None = None,
    token: str | None = None,
    lock: Any = None,
) -> Any:
    """A FastAPI router serving *get_cloud()* under ``/cloud``."""
    import contextlib

    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/cloud", tags=["cloud"])
    guard = lock if lock is not None else contextlib.nullcontext()

    def auth(authorization: str | None) -> None:
        if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "missing or bad token")

    @router.post("/upsert")
    def upsert(body: JsonBody, authorization: AuthHeader = None) -> dict:
        auth(authorization)
        points = body.get("points") or []
        if not isinstance(points, list) or len(points) > 5_000:
            raise HTTPException(400, "points must be a list of at most 5000")
        with guard:
            received = get_cloud().upsert(points)
        if on_upsert is not None and received:
            on_upsert(len(points))
        return {"received": received, "wire_bytes": sum(wire_bytes(p) for p in points)}

    @router.get("/index")
    def index(authorization: AuthHeader = None) -> dict:
        auth(authorization)
        with guard:
            return {k: list(v) for k, v in get_cloud().index().items()}

    @router.get("/count")
    def count(authorization: AuthHeader = None) -> dict:
        auth(authorization)
        with guard:
            return {"count": get_cloud().count()}

    @router.post("/get")
    def get(body: JsonBody, authorization: AuthHeader = None) -> list:
        auth(authorization)
        ids = [str(i) for i in (body.get("ids") or [])][:5_000]
        with guard:
            return list(get_cloud().get(ids))

    @router.post("/search")
    def search(body: JsonBody, authorization: AuthHeader = None) -> list:
        auth(authorization)
        vec = body.get("vector")
        if not isinstance(vec, list) or not vec:
            raise HTTPException(400, "vector required")
        limit = max(1, min(int(body.get("limit", 8)), 50))
        with guard:
            return list(
                get_cloud().search(
                    [float(v) for v in vec], limit, current_only=bool(body.get("current_only"))
                )
            )

    @router.post("/heartbeat")
    def heartbeat(body: JsonBody, authorization: AuthHeader = None) -> dict:
        auth(authorization)
        return on_heartbeat(body) if on_heartbeat is not None else {}

    return router
