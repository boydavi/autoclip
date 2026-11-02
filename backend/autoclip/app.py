"""FastAPI application — REST API plus the built frontend, on one port.

Serving the compiled React bundle from the same origin as the API is what keeps
AutoClip a single process with no CORS configuration, no second port, and no
Node.js at runtime.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__, db, paths
from .api import api_router
from .db import store
from .jobs.events import broker
from .jobs.queue import queue

log = logging.getLogger(__name__)

#: Set to "1" to serve the API without starting the background job worker.
ENV_NO_WORKER = "AUTOCLIP_NO_WORKER"

#: When set, every /api/* request must carry "Authorization: Bearer <token>".
#: The health endpoint is intentionally exempt so monitoring tools keep working.
ENV_API_KEY = "AUTOCLIP_API_KEY"


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject /api/* requests (except /api/health) unless a valid bearer token is present."""

    def __init__(self, app, *, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path != "/api/health":
            auth = request.headers.get("Authorization", "")
            # Use compare_digest to guard against timing attacks.
            provided = auth.removeprefix("Bearer ").strip()
            if not secrets.compare_digest(provided, self._token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)


#: Extra browser origins (comma-separated, e.g. "https://tool.example,http://10.0.0.5:3000")
#: allowed to make state-changing requests. The UI's own origin is always allowed.
ENV_ALLOWED_ORIGINS = "AUTOCLIP_ALLOWED_ORIGINS"

#: Vite's dev server (`npm run dev`), which proxies /api to this backend.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def origin_allowed(origin: str, host: str, allowed: frozenset[str]) -> bool:
    """Whether a browser-supplied ``Origin`` may make a state-changing request.

    Same-origin (the Origin's host:port equals the request's Host) is accepted
    only when that Host is ``localhost`` or an IP address. A DNS-rebinding page
    (``evil.example`` re-pointed at 127.0.0.1) sends matching Origin and Host
    headers, so a name-based match proves nothing; reaching AutoClip under a
    hostname requires listing its origin explicitly. The literal ``null``
    origin (sandboxed iframes, file:// pages) is never allowed.
    """
    origin = origin.strip().lower()
    if not origin or origin == "null":
        return False
    if origin in allowed:
        return True
    netloc = urlsplit(origin).netloc
    if not netloc or netloc != host.strip().lower():
        return False
    return _is_trusted_host(host)


def _is_trusted_host(host: str) -> bool:
    """``localhost`` or an IP literal: names an attacker's DNS cannot hijack."""
    name = urlsplit(f"//{host.strip()}").hostname or ""
    if name == "localhost":
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


class _OriginCheckMiddleware(BaseHTTPMiddleware):
    """Block cross-site state changes against /api/*.

    Browsers let any web page POST a multipart form to ``localhost`` without a
    CORS preflight, so a page you merely visit could upload files or start jobs.
    Browsers always attach ``Origin`` to such requests and pages cannot forge or
    drop it, so rejecting unknown origins closes that hole. Requests with no
    Origin (curl, scripts, other servers) are not browser-driven and pass; they
    are covered by ``AUTOCLIP_API_KEY`` when set. Reads (GET) are untouched.
    """

    def __init__(self, app, *, allowed: frozenset[str]) -> None:
        super().__init__(app)
        self._allowed = allowed

    async def dispatch(self, request: Request, call_next):
        if request.method in _UNSAFE_METHODS and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin is not None and not origin_allowed(
                origin, request.headers.get("host", ""), self._allowed
            ):
                log.warning("Blocked cross-origin %s %s from %r.", request.method, request.url.path, origin)
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Cross-origin request blocked.",
                        "hint": f"Open AutoClip from its own address, or add this origin to {ENV_ALLOWED_ORIGINS}.",
                    },
                )
        return await call_next(request)


def _allowed_origins() -> frozenset[str]:
    extra = os.environ.get(ENV_ALLOWED_ORIGINS, "")
    configured = {o.strip().rstrip("/").lower() for o in extra.split(",") if o.strip()}
    return frozenset({*DEV_ORIGINS, *configured})


#: Where the built frontend is looked for, in priority order. The first is the
#: packaged location (inside the wheel); the second is the dev build output.
_STATIC_CANDIDATES = (
    Path(__file__).resolve().parent / "static",
    Path(__file__).resolve().parents[2] / "frontend" / "dist",
)


def static_dir() -> Path | None:
    for candidate in _STATIC_CANDIDATES:
        if (candidate / "index.html").exists():
            return candidate
    return None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    import asyncio

    paths.ensure_layout()
    db.init()

    interrupted_sources, interrupted_exports = store.fail_interrupted_work()
    if interrupted_sources or interrupted_exports:
        log.warning(
            "Marked %d pending source(s) and %d pending export(s) as failed after restart.",
            interrupted_sources,
            interrupted_exports,
        )

    broker.bind_loop(asyncio.get_running_loop())

    # Serving the API without a worker is useful for debugging a stuck queue and
    # makes API tests deterministic — jobs stay queued instead of racing off.
    worker_enabled = os.environ.get(ENV_NO_WORKER) != "1"
    if worker_enabled:
        await queue.start()
    else:
        log.info("Job worker disabled by %s=1.", ENV_NO_WORKER)

    log.info("AutoClip %s ready. Artifacts in %s", __version__, paths.root())

    try:
        yield
    finally:
        if worker_enabled:
            await queue.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoClip",
        version=__version__,
        description="Local-first AI video clipper.",
        lifespan=lifespan,
    )

    api_key = os.environ.get(ENV_API_KEY, "").strip()
    if api_key:
        app.add_middleware(_BearerAuthMiddleware, token=api_key)
        log.info("API authentication enabled (AUTOCLIP_API_KEY is set).")
    else:
        log.warning(
            "AUTOCLIP_API_KEY is not set — all API endpoints are unauthenticated. "
            "Set it before exposing AutoClip beyond localhost."
        )

    # Added after the auth middleware so it runs first (the last one added is
    # outermost): a cross-site request is refused before auth is even consulted.
    app.add_middleware(_OriginCheckMiddleware, allowed=_allowed_origins())

    app.include_router(api_router)

    @app.get("/api/health")
    async def health() -> dict:
        status = queue.status()
        return {
            "status": "ok",
            "version": __version__,
            "running_job": status.running_job_id,
            "queued": status.queued,
        }

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    directory = static_dir()

    if directory is None:

        @app.get("/", include_in_schema=False)
        async def missing_frontend() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "The frontend has not been built.",
                    "hint": (
                        "Run `npm install && npm run build` in the frontend/ "
                        "directory, then restart. The API itself is available "
                        "at /docs."
                    ),
                },
            )

        log.warning("No built frontend found; serving the API only.")
        return

    app.mount(
        "/assets",
        StaticFiles(directory=directory / "assets"),
        name="assets",
    )

    # response_model=None because the return type is a union of Response
    # subclasses, which FastAPI would otherwise try to turn into a Pydantic
    # response model and reject.
    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def spa(request: Request, full_path: str) -> FileResponse | JSONResponse:
        """Serve the SPA, letting the client router own unknown paths.

        API routes are registered first and win; anything else that isn't a real
        file falls through to index.html so a deep link like /jobs/abc123 works
        on a hard refresh.
        """
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not found."})

        candidate = directory / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(directory / "index.html")

    log.info("Serving the frontend from %s", directory)


app = create_app()
