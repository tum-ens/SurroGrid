"""FastAPI application of the GridExpand API (``gridexpand api``)."""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from gridexpand import paths
from gridexpand.api import API_VERSION, CSRF_HEADER, db, environment
from gridexpand.api.jobs import JobManager
from gridexpand.api.routers import jobs, meta, results, scenarios
from gridexpand.api.settings import ServiceSettings

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
log = logging.getLogger("gridexpand.api")


class RootPathMiddleware:
    """Serve the app below ``root_path`` whether or not a reverse proxy strips the prefix.

    ASGI expects ``path`` to include ``root_path``. A proxy that strips ``/gridexpand`` sends
    ``/api/health``, so the prefix is added back; direct requests work with and without it.
    """

    def __init__(self, app, root_path: str) -> None:
        self.app = app
        self.root_path = root_path.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and self.root_path:
            path = scope["path"]
            if path != self.root_path and not path.startswith(self.root_path + "/"):
                path = self.root_path + path
            scope = dict(scope, path=path, raw_path=path.encode(), root_path=self.root_path)
        await self.app(scope, receive, send)


def _version() -> str:
    try:
        return importlib.metadata.version("gridexpand")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Build the API app.

    Args:
        settings: Runtime settings (default: :meth:`ServiceSettings.from_env`).
    """
    settings = settings or ServiceSettings.from_env()
    hosts = settings.host_allowlist()
    manager = JobManager(settings.jobs_dir, cwd=paths.PROJECT_DIR, max_running=settings.max_running_jobs,
                         env={"GRIDEXPAND_ENV_FILE": str(paths.ENV_FILE), "GRIDEXPAND_WORK_DIR": str(paths.WORK_DIR),
                              "GRIDEXPAND_SOLVER": settings.solver})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The first licence check takes a few seconds; do it before the UI asks.
        threading.Thread(target=environment.check_gurobi, name="gurobi-check", daemon=True).start()
        yield
        manager.shutdown()
        db.reset_engine()

    app = FastAPI(
        title="GridExpand API", version=_version(), lifespan=lifespan,
        description="Jobs, results and scenario files of GridExpand (load allocation, urbs optimisation, power "
                    f"flow, grid expansion) for the GridPlanner UI. API version {API_VERSION}.",
    )
    app.state.settings = settings
    app.state.jobs = manager
    root = settings.root_path.rstrip("/")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").lower()
        if not settings.allow_any_host and host not in hosts:
            return JSONResponse({"detail": f"Host '{host}' is not allowed"}, status_code=421)
        # State-changing calls must come from the UI itself: browsers send this custom header
        # only from same-origin scripts (or allowed CORS origins), which blocks CSRF. Checked for
        # every path, so no prefix spelling can bypass it.
        if request.method not in SAFE_METHODS and request.headers.get(CSRF_HEADER.lower()) != "1":
            return JSONResponse({"detail": f"Missing {CSRF_HEADER} header"}, status_code=403)
        path = request.url.path
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        response = await call_next(request)
        if path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    if settings.cors_origins:  # development only: a UI served from another origin
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Content-Type", CSRF_HEADER, "Last-Event-ID"], max_age=600)
    if root:  # outermost: every other layer sees the ASGI form path = root_path + route path
        app.add_middleware(RootPathMiddleware, root_path=root)

    @app.exception_handler(db.DatabaseUnavailable)
    async def database_unavailable(_: Request, exc: db.DatabaseUnavailable):
        return JSONResponse({"detail": f"Database not reachable: {exc}"}, status_code=503)

    @app.exception_handler(OperationalError)
    async def operational_error(_: Request, exc: OperationalError):
        return JSONResponse({"detail": f"Database error: {db.error_message(exc)}"}, status_code=503)

    @app.exception_handler(ProgrammingError)
    async def programming_error(_: Request, exc: ProgrammingError):
        return JSONResponse({"detail": "The database schema is missing or incomplete "
                                       f"({db.error_message(exc)})"}, status_code=409)

    @app.exception_handler(DBAPIError)
    async def database_error(_: Request, exc: DBAPIError):
        return JSONResponse({"detail": f"Database error: {db.error_message(exc)}"}, status_code=500)

    for router in (meta.router, scenarios.router, jobs.router, results.router):
        app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {"service": "gridexpand-api", "version": app.version, "api": API_VERSION,
                "docs": "docs", "openapi": "openapi.json"}

    return app
