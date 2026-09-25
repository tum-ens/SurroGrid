"""FastAPI application of the GridExpand service."""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from gridexpand import paths
from gridexpand.service import API_VERSION, CSRF_HEADER, db, environment
from gridexpand.service.jobs import JobManager
from gridexpand.service.routers import jobs, meta, results, ui
from gridexpand.service.settings import ServiceSettings

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
log = logging.getLogger("gridexpand.service")


class NoCacheStaticFiles(StaticFiles):
    """Static plugin modules: always revalidated, so a new service version is picked up."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _version() -> str:
    try:
        return importlib.metadata.version("gridexpand")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Build the service app.

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
        title="GridExpand service", version=_version(), root_path=settings.root_path, lifespan=lifespan,
        description="Job API and result queries of GridExpand (load allocation, urbs optimisation, power flow, "
                    f"grid expansion) and the pylovo-ui plugin panels. API version {API_VERSION}.",
    )
    app.state.settings = settings
    app.state.jobs = manager

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").lower()
        if not settings.allow_any_host and host not in hosts:
            return JSONResponse({"detail": f"Host '{host}' is not allowed"}, status_code=421)
        # State-changing calls must come from the UI itself: browsers send this custom header
        # only from same-origin scripts (or allowed CORS origins), which blocks CSRF.
        if (request.method not in SAFE_METHODS and request.url.path.startswith("/api/")
                and request.headers.get(CSRF_HEADER.lower()) != "1"):
            return JSONResponse({"detail": f"Missing {CSRF_HEADER} header"}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    if settings.cors_origins:  # development only: pylovo-ui served from another origin
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", CSRF_HEADER, "Last-Event-ID"], max_age=600)

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

    for router in (meta.router, jobs.router, results.router, ui.router):
        app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {"service": "gridexpand", "version": app.version, "api": API_VERSION,
                "docs": "docs", "plugin_manifest": "ui/manifest.json"}

    app.mount("/ui", NoCacheStaticFiles(directory=ui.UI_DIR), name="ui")
    return app
