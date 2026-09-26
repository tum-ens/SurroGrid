"""Health, status and candidate grids (scenario files: :mod:`.scenarios`)."""

from __future__ import annotations

import platform
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from gridexpand import paths
from gridexpand.service import API_VERSION, environment, queries
from gridexpand.service.settings import ServiceSettings

router = APIRouter(prefix="/api", tags=["status"])


def settings_of(request: Request) -> ServiceSettings:
    return request.app.state.settings


def parse_ags(value: int | str | None) -> int | None:
    """AGS as the integer GridExpand stores (``"09184137"`` -> ``9184137``)."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if not text.isdigit() or len(text) > 12:
        raise HTTPException(400, f"Invalid AGS '{value}'")
    return int(text)


def resolve_ags(ags: int | str | None, plz: int | None) -> tuple[int, list[int]]:
    """The AGS of a request: given, or the only AGS of the PLZ in the municipal register."""
    explicit = parse_ags(ags)
    if explicit is not None:
        return explicit, [explicit]
    if plz is None:
        raise HTTPException(400, "Give an AGS or a PLZ")
    options = queries.ags_for_plz(plz)
    if not options:
        raise HTTPException(404, f"PLZ {plz} is not in pylovo.municipal_register")
    if len(options) > 1:
        raise HTTPException(409, f"PLZ {plz} belongs to several AGS {options}; choose one")
    return options[0], options


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe (no database access)."""
    return {"ok": True, "service": "gridexpand", "api": API_VERSION}


@router.get("/status")
def status(request: Request) -> dict[str, Any]:
    """Database, schema, pylovo versions, solvers, large input files, disk and jobs."""
    settings = settings_of(request)
    return {
        "service": {
            "api": API_VERSION,
            "version": request.app.version,
            "python": platform.python_version(),
            "project_dir": str(paths.PROJECT_DIR),
            "work_dir": str(paths.WORK_DIR),
            "env_file": str(paths.ENV_FILE) if paths.ENV_FILE.is_file() else None,  # None: environment only
            "root_path": settings.root_path,
        },
        "database": queries.database_status(),
        "solvers": environment.solver_status(settings.solver),
        "data_assets": environment.data_assets(),
        "disk": environment.disk_usage(paths.WORK_DIR),
        "jobs": request.app.state.jobs.counts(),
    }


@router.get("/grids")
def grids(
    ags: str | None = None,
    plz: int | None = Query(None, ge=1, le=99999),
    pylovo_version_id: str = Query(..., pattern=r"^[\w.-]{1,10}$"),
    min_buildings: int = Query(5, ge=1, le=100_000),
) -> dict[str, Any]:
    """Candidate grids with the runner's numbering (all of the AGS, or one PLZ)."""
    resolved, options = resolve_ags(ags, plz)
    candidates = queries.grid_candidates(resolved, pylovo_version_id, min_buildings, plz)
    return {"ags": resolved, "ags_options": options, "plz": plz, "pylovo_version_id": pylovo_version_id,
            "min_buildings": min_buildings, "candidates": candidates}
