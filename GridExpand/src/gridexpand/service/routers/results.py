"""Results: expansion analyses, their grids and map layers, power-flow summaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from gridexpand.service import queries
from gridexpand.service.routers.meta import parse_ags

router = APIRouter(prefix="/api/results", tags=["results"])

VERSION_QUERY = Query(None, pattern=r"^[\w.-]{1,10}$")
PLZ_QUERY = Query(None, ge=1, le=99999)


@router.get("/analyses")
def analyses(ags: str | None = None, plz: int | None = PLZ_QUERY,
             pylovo_version_id: str | None = VERSION_QUERY) -> list[dict[str, Any]]:
    """``expansion_analysis_run`` rows with totals (cost, cables and transformers to reinforce).

    With ``plz`` (and/or ``pylovo_version_id``) the totals cover only those grids and
    analyses without such grids are left out.
    """
    return queries.analyses(parse_ags(ags), plz, pylovo_version_id)


@router.get("/analyses/{analysis_key}/grids")
def analysis_grids(analysis_key: str, plz: int | None = PLZ_QUERY,
                   pylovo_version_id: str | None = VERSION_QUERY) -> list[dict[str, Any]]:
    """Per-grid cost, cables to reinforce and transformer loading of one analysis."""
    rows = queries.analysis_grids(analysis_key, plz, pylovo_version_id)
    if rows is None:
        raise HTTPException(404, f"Analysis '{analysis_key}' not found")
    return rows


@router.get("/analyses/{analysis_key}/geojson")
def analysis_geojson(analysis_key: str, plz: int | None = PLZ_QUERY,
                     pylovo_version_id: str | None = VERSION_QUERY) -> dict[str, Any]:
    """Cables (``action``: none / add_1 / add_2plus) and transformers of one analysis (EPSG:4326)."""
    data = queries.analysis_geojson(analysis_key, plz, pylovo_version_id)
    if data is None:
        raise HTTPException(404, f"Analysis '{analysis_key}' not found")
    return data


@router.get("/powerflow")
def powerflow(ags: str | None = None, plz: int | None = PLZ_QUERY, pylovo_version_id: str | None = VERSION_QUERY,
              scenario_key: str | None = Query(None, max_length=200)) -> list[dict[str, Any]]:
    """``powerflow_summary`` per grid, run (model case) and stage: loading percentiles, voltage p05, hours above limits."""
    return queries.powerflow_summaries(parse_ags(ags), plz, pylovo_version_id, scenario_key)
