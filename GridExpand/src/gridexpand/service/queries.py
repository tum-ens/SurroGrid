"""Read queries of the service (plain SQL on stable ``pylovo`` and ``surrogrid`` names).

Only table and view names that notebooks and QGIS projects already rely on are used:
``pylovo.version``, ``grid_result``, ``municipal_register``; ``surrogrid.grid_case``,
``scenario``, ``powerflow_run``, ``powerflow_summary``, ``expansion_analysis_run``,
``expansion_line_result``, ``expansion_transformer_result`` and the QGIS materialized views
``expansion_line_qgis_mv`` / ``expansion_transformer_qgis_mv`` (they carry the geometry).
Geometries are returned as GeoJSON in EPSG:4326 with 6 decimals.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from sqlalchemy.engine import Connection

from gridexpand.common.timeframe import TIMEFRAME_MODES
from gridexpand.service.db import (
    DatabaseUnavailable,
    connect,
    connection_info,
    fetch_all,
    fetch_one,
)

PYLOVO_RELATIONS = ("pylovo.version", "pylovo.grid_result", "pylovo.buildings_result", "pylovo.municipal_register",
                    "pylovo.lines_result_view", "pylovo.transformer_positions_with_grid")
SURROGRID_RELATIONS = ("surrogrid.grid_case", "surrogrid.scenario", "surrogrid.powerflow_run",
                       "surrogrid.powerflow_summary", "surrogrid.expansion_analysis_run",
                       "surrogrid.expansion_line_result", "surrogrid.expansion_transformer_result",
                       "surrogrid.expansion_line_qgis_mv", "surrogrid.expansion_transformer_qgis_mv")

PROFILE_TOKENS = ("status_quo", "post_electrification", "electricity_heat_mobility", "electricity_heat",
                  "electricity_mobility")
RUN_CASES = ("pre", "post-inflex-heuristic", "post-hems-optimized", "post-hems-heuristic")
RUN_MODES = ("summary_inflex", "raw_inflex", "summary", "raw")
_RUN_NAME_RE = re.compile(
    rf"^(?P<prefix>.+?)_(?P<profile>{'|'.join(PROFILE_TOKENS)})"
    rf"(?:_(?P<case>{'|'.join(map(re.escape, RUN_CASES))}))?_(?P<mode>{'|'.join(RUN_MODES)})_powerflow$"
)


# --------------------------------------------------------------------------- helpers
def parse_run_name(run_name: str) -> dict[str, Any]:
    """Profile, model case and output mode encoded in a synthetic power-flow run name.

    ``<scenario_key>_<profile>[_<model case>]_<mode>_powerflow`` (see
    ``synthetic_ags_runner.powerflow_run_name``). Runs without a model case are ``pre``
    for status-quo profiles and unknown otherwise.
    """
    match = _RUN_NAME_RE.match(run_name or "")
    if not match:
        return {"profile": None, "model_case": None, "mode": None}
    case = match["case"] or ("pre" if match["profile"] == "status_quo" else None)
    return {"profile": match["profile"], "model_case": case, "mode": match["mode"]}


def split_scenario_key(scenario_key: str | None) -> dict[str, Any]:
    """Scenario identity and timeframe mode of a pipeline scenario key."""
    key = scenario_key or ""
    for mode in TIMEFRAME_MODES:
        if key.endswith(f"_{mode}"):
            return {"scenario_key": key, "scenario": key[: -len(mode) - 1], "timeframe_mode": mode}
    return {"scenario_key": key or None, "scenario": key or None, "timeframe_mode": None}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean(value) for key, value in row.items()}


def relations(conn: Connection, names: tuple[str, ...]) -> dict[str, bool]:
    """Which of these tables/views exist."""
    rows = fetch_all("SELECT n AS name, to_regclass(n) IS NOT NULL AS present FROM unnest(CAST(:names AS text[])) n",
                     conn, names=list(names))
    return {row["name"]: bool(row["present"]) for row in rows}


def surrogrid_ready(conn: Connection) -> bool:
    return all(relations(conn, SURROGRID_RELATIONS).values())


# --------------------------------------------------------------------------- status
def database_status() -> dict[str, Any]:
    """Connection, server versions, schema state and pylovo versions with grid counts."""
    info: dict[str, Any] = {"connected": False, "error": None, **connection_info()}
    try:
        with connect() as conn:
            server = fetch_one(
                "SELECT current_setting('server_version') AS postgres,"
                " (SELECT extversion FROM pg_extension WHERE extname = 'postgis') AS postgis,"
                " (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') AS timescaledb", conn)
            present = relations(conn, PYLOVO_RELATIONS + SURROGRID_RELATIONS)
            versions = []
            if present["pylovo.version"] and present["pylovo.grid_result"]:
                versions = fetch_all(
                    """SELECT v.version_id, v.version_comment, v.created_at,
                              count(g.grid_result_id) AS grids, count(DISTINCT g.plz) AS plz_count
                       FROM pylovo.version v LEFT JOIN pylovo.grid_result g ON g.version_id = v.version_id
                       GROUP BY v.version_id, v.version_comment, v.created_at
                       ORDER BY v.created_at, v.version_id""", conn)
    except DatabaseUnavailable as exc:
        info["error"] = str(exc)
        return info
    missing = [name for name, ok in present.items() if not ok]
    info.update(
        connected=True,
        server=server,
        schema={
            "pylovo_ready": all(present[name] for name in PYLOVO_RELATIONS),
            "surrogrid_ready": all(present[name] for name in SURROGRID_RELATIONS),
            "missing": missing,
        },
        pylovo_versions=versions,
    )
    return info


# --------------------------------------------------------------------------- regions and grids
def ags_for_plz(plz: int) -> list[int]:
    """AGS codes of a PLZ in pylovo's municipal register."""
    rows = fetch_all("SELECT DISTINCT ags FROM pylovo.municipal_register WHERE plz = :plz ORDER BY ags", plz=plz)
    return [int(row["ags"]) for row in rows]


def grid_candidates(ags: int, pylovo_version_id: str, min_buildings: int, plz: int | None = None) -> list[dict[str, Any]]:
    """Candidate grids of an AGS with the runner's numbering, optionally one PLZ only.

    Every candidate lists the model cases that already have power-flow summaries.
    """
    from gridexpand.scenario.synthetic_ags_runner import get_candidates

    candidates = get_candidates(str(ags), int(min_buildings), "all", str(pylovo_version_id))
    if plz is not None:
        candidates = [c for c in candidates if int(c["plz"]) == int(plz)]
    ids = [int(c["grid_result_id"]) for c in candidates]
    results: dict[int, list[dict[str, Any]]] = {}
    if ids:
        with connect() as conn:
            if surrogrid_ready(conn):
                rows = fetch_all(
                    """SELECT gc.pylovo_grid_result_id AS grid_result_id, pr.run_name, s.scenario_key,
                              max(pr.updated_at) AS updated_at
                       FROM surrogrid.grid_case gc
                       JOIN surrogrid.powerflow_run pr ON pr.grid_case_id = gc.grid_case_id
                       JOIN surrogrid.scenario s ON s.scenario_id = pr.scenario_id
                       JOIN surrogrid.powerflow_summary ps ON ps.powerflow_run_id = pr.powerflow_run_id
                       WHERE gc.pylovo_grid_result_id = ANY(:ids)
                       GROUP BY 1, 2, 3 ORDER BY 1, 4""", conn, ids=ids)
                for row in rows:
                    parsed = parse_run_name(row["run_name"])
                    results.setdefault(int(row["grid_result_id"]), []).append(
                        {"model_case": parsed["model_case"], "mode": parsed["mode"],
                         **split_scenario_key(row["scenario_key"]), "updated_at": row["updated_at"]})
    return [c | {"results": results.get(int(c["grid_result_id"]), [])} for c in candidates]


# --------------------------------------------------------------------------- expansion analyses
_ANALYSES_SQL = """
WITH lines AS (
    SELECT expansion_analysis_run_id,
           count(*) AS lines_total,
           count(*) FILTER (WHERE requires_expansion) AS lines_to_reinforce,
           coalesce(sum(length_km) FILTER (WHERE requires_expansion), 0) AS km_to_reinforce,
           coalesce(sum(additional_parallel), 0) AS additional_cables,
           coalesce(sum(estimated_cost_eur), 0) AS line_cost_eur,
           max(loading_percent) AS max_line_loading_percent,
           count(DISTINCT grid_case_id) AS grids,
           array_agg(DISTINCT plz) AS plz_list,
           array_agg(DISTINCT pylovo_version_id) AS versions
    FROM surrogrid.expansion_line_result
    WHERE (CAST(:plz AS integer) IS NULL OR plz = CAST(:plz AS integer))
      AND (CAST(:version AS text) IS NULL OR pylovo_version_id = CAST(:version AS text))
    GROUP BY expansion_analysis_run_id
), trafos AS (
    SELECT expansion_analysis_run_id,
           count(*) AS transformers_total,
           count(*) FILTER (WHERE requires_expansion) AS transformers_to_reinforce,
           coalesce(sum(estimated_cost_eur), 0) AS transformer_cost_eur,
           max(loading_percent) AS max_transformer_loading_percent,
           coalesce(sum(additional_transformer_kva), 0) AS additional_transformer_kva,
           count(DISTINCT grid_case_id) AS grids
    FROM surrogrid.expansion_transformer_result
    WHERE (CAST(:plz AS integer) IS NULL OR plz = CAST(:plz AS integer))
      AND (CAST(:version AS text) IS NULL OR pylovo_version_id = CAST(:version AS text))
    GROUP BY expansion_analysis_run_id
), runs AS (
    SELECT DISTINCT ON (pr.run_name) pr.run_name, s.scenario_key
    FROM surrogrid.powerflow_run pr JOIN surrogrid.scenario s ON s.scenario_id = pr.scenario_id
    ORDER BY pr.run_name, pr.powerflow_run_id DESC
)
SELECT ar.expansion_analysis_run_id, ar.analysis_key, ar.assumption_key, ar.run_name, ar.stage, ar.ags,
       ar.plz AS analysis_plz, ar.created_at, ar.note, runs.scenario_key,
       greatest(coalesce(l.grids, 0), coalesce(t.grids, 0)) AS grids,
       coalesce(l.lines_total, 0) AS lines_total, coalesce(l.lines_to_reinforce, 0) AS lines_to_reinforce,
       coalesce(l.km_to_reinforce, 0) AS km_to_reinforce, coalesce(l.additional_cables, 0) AS additional_cables,
       coalesce(l.line_cost_eur, 0) AS line_cost_eur, l.max_line_loading_percent,
       coalesce(t.transformers_total, 0) AS transformers_total,
       coalesce(t.transformers_to_reinforce, 0) AS transformers_to_reinforce,
       coalesce(t.transformer_cost_eur, 0) AS transformer_cost_eur, t.max_transformer_loading_percent,
       coalesce(t.additional_transformer_kva, 0) AS additional_transformer_kva,
       coalesce(l.line_cost_eur, 0) + coalesce(t.transformer_cost_eur, 0) AS total_cost_eur,
       l.plz_list, l.versions
FROM surrogrid.expansion_analysis_run ar
LEFT JOIN lines l ON l.expansion_analysis_run_id = ar.expansion_analysis_run_id
LEFT JOIN trafos t ON t.expansion_analysis_run_id = ar.expansion_analysis_run_id
LEFT JOIN runs ON runs.run_name = ar.run_name
WHERE coalesce(ar.data_source, 'Synthetic') = 'Synthetic'
  AND (CAST(:ags AS bigint) IS NULL OR ar.ags = CAST(:ags AS bigint))
  AND ((CAST(:plz AS integer) IS NULL AND CAST(:version AS text) IS NULL)
       OR l.expansion_analysis_run_id IS NOT NULL OR t.expansion_analysis_run_id IS NOT NULL)
ORDER BY ar.created_at DESC, ar.analysis_key
"""


def analysis_label(model_case: str | None, stage: str) -> str:
    """Short label: the status-quo reference or the post case of an analysis."""
    if stage == "pre":
        return "Status quo" if model_case in (None, "pre") else f"Status quo (reference of {model_case})"
    return model_case or stage


def analyses(ags: int | None = None, plz: int | None = None, pylovo_version_id: str | None = None) -> list[dict[str, Any]]:
    """Expansion analyses with totals (restricted to the grids of a PLZ / version if given)."""
    with connect() as conn:
        if not surrogrid_ready(conn):
            return []
        rows = fetch_all(_ANALYSES_SQL, conn, ags=ags, plz=plz, version=pylovo_version_id)
    out = []
    for row in rows:
        parsed = parse_run_name(row["run_name"])
        out.append(_clean_row(row) | parsed | split_scenario_key(row["scenario_key"])
                   | {"label": analysis_label(parsed["model_case"], row["stage"]),
                      "plz_list": sorted(p for p in (row["plz_list"] or []) if p is not None),
                      "versions": sorted(v for v in (row["versions"] or []) if v is not None)})
    return out


def _analysis_id(conn: Connection, analysis_key: str) -> int | None:
    row = fetch_one("SELECT expansion_analysis_run_id AS id FROM surrogrid.expansion_analysis_run "
                    "WHERE analysis_key = :key", conn, key=analysis_key)
    return int(row["id"]) if row else None


_ANALYSIS_GRIDS_SQL = """
WITH lines AS (
    SELECT grid_case_id, powerflow_run_id, plz, kcid, bcid, pylovo_grid_result_id, pylovo_version_id,
           count(*) AS lines_total,
           count(*) FILTER (WHERE requires_expansion) AS lines_to_reinforce,
           coalesce(sum(length_km) FILTER (WHERE requires_expansion), 0) AS km_to_reinforce,
           coalesce(sum(additional_parallel), 0) AS additional_cables,
           coalesce(sum(estimated_cost_eur), 0) AS line_cost_eur,
           max(loading_percent) AS max_line_loading_percent
    FROM surrogrid.expansion_line_result
    WHERE expansion_analysis_run_id = :id
    GROUP BY grid_case_id, powerflow_run_id, plz, kcid, bcid, pylovo_grid_result_id, pylovo_version_id
), trafos AS (
    SELECT grid_case_id, powerflow_run_id, plz, kcid, bcid, pylovo_grid_result_id, pylovo_version_id,
           transformer_rated_power_kva, loading_percent AS transformer_loading_percent, required_transformer_kva,
           additional_transformer_kva, requires_expansion AS transformer_requires_expansion,
           estimated_cost_eur AS transformer_cost_eur, transformer_cost_basis, critical_ts AS transformer_critical_ts
    FROM surrogrid.expansion_transformer_result
    WHERE expansion_analysis_run_id = :id
)
SELECT coalesce(t.grid_case_id, l.grid_case_id) AS grid_case_id,
       coalesce(t.powerflow_run_id, l.powerflow_run_id) AS powerflow_run_id,
       coalesce(t.plz, l.plz) AS plz, coalesce(t.kcid, l.kcid) AS kcid, coalesce(t.bcid, l.bcid) AS bcid,
       coalesce(t.pylovo_grid_result_id, l.pylovo_grid_result_id) AS pylovo_grid_result_id,
       coalesce(t.pylovo_version_id, l.pylovo_version_id) AS pylovo_version_id,
       t.transformer_rated_power_kva, t.transformer_loading_percent, t.required_transformer_kva,
       t.additional_transformer_kva, coalesce(t.transformer_requires_expansion, false) AS transformer_requires_expansion,
       coalesce(t.transformer_cost_eur, 0) AS transformer_cost_eur, t.transformer_cost_basis, t.transformer_critical_ts,
       coalesce(l.lines_total, 0) AS lines_total, coalesce(l.lines_to_reinforce, 0) AS lines_to_reinforce,
       coalesce(l.km_to_reinforce, 0) AS km_to_reinforce, coalesce(l.additional_cables, 0) AS additional_cables,
       coalesce(l.line_cost_eur, 0) AS line_cost_eur, l.max_line_loading_percent,
       coalesce(l.line_cost_eur, 0) + coalesce(t.transformer_cost_eur, 0) AS total_cost_eur
FROM trafos t FULL OUTER JOIN lines l ON l.grid_case_id = t.grid_case_id AND l.powerflow_run_id = t.powerflow_run_id
"""


def analysis_grids(analysis_key: str, plz: int | None = None, pylovo_version_id: str | None = None) -> list[dict[str, Any]] | None:
    """Per-grid totals of one analysis (``None`` if the analysis does not exist)."""
    with connect() as conn:
        if not surrogrid_ready(conn):
            return None
        analysis_id = _analysis_id(conn, analysis_key)
        if analysis_id is None:
            return None
        rows = fetch_all(_ANALYSIS_GRIDS_SQL, conn, id=analysis_id)
    rows = [row for row in rows if (plz is None or row["plz"] == plz)
            and (pylovo_version_id is None or row["pylovo_version_id"] == str(pylovo_version_id))]
    return [_clean_row(row) for row in sorted(rows, key=lambda r: (r["plz"], r["kcid"], r["bcid"]))]


def _line_action(row: dict[str, Any]) -> str:
    if not row["requires_expansion"]:
        return "none"
    return "add_2plus" if (row["additional_parallel"] or 0) >= 2 else "add_1"


def _reinforcement_text(row: dict[str, Any]) -> str:
    parts = [f"{row[key]}× NAYY 4×{size}" for key, size in
             (("reinforcement_150_count", 150), ("reinforcement_185_count", 185), ("reinforcement_240_count", 240))
             if row.get(key)]
    return ", ".join(parts)


def analysis_geojson(analysis_key: str, plz: int | None = None, pylovo_version_id: str | None = None) -> dict[str, Any] | None:
    """Cables and transformers of one analysis as two GeoJSON feature collections."""
    filters = ("analysis_key = :key AND geom IS NOT NULL AND (CAST(:plz AS integer) IS NULL OR plz = CAST(:plz AS integer))"
               " AND (CAST(:version AS text) IS NULL OR pylovo_version_id = CAST(:version AS text))")
    with connect() as conn:
        if not surrogrid_ready(conn) or _analysis_id(conn, analysis_key) is None:
            return None
        params = {"key": analysis_key, "plz": plz, "version": pylovo_version_id}
        lines = fetch_all(
            f"""SELECT qgis_id, grid_case_id, plz, kcid, bcid, pylovo_grid_result_id, visible_line_id,
                       visible_line_name, visible_std_type, length_km, loading_percent, max_i_ka, required_parallel,
                       additional_parallel, reinforcement_150_count, reinforcement_185_count, reinforcement_240_count,
                       requires_expansion, overloaded_at_100_percent, estimated_cost_eur, critical_component_cost_basis,
                       critical_ts, ST_AsGeoJSON(ST_Transform(geom, 4326), 6) AS geometry
                FROM surrogrid.expansion_line_qgis_mv WHERE {filters} ORDER BY qgis_id""", conn, **params)
        trafos = fetch_all(
            f"""SELECT qgis_id, grid_case_id, plz, kcid, bcid, pylovo_grid_result_id, transformer_rated_power_kva,
                       transformer_equipment_name, loading_percent, max_s_mva, required_transformer_kva,
                       additional_transformer_kva, requires_expansion, overloaded_at_100_percent, estimated_cost_eur,
                       transformer_cost_basis, critical_ts, osm_id,
                       ST_AsGeoJSON(ST_Transform(geom, 4326), 6) AS geometry
                FROM surrogrid.expansion_transformer_qgis_mv WHERE {filters} ORDER BY qgis_id""", conn, **params)
        bounds = fetch_one(
            f"""SELECT ST_XMin(b) AS minx, ST_YMin(b) AS miny, ST_XMax(b) AS maxx, ST_YMax(b) AS maxy
                FROM (SELECT ST_Transform(ST_SetSRID(ST_Extent(geom)::geometry, 25832), 4326) AS b
                      FROM surrogrid.expansion_line_qgis_mv WHERE {filters}) e""", conn, **params)

    def feature(row: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
        return {"type": "Feature", "id": int(row["qgis_id"]), "geometry": json.loads(row["geometry"]),
                "properties": {key: _clean(value) for key, value in props.items()}}

    line_features = [feature(row, {
        "grid_case_id": row["grid_case_id"], "plz": row["plz"], "kcid": row["kcid"], "bcid": row["bcid"],
        "grid_result_id": row["pylovo_grid_result_id"], "line_id": row["visible_line_id"],
        "name": row["visible_line_name"], "std_type": row["visible_std_type"], "length_km": row["length_km"],
        "loading_percent": row["loading_percent"], "max_i_ka": row["max_i_ka"],
        "required_parallel": row["required_parallel"], "additional_parallel": row["additional_parallel"],
        "reinforcement": _reinforcement_text(row), "requires_expansion": row["requires_expansion"],
        "overloaded": row["overloaded_at_100_percent"], "cost_eur": row["estimated_cost_eur"],
        "cost_basis": row["critical_component_cost_basis"],
        "critical_ts": row["critical_ts"].isoformat() if row["critical_ts"] else None, "action": _line_action(row),
    }) for row in lines]
    trafo_features = [feature(row, {
        "grid_case_id": row["grid_case_id"], "plz": row["plz"], "kcid": row["kcid"], "bcid": row["bcid"],
        "grid_result_id": row["pylovo_grid_result_id"], "rated_kva": row["transformer_rated_power_kva"],
        "equipment": row["transformer_equipment_name"], "loading_percent": row["loading_percent"],
        "max_s_mva": row["max_s_mva"], "required_kva": row["required_transformer_kva"],
        "additional_kva": row["additional_transformer_kva"], "requires_expansion": row["requires_expansion"],
        "overloaded": row["overloaded_at_100_percent"], "cost_eur": row["estimated_cost_eur"],
        "cost_basis": row["transformer_cost_basis"], "osm_id": row["osm_id"],
        "critical_ts": row["critical_ts"].isoformat() if row["critical_ts"] else None,
    }) for row in trafos]
    box = None
    if bounds and bounds["minx"] is not None:
        box = [round(float(bounds[key]), 6) for key in ("minx", "miny", "maxx", "maxy")]
    return {"analysis_key": analysis_key, "bounds": box,
            "lines": {"type": "FeatureCollection", "features": line_features},
            "transformers": {"type": "FeatureCollection", "features": trafo_features}}


# --------------------------------------------------------------------------- power-flow summaries
_POWERFLOW_SQL = """
SELECT ps.powerflow_run_id, ps.stage, pr.run_name, s.scenario_key,
       gc.ags, gc.plz, gc.kcid, gc.bcid, gc.pylovo_grid_result_id, gc.pylovo_version_id,
       ps.n_timesteps, ps.transformer_s_rated_mva, ps.trafo_max_s_mva,
       ps.trafo_loading_p50_time_percent, ps.trafo_loading_p90_time_percent, ps.trafo_loading_p95_time_percent,
       ps.trafo_loading_p99_time_percent, ps.trafo_loading_max_time_percent, ps.trafo_loading_hours_above_100,
       ps.cable_loading_p95_asset_percent, ps.cable_hours_above_100_p95_asset,
       ps.voltage_p05_load_bus_hour_pu, ps.voltage_hours_below_0_90_p95_asset,
       ps.voltage_hours_above_1_03_p95_asset, ps.voltage_hours_above_1_10_p95_asset, ps.created_at
FROM surrogrid.powerflow_summary ps
JOIN surrogrid.powerflow_run pr ON pr.powerflow_run_id = ps.powerflow_run_id
JOIN surrogrid.grid_case gc ON gc.grid_case_id = pr.grid_case_id
JOIN surrogrid.scenario s ON s.scenario_id = pr.scenario_id
WHERE (CAST(:ags AS bigint) IS NULL OR gc.ags = CAST(:ags AS bigint))
  AND (CAST(:plz AS integer) IS NULL OR gc.plz = CAST(:plz AS integer))
  AND (CAST(:version AS text) IS NULL OR gc.pylovo_version_id = CAST(:version AS text))
  AND (CAST(:scenario_key AS text) IS NULL OR s.scenario_key = CAST(:scenario_key AS text))
ORDER BY gc.plz, gc.kcid, gc.bcid, pr.run_name, ps.stage
"""


def powerflow_summaries(ags: int | None = None, plz: int | None = None, pylovo_version_id: str | None = None,
                        scenario_key: str | None = None) -> list[dict[str, Any]]:
    """``powerflow_summary`` rows per grid, run and stage, with the model case of the run."""
    with connect() as conn:
        if not surrogrid_ready(conn):
            return []
        rows = fetch_all(_POWERFLOW_SQL, conn, ags=ags, plz=plz, version=pylovo_version_id,
                         scenario_key=scenario_key)
    return [_clean_row(row) | parse_run_name(row["run_name"]) | split_scenario_key(row["scenario_key"])
            for row in rows]
