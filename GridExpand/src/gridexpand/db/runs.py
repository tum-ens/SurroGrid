"""Scenario, pipeline-run and Step 2/Step 4 run rows.

Creating a run upserts it by ``(grid case, scenario, run_name)`` and deletes
its previous result rows in the same transaction, so a re-run replaces the
old results. Scenario rows keep only scenario-level assumptions; per-grid
facts (selected week, profile hashes, model case) live on the run rows.
"""

from __future__ import annotations

import json
import math
import numbers
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from gridexpand.common.timeframe import build_full_year_metadata, build_initial_metadata
from gridexpand.db.grids import get_or_create_grid_case
from gridexpand.db.schema import ensure_schema

DEFAULT_SCENARIO_KEY = "baseline_static"
DEFAULT_SCENARIO_LABEL = "Baseline static assumptions"
DEFAULT_SCENARIO_DESCRIPTION = (
    "Initial static full-pipeline scenario. Explicit scenario dimensions will be "
    "added once scenario variation is introduced."
)
DEFAULT_SCENARIO_ASSUMPTIONS = {
    "pipeline": "GridExpand",
    "variant": "static",
    **build_full_year_metadata(),
}
SCENARIO_IDENTITY_KEYS = ("scenario_id", "scenario_hash", "scenario_key")

# Result tables of each run type, deleted when the run is re-created.
DEMAND_ALLOCATION_CHILDREN = (
    "allocated_vehicle",
    "electrification_assignment",
    "allocated_eff_factor",
    "demand_component_audit",
    "allocated_demand",
)
POWERFLOW_SUMMARY_TABLES = (
    "powerflow_transformer_diagnostic",
    "powerflow_tail_value",
    "powerflow_cable_summary",
    "powerflow_bus_voltage_summary",
    "powerflow_summary",
)
POWERFLOW_RAW_TABLES = (
    "powerflow_reactive_component",
    "powerflow_line_result",
    "powerflow_bus_voltage",
    "powerflow_import",
    "powerflow_demand",
)
POWERFLOW_CHILDREN = ("powerflow_asset", *POWERFLOW_SUMMARY_TABLES, *POWERFLOW_RAW_TABLES)
REAL_POWERFLOW_CHILDREN = (
    "real_powerflow_tail_value",
    "real_powerflow_cable_summary",
    "real_powerflow_bus_voltage_summary",
    "real_powerflow_summary",
)


def json_safe(value: Any) -> Any:
    """Replace non-finite numbers (JSONB has no NaN/inf) and numpy scalars."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(float(value)) else None
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(json_safe(value), allow_nan=False)


def scenario_level_assumptions(assumptions: dict[str, Any]) -> dict[str, Any]:
    """Keep the values that every grid and model case of one scenario key shares.

    Grid- and case-specific metadata belongs to the run rows; storing it on the
    scenario made the row depend on which run wrote last.
    """
    timeframe_mode = assumptions.get("timeframe_mode", "full_year")
    shared = {**DEFAULT_SCENARIO_ASSUMPTIONS, **build_initial_metadata(timeframe_mode)}
    shared.update({key: assumptions[key] for key in SCENARIO_IDENTITY_KEYS if key in assumptions})
    return shared


def default_pipeline_run_name(scenario_key: str) -> str:
    return f"{scenario_key}_pipeline"


def default_demand_allocation_run_name(scenario_key: str, profiles: str, mobility_source: str) -> str:
    return f"{scenario_key}_{profiles}_{mobility_source}_demand_allocation"


def default_powerflow_run_name(scenario_key: str, pre_only: bool) -> str:
    return f"{scenario_key}_{'pre' if pre_only else 'full'}_powerflow"


def delete_children(conn: Connection, tables: tuple[str, ...], id_column: str, run_id: int) -> None:
    """Delete the rows of one run from each of ``tables``."""
    for table in tables:
        conn.execute(text(f"DELETE FROM surrogrid.{table} WHERE {id_column} = :run_id"), {"run_id": int(run_id)})


def ensure_scenario(
    engine: Engine,
    *,
    scenario_key: str = DEFAULT_SCENARIO_KEY,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
    description: str = DEFAULT_SCENARIO_DESCRIPTION,
    assumptions: dict[str, Any] | None = None,
) -> int:
    """Upsert a scenario row; assumptions (if given) are reduced to scenario-level keys."""
    query = text(
        """
        INSERT INTO surrogrid.scenario (scenario_key, scenario_label, description, assumptions)
        VALUES (:scenario_key, :scenario_label, :description, CAST(:assumptions AS JSONB))
        ON CONFLICT (scenario_key) DO UPDATE SET
            scenario_label = EXCLUDED.scenario_label,
            description = EXCLUDED.description,
            assumptions = CASE
                WHEN :has_assumptions THEN EXCLUDED.assumptions
                ELSE surrogrid.scenario.assumptions
            END,
            updated_at = NOW()
        RETURNING scenario_id
        """
    )
    with engine.begin() as conn:
        return int(
            conn.execute(
                query,
                {
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label,
                    "description": description,
                    "assumptions": json_dumps(scenario_level_assumptions(assumptions or {})),
                    "has_assumptions": assumptions is not None,
                },
            ).scalar_one()
        )


def ensure_pipeline_run(engine: Engine, *, grid_case_id: int, scenario_id: int, run_name: str) -> int:
    """Upsert the pipeline run of one grid case and scenario."""
    query = text(
        """
        INSERT INTO surrogrid.pipeline_run (grid_case_id, scenario_id, run_name)
        VALUES (:grid_case_id, :scenario_id, :run_name)
        ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET updated_at = NOW()
        RETURNING pipeline_run_id
        """
    )
    with engine.begin() as conn:
        return int(
            conn.execute(
                query, {"grid_case_id": int(grid_case_id), "scenario_id": int(scenario_id), "run_name": run_name}
            ).scalar_one()
        )


def _run_parents(
    engine: Engine,
    grid_ref: dict[str, Any],
    *,
    scenario_key: str,
    scenario_label: str,
    assumptions: dict[str, Any] | None,
) -> tuple[int, int, int]:
    grid_case_id = get_or_create_grid_case(engine, grid_ref)
    scenario_id = ensure_scenario(
        engine, scenario_key=scenario_key, scenario_label=scenario_label, assumptions=assumptions
    )
    pipeline_run_id = ensure_pipeline_run(
        engine,
        grid_case_id=grid_case_id,
        scenario_id=scenario_id,
        run_name=default_pipeline_run_name(scenario_key),
    )
    return grid_case_id, scenario_id, pipeline_run_id


def create_demand_allocation_run(
    engine: Engine,
    grid_ref: dict[str, Any],
    *,
    bridge_filename: str,
    profiles: str,
    mobility_source: str,
    scenario_key: str = DEFAULT_SCENARIO_KEY,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
    run_name: str | None = None,
    assumptions: dict[str, Any] | None = None,
) -> int:
    """Create (or reset) one Step 2 run and return its id."""
    grid_case_id, scenario_id, pipeline_run_id = _run_parents(
        engine, grid_ref, scenario_key=scenario_key, scenario_label=scenario_label, assumptions=assumptions
    )
    run_name = run_name or default_demand_allocation_run_name(scenario_key, profiles, mobility_source)
    query = text(
        """
        INSERT INTO surrogrid.demand_allocation_run (
            pipeline_run_id, grid_case_id, scenario_id, run_name, bridge_filename,
            storage_mode, profiles, mobility_source, assumptions
        )
        VALUES (
            :pipeline_run_id, :grid_case_id, :scenario_id, :run_name, :bridge_filename,
            'db', :profiles, :mobility_source, CAST(:assumptions AS JSONB)
        )
        ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            bridge_filename = EXCLUDED.bridge_filename,
            storage_mode = EXCLUDED.storage_mode,
            profiles = EXCLUDED.profiles,
            mobility_source = EXCLUDED.mobility_source,
            assumptions = EXCLUDED.assumptions,
            updated_at = NOW()
        RETURNING demand_allocation_run_id
        """
    )
    with engine.begin() as conn:
        run_id = int(
            conn.execute(
                query,
                {
                    "pipeline_run_id": pipeline_run_id,
                    "grid_case_id": grid_case_id,
                    "scenario_id": scenario_id,
                    "run_name": run_name,
                    "bridge_filename": bridge_filename,
                    "profiles": profiles,
                    "mobility_source": mobility_source,
                    "assumptions": json_dumps(assumptions or {}),
                },
            ).scalar_one()
        )
        delete_children(conn, DEMAND_ALLOCATION_CHILDREN, "demand_allocation_run_id", run_id)
    return run_id


def update_demand_allocation_run_assumptions(engine: Engine, run_id: int, assumptions: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE surrogrid.demand_allocation_run
                SET assumptions = CAST(:assumptions AS JSONB), updated_at = NOW()
                WHERE demand_allocation_run_id = :run_id
                """
            ),
            {"run_id": int(run_id), "assumptions": json_dumps(assumptions)},
        )


def create_powerflow_run(
    engine: Engine,
    grid_ref: dict[str, Any],
    *,
    urbs_input_file: str,
    pre_only: bool,
    scenario_key: str = DEFAULT_SCENARIO_KEY,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
    run_name: str | None = None,
    assumptions: dict[str, Any] | None = None,
) -> int:
    """Create (or reset) one Step 4 run and return its id."""
    grid_case_id, scenario_id, pipeline_run_id = _run_parents(
        engine, grid_ref, scenario_key=scenario_key, scenario_label=scenario_label, assumptions=assumptions
    )
    run_name = run_name or default_powerflow_run_name(scenario_key, pre_only)
    query = text(
        """
        INSERT INTO surrogrid.powerflow_run (
            pipeline_run_id, grid_case_id, scenario_id, run_name,
            urbs_input_file, storage_mode, pre_only, assumptions
        )
        VALUES (
            :pipeline_run_id, :grid_case_id, :scenario_id, :run_name,
            :urbs_input_file, 'db', :pre_only, CAST(:assumptions AS JSONB)
        )
        ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            urbs_input_file = EXCLUDED.urbs_input_file,
            storage_mode = EXCLUDED.storage_mode,
            pre_only = EXCLUDED.pre_only,
            assumptions = EXCLUDED.assumptions,
            updated_at = NOW()
        RETURNING powerflow_run_id
        """
    )
    with engine.begin() as conn:
        run_id = int(
            conn.execute(
                query,
                {
                    "pipeline_run_id": pipeline_run_id,
                    "grid_case_id": grid_case_id,
                    "scenario_id": scenario_id,
                    "run_name": run_name,
                    "urbs_input_file": urbs_input_file,
                    "pre_only": bool(pre_only),
                    "assumptions": json_dumps(assumptions or {}),
                },
            ).scalar_one()
        )
        delete_children(conn, POWERFLOW_CHILDREN, "powerflow_run_id", run_id)
    return run_id


STAGING_MARKER = "#staging-"


def staging_run_name(run_name: str, token: str) -> str:
    """Name under which a Step 4 run is written before it replaces ``run_name``."""
    return f"{run_name}{STAGING_MARKER}{token}"


def promote_powerflow_run(engine: Engine, staging_run_id: int, run_name: str) -> None:
    """Replace the run ``run_name`` of the same grid and scenario by a completed staging run.

    The previous run (and its result rows, via ``ON DELETE CASCADE``) and any leftover
    staging runs of earlier, interrupted attempts are deleted in the same transaction in
    which the staging run takes over the final name, so readers see either the old or
    the new results, never a half-written run.
    """
    with engine.begin() as conn:
        grid_case_id, scenario_id = conn.execute(
            text(
                "SELECT grid_case_id, scenario_id FROM surrogrid.powerflow_run "
                "WHERE powerflow_run_id = :run_id"
            ),
            {"run_id": int(staging_run_id)},
        ).one()
        conn.execute(
            text(
                """
                DELETE FROM surrogrid.powerflow_run
                WHERE grid_case_id = :grid_case_id AND scenario_id = :scenario_id
                  AND powerflow_run_id <> :run_id
                  AND (run_name = :run_name OR starts_with(run_name, :staging_prefix))
                """
            ),
            {
                "grid_case_id": grid_case_id,
                "scenario_id": scenario_id,
                "run_id": int(staging_run_id),
                "run_name": run_name,
                "staging_prefix": f"{run_name}{STAGING_MARKER}",
            },
        )
        conn.execute(
            text(
                "UPDATE surrogrid.powerflow_run SET run_name = :run_name, updated_at = NOW() "
                "WHERE powerflow_run_id = :run_id"
            ),
            {"run_name": run_name, "run_id": int(staging_run_id)},
        )


def discard_powerflow_run(engine: Engine, run_id: int) -> None:
    """Delete one (staging) Step 4 run and its rows."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM surrogrid.powerflow_run WHERE powerflow_run_id = :run_id"),
            {"run_id": int(run_id)},
        )


def get_or_create_real_grid_case(engine: Engine, grid_ref: dict[str, Any]) -> int:
    """Upsert a real (DSO) grid by ``(source, source_file)``."""
    ensure_schema(engine)
    query = text(
        """
        INSERT INTO surrogrid.real_grid_case (
            source, plz, lv_id, variant, category, load_status, status,
            source_file, bus_count, line_count, load_count, assumptions
        )
        VALUES (
            :source, :plz, :lv_id, :variant, :category, :load_status, :status,
            :source_file, :bus_count, :line_count, :load_count, CAST(:assumptions AS JSONB)
        )
        ON CONFLICT (source, source_file) DO UPDATE SET
            plz = EXCLUDED.plz,
            lv_id = EXCLUDED.lv_id,
            variant = EXCLUDED.variant,
            category = EXCLUDED.category,
            load_status = EXCLUDED.load_status,
            status = EXCLUDED.status,
            bus_count = EXCLUDED.bus_count,
            line_count = EXCLUDED.line_count,
            load_count = EXCLUDED.load_count,
            assumptions = EXCLUDED.assumptions,
            updated_at = NOW()
        RETURNING real_grid_case_id
        """
    )
    params = {
        "source": str(grid_ref.get("source", "swf")),
        "plz": grid_ref.get("plz"),
        "lv_id": str(grid_ref["lv_id"]),
        "variant": grid_ref.get("variant"),
        "category": grid_ref.get("category"),
        "load_status": grid_ref.get("load_status"),
        "status": grid_ref.get("status"),
        "source_file": str(grid_ref["source_file"]),
        "bus_count": grid_ref.get("bus_count"),
        "line_count": grid_ref.get("line_count"),
        "load_count": grid_ref.get("load_count"),
        "assumptions": json_dumps(grid_ref.get("assumptions") or {}),
    }
    with engine.begin() as conn:
        return int(conn.execute(query, params).scalar_one())


def create_real_powerflow_run(
    engine: Engine,
    grid_ref: dict[str, Any],
    *,
    run_name: str,
    scenario_key: str = DEFAULT_SCENARIO_KEY,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
    assumptions: dict[str, Any] | None = None,
) -> int:
    """Create (or reset) one real-grid power-flow run and return its id."""
    real_grid_case_id = get_or_create_real_grid_case(engine, grid_ref)
    scenario_id = ensure_scenario(
        engine, scenario_key=scenario_key, scenario_label=scenario_label, assumptions=assumptions
    )
    query = text(
        """
        INSERT INTO surrogrid.real_powerflow_run (
            real_grid_case_id, scenario_id, run_name, storage_mode, pre_only, assumptions
        )
        VALUES (:real_grid_case_id, :scenario_id, :run_name, 'db', TRUE, CAST(:assumptions AS JSONB))
        ON CONFLICT (real_grid_case_id, scenario_id, run_name) DO UPDATE SET
            storage_mode = EXCLUDED.storage_mode,
            pre_only = EXCLUDED.pre_only,
            assumptions = EXCLUDED.assumptions,
            updated_at = NOW()
        RETURNING real_powerflow_run_id
        """
    )
    with engine.begin() as conn:
        run_id = int(
            conn.execute(
                query,
                {
                    "real_grid_case_id": real_grid_case_id,
                    "scenario_id": scenario_id,
                    "run_name": run_name,
                    "assumptions": json_dumps(assumptions or {}),
                },
            ).scalar_one()
        )
        delete_children(conn, REAL_POWERFLOW_CHILDREN, "real_powerflow_run_id", run_id)
    return run_id


def find_powerflow_run(
    engine: Engine,
    *,
    run_name: str | None = None,
    urbs_input_file: str | None = None,
    pre_only: bool | None = None,
    scenario_id: int | None = None,
    grid_case_id: int | None = None,
) -> dict[str, Any] | None:
    """The most recently updated power-flow run matching every given filter.

    Returns:
        ``powerflow_run_id, run_name, pre_only, scenario_id, scenario_key,
        grid_case_id, urbs_input_file, updated_at`` or None.
    """
    query = text(
        """
        SELECT pr.powerflow_run_id, pr.run_name, pr.pre_only, pr.scenario_id, sc.scenario_key,
               pr.grid_case_id, pr.urbs_input_file, pr.updated_at
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.scenario sc USING (scenario_id)
        WHERE (CAST(:run_name AS TEXT) IS NULL OR pr.run_name = :run_name)
          AND (CAST(:urbs_input_file AS TEXT) IS NULL OR pr.urbs_input_file = :urbs_input_file)
          AND (CAST(:pre_only AS BOOLEAN) IS NULL OR pr.pre_only = :pre_only)
          AND (CAST(:scenario_id AS BIGINT) IS NULL OR pr.scenario_id = :scenario_id)
          AND (CAST(:grid_case_id AS BIGINT) IS NULL OR pr.grid_case_id = :grid_case_id)
          AND strpos(pr.run_name, '#staging-') = 0
        ORDER BY pr.updated_at DESC, pr.powerflow_run_id DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "run_name": run_name,
                "urbs_input_file": urbs_input_file,
                "pre_only": pre_only,
                "scenario_id": scenario_id,
                "grid_case_id": grid_case_id,
            },
        ).mappings().first()
    return None if row is None else dict(row)


def list_powerflow_runs(
    engine: Engine,
    *,
    run_name: str | None = None,
    stages: tuple[str, ...] = ("pre", "post"),
    scenario_id: int | None = None,
    ags: int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
) -> pd.DataFrame:
    """Power-flow runs with raw results, one row per run (without raw-table scans).

    Timestep counts come from ``powerflow_import`` (one row per timestep and
    stage), not from the per-bus tables.
    """
    query = text(
        """
        SELECT pr.powerflow_run_id, pr.run_name, pr.pre_only, pr.scenario_id,
               sc.scenario_key, sc.scenario_label, gc.grid_case_id, gc.ags, gc.plz,
               gc.kcid, gc.bcid, gc.cell_id, gc.pylovo_grid_result_id,
               MIN(pi.t_index) AS min_timestep, MAX(pi.t_index) AS max_timestep,
               COUNT(DISTINCT pi.t_index) AS n_timesteps,
               ARRAY_AGG(DISTINCT pi.stage ORDER BY pi.stage) AS stages,
               pr.updated_at
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.powerflow_import pi USING (powerflow_run_id)
        WHERE strpos(pr.run_name, '#staging-') = 0
          AND (CAST(:run_name AS TEXT) IS NULL OR pr.run_name = :run_name)
          AND (CAST(:scenario_id AS BIGINT) IS NULL OR pr.scenario_id = :scenario_id)
          AND (CAST(:ags AS BIGINT) IS NULL OR gc.ags = :ags)
          AND (CAST(:plz AS INTEGER) IS NULL OR gc.plz = :plz)
          AND (CAST(:kcid AS INTEGER) IS NULL OR gc.kcid = :kcid)
          AND (CAST(:bcid AS INTEGER) IS NULL OR gc.bcid = :bcid)
          AND pi.stage = ANY(:stages)
        GROUP BY pr.powerflow_run_id, sc.scenario_key, sc.scenario_label, gc.grid_case_id
        ORDER BY gc.ags, gc.plz, gc.kcid, gc.bcid, pr.run_name, pr.powerflow_run_id
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stages": list(stages),
                "scenario_id": scenario_id,
                "ags": ags,
                "plz": plz,
                "kcid": kcid,
                "bcid": bcid,
            },
        )
