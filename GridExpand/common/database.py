"""Central database access for GridExpand's DB-backed storage mode."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from common.timeframe import build_full_year_metadata


GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = GRIDEXPAND_DIR / ".env"
SCHEMA_SQL_PATH = Path(__file__).with_name("surrogrid_schema.sql")
TIME_INDEX_START = "2009-01-01 00:00:00+00:00"
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


def get_pylovo_version_id() -> str | None:
    value = os.getenv("PYLOVO_VERSION_ID")
    if value is None:
        return None
    value = value.strip().strip('\"').strip("'")
    return value or None


def normalize_ags(value: str | int) -> int:
    """Store AGS as an integer, without a leading zero."""
    return int(str(value).strip().lstrip("0") or "0")


class SurroGridDatabase:
    """PostgreSQL/PostGIS read/write helper for SurroGrid pipeline data."""

    def __init__(self) -> None:
        load_dotenv(ENV_PATH, override=True)

        self.pylovo_version_id = get_pylovo_version_id()

        self.engine = create_engine(
            "postgresql+psycopg2://"
            f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}"
        )

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(916200005)"))
            if self._schema_ready(conn):
                self._ensure_powerflow_summary_columns(conn)
                self._ensure_real_powerflow_schema(conn)
                return
            sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
            statements = [statement.strip() for statement in sql.split(";") if statement.strip()]
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS surrogrid"))
            self._migrate_legacy_scenario_schema(conn)
            for statement in statements:
                conn.execute(text(statement))
            self._ensure_powerflow_summary_columns(conn)
            self._ensure_real_powerflow_schema(conn)

    def _ensure_powerflow_summary_columns(self, conn) -> None:
        column_specs = {
            "powerflow_summary": {
                "n_converged_timesteps": "INTEGER",
                "trafo_mean_s_mva": "DOUBLE PRECISION",
                "trafo_max_s_mva": "DOUBLE PRECISION",
                "trafo_max_p_mw": "DOUBLE PRECISION",
                "trafo_max_q_mvar": "DOUBLE PRECISION",
                "trafo_critical_t_index": "INTEGER",
                "trafo_critical_ts": "TIMESTAMPTZ",
                "n_failed_timesteps": "INTEGER",
                "trafo_loading_p50_time_percent": "DOUBLE PRECISION",
                "trafo_loading_p90_time_percent": "DOUBLE PRECISION",
                "trafo_loading_p99_time_percent": "DOUBLE PRECISION",
                "trafo_loading_max_time_percent": "DOUBLE PRECISION",
                "trafo_loading_hours_above_100": "INTEGER",
                "cable_hours_above_100_p95_asset": "DOUBLE PRECISION",
                "voltage_hours_below_0_90_p95_asset": "DOUBLE PRECISION",
                "voltage_hours_above_1_03_p95_asset": "DOUBLE PRECISION",
                "voltage_hours_above_1_10_p95_asset": "DOUBLE PRECISION",
            },
            "powerflow_cable_summary": {
                "cable_loading_p50_time_percent": "DOUBLE PRECISION",
                "cable_loading_p90_time_percent": "DOUBLE PRECISION",
                "cable_loading_p95_time_percent": "DOUBLE PRECISION",
                "cable_loading_p99_time_percent": "DOUBLE PRECISION",
                "cable_loading_max_t_index": "INTEGER",
                "cable_loading_hours_above_100": "INTEGER",
                "cable_parallel": "DOUBLE PRECISION",
                "cable_installed_capacity_ka": "DOUBLE PRECISION",
            },
            "real_powerflow_cable_summary": {
                "cable_parallel": "DOUBLE PRECISION",
                "cable_installed_capacity_ka": "DOUBLE PRECISION",
            },
            "powerflow_bus_voltage_summary": {
                "voltage_p50_time_pu": "DOUBLE PRECISION",
                "voltage_p10_time_pu": "DOUBLE PRECISION",
                "voltage_p01_time_pu": "DOUBLE PRECISION",
                "voltage_min_time_pu": "DOUBLE PRECISION",
                "voltage_max_time_pu": "DOUBLE PRECISION",
                "voltage_hours_below_0_90": "INTEGER",
                "voltage_hours_above_1_03": "INTEGER",
                "voltage_hours_above_1_10": "INTEGER",
            },
        }
        for table_name, columns in column_specs.items():
            for column_name, column_type in columns.items():
                conn.execute(
                    text(
                        f"ALTER TABLE IF EXISTS surrogrid.{table_name} "
                        f"ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                    )
                )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.powerflow_tail_value (
                    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    asset_id INTEGER NOT NULL,
                    tail TEXT NOT NULL,
                    threshold_value DOUBLE PRECISION,
                    t_index INTEGER NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_powerflow_tail_value UNIQUE (
                        powerflow_run_id, stage, metric, asset_type, asset_id, tail, t_index
                    )
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_powerflow_tail_value_run_stage_metric
                    ON surrogrid.powerflow_tail_value (powerflow_run_id, stage, metric)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_powerflow_tail_value_asset
                    ON surrogrid.powerflow_tail_value (metric, asset_type, asset_id)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.powerflow_transformer_diagnostic (
                    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    diagnostic TEXT NOT NULL,
                    point_index INTEGER NOT NULL,
                    x_value DOUBLE PRECISION NOT NULL,
                    t_index INTEGER,
                    ts TIMESTAMPTZ,
                    p_mw DOUBLE PRECISION,
                    q_mvar DOUBLE PRECISION,
                    q_abs_mvar DOUBLE PRECISION,
                    s_mva DOUBLE PRECISION,
                    mean_s_mva DOUBLE PRECISION,
                    max_s_mva DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_powerflow_transformer_diagnostic UNIQUE (
                        powerflow_run_id, stage, diagnostic, point_index
                    )
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_powerflow_transformer_diagnostic_run_stage
                    ON surrogrid.powerflow_transformer_diagnostic (powerflow_run_id, stage, diagnostic)
                """
            )
        )

    def _ensure_real_powerflow_schema(self, conn) -> None:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_grid_case (
                    real_grid_case_id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    plz INTEGER,
                    lv_id TEXT NOT NULL,
                    variant TEXT,
                    category TEXT,
                    load_status TEXT,
                    status TEXT,
                    source_file TEXT NOT NULL,
                    bus_count INTEGER,
                    line_count INTEGER,
                    load_count INTEGER,
                    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_grid_case_source_file UNIQUE (source, source_file)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_powerflow_run (
                    real_powerflow_run_id BIGSERIAL PRIMARY KEY,
                    real_grid_case_id BIGINT NOT NULL REFERENCES surrogrid.real_grid_case(real_grid_case_id) ON DELETE CASCADE,
                    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
                    run_name TEXT NOT NULL,
                    storage_mode TEXT NOT NULL DEFAULT 'db',
                    pre_only BOOLEAN NOT NULL DEFAULT TRUE,
                    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_powerflow_run_case_scenario_name UNIQUE (real_grid_case_id, scenario_id, run_name)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_powerflow_summary (
                    real_powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.real_powerflow_run(real_powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    n_timesteps INTEGER NOT NULL,
                    n_voltage_buses INTEGER NOT NULL,
                    n_cables INTEGER NOT NULL,
                    n_converged_timesteps INTEGER,
                    n_failed_timesteps INTEGER,
                    transformer_s_rated_mva DOUBLE PRECISION,
                    trafo_loading_p50_time_percent DOUBLE PRECISION,
                    trafo_loading_p90_time_percent DOUBLE PRECISION,
                    trafo_loading_p95_time_percent DOUBLE PRECISION,
                    trafo_loading_p99_time_percent DOUBLE PRECISION,
                    trafo_loading_max_time_percent DOUBLE PRECISION,
                    trafo_loading_hours_above_100 INTEGER,
                    cable_loading_p95_asset_percent DOUBLE PRECISION,
                    cable_hours_above_100_p95_asset DOUBLE PRECISION,
                    voltage_p05_load_bus_hour_pu DOUBLE PRECISION,
                    voltage_hours_below_0_90_p95_asset DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_powerflow_summary_run_stage UNIQUE (real_powerflow_run_id, stage)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_powerflow_cable_summary (
                    real_powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.real_powerflow_run(real_powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    cable INTEGER NOT NULL,
                    cable_loading_p50_time_percent DOUBLE PRECISION,
                    cable_loading_p90_time_percent DOUBLE PRECISION,
                    cable_loading_p95_time_percent DOUBLE PRECISION,
                    cable_loading_p99_time_percent DOUBLE PRECISION,
                    cable_loading_max_time_percent DOUBLE PRECISION,
                    cable_loading_hours_above_100 INTEGER,
                    cable_max_i_ka DOUBLE PRECISION,
                    cable_parallel DOUBLE PRECISION,
                    cable_installed_capacity_ka DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_powerflow_cable_summary_run_stage_cable UNIQUE (real_powerflow_run_id, stage, cable)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_powerflow_bus_voltage_summary (
                    real_powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.real_powerflow_run(real_powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    bus INTEGER NOT NULL,
                    voltage_p50_time_pu DOUBLE PRECISION,
                    voltage_p10_time_pu DOUBLE PRECISION,
                    voltage_p05_time_pu DOUBLE PRECISION,
                    voltage_p01_time_pu DOUBLE PRECISION,
                    voltage_min_time_pu DOUBLE PRECISION,
                    voltage_hours_below_0_90 INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_powerflow_bus_voltage_summary_run_stage_bus UNIQUE (real_powerflow_run_id, stage, bus)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.real_powerflow_tail_value (
                    real_powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.real_powerflow_run(real_powerflow_run_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    asset_id INTEGER NOT NULL,
                    tail TEXT NOT NULL,
                    threshold_value DOUBLE PRECISION,
                    t_index INTEGER NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_real_powerflow_tail_value UNIQUE (
                        real_powerflow_run_id, stage, metric, asset_type, asset_id, tail, t_index
                    )
                )
                """
            )
        )
        for index_sql in (
            "CREATE INDEX IF NOT EXISTS idx_real_grid_case_plz ON surrogrid.real_grid_case (plz)",
            "CREATE INDEX IF NOT EXISTS idx_real_powerflow_run_scenario ON surrogrid.real_powerflow_run (scenario_id)",
            "CREATE INDEX IF NOT EXISTS idx_real_powerflow_summary_run_stage ON surrogrid.real_powerflow_summary (real_powerflow_run_id, stage)",
            "CREATE INDEX IF NOT EXISTS idx_real_powerflow_cable_summary_run_stage ON surrogrid.real_powerflow_cable_summary (real_powerflow_run_id, stage)",
            "CREATE INDEX IF NOT EXISTS idx_real_powerflow_bus_voltage_summary_run_stage ON surrogrid.real_powerflow_bus_voltage_summary (real_powerflow_run_id, stage)",
            "CREATE INDEX IF NOT EXISTS idx_real_powerflow_tail_value_run_stage_metric ON surrogrid.real_powerflow_tail_value (real_powerflow_run_id, stage, metric)",
        ):
            conn.execute(text(index_sql))

        for column_name in ("n_converged_timesteps", "n_failed_timesteps"):
            conn.execute(
                text(
                    f"ALTER TABLE IF EXISTS surrogrid.real_powerflow_summary "
                    f"ADD COLUMN IF NOT EXISTS {column_name} INTEGER"
                )
            )

    def _schema_ready(self, conn=None) -> bool:
        query = text(
            """
            WITH required_columns(table_name, column_name) AS (
                VALUES
                    ('demand_allocation_run', 'assumptions'),
                    ('powerflow_run', 'assumptions'),
                    ('powerflow_summary', 'trafo_mean_s_mva'),
                    ('powerflow_summary', 'trafo_max_s_mva'),
                    ('powerflow_summary', 'trafo_max_p_mw'),
                    ('powerflow_summary', 'trafo_max_q_mvar'),
                    ('powerflow_summary', 'trafo_critical_t_index'),
                    ('powerflow_summary', 'trafo_critical_ts'),
                    ('powerflow_summary', 'trafo_loading_p50_time_percent'),
                    ('powerflow_summary', 'trafo_loading_p90_time_percent'),
                    ('powerflow_summary', 'trafo_loading_p99_time_percent'),
                    ('powerflow_summary', 'trafo_loading_max_time_percent'),
                    ('powerflow_summary', 'trafo_loading_hours_above_100'),
                    ('powerflow_summary', 'cable_hours_above_100_p95_asset'),
                    ('powerflow_summary', 'voltage_hours_below_0_90_p95_asset'),
                    ('powerflow_summary', 'voltage_hours_above_1_03_p95_asset'),
                    ('powerflow_summary', 'voltage_hours_above_1_10_p95_asset'),
                    ('powerflow_cable_summary', 'cable_loading_p50_time_percent'),
                    ('powerflow_cable_summary', 'cable_loading_p90_time_percent'),
                    ('powerflow_cable_summary', 'cable_loading_p95_time_percent'),
                    ('powerflow_cable_summary', 'cable_loading_p99_time_percent'),
                    ('powerflow_cable_summary', 'cable_loading_max_t_index'),
                    ('powerflow_cable_summary', 'cable_loading_hours_above_100'),
                    ('powerflow_cable_summary', 'cable_parallel'),
                    ('powerflow_cable_summary', 'cable_installed_capacity_ka'),
                    ('powerflow_bus_voltage_summary', 'voltage_p50_time_pu'),
                    ('powerflow_bus_voltage_summary', 'voltage_p10_time_pu'),
                    ('powerflow_bus_voltage_summary', 'voltage_p01_time_pu'),
                    ('powerflow_bus_voltage_summary', 'voltage_min_time_pu'),
                    ('powerflow_bus_voltage_summary', 'voltage_max_time_pu'),
                    ('powerflow_bus_voltage_summary', 'voltage_hours_below_0_90'),
                    ('powerflow_bus_voltage_summary', 'voltage_hours_above_1_03'),
                    ('powerflow_bus_voltage_summary', 'voltage_hours_above_1_10')
            )
            SELECT
                to_regclass('surrogrid.grid_case') IS NOT NULL
                AND to_regclass('surrogrid.scenario') IS NOT NULL
                AND to_regclass('surrogrid.pipeline_run') IS NOT NULL
                AND to_regclass('surrogrid.demand_allocation_run') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_run') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_reactive_component') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_summary') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_cable_summary') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_bus_voltage_summary') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_tail_value') IS NOT NULL
                AND to_regclass('surrogrid.powerflow_transformer_diagnostic') IS NOT NULL
                AND to_regclass('surrogrid.real_grid_case') IS NOT NULL
                AND to_regclass('surrogrid.real_powerflow_run') IS NOT NULL
                AND to_regclass('surrogrid.real_powerflow_summary') IS NOT NULL
                AND to_regclass('surrogrid.real_powerflow_cable_summary') IS NOT NULL
                AND to_regclass('surrogrid.real_powerflow_bus_voltage_summary') IS NOT NULL
                AND to_regclass('surrogrid.real_powerflow_tail_value') IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM required_columns rc
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns c
                        WHERE c.table_schema = 'surrogrid'
                          AND c.table_name = rc.table_name
                          AND c.column_name = rc.column_name
                    )
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conrelid = to_regclass('surrogrid.pipeline_run')
                      AND conname = 'fk_pipeline_run_scenario'
                      AND confdeltype = 'c'
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conrelid = to_regclass('surrogrid.demand_allocation_run')
                      AND conname = 'fk_demand_allocation_run_scenario'
                      AND confdeltype = 'c'
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conrelid = to_regclass('surrogrid.powerflow_run')
                      AND conname = 'fk_powerflow_run_scenario'
                      AND confdeltype = 'c'
                ) AS ready
            """
        )
        if conn is not None:
            return bool(conn.execute(query).scalar_one())
        try:
            with self.engine.connect() as check_conn:
                return bool(check_conn.execute(query).scalar_one())
        except Exception:
            return False

    def _migrate_legacy_scenario_schema(self, conn) -> None:
        exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'surrogrid'
                      AND table_name = 'scenario'
                )
                """
            )
        ).scalar_one()
        if not exists:
            return

        scenario_id_type = conn.execute(
            text(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'surrogrid'
                  AND table_name = 'scenario'
                  AND column_name = 'scenario_id'
                """
            )
        ).scalar_one_or_none()
        if scenario_id_type not in {"text", "character varying"}:
            return

        conn.execute(text("ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_scenario"))
        conn.execute(text("DROP INDEX IF EXISTS surrogrid.idx_powerflow_run_scenario"))
        conn.execute(text("DROP INDEX IF EXISTS surrogrid.uq_powerflow_run_grid_scenario_name"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.scenario_int (
                    scenario_id BIGSERIAL PRIMARY KEY,
                    scenario_key TEXT NOT NULL UNIQUE,
                    scenario_label TEXT NOT NULL,
                    description TEXT,
                    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO surrogrid.scenario_int (
                    scenario_key, scenario_label, description,
                    assumptions, created_at, updated_at
                )
                SELECT
                    scenario_id AS scenario_key,
                    scenario_label,
                    description,
                    COALESCE(assumptions, '{}'::JSONB) - 'timeframe_mode' AS assumptions,
                    created_at,
                    updated_at
                FROM surrogrid.scenario
                ON CONFLICT (scenario_key) DO UPDATE SET
                    scenario_label = EXCLUDED.scenario_label,
                    description = EXCLUDED.description,
                    assumptions = EXCLUDED.assumptions,
                    updated_at = NOW()
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO surrogrid.scenario_int (
                    scenario_key, scenario_label, description, assumptions
                )
                VALUES (
                    'baseline_static',
                    'Baseline static assumptions',
                    'Initial static full-pipeline scenario. Explicit scenario dimensions will be added once scenario variation is introduced.',
                    '{"pipeline": "GridExpand", "variant": "static", "timeframe_mode": "full_year", "horizon_hours": 8760, "timeframe_start": "2009-01-01T00:00:00+00:00", "timeframe_end": "2009-12-31T23:00:00+00:00", "source_year_or_reference_year": 2009, "timeframe_kind": "full_year", "methodological_note": "Full 8760-hour reference-year run. Cost and investment outputs keep their existing annual interpretation.", "cost_investment_interpretation": "annual_valid", "annual_valid": true}'::JSONB
                )
                ON CONFLICT (scenario_key) DO NOTHING
                """
            )
        )
        powerflow_run_exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'surrogrid'
                      AND table_name = 'powerflow_run'
                )
                """
            )
        ).scalar_one()
        if powerflow_run_exists:
            conn.execute(text("ALTER TABLE surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS scenario_key TEXT"))
            conn.execute(text("UPDATE surrogrid.powerflow_run SET scenario_key = scenario_id WHERE scenario_key IS NULL"))
            conn.execute(text("ALTER TABLE surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS scenario_id_int BIGINT"))
            conn.execute(
                text(
                    """
                    UPDATE surrogrid.powerflow_run pr
                    SET scenario_id_int = sc.scenario_id
                    FROM surrogrid.scenario_int sc
                    WHERE pr.scenario_id_int IS NULL
                      AND sc.scenario_key = pr.scenario_key
                    """
                )
            )
            conn.execute(
                text(
                    """
                    UPDATE surrogrid.powerflow_run pr
                    SET scenario_id_int = sc.scenario_id
                    FROM surrogrid.scenario_int sc
                    WHERE pr.scenario_id_int IS NULL
                      AND sc.scenario_key = 'baseline_static'
                    """
                )
            )
            conn.execute(text("ALTER TABLE surrogrid.powerflow_run DROP COLUMN scenario_id"))
            conn.execute(text("ALTER TABLE surrogrid.powerflow_run RENAME COLUMN scenario_id_int TO scenario_id"))
            conn.execute(
                text(
                    """
                    UPDATE surrogrid.powerflow_run
                    SET run_name = CASE
                        WHEN pre_only THEN 'baseline_static_pre_powerflow'
                        ELSE 'baseline_static_full_powerflow'
                    END
                    WHERE run_name IN (
                        'baseline_static_pre_powerflow',
                        'baseline_static_full_powerflow',
                        'baseline_static_full_year_pre_powerflow',
                        'baseline_static_full_year_full_powerflow'
                    )
                    """
                )
            )
        conn.execute(text("DROP TABLE surrogrid.scenario"))
        conn.execute(text("ALTER TABLE surrogrid.scenario_int RENAME TO scenario"))

    def resolve_grid_identifier(
        self,
        input_id: str | int,
        *,
        plz: int | None = None,
        kcid: int | None = None,
        bcid: int | None = None,
        candidate_index: int = 0,
        min_buildings: int = 5,
        demand_scope: str = "all",
    ) -> dict[str, Any]:
        """Resolve a CLI identifier to one concrete pylovo grid.

        DB mode accepts either an AGS, such as ``09278140`` or ``9278140``, or a
        generated GridExpand bridge filename like ``9278140-00_94342_1_-1.h5``.
        """
        input_id_str = str(input_id).strip()
        if input_id_str.endswith(".h5"):
            parsed = self.parse_grid_filename(input_id_str)
            return self._grid_ref_from_specs(
                ags=parsed["ags"],
                plz=parsed["plz"],
                kcid=parsed["kcid"],
                bcid=parsed["bcid"],
                candidate_index=parsed["candidate_index"],
                bridge_filename=input_id_str,
            )

        match = re.match(r"^0*(\d+)(?:-(\d+))?$", input_id_str)
        if not match:
            raise ValueError(
                "DB storage expects inputfile_id as AGS, for example 09278140, "
                "or a DB-mode filename like 9278140-00_94342_1_-1.h5."
            )

        ags = normalize_ags(match.group(1))
        if match.group(2) is not None:
            candidate_index = int(match.group(2))

        if (plz is None) != (kcid is None) or (plz is None) != (bcid is None):
            raise ValueError("Provide --plz, --kcid, and --bcid together, or omit all three.")

        if plz is not None and kcid is not None and bcid is not None:
            return self._grid_ref_from_specs(
                ags=ags,
                plz=int(plz),
                kcid=int(kcid),
                bcid=int(bcid),
                candidate_index=candidate_index,
            )

        query = text(
            """
            WITH ags_plz AS (
                SELECT DISTINCT plz
                FROM pylovo.municipal_register
                WHERE ags = :ags
            ),
            building_counts AS (
                SELECT
                    grid_result_id,
                    version_id,
                    COUNT(*) AS n_buildings,
                    COUNT(*) FILTER (
                        WHERE
                            CASE
                                WHEN UPPER(COALESCE(TRIM(building_type), TRIM(type), '')) IN ('AB', 'MFH', 'TH', 'SFH')
                                THEN 'Residential'
                                WHEN LOWER(COALESCE(TRIM(building_use), TRIM(type), '')) LIKE '%%public%%'
                                THEN 'Public'
                                WHEN LOWER(COALESCE(TRIM(building_use), TRIM(type), '')) LIKE '%%commercial%%'
                                THEN 'Commercial'
                                ELSE 'Commercial'
                            END = 'Residential'
                    ) AS n_residential_buildings
                FROM pylovo.buildings_result
                GROUP BY grid_result_id, version_id
            ),
            latest AS (
                SELECT DISTINCT ON (gr.plz, gr.kcid, gr.bcid)
                    gr.grid_result_id,
                    gr.version_id,
                    gr.plz,
                    gr.kcid,
                    gr.bcid,
                    bc.n_buildings,
                    bc.n_residential_buildings
                FROM pylovo.grid_result gr
                JOIN ags_plz ap ON ap.plz = gr.plz
                JOIN building_counts bc
                  ON bc.grid_result_id = gr.grid_result_id
                 AND bc.version_id = gr.version_id
                WHERE
                  CASE
                    WHEN :demand_scope = 'residential' THEN bc.n_residential_buildings
                    ELSE bc.n_buildings
                  END >= :min_buildings
                  AND (:pylovo_version_id IS NULL OR gr.version_id::text = :pylovo_version_id)
                ORDER BY gr.plz, gr.kcid, gr.bcid, gr.version_id DESC
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY plz, kcid, bcid) - 1 AS candidate_index
                FROM latest
            )
            SELECT *
            FROM numbered
            WHERE candidate_index = :candidate_index
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                query,
                {
                    "ags": ags,
                    "candidate_index": int(candidate_index),
                    "min_buildings": int(min_buildings),
                    "demand_scope": demand_scope,
                    "pylovo_version_id": self.pylovo_version_id,
                },
            ).mappings().first()

        if row is None:
            raise ValueError(
                f"No pylovo grid candidate found for AGS={ags}, candidate_index={candidate_index}."
            )

        return self._format_grid_ref(ags=ags, row=dict(row), bridge_filename=None)

    def parse_grid_filename(self, filename: str) -> dict[str, Any]:
        stem = Path(filename).name.removesuffix(".h5")
        parts = stem.split("_")
        if len(parts) < 4:
            raise ValueError(
                "DB storage filenames must follow <cell_id>_<plz>_<kcid>_<bcid>.h5."
            )

        cell_id = parts[0]
        ags_match = re.match(r"^0*(\d+)(?:-(\d+))?$", cell_id)
        if ags_match is None:
            raise ValueError("DB storage filenames must begin with an AGS-based cell_id.")

        return {
            "ags": normalize_ags(ags_match.group(1)),
            "candidate_index": int(ags_match.group(2) or 0),
            "cell_id": cell_id,
            "plz": int(parts[1]),
            "kcid": int(parts[2]),
            "bcid": int(parts[3]),
        }

    def _grid_ref_from_specs(
        self,
        *,
        ags: int,
        plz: int,
        kcid: int,
        bcid: int,
        candidate_index: int,
        bridge_filename: str | None = None,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT grid_result_id, version_id, plz, kcid, bcid
            FROM pylovo.grid_result
            WHERE plz = :plz AND kcid = :kcid AND bcid = :bcid
              AND (:pylovo_version_id IS NULL OR version_id::text = :pylovo_version_id)
            ORDER BY version_id DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                query,
                {
                    "plz": int(plz),
                    "kcid": int(kcid),
                    "bcid": int(bcid),
                    "pylovo_version_id": self.pylovo_version_id,
                },
            ).mappings().first()

        if row is None:
            version_hint = f" and PYLOVO_VERSION_ID={self.pylovo_version_id}" if self.pylovo_version_id else ""
            raise ValueError(f"No pylovo grid found for PLZ={plz}, KCID={kcid}, BCID={bcid}{version_hint}.")

        return self._format_grid_ref(
            ags=ags,
            row=dict(row),
            candidate_index=candidate_index,
            bridge_filename=bridge_filename,
        )

    def _format_grid_ref(
        self,
        *,
        ags: int,
        row: dict[str, Any],
        candidate_index: int | None = None,
        bridge_filename: str | None = None,
    ) -> dict[str, Any]:
        if candidate_index is None:
            candidate_index = int(row.get("candidate_index", 0))
        cell_id = f"{int(ags)}-{int(candidate_index):02d}"
        bridge_filename = bridge_filename or f"{cell_id}_{int(row['plz'])}_{int(row['kcid'])}_{int(row['bcid'])}.h5"
        return {
            "ags": int(ags),
            "candidate_index": int(candidate_index),
            "cell_id": cell_id,
            "bridge_filename": bridge_filename,
            "grid_result_id": int(row["grid_result_id"]),
            "version_id": str(row["version_id"]),
            "plz": int(row["plz"]),
            "kcid": int(row["kcid"]),
            "bcid": int(row["bcid"]),
        }

    def get_or_create_grid_case(self, grid_ref: dict[str, Any]) -> int:
        self.ensure_schema()
        query = text(
            """
            INSERT INTO surrogrid.grid_case (
                ags, plz, kcid, bcid, pylovo_grid_result_id,
                pylovo_version_id, cell_id
            )
            VALUES (
                :ags, :plz, :kcid, :bcid, :grid_result_id,
                :version_id, :cell_id
            )
            ON CONFLICT (ags, plz, kcid, bcid, pylovo_grid_result_id)
            DO UPDATE SET
                pylovo_version_id = EXCLUDED.pylovo_version_id,
                cell_id = EXCLUDED.cell_id
            RETURNING grid_case_id
            """
        )
        with self.engine.begin() as conn:
            return int(conn.execute(query, grid_ref).scalar_one())

    def read_step2_input_data(
        self, grid_ref: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        self.get_or_create_grid_case(grid_ref)
        df_buildings = self.read_buildings(grid_ref)
        df_region = self.read_region(grid_ref)
        return df_buildings, df_region, None

    def read_region(self, grid_ref: dict[str, Any]) -> pd.DataFrame:
        query = text(
            """
            WITH selected_grid AS (
                SELECT grid_result_id
                FROM pylovo.grid_result
                WHERE grid_result_id = :grid_result_id
            ),
            trafo AS (
                SELECT
                    ST_Y(ST_Transform(tp.geom, 4326)) AS lat,
                    ST_X(ST_Transform(tp.geom, 4326)) AS lon
                FROM pylovo.transformer_positions tp
                JOIN selected_grid sg ON sg.grid_result_id = tp.grid_result_id
                LIMIT 1
            )
            SELECT
                mr.plz,
                mr.pop,
                mr.area,
                mr.name_city,
                mr.pop_den,
                mr.regio7,
                COALESCE((SELECT lat FROM trafo), mr.lat) AS lat,
                COALESCE((SELECT lon FROM trafo), mr.lon) AS lon,
                :kcid AS kcid,
                :bcid AS bcid,
                :ags AS ags
            FROM pylovo.municipal_register mr
            WHERE mr.ags = :ags AND mr.plz = :plz
            LIMIT 1
            """
        )
        with self.engine.connect() as conn:
            df_region = pd.read_sql_query(
                query,
                conn,
                params={
                    "grid_result_id": grid_ref["grid_result_id"],
                    "ags": grid_ref["ags"],
                    "plz": grid_ref["plz"],
                    "kcid": grid_ref["kcid"],
                    "bcid": grid_ref["bcid"],
                },
            )

        if df_region.empty:
            raise ValueError(
                f"No municipal_register row found for AGS={grid_ref['ags']} and PLZ={grid_ref['plz']}."
            )
        return df_region

    def read_buildings(self, grid_ref: dict[str, Any]) -> pd.DataFrame:
        query = text(
            """
            SELECT
                b.objectid,
                b.id,
                b.feature_id,
                b.vertice_id,
                b.height,
                b.floor_area,
                b.floor_number,
                CASE
                    WHEN UPPER(COALESCE(TRIM(b.building_type), TRIM(b.type), '')) IN ('AB', 'MFH', 'TH', 'SFH')
                    THEN UPPER(COALESCE(TRIM(b.building_type), TRIM(b.type)))
                    WHEN LOWER(COALESCE(TRIM(b.building_use), TRIM(b.type), '')) LIKE '%%public%%'
                    THEN 'public'
                    WHEN LOWER(COALESCE(TRIM(b.building_use), TRIM(b.type), '')) LIKE '%%commercial%%'
                    THEN 'commercial'
                    ELSE 'commercial'
                END AS building_type,
                CASE
                    WHEN UPPER(COALESCE(TRIM(b.building_type), TRIM(b.type), '')) IN ('AB', 'MFH', 'TH', 'SFH')
                    THEN 'Residential'
                    WHEN LOWER(COALESCE(TRIM(b.building_use), TRIM(b.type), '')) LIKE '%%public%%'
                    THEN 'Public'
                    WHEN LOWER(COALESCE(TRIM(b.building_use), TRIM(b.type), '')) LIKE '%%commercial%%'
                    THEN 'Commercial'
                    ELSE 'Commercial'
                END AS building_use,
                b.occupants,
                b.households,
                CAST(b.construction_year AS VARCHAR) AS construction_year,
                b.postcode,
                b.address_street_id,
                b.street,
                b.house_number,
                b.gemeindeschluessel,
                b.assigned_way_id,
                b.peak_load_in_kw,
                b.connection_point,
                ST_Y(ST_Transform(b.centroid, 4326)) AS lat,
                ST_X(ST_Transform(b.centroid, 4326)) AS lon
            FROM pylovo.buildings_result b
            WHERE b.grid_result_id = :grid_result_id
              AND b.version_id = :version_id
            """
        )
        with self.engine.connect() as conn:
            df_buildings = pd.read_sql_query(
                query,
                conn,
                params={
                    "grid_result_id": grid_ref["grid_result_id"],
                    "version_id": grid_ref["version_id"],
                },
            )
            df_bus = pd.read_sql_query(
                text(
                    """
                    SELECT pp_index AS bus, name
                    FROM pylovo.pandapower_bus
                    WHERE grid_result_id = :grid_result_id
                    """
                ),
                conn,
                params={"grid_result_id": grid_ref["grid_result_id"]},
            )

        if df_buildings.empty:
            raise ValueError(
                f"No buildings found for pylovo grid_result_id={grid_ref['grid_result_id']}."
            )

        if not df_bus.empty:
            df_id = pd.DataFrame()
            df_id["vertice_id"] = (
                df_bus["name"].astype(str).str.extract(r"^Consumer Nodebus (\d+)$")[0]
            )
            df_id = df_id.dropna()
            df_id["vertice_id"] = df_id["vertice_id"].astype(int)
            df_id["bus"] = df_bus.loc[df_id.index, "bus"].astype(int)
            df_buildings = df_buildings.merge(df_id, on="vertice_id", how="left")
        else:
            df_buildings["bus"] = pd.NA

        if "connection_point" in df_buildings.columns:
            df_buildings["bus"] = df_buildings["bus"].fillna(df_buildings["connection_point"])
            df_buildings.drop(columns=["connection_point"], inplace=True)

        if {"occupants", "households", "building_use"}.issubset(df_buildings.columns):
            residential_mask = df_buildings["building_use"].eq("Residential")
            fallback_occ = df_buildings.loc[residential_mask, "households"].fillna(1)
            fallback_occ = fallback_occ.clip(lower=1) * 2
            df_buildings.loc[residential_mask, "occupants"] = (
                df_buildings.loc[residential_mask, "occupants"].fillna(fallback_occ)
            )

        cols = df_buildings.columns.tolist()
        cols.insert(0, cols.pop(cols.index("bus")))
        df_buildings = df_buildings[cols]
        return df_buildings.sort_values(by="bus").reset_index(drop=True)

    def read_pandapower_grid(self, grid_ref: dict[str, Any]):
        import pandapower as pp

        query = text(
            """
            SELECT grid
            FROM pylovo.grid_result
            WHERE grid_result_id = :grid_result_id
            """
        )
        with self.engine.connect() as conn:
            payload = conn.execute(
                query,
                {"grid_result_id": grid_ref["grid_result_id"]},
            ).scalar_one()

        grid_json = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
        return pp.from_json_string(grid_json)

    def ensure_scenario(
        self,
        *,
        scenario_key: str = DEFAULT_SCENARIO_KEY,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
        description: str = DEFAULT_SCENARIO_DESCRIPTION,
        assumptions: dict[str, Any] | None = None,
    ) -> int:
        scenario_assumptions = dict(DEFAULT_SCENARIO_ASSUMPTIONS)
        if assumptions:
            scenario_assumptions.update(assumptions)
        query = text(
            """
            INSERT INTO surrogrid.scenario (
                scenario_key, scenario_label, description, assumptions
            )
            VALUES (
                :scenario_key, :scenario_label,
                :description, CAST(:assumptions AS JSONB)
            )
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
        with self.engine.begin() as conn:
            return int(
                conn.execute(
                    query,
                    {
                        "scenario_key": scenario_key,
                        "scenario_label": scenario_label,
                        "description": description,
                        "assumptions": json.dumps(scenario_assumptions),
                        "has_assumptions": assumptions is not None,
                    },
                ).scalar_one()
            )

    def count_scenario_data(self, scenario_key: str) -> dict[str, int]:
        """Count SurroGrid rows related to one scenario key."""
        self.ensure_schema()
        with self.engine.connect() as conn:
            scenario_id = self._scenario_id_for_key(conn, scenario_key)
            if scenario_id is None:
                raise ValueError(f"No surrogrid.scenario row found for scenario_key={scenario_key!r}.")
            return self._count_scenario_data(conn, scenario_id)

    def delete_scenario_data(
        self,
        scenario_key: str,
        *,
        keep_demands: bool = False,
        dry_run: bool = True,
        refresh_expansion_views: bool = True,
    ) -> dict[str, int]:
        """Delete DB-backed SurroGrid data for a scenario key.

        By default the scenario row is deleted, cascading all dependent
        SurroGrid data. Set ``keep_demands=True`` to keep the scenario,
        pipeline, and Step 2 demand-allocation rows while deleting downstream
        power-flow and expansion results.
        """
        self.ensure_schema()
        with self.engine.begin() as conn:
            scenario_id = self._scenario_id_for_key(conn, scenario_key)
            if scenario_id is None:
                raise ValueError(f"No surrogrid.scenario row found for scenario_key={scenario_key!r}.")
            counts = self._count_scenario_data(conn, scenario_id)
            if dry_run:
                return counts

            self._delete_expansion_scenario_data(conn, scenario_id)
            if keep_demands:
                if self._relation_exists(conn, "surrogrid.powerflow_run"):
                    conn.execute(
                        text("DELETE FROM surrogrid.powerflow_run WHERE scenario_id = :scenario_id"),
                        {"scenario_id": scenario_id},
                    )
            else:
                conn.execute(
                    text("DELETE FROM surrogrid.scenario WHERE scenario_id = :scenario_id"),
                    {"scenario_id": scenario_id},
                )

        if refresh_expansion_views:
            self.refresh_expansion_materialized_views()
        return counts

    def refresh_expansion_materialized_views(self) -> None:
        """Refresh optional Step 5 QGIS materialized views if they exist."""
        with self.engine.begin() as conn:
            for view_name in (
                "surrogrid.expansion_line_qgis_mv",
                "surrogrid.expansion_transformer_qgis_mv",
            ):
                if self._relation_exists(conn, view_name):
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))

    def _scenario_id_for_key(self, conn, scenario_key: str) -> int | None:
        value = conn.execute(
            text("SELECT scenario_id FROM surrogrid.scenario WHERE scenario_key = :scenario_key"),
            {"scenario_key": scenario_key},
        ).scalar_one_or_none()
        return None if value is None else int(value)

    def _relation_exists(self, conn, relation_name: str) -> bool:
        return bool(
            conn.execute(
                text("SELECT to_regclass(:relation_name) IS NOT NULL"),
                {"relation_name": relation_name},
            ).scalar_one()
        )

    def _delete_expansion_scenario_data(self, conn, scenario_id: int) -> None:
        for table_name in ("expansion_line_result", "expansion_transformer_result", "expansion_analysis_run"):
            relation_name = f"surrogrid.{table_name}"
            if self._relation_exists(conn, relation_name):
                conn.execute(
                    text(f"DELETE FROM {relation_name} WHERE scenario_id = :scenario_id"),
                    {"scenario_id": scenario_id},
                )

    def _count_scenario_data(self, conn, scenario_id: int) -> dict[str, int]:
        counts = {
            "scenario": self._count_query(
                conn,
                "surrogrid.scenario",
                "SELECT COUNT(*) FROM surrogrid.scenario WHERE scenario_id = :scenario_id",
                scenario_id,
            )
        }
        count_specs = (
            ("pipeline_run", "surrogrid.pipeline_run", "SELECT COUNT(*) FROM surrogrid.pipeline_run WHERE scenario_id = :scenario_id"),
            ("demand_allocation_run", "surrogrid.demand_allocation_run", "SELECT COUNT(*) FROM surrogrid.demand_allocation_run WHERE scenario_id = :scenario_id"),
            ("allocated_demand", "surrogrid.allocated_demand", """
                SELECT COUNT(*)
                FROM surrogrid.allocated_demand child
                JOIN surrogrid.demand_allocation_run parent USING (demand_allocation_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("allocated_eff_factor", "surrogrid.allocated_eff_factor", """
                SELECT COUNT(*)
                FROM surrogrid.allocated_eff_factor child
                JOIN surrogrid.demand_allocation_run parent USING (demand_allocation_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("allocated_vehicle", "surrogrid.allocated_vehicle", """
                SELECT COUNT(*)
                FROM surrogrid.allocated_vehicle child
                JOIN surrogrid.demand_allocation_run parent USING (demand_allocation_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("powerflow_run", "surrogrid.powerflow_run", "SELECT COUNT(*) FROM surrogrid.powerflow_run WHERE scenario_id = :scenario_id"),
            ("powerflow_demand", "surrogrid.powerflow_demand", """
                SELECT COUNT(*)
                FROM surrogrid.powerflow_demand child
                JOIN surrogrid.powerflow_run parent USING (powerflow_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("powerflow_import", "surrogrid.powerflow_import", """
                SELECT COUNT(*)
                FROM surrogrid.powerflow_import child
                JOIN surrogrid.powerflow_run parent USING (powerflow_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("powerflow_bus_voltage", "surrogrid.powerflow_bus_voltage", """
                SELECT COUNT(*)
                FROM surrogrid.powerflow_bus_voltage child
                JOIN surrogrid.powerflow_run parent USING (powerflow_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("powerflow_line_result", "surrogrid.powerflow_line_result", """
                SELECT COUNT(*)
                FROM surrogrid.powerflow_line_result child
                JOIN surrogrid.powerflow_run parent USING (powerflow_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("powerflow_reactive_component", "surrogrid.powerflow_reactive_component", """
                SELECT COUNT(*)
                FROM surrogrid.powerflow_reactive_component child
                JOIN surrogrid.powerflow_run parent USING (powerflow_run_id)
                WHERE parent.scenario_id = :scenario_id
            """),
            ("expansion_analysis_run", "surrogrid.expansion_analysis_run", "SELECT COUNT(*) FROM surrogrid.expansion_analysis_run WHERE scenario_id = :scenario_id"),
            ("expansion_line_result", "surrogrid.expansion_line_result", "SELECT COUNT(*) FROM surrogrid.expansion_line_result WHERE scenario_id = :scenario_id"),
            ("expansion_transformer_result", "surrogrid.expansion_transformer_result", "SELECT COUNT(*) FROM surrogrid.expansion_transformer_result WHERE scenario_id = :scenario_id"),
            ("expansion_line_qgis_mv", "surrogrid.expansion_line_qgis_mv", "SELECT COUNT(*) FROM surrogrid.expansion_line_qgis_mv WHERE scenario_id = :scenario_id"),
            ("expansion_transformer_qgis_mv", "surrogrid.expansion_transformer_qgis_mv", "SELECT COUNT(*) FROM surrogrid.expansion_transformer_qgis_mv WHERE scenario_id = :scenario_id"),
        )
        for key, relation_name, query in count_specs:
            counts[key] = self._count_query(conn, relation_name, query, scenario_id)
        return counts

    def _count_query(self, conn, relation_name: str, query: str, scenario_id: int) -> int:
        if not self._relation_exists(conn, relation_name):
            return 0
        return int(conn.execute(text(query), {"scenario_id": scenario_id}).scalar_one())

    def ensure_pipeline_run(
        self,
        *,
        grid_case_id: int,
        scenario_id: int,
        run_name: str,
    ) -> int:
        query = text(
            """
            INSERT INTO surrogrid.pipeline_run (
                grid_case_id, scenario_id, run_name
            )
            VALUES (
                :grid_case_id, :scenario_id, :run_name
            )
            ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET
                updated_at = NOW()
            RETURNING pipeline_run_id
            """
        )
        with self.engine.begin() as conn:
            return int(
                conn.execute(
                    query,
                    {
                        "grid_case_id": int(grid_case_id),
                        "scenario_id": int(scenario_id),
                        "run_name": run_name,
                    },
                ).scalar_one()
            )

    def default_pipeline_run_name(self, scenario_key: str) -> str:
        return f"{scenario_key}_pipeline"

    def create_demand_allocation_run(
        self,
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
        grid_case_id = self.get_or_create_grid_case(grid_ref)
        scenario_id = self.ensure_scenario(
            scenario_key=scenario_key,
            scenario_label=scenario_label,
            assumptions=assumptions,
        )
        pipeline_run_id = self.ensure_pipeline_run(
            grid_case_id=grid_case_id,
            scenario_id=scenario_id,
            run_name=self.default_pipeline_run_name(scenario_key),
        )
        run_name = run_name or self.default_demand_allocation_run_name(
            scenario_key,
            profiles,
            mobility_source,
        )
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
        with self.engine.begin() as conn:
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
                        "assumptions": json.dumps(assumptions or {}),
                    },
                ).scalar_one()
            )
            self._clear_demand_allocation_run(conn, run_id)
        return run_id

    def update_demand_allocation_run_assumptions(self, run_id: int, assumptions: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE surrogrid.demand_allocation_run
                    SET assumptions = CAST(:assumptions AS JSONB),
                        updated_at = NOW()
                    WHERE demand_allocation_run_id = :run_id
                    """
                ),
                {"run_id": int(run_id), "assumptions": json.dumps(assumptions)},
            )

    def default_demand_allocation_run_name(
        self,
        scenario_key: str,
        profiles: str,
        mobility_source: str,
    ) -> str:
        return f"{scenario_key}_{profiles}_{mobility_source}_demand_allocation"

    def _clear_demand_allocation_run(self, conn, run_id: int) -> None:
        for table_name in (
            "allocated_vehicle",
            "allocated_eff_factor",
            "allocated_demand",
        ):
            conn.execute(
                text(
                    f"DELETE FROM surrogrid.{table_name} "
                    "WHERE demand_allocation_run_id = :run_id"
                ),
                {"run_id": run_id},
            )

    def write_allocated_demand(self, run_id: int, df: pd.DataFrame) -> None:
        self._write_allocated_timeseries(run_id, df, "commodity", "allocated_demand")

    def write_allocated_eff_factor(self, run_id: int, df: pd.DataFrame) -> None:
        self._write_allocated_timeseries(run_id, df, "component", "allocated_eff_factor")

    def write_allocated_vehicles(self, run_id: int, df_buildings: pd.DataFrame, battery_dict: dict | None = None) -> None:
        battery_dict = battery_dict or {}
        rows = []
        for car_dict in df_buildings.get("car_dict", []):
            if not isinstance(car_dict, dict):
                continue
            for (bus, vehicle_id), cfg in car_dict.items():
                rows.append(
                    {
                        "demand_allocation_run_id": run_id,
                        "bus": int(bus),
                        "vehicle_id": int(vehicle_id),
                        "model": str(cfg["model"]),
                        "schedule": str(cfg["schedule"]),
                        "seed": int(cfg["seed"]),
                        "profile_id": cfg.get("profile_id"),
                        "battery_cap_kwh": cfg.get("battery_cap_kwh", battery_dict.get((int(bus), int(vehicle_id)))),
                    }
                )
        if rows:
            self._append(pd.DataFrame(rows), "allocated_vehicle")

    def _write_allocated_timeseries(
        self,
        run_id: int,
        df: pd.DataFrame,
        label_column: str,
        table_name: str,
    ) -> None:
        if df.empty:
            return
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
            raise ValueError(f"Expected MultiIndex columns for {table_name}.")

        out = df.copy().reset_index(drop=True)
        out.columns = pd.MultiIndex.from_tuples(
            [(int(col[0]), str(col[1])) for col in out.columns.to_flat_index()],
            names=["bus", label_column],
        )
        out.index = pd.RangeIndex(len(out), name="t_index")
        out = out.stack(["bus", label_column], future_stack=True).rename("value").reset_index()
        out.insert(1, "ts", self._timestamps_for_demand_allocation_run(run_id, len(df)).take(out["t_index"].to_numpy()))
        out.insert(0, "demand_allocation_run_id", run_id)
        out = out.dropna(subset=["value"])
        self._append(out, table_name)

    def get_or_create_real_grid_case(self, grid_ref: dict[str, Any]) -> int:
        self.ensure_schema()
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
            "assumptions": json.dumps(grid_ref.get("assumptions") or {}),
        }
        with self.engine.begin() as conn:
            return int(conn.execute(query, params).scalar_one())

    def create_real_powerflow_run(
        self,
        grid_ref: dict[str, Any],
        *,
        run_name: str,
        scenario_key: str = DEFAULT_SCENARIO_KEY,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
        assumptions: dict[str, Any] | None = None,
    ) -> int:
        real_grid_case_id = self.get_or_create_real_grid_case(grid_ref)
        scenario_id = self.ensure_scenario(
            scenario_key=scenario_key,
            scenario_label=scenario_label,
            assumptions=assumptions,
        )
        query = text(
            """
            INSERT INTO surrogrid.real_powerflow_run (
                real_grid_case_id, scenario_id, run_name, storage_mode, pre_only, assumptions
            )
            VALUES (
                :real_grid_case_id, :scenario_id, :run_name, 'db', TRUE, CAST(:assumptions AS JSONB)
            )
            ON CONFLICT (real_grid_case_id, scenario_id, run_name) DO UPDATE SET
                storage_mode = EXCLUDED.storage_mode,
                pre_only = EXCLUDED.pre_only,
                assumptions = EXCLUDED.assumptions,
                updated_at = NOW()
            RETURNING real_powerflow_run_id
            """
        )
        with self.engine.begin() as conn:
            run_id = int(
                conn.execute(
                    query,
                    {
                        "real_grid_case_id": real_grid_case_id,
                        "scenario_id": scenario_id,
                        "run_name": run_name,
                        "assumptions": json.dumps(assumptions or {}),
                    },
                ).scalar_one()
            )
            self._clear_real_powerflow_run(conn, run_id)
        return run_id

    def _clear_real_powerflow_run(self, conn, run_id: int) -> None:
        for table_name in (
            "real_powerflow_tail_value",
            "real_powerflow_cable_summary",
            "real_powerflow_bus_voltage_summary",
            "real_powerflow_summary",
        ):
            conn.execute(
                text(f"DELETE FROM surrogrid.{table_name} WHERE real_powerflow_run_id = :run_id"),
                {"run_id": run_id},
            )

    def write_real_powerflow_summary(self, run_id: int, stage: str, summary: dict[str, Any]) -> None:
        grid_summary = summary.get("grid_summary", summary)
        out = pd.DataFrame(
            [
                {
                    "real_powerflow_run_id": int(run_id),
                    "stage": stage,
                    "n_timesteps": int(grid_summary.get("n_timesteps", 0)),
                    "n_converged_timesteps": grid_summary.get("n_converged_timesteps"),
                    "n_failed_timesteps": grid_summary.get("n_failed_timesteps"),
                    "n_voltage_buses": int(grid_summary.get("n_voltage_buses", 0)),
                    "n_cables": int(grid_summary.get("n_cables", 0)),
                    "transformer_s_rated_mva": grid_summary.get("transformer_s_rated_mva"),
                    "trafo_loading_p50_time_percent": grid_summary.get("trafo_loading_p50_time_percent"),
                    "trafo_loading_p90_time_percent": grid_summary.get("trafo_loading_p90_time_percent"),
                    "trafo_loading_p95_time_percent": grid_summary.get("trafo_loading_p95_time_percent"),
                    "trafo_loading_p99_time_percent": grid_summary.get("trafo_loading_p99_time_percent"),
                    "trafo_loading_max_time_percent": grid_summary.get("trafo_loading_max_time_percent"),
                    "trafo_loading_hours_above_100": grid_summary.get("trafo_loading_hours_above_100"),
                    "cable_loading_p95_asset_percent": grid_summary.get("cable_loading_p95_asset_percent"),
                    "cable_hours_above_100_p95_asset": grid_summary.get("cable_hours_above_100_p95_asset"),
                    "voltage_p05_load_bus_hour_pu": grid_summary.get("voltage_p05_load_bus_hour_pu"),
                    "voltage_hours_below_0_90_p95_asset": grid_summary.get("voltage_hours_below_0_90_p95_asset"),
                }
            ]
        )
        self._append(out, "real_powerflow_summary")

        cable_summary = summary.get("cable_summary")
        if isinstance(cable_summary, pd.DataFrame) and not cable_summary.empty:
            cable_out = cable_summary.copy()
            real_cable_columns = [
                "cable",
                "cable_loading_p50_time_percent",
                "cable_loading_p90_time_percent",
                "cable_loading_p95_time_percent",
                "cable_loading_p99_time_percent",
                "cable_loading_max_time_percent",
                "cable_loading_hours_above_100",
                "cable_max_i_ka",
                "cable_parallel",
                "cable_installed_capacity_ka",
            ]
            cable_out = cable_out[[col for col in real_cable_columns if col in cable_out.columns]]
            cable_out.insert(0, "stage", stage)
            cable_out.insert(0, "real_powerflow_run_id", int(run_id))
            cable_out["cable"] = cable_out["cable"].astype(int)
            self._append(cable_out, "real_powerflow_cable_summary")

        bus_voltage_summary = summary.get("bus_voltage_summary")
        if isinstance(bus_voltage_summary, pd.DataFrame) and not bus_voltage_summary.empty:
            bus_out = bus_voltage_summary.copy()
            real_voltage_columns = [
                "bus",
                "voltage_p50_time_pu",
                "voltage_p10_time_pu",
                "voltage_p05_time_pu",
                "voltage_p01_time_pu",
                "voltage_min_time_pu",
                "voltage_hours_below_0_90",
            ]
            bus_out = bus_out[[col for col in real_voltage_columns if col in bus_out.columns]]
            bus_out.insert(0, "stage", stage)
            bus_out.insert(0, "real_powerflow_run_id", int(run_id))
            bus_out["bus"] = bus_out["bus"].astype(int)
            self._append(bus_out, "real_powerflow_bus_voltage_summary")

        tail_summary = summary.get("tail_summary")
        if isinstance(tail_summary, pd.DataFrame) and not tail_summary.empty:
            tail_out = tail_summary.copy()
            tail_out.insert(0, "stage", stage)
            tail_out.insert(0, "real_powerflow_run_id", int(run_id))
            tail_out["asset_id"] = tail_out["asset_id"].astype(int)
            tail_out["t_index"] = tail_out["t_index"].astype(int)
            self._append(tail_out, "real_powerflow_tail_value")

    def create_powerflow_run(
        self,
        grid_ref: dict[str, Any],
        *,
        urbs_input_file: str,
        pre_only: bool,
        scenario_key: str = DEFAULT_SCENARIO_KEY,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
        run_name: str | None = None,
        assumptions: dict[str, Any] | None = None,
    ) -> int:
        grid_case_id = self.get_or_create_grid_case(grid_ref)
        scenario_id = self.ensure_scenario(
            scenario_key=scenario_key,
            scenario_label=scenario_label,
            assumptions=assumptions,
        )
        pipeline_run_id = self.ensure_pipeline_run(
            grid_case_id=grid_case_id,
            scenario_id=scenario_id,
            run_name=self.default_pipeline_run_name(scenario_key),
        )
        run_name = run_name or self.default_powerflow_run_name(scenario_key, pre_only)
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
        with self.engine.begin() as conn:
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
                        "assumptions": json.dumps(assumptions or {}),
                    },
                ).scalar_one()
            )
            self._clear_powerflow_run(conn, run_id)
        return run_id

    def default_powerflow_run_name(self, scenario_key: str, pre_only: bool) -> str:
        stage = "pre" if pre_only else "full"
        return f"{scenario_key}_{stage}_powerflow"

    def _clear_powerflow_run(self, conn, run_id: int) -> None:
        for table_name in (
            "powerflow_transformer_diagnostic",
            "powerflow_tail_value",
            "powerflow_cable_summary",
            "powerflow_bus_voltage_summary",
            "powerflow_summary",
            "powerflow_reactive_component",
            "powerflow_line_result",
            "powerflow_bus_voltage",
            "powerflow_import",
            "powerflow_demand",
        ):
            conn.execute(
                text(f"DELETE FROM surrogrid.{table_name} WHERE powerflow_run_id = :run_id"),
                {"run_id": run_id},
            )

    def write_powerflow_summary(self, run_id: int, stage: str, summary: dict[str, Any]) -> None:
        grid_summary = summary.get("grid_summary", summary)
        out = pd.DataFrame(
            [
                {
                    "powerflow_run_id": int(run_id),
                    "stage": stage,
                    "n_timesteps": int(grid_summary.get("n_timesteps", 0)),
                    "n_converged_timesteps": grid_summary.get("n_converged_timesteps"),
                    "n_failed_timesteps": grid_summary.get("n_failed_timesteps"),
                    "n_voltage_buses": int(grid_summary.get("n_voltage_buses", 0)),
                    "n_cables": int(grid_summary.get("n_cables", 0)),
                    "transformer_s_rated_mva": grid_summary.get("transformer_s_rated_mva"),
                    "trafo_mean_s_mva": grid_summary.get("trafo_mean_s_mva"),
                    "trafo_max_s_mva": grid_summary.get("trafo_max_s_mva"),
                    "trafo_max_p_mw": grid_summary.get("trafo_max_p_mw"),
                    "trafo_max_q_mvar": grid_summary.get("trafo_max_q_mvar"),
                    "trafo_critical_t_index": grid_summary.get("trafo_critical_t_index"),
                    "trafo_critical_ts": self._timestamp_for_powerflow_index(run_id, grid_summary.get("trafo_critical_t_index")),
                    "trafo_loading_p50_time_percent": grid_summary.get("trafo_loading_p50_time_percent"),
                    "trafo_loading_p90_time_percent": grid_summary.get("trafo_loading_p90_time_percent"),
                    "trafo_loading_p95_time_percent": grid_summary.get("trafo_loading_p95_time_percent"),
                    "trafo_loading_p99_time_percent": grid_summary.get("trafo_loading_p99_time_percent"),
                    "trafo_loading_max_time_percent": grid_summary.get("trafo_loading_max_time_percent"),
                    "trafo_loading_hours_above_100": grid_summary.get("trafo_loading_hours_above_100"),
                    "cable_loading_p95_asset_percent": grid_summary.get("cable_loading_p95_asset_percent"),
                    "cable_hours_above_100_p95_asset": grid_summary.get("cable_hours_above_100_p95_asset"),
                    "voltage_p05_load_bus_hour_pu": grid_summary.get("voltage_p05_load_bus_hour_pu"),
                    "voltage_hours_below_0_90_p95_asset": grid_summary.get("voltage_hours_below_0_90_p95_asset"),
                    "voltage_hours_above_1_03_p95_asset": grid_summary.get("voltage_hours_above_1_03_p95_asset"),
                    "voltage_hours_above_1_10_p95_asset": grid_summary.get("voltage_hours_above_1_10_p95_asset"),
                }
            ]
        )
        self._append(out, "powerflow_summary")

        cable_summary = summary.get("cable_summary")
        if isinstance(cable_summary, pd.DataFrame) and not cable_summary.empty:
            cable_out = cable_summary.copy()
            cable_out.insert(0, "stage", stage)
            cable_out.insert(0, "powerflow_run_id", int(run_id))
            cable_out["cable"] = cable_out["cable"].astype(int)
            self._append(cable_out, "powerflow_cable_summary")

        bus_voltage_summary = summary.get("bus_voltage_summary")
        if isinstance(bus_voltage_summary, pd.DataFrame) and not bus_voltage_summary.empty:
            bus_out = bus_voltage_summary.copy()
            bus_out.insert(0, "stage", stage)
            bus_out.insert(0, "powerflow_run_id", int(run_id))
            bus_out["bus"] = bus_out["bus"].astype(int)
            self._append(bus_out, "powerflow_bus_voltage_summary")

        tail_summary = summary.get("tail_summary")
        if isinstance(tail_summary, pd.DataFrame) and not tail_summary.empty:
            tail_out = tail_summary.copy()
            tail_out.insert(0, "stage", stage)
            tail_out.insert(0, "powerflow_run_id", int(run_id))
            tail_out["asset_id"] = tail_out["asset_id"].astype(int)
            tail_out["t_index"] = tail_out["t_index"].astype(int)
            self._append(tail_out, "powerflow_tail_value")

        transformer_diagnostic = summary.get("transformer_diagnostic")
        if isinstance(transformer_diagnostic, pd.DataFrame) and not transformer_diagnostic.empty:
            diag_out = transformer_diagnostic.copy()
            diag_out.insert(0, "stage", stage)
            diag_out.insert(0, "powerflow_run_id", int(run_id))
            diag_out["point_index"] = diag_out["point_index"].astype(int)
            diag_out["x_value"] = diag_out["x_value"].astype(float)
            diag_out["t_index"] = pd.to_numeric(diag_out["t_index"], errors="coerce").astype("Int64")
            diag_out["ts"] = [
                self._timestamp_for_powerflow_index(run_id, value)
                if pd.notna(value) else pd.NaT
                for value in diag_out["t_index"]
            ]
            self._append(diag_out, "powerflow_transformer_diagnostic")

    def write_powerflow_demand(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        ts = self._timestamps_for_powerflow_run(run_id, len(df))
        buses = sorted({int(col[0]) for col in df.columns})
        chunk_size = 25
        for start in range(0, len(buses), chunk_size):
            rows = []
            for bus in buses[start:start + chunk_size]:
                rows.append(
                    pd.DataFrame(
                        {
                            "powerflow_run_id": run_id,
                            "stage": stage,
                            "ts": ts,
                            "t_index": range(len(df)),
                            "bus": bus,
                            "p_kw": self._series_or_none(df, (bus, "electricity")),
                            "q_kvar": self._series_or_none(df, (bus, "electricity-reactive")),
                        }
                    )
                )
            if rows:
                self._append(pd.concat(rows, ignore_index=True), "powerflow_demand")

    def write_powerflow_import(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        out = pd.DataFrame(
            {
                "powerflow_run_id": run_id,
                "stage": stage,
                "ts": self._timestamps_for_powerflow_run(run_id, len(df)),
                "t_index": range(len(df)),
                "p_mw": df.get("p_mw"),
                "q_mvar": df.get("q_mvar"),
            }
        )
        self._append(out, "powerflow_import")

    def write_powerflow_bus_voltage(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        columns = [int(col) for col in df.columns]
        ts = self._timestamps_for_powerflow_run(run_id, len(df))
        chunk_size = 25
        for start in range(0, len(columns), chunk_size):
            chunk_columns = columns[start:start + chunk_size]
            out = df.iloc[:, start:start + chunk_size].copy()
            out.columns = chunk_columns
            out.insert(0, "t_index", range(len(out)))
            out.insert(0, "ts", ts)
            out.insert(0, "stage", stage)
            out.insert(0, "powerflow_run_id", run_id)
            out = out.melt(
                id_vars=["powerflow_run_id", "stage", "ts", "t_index"],
                var_name="bus",
                value_name="vm_pu",
            )
            self._append(out, "powerflow_bus_voltage")

    def write_powerflow_line_result(self, run_id: int, stage: str, df: pd.DataFrame) -> None:
        ts = self._timestamps_for_powerflow_run(run_id, len(df))
        lines = sorted({int(col[0]) for col in df.columns})
        chunk_size = 25
        for start in range(0, len(lines), chunk_size):
            rows = []
            for line in lines[start:start + chunk_size]:
                rows.append(
                    pd.DataFrame(
                        {
                            "powerflow_run_id": run_id,
                            "stage": stage,
                            "ts": ts,
                            "t_index": range(len(df)),
                            "line": line,
                            "p_from_mw": self._series_or_none(df, (line, "p_from_mw")),
                            "q_from_mvar": self._series_or_none(df, (line, "q_from_mvar")),
                            "i_from_ka": self._series_or_none(df, (line, "i_from_ka")),
                        }
                    )
                )
            if rows:
                self._append(pd.concat(rows, ignore_index=True), "powerflow_line_result")

    def write_powerflow_reactive(self, run_id: int, df: pd.DataFrame) -> None:
        ts = self._timestamps_for_powerflow_run(run_id, len(df))
        columns = list(df.columns)
        chunk_size = 25
        for start in range(0, len(columns), chunk_size):
            rows = []
            for bus, component, source in columns[start:start + chunk_size]:
                rows.append(
                    pd.DataFrame(
                        {
                            "powerflow_run_id": run_id,
                            "ts": ts,
                            "t_index": range(len(df)),
                            "bus": int(bus),
                            "component": str(component),
                            "source": str(source),
                            "q_kvar": df[(bus, component, source)].to_numpy(),
                        }
                    )
                )
            if rows:
                self._append(pd.concat(rows, ignore_index=True), "powerflow_reactive_component")


    def _timestamps_for_demand_allocation_run(self, run_id: int, n_rows: int) -> pd.DatetimeIndex:
        return self._timestamps_for_run(
            "demand_allocation_run",
            "demand_allocation_run_id",
            run_id,
            n_rows,
        )

    def _timestamp_for_powerflow_index(self, run_id: int, t_index) -> pd.Timestamp | None:
        if t_index is None or pd.isna(t_index):
            return None
        return self._timestamps_for_powerflow_run(run_id, int(t_index) + 1)[int(t_index)]

    def _timestamps_for_powerflow_run(self, run_id: int, n_rows: int) -> pd.DatetimeIndex:
        return self._timestamps_for_run(
            "powerflow_run",
            "powerflow_run_id",
            run_id,
            n_rows,
        )

    def _timestamps_for_run(
        self,
        run_table: str,
        id_column: str,
        run_id: int,
        n_rows: int,
    ) -> pd.DatetimeIndex:
        query = text(
            f"""
            SELECT CASE
                WHEN r.assumptions <> '{{}}'::JSONB THEN r.assumptions
                ELSE sc.assumptions
            END AS assumptions
            FROM surrogrid.{run_table} r
            JOIN surrogrid.scenario sc
              ON sc.scenario_id = r.scenario_id
            WHERE r.{id_column} = :run_id
            """
        )
        with self.engine.connect() as conn:
            assumptions = conn.execute(query, {"run_id": int(run_id)}).scalar_one_or_none()
        if isinstance(assumptions, str):
            try:
                assumptions = json.loads(assumptions)
            except json.JSONDecodeError:
                assumptions = {}
        if not isinstance(assumptions, dict):
            assumptions = {}
        return self._timestamps(n_rows, start=assumptions.get("timeframe_start") or TIME_INDEX_START)

    def _append(self, df: pd.DataFrame, table_name: str) -> None:
        df.to_sql(
            table_name,
            self.engine,
            schema="surrogrid",
            if_exists="append",
            index=False,
            chunksize=10000,
            method="multi",
        )

    def _timestamps(self, n_rows: int, start: str = TIME_INDEX_START) -> pd.DatetimeIndex:
        return pd.date_range(start, periods=n_rows, freq="h")

    def _series_or_none(self, df: pd.DataFrame, column: tuple[Any, str]) -> pd.Series:
        if column in df.columns:
            return df[column].reset_index(drop=True)
        return pd.Series([None] * len(df))
