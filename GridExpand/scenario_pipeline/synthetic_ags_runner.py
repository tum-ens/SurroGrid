#!/usr/bin/env python3
"""Run the DB-backed synthetic GridExpand pipeline for AGS grid candidates.

This helper coordinates synthetic Step 2, optional Step 3, and Step 4 runs for long tmux batches. It keeps
per-candidate logs, can resume completed work, validates DB-backed power-flow
outputs, and records failed grids without aborting the full batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import h5py
import pandas as pd
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.timeframe import (  # noqa: E402
    TIMEFRAME_MODES,
    build_initial_metadata,
    horizon_hours_from_hdf,
    output_filename_for_timeframe,
    read_hdf_metadata,
    scenario_key_for_timeframe,
)
from common.orchestration import (  # noqa: E402
    StatusLog,
    run_batch_command,
    run_command,
    utc_now,
)
from scenario_pipeline.config_loader import load_scenario_config  # noqa: E402

DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "config" / "scenarios"
    / "forchheim_2045.yaml"
)


EXPECTED_POWERFLOW_TABLES = {
    "powerflow_demand": ("pre", "post"),
    "powerflow_import": ("pre", "post"),
    "powerflow_bus_voltage": ("pre", "post"),
    "powerflow_line_result": ("pre", "post"),
}

PROFILE_CHOICES = (
    "status_quo",
    "electricity_heat",
    "electricity_mobility",
    "electricity_heat_mobility",
    "all",
)

POWERFLOW_OUTPUT_CHOICES = ("raw", "summary", "both")
DEMAND_SCOPE_CHOICES = ("all", "residential")
CLEANUP_CHOICES = ("never", "success")


def run_name_profile_token(profile: str) -> str:
    return "post_electrification" if profile == "all" else profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DB-backed synthetic GridExpand pipeline batch for one AGS."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--ags",
        required=True,
        help="AGS identifier for the region to process, e.g. 09162000.",
    )
    parser.add_argument(
        "--pylovo-version-id",
        required=True,
        help="Exact pylovo topology version; supplied by the run configuration.",
    )
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step2-cpus", type=int, default=4)
    parser.add_argument("--step3-cpus", type=int, default=16)
    parser.add_argument("--step3-max-cpus", type=int, default=32)
    parser.add_argument("--step3-target-columns", type=int, default=35)
    parser.add_argument("--step3-cluster-concurrency", type=int, default=1)
    parser.add_argument("--step4-cpus", type=int, default=4)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=481527,
        help="Run-level seed for the physical stochastic profile realization.",
    )
    parser.add_argument(
        "--model-case",
        choices=("pre", "post-inflex-heuristic", "post-hems-optimized", "post-hems-heuristic"),
        default="post-hems-optimized",
        help="Scenario case controlling upstream asset sizing and downstream dispatch.",
    )
    parser.add_argument(
        "--case-qualified-output",
        action="store_true",
        help="Append the model-case name to Step 2--4 HDF5 filenames and power-flow run names.",
    )
    parser.add_argument(
        "--powerflow-output",
        choices=POWERFLOW_OUTPUT_CHOICES,
        default="raw",
        help=(
            "Power-flow output mode. 'raw' stores full pre/post time series, "
            "'summary' stores compact notebook metrics, and 'both' stores both. "
            "For electrification profiles the compact summary includes post results."
        ),
    )
    parser.add_argument(
        "--profiles",
        choices=PROFILE_CHOICES,
        default="all",
        help="Demand profile scope passed to Step 2; use electricity_heat for heat without mobility.",
    )
    parser.add_argument(
        "--demand-scope",
        choices=DEMAND_SCOPE_CHOICES,
        default="all",
        help=(
            "Building scope for Step 2 through Step 4. Use residential for a consistent "
            "household-only URBS and power-flow pipeline."
        ),
    )
    parser.add_argument(
        "--step2-timeseries-storage", choices=["db", "temp", "both"], default="temp"
    )
    parser.add_argument(
        "--timeframe-mode",
        choices=TIMEFRAME_MODES,
        default="full_year",
        help="Simulation timeframe passed to Step 2; one-week modes produce 168-hour stress runs.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pilot-index", type=int, default=0)
    parser.add_argument("--no-pilot-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--cleanup-intermediates",
        choices=CLEANUP_CHOICES,
        default="never",
        help=(
            "Delete per-candidate HDF5 hand-off files after successful validation. "
            "'success' keeps failed-candidate files for debugging and keeps DB summaries/logs."
        ),
    )
    parser.add_argument(
        "--cleanup-completed-only",
        action="store_true",
        help=(
            "Only remove intermediate files for candidates already marked as done in the run log, "
            "then exit. Use this before resuming an interrupted disk-limited run."
        ),
    )
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-dynamic-step3", action="store_true")
    parser.add_argument(
        "--no-materialize-expansion",
        action="store_true",
        help="Do not materialize expansion_analysis_run rows after summary power-flow runs.",
    )
    parser.add_argument(
        "--include-no-flex-powerflow",
        action="store_true",
        help=(
            "For post-electrification profiles, run an additional Step 4 no-flex powerflow after "
            "Step 3. Heat is reconstructed from fixed demand using optimized post-flex "
            "heat-pump and auxiliary-heater capacities; PV and EV profiles remain fixed."
        ),
    )
    parser.add_argument(
        "--no-flex-only",
        action="store_true",
        help=(
            "Run Step 3 optimization to obtain post-flex capacities, then run only the "
            "no-flex post-electrification power flow."
        ),
    )
    parser.add_argument(
        "--no-flex-ev-charger-kw",
        type=float,
        help="Optional EV charger cap passed to Step 4 --post-demand-mode no-flex.",
    )
    parser.add_argument(
        "--expansion-analysis-prefix",
        help=(
            "Optional prefix for automatic expansion analysis keys. "
            "Defaults to '<timeframe_mode>_<profiles>[_hh_only][_tsam]'."
        ),
    )
    args = parser.parse_args()
    args.scenario_config = args.scenario_config.resolve()
    scenario, args.scenario_hash = load_scenario_config(args.scenario_config)
    args.tsam = scenario.time_aggregation.enabled
    args.tsam_periods = scenario.time_aggregation.number_of_typical_periods
    args.tsam_hours_per_period = scenario.time_aggregation.hours_per_period
    args.tsam_extreme_method = scenario.time_aggregation.extreme_period_method
    if args.model_case == "pre" and args.profiles != "status_quo":
        parser.error("The pre model case requires --profiles status_quo.")
    if args.model_case != "pre" and args.profiles == "status_quo":
        parser.error("Post model cases require post-electrification profiles.")
    if args.include_no_flex_powerflow and args.profiles == "status_quo":
        parser.error(
            "--include-no-flex-powerflow requires post-electrification profiles, not status_quo."
        )
    if args.no_flex_only and args.profiles == "status_quo":
        parser.error(
            "--no-flex-only requires post-electrification profiles, not status_quo."
        )
    if args.no_flex_only and args.include_no_flex_powerflow:
        parser.error(
            "Use either --no-flex-only or --include-no-flex-powerflow, not both."
        )
    if (args.include_no_flex_powerflow or args.no_flex_only) and args.model_case == "post-hems-optimized":
        parser.error("No-flex dispatch requires a heuristic model case with fixed asset capacities.")
    if args.no_flex_ev_charger_kw is not None and not (
        args.include_no_flex_powerflow or args.no_flex_only
    ):
        parser.error(
            "--no-flex-ev-charger-kw requires --include-no-flex-powerflow or --no-flex-only."
        )
    return args


def step_paths(repo_root: Path) -> dict[str, Path]:
    gridexpand = repo_root / "GridExpand"
    return {
        "step2_results": gridexpand / "2.demand_allocation" / "gridalloc" / "results",
        "step3_input": gridexpand / "3.urbs" / "Input",
        "step4_input": gridexpand / "4.powerflow" / "Input",
    }


def configure_imports(repo_root: Path) -> None:
    gridexpand_dir = repo_root / "GridExpand"
    if str(gridexpand_dir) not in sys.path:
        sys.path.insert(0, str(gridexpand_dir))


def normalize_ags(value: str) -> int:
    return int(str(value).strip().lstrip("0") or "0")


def get_candidates(
    repo_root: Path,
    ags: str,
    min_buildings: int,
    demand_scope: str = "all",
    pylovo_version_id: str | None = None,
) -> list[dict[str, object]]:
    configure_imports(repo_root)
    from common.database import SurroGridDatabase

    db = SurroGridDatabase()
    db.pylovo_version_id = pylovo_version_id
    count_column = (
        "n_residential_buildings" if demand_scope == "residential" else "n_buildings"
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
        )
        SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY plz, kcid, bcid) - 1 AS candidate_index
        FROM latest
        ORDER BY candidate_index
        """
    )
    with db.engine.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                query,
                {
                    "ags": normalize_ags(ags),
                    "min_buildings": int(min_buildings),
                    "demand_scope": demand_scope,
                    "pylovo_version_id": db.pylovo_version_id,
                },
            ).mappings()
        ]
    return [
        db._format_grid_ref(
            ags=normalize_ags(ags),
            row=row,
            candidate_index=int(row["candidate_index"]),
        )
        | {
            "n_buildings": int(row["n_buildings"]),
            "n_selected_buildings": int(row[count_column]),
            "n_residential_buildings": int(row["n_residential_buildings"]),
        }
        for row in rows
    ]


def scenario_base_key(args: argparse.Namespace) -> str:
    return (
        "baseline_static_hh_only"
        if args.demand_scope == "residential"
        else "baseline_static"
    )


def pipeline_scenario_key(args: argparse.Namespace) -> str:
    return scenario_key_for_timeframe(
        args.timeframe_mode, base_key=scenario_base_key(args)
    )


def powerflow_run_name(args: argparse.Namespace, mode: str) -> str:
    case = f"_{args.model_case}" if args.case_qualified_output else ""
    return f"{pipeline_scenario_key(args)}_{run_name_profile_token(args.profiles)}{case}_{mode}_powerflow"


def case_qualified_filename(filename: str, args: argparse.Namespace) -> str:
    if not args.case_qualified_output:
        return filename
    path = Path(filename)
    return f"{path.stem}_{args.model_case}{path.suffix}"


def expansion_analysis_prefix(args: argparse.Namespace) -> str:
    if args.expansion_analysis_prefix:
        return args.expansion_analysis_prefix
    scope_suffix = "_hh_only" if args.demand_scope == "residential" else ""
    tsam_suffix = "_tsam" if args.tsam else ""
    return f"{args.timeframe_mode}_{run_name_profile_token(args.profiles)}{scope_suffix}{tsam_suffix}"


def materialize_expansion_analyses(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    status: StatusLog,
) -> list[dict[str, str]]:
    if args.no_materialize_expansion or args.powerflow_output not in {
        "summary",
        "both",
    }:
        return []

    summary_pre_only = args.profiles == "status_quo"
    prefix = expansion_analysis_prefix(args)
    postprocessing_dir = repo_root / "GridExpand" / "5.postprocessing"
    log_path = args.run_dir / "expansion_materialization.log"
    materialized = []

    def materialize_one(
        run_name: str, stage: str, analysis_key: str, note: str, log_stage: str
    ) -> None:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "expansion.grid_expansion",
            "--run-name",
            run_name,
            "--stage",
            stage,
            "--ags",
            str(args.ags),
            "--analysis-key",
            analysis_key,
            "--note",
            note,
            "--replace",
        ]
        run_batch_command(
            cmd=cmd,
            cwd=postprocessing_dir,
            log_path=log_path,
            status=status,
            stage=log_stage,
        )

    if args.no_flex_only:
        no_flex_run_name = powerflow_run_name(args, "summary_no_flex")
        materialize_one(
            no_flex_run_name,
            "pre",
            f"{prefix}_pre",
            (
                f"Automatically materialized by synthetic_ags_runner from {no_flex_run_name} "
                "summary stage=pre."
            ),
            "expansion_materialize_pre",
        )
        materialized.append({"stage": "pre", "analysis_key": f"{prefix}_pre"})
        materialize_one(
            no_flex_run_name,
            "post",
            f"{prefix}_post_no_flex",
            (
                f"Automatically materialized by synthetic_ags_runner from {no_flex_run_name} "
                "summary stage=post using fixed no-flex demand with post-flex heat capacity split."
            ),
            "expansion_materialize_post_no_flex",
        )
        materialized.append(
            {"stage": "post_no_flex", "analysis_key": f"{prefix}_post_no_flex"}
        )
        return materialized

    stages = ("pre",) if summary_pre_only else ("pre", "post")
    summary_run_name = powerflow_run_name(args, "summary")
    for stage in stages:
        analysis_key = f"{prefix}_{stage}"
        materialize_one(
            summary_run_name,
            stage,
            analysis_key,
            (
                f"Automatically materialized by synthetic_ags_runner from {summary_run_name} "
                f"summary stage={stage}."
            ),
            f"expansion_materialize_{stage}",
        )
        materialized.append({"stage": stage, "analysis_key": analysis_key})

    if args.include_no_flex_powerflow and not summary_pre_only:
        no_flex_run_name = powerflow_run_name(args, "summary_no_flex")
        no_flex_analysis_key = f"{prefix}_post_no_flex"
        materialize_one(
            no_flex_run_name,
            "post",
            no_flex_analysis_key,
            (
                f"Automatically materialized by synthetic_ags_runner from {no_flex_run_name} "
                "summary stage=post using fixed no-flex demand with post-flex heat capacity split."
            ),
            "expansion_materialize_post_no_flex",
        )
        materialized.append(
            {"stage": "post_no_flex", "analysis_key": no_flex_analysis_key}
        )

    return materialized


def hdf_column_count(path: Path, key: str) -> int:
    group_name = key.strip("/")
    with h5py.File(path, mode="r") as h5:
        if group_name not in h5:
            return 0
        group = h5[group_name]
        total = 0
        for name, node in group.items():
            if (
                name.startswith("block")
                and name.endswith("_values")
                and len(node.shape) == 2
            ):
                total += int(node.shape[1])
        return total


def _labels_from_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        return [str(value).lower() for value in df.columns.get_level_values(-1)]
    return [str(value).lower() for value in df.columns]


def _labels_from_column(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return [str(value).lower() for value in df[column].dropna().unique()]


def _contains_any(labels: list[str], tokens: tuple[str, ...]) -> bool:
    return any(any(token in label for token in tokens) for label in labels)


def scenario_suffix_from_hdf(path: Path) -> str:
    with pd.HDFStore(path, mode="r") as store:
        demand = (
            store["/urbs_in/demand"] if "/urbs_in/demand" in store else pd.DataFrame()
        )
        supim = store["/urbs_in/supim"] if "/urbs_in/supim" in store else pd.DataFrame()
        process = (
            store["/urbs_in/process"] if "/urbs_in/process" in store else pd.DataFrame()
        )
        commodity = (
            store["/urbs_in/commodity"]
            if "/urbs_in/commodity" in store
            else pd.DataFrame()
        )

    demand_labels = _labels_from_columns(demand)
    supim_labels = _labels_from_columns(supim)
    process_labels = _labels_from_column(process, "Process")
    commodity_labels = _labels_from_column(commodity, "Commodity")

    has_pv = bool(supim_labels) or _contains_any(process_labels, ("pv", "rooftop"))
    has_heat = (
        _contains_any(demand_labels, ("space_heat", "water_heat", "heat"))
        or _contains_any(process_labels, ("heatpump", "heat_dummy", "heat"))
        or _contains_any(commodity_labels, ("space_heat", "water_heat", "common_heat"))
    )
    has_ev = (
        _contains_any(demand_labels, ("mobility", "bev", "ev"))
        or _contains_any(process_labels, ("charging_station", "bev", "mobility"))
        or _contains_any(commodity_labels, ("mobility", "bev"))
    )

    return (
        f"PV{100 if has_pv else 0}_"
        f"HP{100 if has_heat else 0}_"
        f"EV{100 if has_ev else 0}_"
        "VarTar0_CapPr0"
    )


def choose_step3_settings(
    step2_output: Path, args: argparse.Namespace
) -> tuple[int, int, dict[str, int]]:
    if args.no_dynamic_step3:
        return int(args.step3_cpus), int(args.step3_cluster_concurrency), {}

    stats: dict[str, int] = {}
    for key, name in (
        ("urbs_in/demand", "demand_columns"),
        ("urbs_in/eff_factor", "eff_factor_columns"),
    ):
        try:
            stats[name] = hdf_column_count(step2_output, key)
        except Exception:
            stats[name] = 0
    largest_columns = max(stats.values() or [0])
    required = (
        math.ceil(largest_columns / max(1, int(args.step3_target_columns)))
        if largest_columns
        else args.step3_cpus
    )
    choices = [4, 8, 12, 16, 24, 32]
    max_cpus = max(1, int(args.step3_max_cpus))
    choices = [value for value in choices if value <= max_cpus] or [max_cpus]
    selected = next((value for value in choices if value >= required), choices[-1])
    selected = max(selected, int(args.step3_cpus))
    selected = min(selected, max_cpus)
    return selected, int(args.step3_cluster_concurrency), stats


def validate_powerflow_db(
    repo_root: Path,
    scenario_filename: str,
    *,
    summary_only: bool = False,
    pre_only: bool = False,
    run_name: str | None = None,
    expected_summary_stages: tuple[str, ...] = ("pre",),
) -> dict[str, Any]:
    configure_imports(repo_root)
    scenario_path = (
        repo_root / "GridExpand" / "4.powerflow" / "Input" / scenario_filename
    )
    with pd.HDFStore(scenario_path, mode="r") as store:
        if "/urbs_out/MILP/tau_pro" in store:
            tau_pro = store["/urbs_out/MILP/tau_pro"]
            time_level = "t" if "t" in tau_pro.index.names else 0
            expected_horizon = tau_pro.index.get_level_values(time_level).nunique()
        else:
            expected_horizon = horizon_hours_from_hdf(scenario_path)
    expected_max_t = expected_horizon - 1
    from common.database import SurroGridDatabase

    db = SurroGridDatabase()
    with db.engine.connect() as conn:
        run = (
            conn.execute(
                text(
                    """
                SELECT powerflow_run_id, run_name, pre_only, updated_at
                FROM surrogrid.powerflow_run
                WHERE urbs_input_file = :scenario_filename
                  AND pre_only = :pre_only
                  AND (:run_name IS NULL OR run_name = :run_name)
                ORDER BY updated_at DESC, powerflow_run_id DESC
                LIMIT 1
                """
                ),
                {
                    "scenario_filename": scenario_filename,
                    "pre_only": pre_only,
                    "run_name": run_name,
                },
            )
            .mappings()
            .first()
        )
        if run is None:
            mode = "summary" if summary_only else "raw"
            raise RuntimeError(f"No {mode} powerflow_run found for {scenario_filename}")

        run_id = int(run["powerflow_run_id"])
        validation: dict[str, Any] = {
            "powerflow_run_id": run_id,
            "run_name": run["run_name"],
            "mode": "summary" if summary_only else "raw",
            "tables": {},
            "expected_horizon_hours": expected_horizon,
        }
        missing: list[str] = []
        if summary_only:
            for table_name in (
                "powerflow_summary",
                "powerflow_cable_summary",
                "powerflow_bus_voltage_summary",
            ):
                rows = (
                    conn.execute(
                        text(
                            f"""
                        SELECT stage, count(*) AS rows
                        FROM surrogrid.{table_name}
                        WHERE powerflow_run_id = :run_id
                        GROUP BY stage
                        ORDER BY stage
                        """
                        ),
                        {"run_id": run_id},
                    )
                    .mappings()
                    .all()
                )
                by_stage = {str(row["stage"]): dict(row) for row in rows}
                validation["tables"][table_name] = by_stage
                for stage in expected_summary_stages:
                    row = by_stage.get(stage)
                    if not row or int(row["rows"]) <= 0:
                        missing.append(f"{table_name}:{stage}:missing")
            if missing:
                raise RuntimeError(
                    "Incomplete Step 4 DB summary results: " + ", ".join(missing)
                )
            return validation

        expected_stage_filter = ("pre",) if pre_only else None
        for table_name, expected_stages in EXPECTED_POWERFLOW_TABLES.items():
            if expected_stage_filter is not None:
                expected_stages = expected_stage_filter
            rows = (
                conn.execute(
                    text(
                        f"""
                    SELECT stage, count(*) AS rows, min(t_index) AS min_t, max(t_index) AS max_t
                    FROM surrogrid.{table_name}
                    WHERE powerflow_run_id = :run_id
                    GROUP BY stage
                    ORDER BY stage
                    """
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .all()
            )
            by_stage = {str(row["stage"]): dict(row) for row in rows}
            validation["tables"][table_name] = by_stage
            for stage in expected_stages:
                row = by_stage.get(stage)
                if not row:
                    missing.append(f"{table_name}:{stage}:missing")
                    continue
                if (
                    int(row["rows"]) <= 0
                    or int(row["min_t"]) != 0
                    or int(row["max_t"]) != expected_max_t
                ):
                    missing.append(f"{table_name}:{stage}:incomplete")

        if not pre_only:
            reactive_rows = (
                conn.execute(
                    text(
                        """
                    SELECT count(*) AS rows, min(t_index) AS min_t, max(t_index) AS max_t
                    FROM surrogrid.powerflow_reactive_component
                    WHERE powerflow_run_id = :run_id
                    """
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .one()
            )
            validation["tables"]["powerflow_reactive_component"] = {
                "all": dict(reactive_rows)
            }
            if (
                int(reactive_rows["rows"]) <= 0
                or int(reactive_rows["min_t"]) != 0
                or int(reactive_rows["max_t"]) != expected_max_t
            ):
                missing.append("powerflow_reactive_component:all:incomplete")

        if missing:
            raise RuntimeError("Incomplete Step 4 DB results: " + ", ".join(missing))
        return validation


def candidate_failed_payload(
    candidate: dict[str, object],
    stage: str,
    message: str,
    seconds: float,
    log_file: Path,
) -> dict[str, object]:
    return {
        "candidate_index": int(candidate["candidate_index"]),
        "ags": candidate.get("ags"),
        "plz": candidate.get("plz"),
        "kcid": candidate.get("kcid"),
        "bcid": candidate.get("bcid"),
        "n_buildings": candidate.get("n_buildings"),
        "bridge_filename": candidate.get("bridge_filename"),
        "stage": stage,
        "message": message,
        "seconds": seconds,
        "log_file": str(log_file),
    }


def candidate_intermediate_files(
    repo_root: Path, candidate: dict[str, object], args: argparse.Namespace
) -> list[Path]:
    paths = step_paths(repo_root)
    step2_filename = case_qualified_filename(
        output_filename_for_timeframe(
            str(candidate["bridge_filename"]), args.timeframe_mode
        ),
        args,
    )
    step2_stem = Path(step2_filename).stem
    files = [
        paths["step2_results"] / step2_filename,
        paths["step3_input"] / step2_filename,
        paths["step4_input"] / step2_filename,
    ]
    files.extend(sorted(paths["step4_input"].glob(f"{step2_stem}_*.h5")))
    seen: set[Path] = set()
    unique_files = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def cleanup_candidate_intermediates(
    repo_root: Path,
    candidate: dict[str, object],
    args: argparse.Namespace,
    status: StatusLog,
    *,
    reason: str,
) -> dict[str, object]:
    removed_files = []
    removed_bytes = 0
    for file_path in candidate_intermediate_files(repo_root, candidate, args):
        if not file_path.exists() or not file_path.is_file():
            continue
        size = file_path.stat().st_size
        file_path.unlink()
        removed_files.append(str(file_path))
        removed_bytes += size
    payload = {
        "event": "candidate_intermediates_cleaned",
        "candidate_index": int(candidate["candidate_index"]),
        "reason": reason,
        "removed_files": len(removed_files),
        "removed_bytes": removed_bytes,
    }
    status.event(**payload)
    return {**payload, "files": removed_files}


def completed_candidate_indexes(status: StatusLog) -> set[int]:
    completed = {
        index for index, row in status.rows.items() if row.get("status") == "done"
    }
    if not status.events_path.exists():
        return completed
    with status.events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event") in {"candidate_done", "pilot_finish"}
                and event.get("status") == "done"
            ):
                candidate_index = event.get("candidate_index")
                if candidate_index is not None:
                    completed.add(int(candidate_index))
    return completed


def cleanup_completed_intermediates(
    repo_root: Path,
    candidates: list[dict[str, object]],
    args: argparse.Namespace,
    status: StatusLog,
) -> dict[str, object]:
    completed = completed_candidate_indexes(status)
    by_index = {
        int(candidate["candidate_index"]): candidate for candidate in candidates
    }
    cleanup_results = []
    for candidate_index in sorted(completed):
        candidate = by_index.get(candidate_index)
        if candidate is None:
            continue
        cleanup_results.append(
            cleanup_candidate_intermediates(
                repo_root,
                candidate,
                args,
                status,
                reason="completed_only",
            )
        )
    summary = {
        "status": "done",
        "candidate_count": len(cleanup_results),
        "removed_files": sum(
            int(result["removed_files"]) for result in cleanup_results
        ),
        "removed_bytes": sum(
            int(result["removed_bytes"]) for result in cleanup_results
        ),
        "finished_at": utc_now(),
    }
    status.event(event="cleanup_completed_finish", **summary)
    return summary


def run_candidate(
    *,
    repo_root: Path,
    ags: str,
    candidate: dict[str, object],
    args: argparse.Namespace,
    status: StatusLog,
) -> dict[str, object]:
    gridexpand = repo_root / "GridExpand"
    step2_dir = gridexpand / "2.demand_allocation" / "gridalloc"
    step3_dir = gridexpand / "3.urbs"
    step4_dir = gridexpand / "4.powerflow"
    candidate_index = int(candidate["candidate_index"])
    bridge_filename = str(candidate["bridge_filename"])
    step2_filename = case_qualified_filename(
        output_filename_for_timeframe(bridge_filename, args.timeframe_mode), args
    )
    scenario_filename = ""
    log_file = (
        args.run_dir / "logs" / f"candidate_{candidate_index:03d}_{step2_filename}.log"
    )
    started = time.monotonic()
    current_stage = "queued"
    timeframe_metadata = build_initial_metadata(args.timeframe_mode)
    status.update(
        candidate_index,
        ags=candidate.get("ags", ags),
        plz=candidate.get("plz", ""),
        kcid=candidate.get("kcid", ""),
        bcid=candidate.get("bcid", ""),
        n_buildings=candidate.get("n_buildings", ""),
        bridge_filename=step2_filename,
        demand_scope=args.demand_scope,
        timeframe_mode=args.timeframe_mode,
        horizon_hours=timeframe_metadata["horizon_hours"],
        timeframe_start=timeframe_metadata["timeframe_start"],
        timeframe_end=timeframe_metadata["timeframe_end"],
        status="queued",
        stage="queued",
        started_at=utc_now(),
        finished_at="",
        seconds="",
        step3_cpus="",
        urbs_cluster_concurrency="",
        log_file=log_file,
        message="",
    )

    try:
        current_stage = "step2_demand_allocation"
        step2_cmd = [
                "uv",
                "run",
                "--project",
                "..",
                "python",
                "main.py",
                ags,
                "--storage",
                "db",
                "--pylovo-version-id",
                str(args.pylovo_version_id),
                "--candidate-index",
                str(candidate_index),
                "--min-buildings",
                str(args.min_buildings),
                "--profiles",
                args.profiles,
                "--demand-scope",
                args.demand_scope,
                "--mobility-source",
                "pool",
                "--timeseries-storage",
                args.step2_timeseries_storage,
                "--timeframe-mode",
                args.timeframe_mode,
                "--model-case",
                args.model_case,
                "--profile-seed",
                str(args.profile_seed),
                "--scenario-config",
                str(args.scenario_config),
                "--n_cpu",
                str(args.step2_cpus),
            ]
        if args.case_qualified_output:
            step2_cmd.append("--case-qualified-output")
        run_command(
            cmd=step2_cmd,
            cwd=step2_dir,
            log_path=log_file,
            status=status,
            candidate_index=candidate_index,
            stage=current_stage,
        )
        step2_output = step2_dir / "results" / step2_filename
        if not step2_output.exists():
            raise FileNotFoundError(f"Missing Step 2 output {step2_output}")
        timeframe_metadata = read_hdf_metadata(step2_output)
        scenario_suffix = scenario_suffix_from_hdf(step2_output)
        scenario_filename = step2_filename.replace(".h5", f"_{scenario_suffix}.h5")
        status.update(
            candidate_index,
            horizon_hours=timeframe_metadata.get("horizon_hours", ""),
            timeframe_start=timeframe_metadata.get("timeframe_start", ""),
            timeframe_end=timeframe_metadata.get("timeframe_end", ""),
            message=json.dumps({"scenario_suffix": scenario_suffix}, sort_keys=True),
        )

        if args.profiles == "status_quo":
            status.update(
                candidate_index,
                step3_cpus="skipped",
                urbs_cluster_concurrency="skipped",
                message=json.dumps(
                    {"scenario_suffix": scenario_suffix, "step3": "skipped_status_quo"},
                    sort_keys=True,
                ),
            )
            shutil.copy2(step2_output, step4_dir / "Input" / step2_filename)
            validations = []

            if args.powerflow_output in {"raw", "both"}:
                current_stage = "step4_powerflow_raw_pre_only"
                raw_run_name = powerflow_run_name(args, "raw")
                raw_cmd = [
                    "uv",
                    "run",
                    "python",
                    "run_pwrflw.py",
                    step2_filename,
                    "--storage",
                    "db",
                    "--pre-only",
                    "--run-name",
                    raw_run_name,
                    "--n_cpu",
                    str(args.step4_cpus),
                    "--pylovo-version-id",
                    str(args.pylovo_version_id),
                ]
                if args.demand_scope == "residential":
                    raw_cmd.append("--hh-only")
                run_command(
                    cmd=raw_cmd,
                    cwd=step4_dir,
                    log_path=log_file,
                    status=status,
                    candidate_index=candidate_index,
                    stage=current_stage,
                )
                current_stage = "step4_validate_raw_pre_only"
                validations.append(
                    validate_powerflow_db(
                        repo_root,
                        step2_filename,
                        summary_only=False,
                        pre_only=True,
                        run_name=raw_run_name,
                    )
                )

            if args.powerflow_output in {"summary", "both"}:
                current_stage = "step4_powerflow_summary_pre_only"
                summary_run_name = powerflow_run_name(args, "summary")
                summary_cmd = [
                    "uv",
                    "run",
                    "python",
                    "run_pwrflw.py",
                    step2_filename,
                    "--storage",
                    "db",
                    "--pre-only",
                    "--summary-only",
                    "--run-name",
                    summary_run_name,
                    "--n_cpu",
                    str(args.step4_cpus),
                    "--pylovo-version-id",
                    str(args.pylovo_version_id),
                ]
                if args.demand_scope == "residential":
                    summary_cmd.append("--hh-only")
                run_command(
                    cmd=summary_cmd,
                    cwd=step4_dir,
                    log_path=log_file,
                    status=status,
                    candidate_index=candidate_index,
                    stage=current_stage,
                )
                current_stage = "step4_validate_summary_pre_only"
                validations.append(
                    validate_powerflow_db(
                        repo_root,
                        step2_filename,
                        summary_only=True,
                        pre_only=True,
                        run_name=summary_run_name,
                        expected_summary_stages=("pre",),
                    )
                )

            with log_file.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{utc_now()}] STEP4 STATUS-QUO VALIDATION OK\n")
                log_handle.write(
                    json.dumps(validations, indent=2, sort_keys=True, default=str)
                    + "\n"
                )

            seconds = round(time.monotonic() - started, 1)
            status.update(
                candidate_index,
                status="done",
                stage="complete",
                finished_at=utc_now(),
                seconds=seconds,
                message="ok",
            )
            if args.cleanup_intermediates == "success":
                cleanup_candidate_intermediates(
                    repo_root, candidate, args, status, reason="success"
                )
            return {
                "candidate_index": candidate_index,
                "status": "done",
                "seconds": seconds,
            }

        if args.no_flex_only:
            validations = []
            shutil.copy2(step2_output, step3_dir / "Input" / step2_filename)

            step3_cpus, cluster_concurrency, step3_stats = choose_step3_settings(
                step2_output, args
            )
            step3_stats = {
                **step3_stats,
                "post_flex_capacity_source": "required_for_no_flex",
            }
            status.update(
                candidate_index,
                step3_cpus=step3_cpus,
                urbs_cluster_concurrency=cluster_concurrency,
                message=json.dumps(step3_stats, sort_keys=True),
            )

            current_stage = "step3_urbs_for_no_flex"
            step3_cmd = [
                "uv",
                "run",
                "python",
                "run_urbs_cluster.py",
                step2_filename,
                "--n_cpu",
                str(step3_cpus),
            ]
            step3_cmd.extend(["--scenario-config", str(args.scenario_config)])
            run_command(
                cmd=step3_cmd,
                cwd=step3_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
                env_extra={"URBS_CLUSTER_CONCURRENCY": str(cluster_concurrency)},
            )
            step3_output = step3_dir / "result" / scenario_filename
            if not step3_output.exists():
                raise FileNotFoundError(f"Missing Step 3 output {step3_output}")
            powerflow_filename = scenario_filename
            shutil.copy2(step3_output, step4_dir / "Input" / powerflow_filename)

            if args.powerflow_output in {"raw", "both"}:
                current_stage = "step4_powerflow_raw_no_flex"
                raw_no_flex_run_name = powerflow_run_name(args, "raw_no_flex")
                raw_no_flex_cmd = [
                    "uv",
                    "run",
                    "python",
                    "run_pwrflw.py",
                    powerflow_filename,
                    "--storage",
                    "db",
                    "--run-name",
                    raw_no_flex_run_name,
                    "--post-demand-mode",
                    "no-flex",
                    "--n_cpu",
                    str(args.step4_cpus),
                    "--pylovo-version-id",
                    str(args.pylovo_version_id),
                ]
                if args.no_flex_ev_charger_kw is not None:
                    raw_no_flex_cmd.extend(
                        ["--no-flex-ev-charger-kw", str(args.no_flex_ev_charger_kw)]
                    )
                if args.demand_scope == "residential":
                    raw_no_flex_cmd.append("--hh-only")
                run_command(
                    cmd=raw_no_flex_cmd,
                    cwd=step4_dir,
                    log_path=log_file,
                    status=status,
                    candidate_index=candidate_index,
                    stage=current_stage,
                )
                current_stage = "step4_validate_raw_no_flex"
                validations.append(
                    validate_powerflow_db(
                        repo_root,
                        powerflow_filename,
                        summary_only=False,
                        pre_only=False,
                        run_name=raw_no_flex_run_name,
                    )
                )

            if args.powerflow_output in {"summary", "both"}:
                current_stage = "step4_powerflow_summary_no_flex"
                summary_no_flex_run_name = powerflow_run_name(args, "summary_no_flex")
                summary_no_flex_cmd = [
                    "uv",
                    "run",
                    "python",
                    "run_pwrflw.py",
                    powerflow_filename,
                    "--storage",
                    "db",
                    "--summary-only",
                    "--run-name",
                    summary_no_flex_run_name,
                    "--post-demand-mode",
                    "no-flex",
                    "--n_cpu",
                    str(args.step4_cpus),
                    "--pylovo-version-id",
                    str(args.pylovo_version_id),
                ]
                if args.no_flex_ev_charger_kw is not None:
                    summary_no_flex_cmd.extend(
                        ["--no-flex-ev-charger-kw", str(args.no_flex_ev_charger_kw)]
                    )
                if args.demand_scope == "residential":
                    summary_no_flex_cmd.append("--hh-only")
                run_command(
                    cmd=summary_no_flex_cmd,
                    cwd=step4_dir,
                    log_path=log_file,
                    status=status,
                    candidate_index=candidate_index,
                    stage=current_stage,
                )
                current_stage = "step4_validate_summary_no_flex"
                validations.append(
                    validate_powerflow_db(
                        repo_root,
                        powerflow_filename,
                        summary_only=True,
                        pre_only=False,
                        run_name=summary_no_flex_run_name,
                        expected_summary_stages=("pre", "post"),
                    )
                )

            with log_file.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{utc_now()}] STEP4 NO-FLEX VALIDATION OK\n")
                log_handle.write(
                    json.dumps(validations, indent=2, sort_keys=True, default=str)
                    + "\n"
                )

            seconds = round(time.monotonic() - started, 1)
            status.update(
                candidate_index,
                status="done",
                stage="complete",
                finished_at=utc_now(),
                seconds=seconds,
                message="ok",
            )
            if args.cleanup_intermediates == "success":
                cleanup_candidate_intermediates(
                    repo_root, candidate, args, status, reason="success"
                )
            return {
                "candidate_index": candidate_index,
                "status": "done",
                "seconds": seconds,
            }

        shutil.copy2(step2_output, step3_dir / "Input" / step2_filename)

        step3_cpus, cluster_concurrency, step3_stats = choose_step3_settings(
            step2_output, args
        )
        status.update(
            candidate_index,
            step3_cpus=step3_cpus,
            urbs_cluster_concurrency=cluster_concurrency,
            message=json.dumps(step3_stats, sort_keys=True),
        )

        current_stage = "step3_urbs"
        step3_cmd = [
            "uv",
            "run",
            "python",
            "run_urbs_cluster.py",
            step2_filename,
            "--n_cpu",
            str(step3_cpus),
        ]
        step3_cmd.extend(["--scenario-config", str(args.scenario_config)])
        run_command(
            cmd=step3_cmd,
            cwd=step3_dir,
            log_path=log_file,
            status=status,
            candidate_index=candidate_index,
            stage=current_stage,
            env_extra={"URBS_CLUSTER_CONCURRENCY": str(cluster_concurrency)},
        )
        step3_output = step3_dir / "result" / scenario_filename
        if not step3_output.exists():
            raise FileNotFoundError(f"Missing Step 3 output {step3_output}")
        shutil.copy2(step3_output, step4_dir / "Input" / scenario_filename)

        validations = []
        if args.powerflow_output in {"raw", "both"}:
            current_stage = "step4_powerflow_raw"
            raw_run_name = powerflow_run_name(args, "raw")
            raw_cmd = [
                "uv",
                "run",
                "python",
                "run_pwrflw.py",
                scenario_filename,
                "--storage",
                "db",
                "--run-name",
                raw_run_name,
                "--n_cpu",
                str(args.step4_cpus),
                "--pylovo-version-id",
                str(args.pylovo_version_id),
            ]
            if args.demand_scope == "residential":
                raw_cmd.append("--hh-only")
            run_command(
                cmd=raw_cmd,
                cwd=step4_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
            )

            current_stage = "step4_validate_raw"
            validations.append(
                validate_powerflow_db(
                    repo_root,
                    scenario_filename,
                    summary_only=False,
                    pre_only=False,
                    run_name=raw_run_name,
                )
            )

        if args.include_no_flex_powerflow and args.powerflow_output in {"raw", "both"}:
            current_stage = "step4_powerflow_raw_no_flex"
            raw_no_flex_run_name = powerflow_run_name(args, "raw_no_flex")
            raw_no_flex_cmd = [
                "uv",
                "run",
                "python",
                "run_pwrflw.py",
                scenario_filename,
                "--storage",
                "db",
                "--run-name",
                raw_no_flex_run_name,
                "--post-demand-mode",
                "no-flex",
                "--n_cpu",
                str(args.step4_cpus),
                "--pylovo-version-id",
                str(args.pylovo_version_id),
            ]
            if args.no_flex_ev_charger_kw is not None:
                raw_no_flex_cmd.extend(
                    ["--no-flex-ev-charger-kw", str(args.no_flex_ev_charger_kw)]
                )
            if args.demand_scope == "residential":
                raw_no_flex_cmd.append("--hh-only")
            run_command(
                cmd=raw_no_flex_cmd,
                cwd=step4_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
            )

            current_stage = "step4_validate_raw_no_flex"
            validations.append(
                validate_powerflow_db(
                    repo_root,
                    scenario_filename,
                    summary_only=False,
                    pre_only=False,
                    run_name=raw_no_flex_run_name,
                )
            )

        if args.powerflow_output in {"summary", "both"}:
            current_stage = "step4_powerflow_summary"
            summary_pre_only = args.profiles == "status_quo"
            expected_summary_stages = ("pre",) if summary_pre_only else ("pre", "post")
            summary_run_name = powerflow_run_name(args, "summary")
            summary_cmd = [
                "uv",
                "run",
                "python",
                "run_pwrflw.py",
                scenario_filename,
                "--storage",
                "db",
                "--summary-only",
                "--run-name",
                summary_run_name,
                "--n_cpu",
                str(args.step4_cpus),
                "--pylovo-version-id",
                str(args.pylovo_version_id),
            ]
            if summary_pre_only:
                summary_cmd.insert(summary_cmd.index("--summary-only"), "--pre-only")
            if args.demand_scope == "residential":
                summary_cmd.append("--hh-only")
            run_command(
                cmd=summary_cmd,
                cwd=step4_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
            )

            current_stage = "step4_validate_summary"
            validations.append(
                validate_powerflow_db(
                    repo_root,
                    scenario_filename,
                    summary_only=True,
                    pre_only=summary_pre_only,
                    run_name=summary_run_name,
                    expected_summary_stages=expected_summary_stages,
                )
            )

        if args.include_no_flex_powerflow and args.powerflow_output in {
            "summary",
            "both",
        }:
            current_stage = "step4_powerflow_summary_no_flex"
            summary_no_flex_run_name = powerflow_run_name(args, "summary_no_flex")
            summary_no_flex_cmd = [
                "uv",
                "run",
                "python",
                "run_pwrflw.py",
                scenario_filename,
                "--storage",
                "db",
                "--summary-only",
                "--run-name",
                summary_no_flex_run_name,
                "--post-demand-mode",
                "no-flex",
                "--n_cpu",
                str(args.step4_cpus),
                "--pylovo-version-id",
                str(args.pylovo_version_id),
            ]
            if args.no_flex_ev_charger_kw is not None:
                summary_no_flex_cmd.extend(
                    ["--no-flex-ev-charger-kw", str(args.no_flex_ev_charger_kw)]
                )
            if args.demand_scope == "residential":
                summary_no_flex_cmd.append("--hh-only")
            run_command(
                cmd=summary_no_flex_cmd,
                cwd=step4_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
            )

            current_stage = "step4_validate_summary_no_flex"
            validations.append(
                validate_powerflow_db(
                    repo_root,
                    scenario_filename,
                    summary_only=True,
                    pre_only=False,
                    run_name=summary_no_flex_run_name,
                    expected_summary_stages=("pre", "post"),
                )
            )

        with log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] STEP4 VALIDATION OK\n")
            log_handle.write(
                json.dumps(validations, indent=2, sort_keys=True, default=str) + "\n"
            )

        seconds = round(time.monotonic() - started, 1)
        status.update(
            candidate_index,
            status="done",
            stage="complete",
            finished_at=utc_now(),
            seconds=seconds,
            message="ok",
        )
        if args.cleanup_intermediates == "success":
            cleanup_candidate_intermediates(
                repo_root, candidate, args, status, reason="success"
            )
        return {
            "candidate_index": candidate_index,
            "status": "done",
            "seconds": seconds,
        }
    except Exception as exc:
        seconds = round(time.monotonic() - started, 1)
        with log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] FAILURE in {current_stage}: {exc}\n")
            log_handle.write(traceback.format_exc())
        payload = candidate_failed_payload(
            candidate, current_stage, str(exc), seconds, log_file
        )
        status.update(
            candidate_index,
            status="failed",
            stage=current_stage,
            finished_at=utc_now(),
            seconds=seconds,
            message=str(exc),
        )
        status.failed_grid(**payload)
        status.event(event="candidate_failed", **payload)
        return {
            "candidate_index": candidate_index,
            "status": "failed",
            "seconds": seconds,
            "message": str(exc),
        }


def filter_candidates(
    candidates: list[dict[str, object]], args: argparse.Namespace, status: StatusLog
) -> list[dict[str, object]]:
    selected = candidates
    if args.start_index is not None:
        selected = [
            candidate
            for candidate in selected
            if int(candidate["candidate_index"]) >= args.start_index
        ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not args.resume:
        return selected

    completed = completed_candidate_indexes(status)
    runnable = []
    for candidate in selected:
        candidate_index = int(candidate["candidate_index"])
        previous = status.status_for(candidate_index)
        if previous == "done" or candidate_index in completed:
            status.event(
                event="candidate_skipped_resume_done", candidate_index=candidate_index
            )
            continue
        if previous == "failed" and not args.rerun_failed:
            status.event(
                event="candidate_skipped_resume_failed", candidate_index=candidate_index
            )
            continue
        runnable.append(candidate)
    return runnable


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    args.run_dir = run_dir
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    status = StatusLog(run_dir, resume=args.resume or args.cleanup_completed_only)

    started_wall = time.monotonic()
    status.event(
        event="batch_start",
        repo_root=str(repo_root),
        ags=args.ags,
        pylovo_version_id=args.pylovo_version_id,
        min_buildings=args.min_buildings,
        workers=args.workers,
        step2_cpus=args.step2_cpus,
        step2_timeseries_storage=args.step2_timeseries_storage,
        profiles=args.profiles,
        demand_scope=args.demand_scope,
        timeframe_mode=args.timeframe_mode,
        step3_cpus=args.step3_cpus,
        step3_max_cpus=args.step3_max_cpus,
        step3_cluster_concurrency=args.step3_cluster_concurrency,
        step4_cpus=args.step4_cpus,
        powerflow_output=args.powerflow_output,
        materialize_expansion=not args.no_materialize_expansion
        and args.powerflow_output in {"summary", "both"},
        expansion_analysis_prefix=expansion_analysis_prefix(args),
        run_dir=str(run_dir),
        resume=args.resume,
        rerun_failed=args.rerun_failed,
        cleanup_intermediates=args.cleanup_intermediates,
        cleanup_completed_only=args.cleanup_completed_only,
    )

    candidates = get_candidates(
        repo_root,
        args.ags,
        args.min_buildings,
        args.demand_scope,
        args.pylovo_version_id,
    )
    (run_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    status.event(event="candidates_loaded", count=len(candidates))
    if not candidates:
        status.event(
            event="batch_finish", status="failed", message="No candidates found"
        )
        return 1

    if args.cleanup_completed_only:
        summary = cleanup_completed_intermediates(repo_root, candidates, args, status)
        (run_dir / "cleanup_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return 0

    candidates = filter_candidates(candidates, args, status)
    status.event(event="candidates_selected", count=len(candidates))
    if not candidates:
        summary = {
            "status": "done",
            "candidate_count": 0,
            "failure_count": 0,
            "total_seconds": round(time.monotonic() - started_wall, 1),
            "finished_at": utc_now(),
            "message": "No runnable candidates after filtering/resume.",
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        status.event(event="batch_finish", **summary)
        return 0

    by_index = {
        int(candidate["candidate_index"]): candidate for candidate in candidates
    }
    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    pilot_candidate = by_index.get(args.pilot_index)
    if pilot_candidate is not None:
        status.event(event="pilot_start", candidate_index=args.pilot_index)
        pilot_result = run_candidate(
            repo_root=repo_root,
            ags=args.ags,
            candidate=pilot_candidate,
            args=args,
            status=status,
        )
        status.event(event="pilot_finish", **pilot_result)
        if pilot_result["status"] == "done":
            completed.append(pilot_result)
        else:
            failures.append(pilot_result)
            if not args.no_pilot_gate:
                total_seconds = round(time.monotonic() - started_wall, 1)
                summary = {
                    "status": "failed",
                    "failed_at": "pilot",
                    "candidate_count": len(candidates),
                    "failure_count": len(failures),
                    "failures": failures,
                    "total_seconds": total_seconds,
                    "finished_at": utc_now(),
                }
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, default=str), encoding="utf-8"
                )
                status.event(event="batch_finish", **summary)
                return 1

    remaining = [
        candidate
        for candidate in candidates
        if int(candidate["candidate_index"]) != args.pilot_index
    ]
    status.event(
        event="batch_workers_start", remaining=len(remaining), workers=args.workers
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {
            executor.submit(
                run_candidate,
                repo_root=repo_root,
                ags=args.ags,
                candidate=candidate,
                args=args,
                status=status,
            ): int(candidate["candidate_index"])
            for candidate in remaining
        }
        for future in as_completed(future_map):
            candidate_index = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "candidate_index": candidate_index,
                    "status": "failed",
                    "message": str(exc),
                }
                status.event(
                    event="candidate_failed_unhandled",
                    candidate_index=candidate_index,
                    message=str(exc),
                )
            if result.get("status") == "done":
                completed.append(result)
                status.event(event="candidate_done", **result)
            else:
                failures.append(result)
                status.event(event="candidate_failed_recorded", **result)

    materialized_expansion = []
    expansion_failure = None
    try:
        materialized_expansion = materialize_expansion_analyses(
            repo_root=repo_root, args=args, status=status
        )
    except Exception as exc:
        expansion_failure = str(exc)
        status.event(
            event="expansion_materialization_failed", message=expansion_failure
        )

    total_seconds = round(time.monotonic() - started_wall, 1)
    batch_status = "done" if not failures else "completed_with_failures"
    if expansion_failure and batch_status == "done":
        batch_status = "completed_with_expansion_failure"
    summary = {
        "status": batch_status,
        "candidate_count": len(candidates),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "materialized_expansion": materialized_expansion,
        "expansion_failure": expansion_failure,
        "total_seconds": total_seconds,
        "finished_at": utc_now(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    status.event(event="batch_finish", **summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
