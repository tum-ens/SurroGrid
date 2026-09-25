#!/usr/bin/env python3
"""Run the DB-backed synthetic pipeline (Steps 2-4 + expansion) for the grids of one AGS.

``gridexpand synthetic`` runs one model case over the candidate grids of an
AGS: it prepares the regional electrification assignment once, then runs
Step 2, Step 3 (post cases) and Step 4 per grid in a worker pool, validates
the Step 4 rows in the database and materializes the regional expansion
analyses. Each grid keeps a log; failed grids are recorded without aborting
the batch; ``--resume`` skips grids already done in the same run directory.
``gridexpand run`` executes the ``synthetic`` pipeline of a run YAML through
:func:`run_batch`, the same function.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from sqlalchemy import text

from gridexpand.common.electrification import (
    assignment_manifest_hash,
    validate_electrification_assignment_config,
)
from gridexpand.common.reproducibility import DEFAULT_PROFILE_SEED
from gridexpand.common.orchestration import (
    CANCEL,
    Cancelled,
    StatusLog,
    install_cancel_handlers,
    run_batch_command,
    run_command,
    utc_now,
)
from gridexpand.common.timeframe import (
    TIMEFRAME_MODES,
    build_initial_metadata,
    horizon_hours_from_hdf,
    output_filename_for_timeframe,
    read_hdf_metadata,
    scenario_key_for_timeframe,
    scenario_output_directory,
)
from gridexpand.db import SurroGridDatabase
from gridexpand.optimization.solver import SOLVER_ENV, SUPPORTED_SOLVERS
from gridexpand.paths import (
    ALLOCATION_RESULTS_DIR,
    OPTIMIZATION_INPUT_DIR,
    OPTIMIZATION_RESULT_DIR,
    POWERFLOW_INPUT_DIR,
    PROJECT_DIR,
    WORK_DIR,
    ensure_dir,
)
from gridexpand.scenario import commands
from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key
from gridexpand.scenario.model_cases import MODEL_CASES, SYNTHETIC_UNSUPPORTED_CASES, get_model_case
from gridexpand.scenario.scenario_config import ScenarioConfig

EXPECTED_POWERFLOW_TABLES = {
    "powerflow_demand": ("pre", "post"),
    "powerflow_import": ("pre", "post"),
    "powerflow_bus_voltage": ("pre", "post"),
    "powerflow_line_result": ("pre", "post"),
}
SUMMARY_TABLES = ("powerflow_summary", "powerflow_cable_summary", "powerflow_bus_voltage_summary")

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
GRID_SCOPE_CHOICES = ("full", "backbone")
MOBILITY_SOURCE = "pool"
EXIT_CANCELLED = 143

SYNTHETIC_INFLEX_UNSUPPORTED = (
    SYNTHETIC_UNSUPPORTED_CASES["post-inflex-heuristic"]
    + " Run post-hems-heuristic without INFLEX, or use a paired pipeline."
)

_BATCH_IDENTITY_FILE = "batch_identity.json"


def run_name_profile_token(profile: str) -> str:
    return "post_electrification" if profile == "all" else profile


@dataclass(frozen=True)
class BatchSettings:
    """Everything one synthetic batch needs (one AGS, one model case).

    Built from the ``gridexpand synthetic`` flags (:func:`settings_from_args`)
    or from a ``pipeline: synthetic`` run YAML (``gridexpand run``). Field names
    follow the CLI flags.
    """

    ags: str
    pylovo_version_id: str
    scenario_config: Path
    scenario: ScenarioConfig
    scenario_hash: str
    run_dir: Path
    model_case: str = "post-hems-optimized"
    profiles: str = "all"
    min_buildings: int = 5
    demand_scope: str = "all"
    timeframe_mode: str = "full_year"
    plz: int | None = None
    kcid: int | None = None
    bcid: int | None = None
    start_index: int | None = None
    limit: int | None = None
    workers: int = 1
    step2_cpus: int = 4
    step2_timeseries_storage: str = "temp"
    step3_cpus: int = 16
    step3_max_cpus: int = 32
    step3_target_columns: int = 35
    step3_cluster_concurrency: int = 1
    dynamic_step3: bool = True
    step4_cpus: int = 4
    solver: str | None = None
    powerflow_output: str = "raw"
    powerflow_grid_scope: str = "full"
    case_qualified_output: bool = False
    profile_seed: int = DEFAULT_PROFILE_SEED
    electrification_assignment: Path | None = None
    pilot_index: int = 0
    pilot_gate: bool = True
    resume: bool = False
    rerun_failed: bool = False
    cleanup_intermediates: str = "never"
    materialize_expansion: bool = True
    include_inflex_powerflow: bool = False
    inflex_only: bool = False
    inflex_ev_charger_kw: float | None = None
    expansion_analysis_prefix: str | None = None

    @property
    def tsam(self) -> bool:
        return bool(self.scenario.time_aggregation.enabled)

    @property
    def assignment_path(self) -> Path:
        """Regional electrification assignment (default ``<run_dir>/electrification_assignment.csv``)."""
        return self.electrification_assignment or self.run_dir / "electrification_assignment.csv"

    @property
    def summary_output(self) -> bool:
        return self.powerflow_output in {"summary", "both"}

    def replace(self, **changes: Any) -> "BatchSettings":
        return dataclasses.replace(self, **changes)


def check_settings(settings: BatchSettings) -> None:
    """Reject inconsistent flag combinations (the same checks for CLI and run YAML).

    Raises:
        ValueError: with the message of the first violated rule.
    """
    case = get_model_case(settings.model_case)
    if case.profiles == "status_quo" and settings.profiles != "status_quo":
        raise ValueError("The pre model case requires --profiles status_quo.")
    if case.profiles != "status_quo" and settings.profiles == "status_quo":
        raise ValueError("Post model cases require post-electrification profiles.")
    if settings.include_inflex_powerflow and settings.inflex_only:
        raise ValueError("Use either --inflex-only or --include-inflex-powerflow, not both.")
    if settings.include_inflex_powerflow or settings.inflex_only:
        raise ValueError(SYNTHETIC_INFLEX_UNSUPPORTED)
    if settings.inflex_ev_charger_kw is not None:
        raise ValueError("--inflex-ev-charger-kw requires --include-inflex-powerflow or --inflex-only.")
    if (settings.kcid is None) != (settings.bcid is None):
        raise ValueError("--kcid and --bcid must be given together.")
    if settings.kcid is not None and settings.plz is None:
        raise ValueError("--plz is required with --kcid/--bcid.")
    for name in ("workers", "step2_cpus", "step3_cpus", "step3_max_cpus", "step3_target_columns",
                 "step3_cluster_concurrency", "step4_cpus"):
        if int(getattr(settings, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand synthetic",
        description="Run the DB-backed synthetic GridExpand pipeline for the grids of one AGS.",
    )
    parser.add_argument("--repo-root", type=Path, help="Deprecated and ignored; directories come from gridexpand.paths.")
    parser.add_argument("--ags", required=True, help="AGS identifier for the region to process, e.g. 09162000.")
    parser.add_argument("--pylovo-version-id", required=True,
                        help="Exact pylovo topology version; supplied by the run configuration.")
    parser.add_argument("--plz", type=int, help="Only the candidate grids of this PLZ (also the assignment scope).")
    parser.add_argument("--kcid", type=int, help="With --plz/--bcid: one grid only (also the assignment scope).")
    parser.add_argument("--bcid", type=int, help="With --plz/--kcid: one grid only.")
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Grids processed in parallel.")
    parser.add_argument("--step2-cpus", type=int, default=4)
    parser.add_argument("--step3-cpus", type=int, default=16,
                        help="Minimum number of Step 3 building clusters (partitions).")
    parser.add_argument("--step3-max-cpus", type=int, default=32)
    parser.add_argument("--step3-target-columns", type=int, default=35)
    parser.add_argument("--step3-cluster-concurrency", type=int, default=1,
                        help="Step 3 clusters solved at the same time.")
    parser.add_argument("--step4-cpus", type=int, default=4)
    parser.add_argument("--solver", choices=SUPPORTED_SOLVERS, default=None,
                        help="Step 3 solver (default: $GRIDEXPAND_SOLVER, else gurobi).")
    parser.add_argument("--scenario-config", type=Path, required=True, help="Scenario YAML (config/scenarios).")
    parser.add_argument(
        "--electrification-assignment",
        type=Path,
        help=(
            "Precomputed regional assignment manifest. If omitted, the runner "
            "prepares one before starting candidate workers."
        ),
    )
    parser.add_argument("--profile-seed", type=int, default=DEFAULT_PROFILE_SEED,
                        help="Run-level seed for the physical stochastic profile realization.")
    parser.add_argument(
        "--model-case",
        choices=tuple(MODEL_CASES),
        default="post-hems-optimized",
        help="Scenario case controlling upstream asset sizing and downstream dispatch.",
    )
    parser.add_argument(
        "--case-qualified-output",
        action="store_true",
        help=(
            "Append the model-case name to Step 2-4 HDF5 filenames, power-flow run "
            "names and expansion analysis keys."
        ),
    )
    parser.add_argument(
        "--powerflow-output",
        choices=POWERFLOW_OUTPUT_CHOICES,
        default="raw",
        help=(
            "Power-flow output mode. 'raw' stores full pre/post time series, "
            "'summary' stores compact notebook metrics, and 'both' stores both "
            "(from one power-flow pass). For electrification profiles the compact "
            "summary includes post results."
        ),
    )
    parser.add_argument("--powerflow-grid-scope", choices=GRID_SCOPE_CHOICES, default="full",
                        help="Assets of the summary statistics (Step 4 --summary-grid-scope).")
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
    parser.add_argument("--step2-timeseries-storage", choices=["db", "temp", "both"], default="temp")
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
        "--include-inflex-powerflow",
        action="store_true",
        help="Additional INFLEX power flow after Step 3 (not available for synthetic grids, see review-optpf B3).",
    )
    parser.add_argument(
        "--inflex-only",
        action="store_true",
        help="Step 3 for post-flex capacities, then only the INFLEX power flow (not available, see above).",
    )
    parser.add_argument("--inflex-ev-charger-kw", type=float,
                        help="Optional EV charger cap passed to Step 4 --post-demand-mode inflex.")
    parser.add_argument(
        "--expansion-analysis-prefix",
        help=(
            "Optional prefix for automatic expansion analysis keys. Defaults to "
            "'<ags:08d>_<timeframe_mode>_<profiles>[_hh_only][_tsam][_<model_case>]' "
            "(the model case with --case-qualified-output)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the flags (no file or database access)."""
    return build_parser().parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> BatchSettings:
    """Load the scenario YAML and turn parsed flags into :class:`BatchSettings`."""
    scenario_config = args.scenario_config.resolve()
    scenario, scenario_hash = load_scenario_config(scenario_config)
    return BatchSettings(
        ags=str(args.ags),
        pylovo_version_id=str(args.pylovo_version_id),
        scenario_config=scenario_config,
        scenario=scenario,
        scenario_hash=scenario_hash,
        run_dir=args.run_dir.resolve(),
        model_case=args.model_case,
        profiles=args.profiles,
        min_buildings=args.min_buildings,
        demand_scope=args.demand_scope,
        timeframe_mode=args.timeframe_mode,
        plz=args.plz,
        kcid=args.kcid,
        bcid=args.bcid,
        start_index=args.start_index,
        limit=args.limit,
        workers=args.workers,
        step2_cpus=args.step2_cpus,
        step2_timeseries_storage=args.step2_timeseries_storage,
        step3_cpus=args.step3_cpus,
        step3_max_cpus=args.step3_max_cpus,
        step3_target_columns=args.step3_target_columns,
        step3_cluster_concurrency=args.step3_cluster_concurrency,
        dynamic_step3=not args.no_dynamic_step3,
        step4_cpus=args.step4_cpus,
        solver=args.solver,
        powerflow_output=args.powerflow_output,
        powerflow_grid_scope=args.powerflow_grid_scope,
        case_qualified_output=args.case_qualified_output,
        profile_seed=args.profile_seed,
        electrification_assignment=(
            args.electrification_assignment.resolve() if args.electrification_assignment is not None else None
        ),
        pilot_index=args.pilot_index,
        pilot_gate=not args.no_pilot_gate,
        resume=args.resume or args.cleanup_completed_only,
        rerun_failed=args.rerun_failed,
        cleanup_intermediates=args.cleanup_intermediates,
        materialize_expansion=not args.no_materialize_expansion,
        include_inflex_powerflow=args.include_inflex_powerflow,
        inflex_only=args.inflex_only,
        inflex_ev_charger_kw=args.inflex_ev_charger_kw,
        expansion_analysis_prefix=args.expansion_analysis_prefix,
    )


# Candidates and names ---------------------------------------------------------------


def get_candidates(
    ags: str,
    min_buildings: int,
    demand_scope: str = "all",
    pylovo_version_id: str | None = None,
) -> list[dict[str, object]]:
    """Candidate grids of an AGS (see :func:`gridexpand.db.grids.list_grid_candidates`)."""
    db = SurroGridDatabase()
    db.pylovo_version_id = pylovo_version_id
    return db.list_grid_candidates(ags, min_buildings=min_buildings, demand_scope=demand_scope)


def select_region(
    candidates: list[dict[str, object]],
    *,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
) -> list[dict[str, object]]:
    """Candidates of one PLZ or one ``(plz, kcid, bcid)`` grid; numbering is kept."""
    selected = candidates
    if plz is not None:
        selected = [c for c in selected if int(c["plz"]) == int(plz)]
    if kcid is not None:
        selected = [c for c in selected if (int(c["kcid"]), int(c["bcid"])) == (int(kcid), int(bcid))]
    return selected


def load_candidates(settings: BatchSettings) -> list[dict[str, object]]:
    """Candidate grids of the batch region (AGS, optionally one PLZ or one grid)."""
    candidates = select_region(
        get_candidates(settings.ags, settings.min_buildings, settings.demand_scope, settings.pylovo_version_id),
        plz=settings.plz, kcid=settings.kcid, bcid=settings.bcid,
    )
    return candidates


def job_key(candidate: dict[str, object]) -> str:
    """Stable identity of a candidate grid: its bridge stem ``<ags>-<idx>_<plz>_<kcid>_<bcid>``."""
    return Path(str(candidate["bridge_filename"])).stem


def pipeline_scenario_key(settings: BatchSettings) -> str:
    return scenario_key_for_timeframe(
        settings.timeframe_mode,
        base_key=scenario_identity_key(settings.scenario.scenario_id, settings.scenario_hash),
    )


def powerflow_run_name(settings: BatchSettings, mode: str) -> str:
    case = f"_{settings.model_case}" if settings.case_qualified_output else ""
    return f"{pipeline_scenario_key(settings)}_{run_name_profile_token(settings.profiles)}{case}_{mode}_powerflow"


def case_qualified_filename(filename: str, settings: BatchSettings) -> str:
    if not settings.case_qualified_output:
        return filename
    path = Path(filename)
    return f"{path.stem}_{settings.model_case}{path.suffix}"


def step2_filename(candidate: dict[str, object], settings: BatchSettings) -> str:
    """Step 2 output name of a candidate (timeframe- and, if requested, case-qualified)."""
    return case_qualified_filename(
        output_filename_for_timeframe(str(candidate["bridge_filename"]), settings.timeframe_mode), settings
    )


def expansion_analysis_prefix(settings: BatchSettings) -> str:
    """Prefix of the batch's expansion analysis keys.

    ``<ags:08d>_<timeframe>_<profiles>[_hh_only][_tsam][_<model_case>]``. The
    AGS keeps two regions apart (review-post B7) and, with
    ``case_qualified_output``, the model case keeps the heuristic and optimized
    batches apart (review-orch B2); before, a later batch replaced the earlier
    one's analyses (``grid_expansion --replace`` deletes by key).
    """
    if settings.expansion_analysis_prefix:
        return settings.expansion_analysis_prefix
    scope_suffix = "_hh_only" if settings.demand_scope == "residential" else ""
    tsam_suffix = "_tsam" if settings.tsam else ""
    case_suffix = f"_{settings.model_case}" if settings.case_qualified_output else ""
    ags = str(int(str(settings.ags).strip() or "0")).zfill(8)
    return (
        f"{ags}_{settings.timeframe_mode}_{run_name_profile_token(settings.profiles)}"
        f"{scope_suffix}{tsam_suffix}{case_suffix}"
    )


# Step 4 passes and validation ------------------------------------------------------------


@dataclass(frozen=True)
class PowerflowPass:
    """One Step 4 call: one demand reconstruction, raw and/or summary outputs.

    Attributes:
        mode: ``pre_only`` (status quo), ``flexible`` or ``inflex``.
        outputs: ``("raw",)``, ``("summary",)`` or ``("raw", "summary")``.
    """

    mode: str
    outputs: tuple[str, ...]

    @property
    def pre_only(self) -> bool:
        return self.mode == "pre_only"

    @property
    def inflex(self) -> bool:
        return self.mode == "inflex"

    @property
    def suffix(self) -> str:
        return {"pre_only": "_pre_only", "flexible": "", "inflex": "_inflex"}[self.mode]

    @property
    def stage(self) -> str:
        return f"step4_powerflow_{'_'.join(self.outputs)}{self.suffix}"

    def run_token(self, output: str) -> str:
        """Run-name mode of one output: raw, summary, raw_inflex or summary_inflex."""
        return f"{output}_inflex" if self.inflex else output

    @property
    def expected_summary_stages(self) -> tuple[str, ...]:
        return ("pre",) if self.pre_only else ("pre", "post")


def powerflow_outputs(powerflow_output: str) -> tuple[str, ...]:
    return {"raw": ("raw",), "summary": ("summary",), "both": ("raw", "summary")}[powerflow_output]


def powerflow_passes(settings: BatchSettings) -> list[PowerflowPass]:
    """The Step 4 calls of one candidate, in execution order."""
    outputs = powerflow_outputs(settings.powerflow_output)
    if settings.profiles == "status_quo":
        return [PowerflowPass("pre_only", outputs)]
    if settings.inflex_only:
        return [PowerflowPass("inflex", outputs)]
    passes = [PowerflowPass("flexible", outputs)]
    if settings.include_inflex_powerflow:
        passes.append(PowerflowPass("inflex", outputs))
    return passes


def powerflow_pass_command(settings: BatchSettings, input_name: str, powerflow_pass: PowerflowPass) -> list[str]:
    raw = "raw" in powerflow_pass.outputs
    summary = "summary" in powerflow_pass.outputs
    first = powerflow_pass.run_token(powerflow_pass.outputs[0])
    return commands.powerflow_command(
        input_name,
        n_cpu=settings.step4_cpus,
        run_name=powerflow_run_name(settings, first),
        outputs=powerflow_pass.outputs,
        summary_run_name=(
            powerflow_run_name(settings, powerflow_pass.run_token("summary")) if raw and summary else None
        ),
        pre_only=powerflow_pass.pre_only,
        post_demand_mode="inflex" if powerflow_pass.inflex else None,
        inflex_ev_charger_kw=settings.inflex_ev_charger_kw if powerflow_pass.inflex else None,
        pylovo_version_id=settings.pylovo_version_id,
        hh_only=settings.demand_scope == "residential",
        summary_grid_scope=None if settings.powerflow_grid_scope == "full" else settings.powerflow_grid_scope,
    )


def _stage_counts(conn, table_name: str, run_id: int, *, with_time: bool) -> dict[str, dict[str, Any]]:
    columns = "count(*) AS rows, min(t_index) AS min_t, max(t_index) AS max_t" if with_time else "count(*) AS rows"
    rows = conn.execute(
        text(
            f"SELECT stage, {columns} FROM surrogrid.{table_name} "
            "WHERE powerflow_run_id = :run_id GROUP BY stage ORDER BY stage"
        ),
        {"run_id": run_id},
    ).mappings().all()
    return {str(row["stage"]): dict(row) for row in rows}


def validate_powerflow_db(
    scenario_filename: str,
    *,
    summary_only: bool = False,
    pre_only: bool = False,
    run_name: str | None = None,
    expected_summary_stages: tuple[str, ...] = ("pre",),
) -> dict[str, Any]:
    """Check that a Step 4 run wrote complete rows (summary or raw tables).

    Raises:
        RuntimeError: no matching run, or missing/incomplete stages.
    """
    scenario_path = POWERFLOW_INPUT_DIR / scenario_filename
    with pd.HDFStore(scenario_path, mode="r") as store:
        if "/urbs_out/MILP/tau_pro" in store:
            tau_pro = store["/urbs_out/MILP/tau_pro"]
            time_level = "t" if "t" in tau_pro.index.names else 0
            expected_horizon = tau_pro.index.get_level_values(time_level).nunique()
        else:
            expected_horizon = horizon_hours_from_hdf(scenario_path)
    expected_max_t = expected_horizon - 1
    db = SurroGridDatabase()
    mode = "summary" if summary_only else "raw"
    run = db.find_powerflow_run(run_name=run_name, urbs_input_file=scenario_filename, pre_only=pre_only)
    if run is None:
        raise RuntimeError(f"No {mode} powerflow_run found for {scenario_filename}")
    run_id = int(run["powerflow_run_id"])
    validation: dict[str, Any] = {
        "powerflow_run_id": run_id,
        "run_name": run["run_name"],
        "mode": mode,
        "tables": {},
        "expected_horizon_hours": expected_horizon,
    }
    missing: list[str] = []
    with db.engine.connect() as conn:
        if summary_only:
            for table_name in SUMMARY_TABLES:
                by_stage = _stage_counts(conn, table_name, run_id, with_time=False)
                validation["tables"][table_name] = by_stage
                for stage in expected_summary_stages:
                    row = by_stage.get(stage)
                    if not row or int(row["rows"]) <= 0:
                        missing.append(f"{table_name}:{stage}:missing")
            if missing:
                raise RuntimeError("Incomplete Step 4 DB summary results: " + ", ".join(missing))
            return validation

        for table_name, expected_stages in EXPECTED_POWERFLOW_TABLES.items():
            if pre_only:
                expected_stages = ("pre",)
            by_stage = _stage_counts(conn, table_name, run_id, with_time=True)
            validation["tables"][table_name] = by_stage
            for stage in expected_stages:
                row = by_stage.get(stage)
                if not row:
                    missing.append(f"{table_name}:{stage}:missing")
                elif int(row["rows"]) <= 0 or int(row["min_t"]) != 0 or int(row["max_t"]) != expected_max_t:
                    missing.append(f"{table_name}:{stage}:incomplete")

        if not pre_only:
            reactive_rows = conn.execute(
                text(
                    "SELECT count(*) AS rows, min(t_index) AS min_t, max(t_index) AS max_t "
                    "FROM surrogrid.powerflow_reactive_component WHERE powerflow_run_id = :run_id"
                ),
                {"run_id": run_id},
            ).mappings().one()
            validation["tables"]["powerflow_reactive_component"] = {"all": dict(reactive_rows)}
            if (
                int(reactive_rows["rows"]) <= 0
                or int(reactive_rows["min_t"]) != 0
                or int(reactive_rows["max_t"]) != expected_max_t
            ):
                missing.append("powerflow_reactive_component:all:incomplete")

    if missing:
        raise RuntimeError("Incomplete Step 4 DB results: " + ", ".join(missing))
    return validation


# Step 3 settings ------------------------------------------------------------------------


def hdf_column_count(path: Path, key: str) -> int:
    group_name = key.strip("/")
    with h5py.File(path, mode="r") as h5:
        if group_name not in h5:
            return 0
        group = h5[group_name]
        total = 0
        for name, node in group.items():
            if name.startswith("block") and name.endswith("_values") and len(node.shape) == 2:
                total += int(node.shape[1])
        return total


def scenario_suffix_from_hdf(path: Path) -> str:
    """Return the canonical scenario suffix stored by Step 2."""
    scenario_key = read_hdf_metadata(path).get("scenario_key")
    if not scenario_key:
        raise ValueError(f"Step-2 HDF is missing scenario_key metadata: {path}")
    return str(scenario_key)


def choose_step3_settings(step2_output: Path, settings: BatchSettings) -> tuple[int, int, dict[str, int]]:
    """Number of Step 3 building clusters and their concurrency for one Step 2 file.

    Picks the smallest of 4/8/12/16/24/32 clusters (at least ``step3_cpus``, at
    most ``step3_max_cpus``) that keeps each cluster near ``step3_target_columns``
    demand columns.
    """
    if not settings.dynamic_step3:
        return int(settings.step3_cpus), int(settings.step3_cluster_concurrency), {}

    stats: dict[str, int] = {}
    for key, name in (("urbs_in/demand", "demand_columns"), ("urbs_in/eff_factor", "eff_factor_columns")):
        try:
            stats[name] = hdf_column_count(step2_output, key)
        except Exception:
            stats[name] = 0
    largest_columns = max(stats.values() or [0])
    required = (
        math.ceil(largest_columns / max(1, int(settings.step3_target_columns)))
        if largest_columns
        else settings.step3_cpus
    )
    choices = [4, 8, 12, 16, 24, 32]
    max_cpus = max(1, int(settings.step3_max_cpus))
    choices = [value for value in choices if value <= max_cpus] or [max_cpus]
    selected = next((value for value in choices if value >= required), choices[-1])
    selected = max(selected, int(settings.step3_cpus))
    selected = min(selected, max_cpus)
    return selected, int(settings.step3_cluster_concurrency), stats


# One candidate ----------------------------------------------------------------------------


@dataclass
class CandidateContext:
    """State of one candidate while it runs."""

    settings: BatchSettings
    status: StatusLog
    candidate: dict[str, object]
    step2_filename: str
    log_file: Path
    started: float
    stage: str = "queued"

    @property
    def index(self) -> int:
        return int(self.candidate["candidate_index"])

    def command(self, cmd: list[str], stage: str) -> None:
        self.stage = stage
        run_command(cmd=cmd, log_path=self.log_file, status=self.status, job=self.index, stage=stage)

    def result(self, status: str, **extra: object) -> dict[str, object]:
        return {
            "candidate_index": self.index,
            "job": job_key(self.candidate),
            "status": status,
            "seconds": round(time.monotonic() - self.started, 1),
            **extra,
        }


def run_step2(ctx: CandidateContext) -> tuple[Path, str, str]:
    """Step 2 of one candidate; returns the Step 2 file, the scenario suffix and file name."""
    settings = ctx.settings
    ctx.command(
        commands.allocation_command(
            settings.ags,
            pylovo_version_id=settings.pylovo_version_id,
            candidate_index=ctx.index,
            min_buildings=settings.min_buildings,
            profiles=settings.profiles,
            demand_scope=settings.demand_scope,
            mobility_source=MOBILITY_SOURCE,
            timeseries_storage=settings.step2_timeseries_storage,
            timeframe_mode=settings.timeframe_mode,
            model_case=settings.model_case,
            profile_seed=settings.profile_seed,
            scenario_config=settings.scenario_config,
            electrification_assignment=settings.assignment_path,
            n_cpu=settings.step2_cpus,
            case_qualified_output=settings.case_qualified_output,
        ),
        "step2_demand_allocation",
    )
    step2_output = scenario_output_directory(ALLOCATION_RESULTS_DIR, pipeline_scenario_key(settings)) / ctx.step2_filename
    if not step2_output.exists():
        raise FileNotFoundError(f"Missing Step 2 output {step2_output}")
    timeframe_metadata = read_hdf_metadata(step2_output)
    scenario_suffix = scenario_suffix_from_hdf(step2_output)
    ctx.status.update(
        ctx.index,
        horizon_hours=timeframe_metadata.get("horizon_hours", ""),
        timeframe_start=timeframe_metadata.get("timeframe_start", ""),
        timeframe_end=timeframe_metadata.get("timeframe_end", ""),
        message=json.dumps({"scenario_suffix": scenario_suffix}, sort_keys=True),
    )
    return step2_output, scenario_suffix, ctx.step2_filename.replace(".h5", f"_{scenario_suffix}.h5")


def run_step3(ctx: CandidateContext, step2_output: Path, scenario_filename: str) -> None:
    """Step 3 of one candidate; copies its result to the Step 4 input directory."""
    settings = ctx.settings
    shutil.copy2(step2_output, ensure_dir(OPTIMIZATION_INPUT_DIR) / ctx.step2_filename)
    step3_cpus, cluster_concurrency, step3_stats = choose_step3_settings(step2_output, settings)
    if settings.inflex_only:
        step3_stats = {**step3_stats, "post_flex_capacity_source": "required_for_inflex"}
    ctx.status.update(
        ctx.index,
        step3_cpus=step3_cpus,
        urbs_cluster_concurrency=cluster_concurrency,
        message=json.dumps(step3_stats, sort_keys=True),
    )
    ctx.command(
        commands.optimization_command(
            ctx.step2_filename,
            n_cpu=step3_cpus,
            scenario_config=settings.scenario_config,
            cluster_concurrency=cluster_concurrency,
            solver=settings.solver,
        ),
        "step3_urbs_for_inflex" if settings.inflex_only else "step3_urbs",
    )
    step3_output = scenario_output_directory(OPTIMIZATION_RESULT_DIR, pipeline_scenario_key(settings)) / scenario_filename
    if not step3_output.exists():
        raise FileNotFoundError(f"Missing Step 3 output {step3_output}")
    shutil.copy2(step3_output, ensure_dir(POWERFLOW_INPUT_DIR) / scenario_filename)


def run_powerflow_pass(ctx: CandidateContext, input_name: str, powerflow_pass: PowerflowPass) -> list[dict[str, Any]]:
    """One Step 4 call and the DB validation of each of its outputs."""
    settings = ctx.settings
    ctx.command(powerflow_pass_command(settings, input_name, powerflow_pass), powerflow_pass.stage)
    validations = []
    for output in powerflow_pass.outputs:
        ctx.stage = f"step4_validate_{output}{powerflow_pass.suffix}"
        validations.append(
            validate_powerflow_db(
                input_name,
                summary_only=output == "summary",
                pre_only=powerflow_pass.pre_only,
                run_name=powerflow_run_name(settings, powerflow_pass.run_token(output)),
                expected_summary_stages=powerflow_pass.expected_summary_stages,
            )
        )
    return validations


def run_candidate(*, candidate: dict[str, object], settings: BatchSettings, status: StatusLog) -> dict[str, object]:
    """Steps 2-4 of one candidate grid; never raises (failures are recorded)."""
    filename = step2_filename(candidate, settings)
    index = int(candidate["candidate_index"])
    ctx = CandidateContext(
        settings=settings,
        status=status,
        candidate=candidate,
        step2_filename=filename,
        log_file=settings.run_dir / "logs" / f"candidate_{index:03d}_{filename}.log",
        started=time.monotonic(),
    )
    if CANCEL.is_set():
        return ctx.result("cancelled", message="cancelled before start")
    timeframe_metadata = build_initial_metadata(settings.timeframe_mode)
    status.update(
        index,
        ags=candidate.get("ags", settings.ags),
        plz=candidate.get("plz", ""),
        kcid=candidate.get("kcid", ""),
        bcid=candidate.get("bcid", ""),
        n_buildings=candidate.get("n_buildings", ""),
        bridge_filename=filename,
        demand_scope=settings.demand_scope,
        timeframe_mode=settings.timeframe_mode,
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
        log_file=ctx.log_file,
        message="",
    )
    try:
        step2_output, scenario_suffix, scenario_filename = run_step2(ctx)
        if settings.profiles == "status_quo":
            status.update(
                index,
                step3_cpus="skipped",
                urbs_cluster_concurrency="skipped",
                message=json.dumps({"scenario_suffix": scenario_suffix, "step3": "skipped_status_quo"}, sort_keys=True),
            )
            shutil.copy2(step2_output, ensure_dir(POWERFLOW_INPUT_DIR) / filename)
            powerflow_input = filename
            banner = "STEP4 STATUS-QUO VALIDATION OK"
        else:
            run_step3(ctx, step2_output, scenario_filename)
            powerflow_input = scenario_filename
            banner = "STEP4 NO-FLEX VALIDATION OK" if settings.inflex_only else "STEP4 VALIDATION OK"
        validations = [
            validation
            for powerflow_pass in powerflow_passes(settings)
            for validation in run_powerflow_pass(ctx, powerflow_input, powerflow_pass)
        ]
        with ctx.log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] {banner}\n")
            log_handle.write(json.dumps(validations, indent=2, sort_keys=True, default=str) + "\n")
        result = ctx.result("done")
        status.update(index, status="done", stage="complete", finished_at=utc_now(), seconds=result["seconds"],
                      message="ok")
        if settings.cleanup_intermediates == "success":
            cleanup_candidate_intermediates(candidate, settings, status, reason="success")
        return result
    except Cancelled as exc:
        result = ctx.result("cancelled", stage=ctx.stage, message=str(exc))
        with ctx.log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] CANCELLED in {ctx.stage}\n")
        status.update(index, status="cancelled", stage=ctx.stage, finished_at=utc_now(), seconds=result["seconds"],
                      message=str(exc))
        status.event(event="candidate_cancelled", **result)
        return result
    except Exception as exc:
        seconds = round(time.monotonic() - ctx.started, 1)
        with ctx.log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] FAILURE in {ctx.stage}: {exc}\n")
            log_handle.write(traceback.format_exc())
        payload = candidate_failed_payload(candidate, ctx.stage, str(exc), seconds, ctx.log_file)
        status.update(index, status="failed", stage=ctx.stage, finished_at=utc_now(), seconds=seconds,
                      message=str(exc))
        status.failed_grid(**payload)
        status.event(event="candidate_failed", **payload)
        return ctx.result("failed", message=str(exc))


def candidate_failed_payload(
    candidate: dict[str, object],
    stage: str,
    message: str,
    seconds: float,
    log_file: Path,
) -> dict[str, object]:
    return {
        "candidate_index": int(candidate["candidate_index"]),
        "job": job_key(candidate),
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


# Intermediates and resume ------------------------------------------------------------------


def candidate_intermediate_files(candidate: dict[str, object], settings: BatchSettings) -> list[Path]:
    filename = step2_filename(candidate, settings)
    stem = Path(filename).stem
    step2_results = scenario_output_directory(ALLOCATION_RESULTS_DIR, pipeline_scenario_key(settings))
    files = [step2_results / filename, OPTIMIZATION_INPUT_DIR / filename, POWERFLOW_INPUT_DIR / filename]
    files.extend(sorted(POWERFLOW_INPUT_DIR.glob(f"{stem}_*.h5")))
    seen: set[Path] = set()
    unique_files = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def cleanup_candidate_intermediates(
    candidate: dict[str, object],
    settings: BatchSettings,
    status: StatusLog,
    *,
    reason: str,
) -> dict[str, object]:
    removed_files = []
    removed_bytes = 0
    for file_path in candidate_intermediate_files(candidate, settings):
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


def _events(status: StatusLog) -> list[dict[str, Any]]:
    if not status.events_path.exists():
        return []
    events = []
    with status.events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def previous_status(
    candidates: list[dict[str, object]], settings: BatchSettings, status: StatusLog
) -> dict[int, str]:
    """``done``/``failed`` of earlier attempts, keyed by candidate index.

    A row counts only if it describes the same grid (its ``bridge_filename``
    equals the candidate's Step 2 file), so a changed candidate numbering never
    marks the wrong grid as done. Events count if they name the same job
    (older events without a job name count by index, as before).
    """
    by_index = {int(candidate["candidate_index"]): candidate for candidate in candidates}
    result: dict[int, str] = {}
    for index, row in status.rows.items():
        candidate = by_index.get(int(index))
        if candidate is None or str(row.get("bridge_filename") or "") != step2_filename(candidate, settings):
            continue
        if row.get("status") in {"done", "failed"}:
            result[int(index)] = str(row["status"])
    for event in _events(status):
        if event.get("event") not in {"candidate_done", "pilot_finish"} or event.get("status") != "done":
            continue
        index = event.get("candidate_index")
        if index is None or int(index) not in by_index:
            continue
        if event.get("job") not in (None, job_key(by_index[int(index)])):
            continue
        result[int(index)] = "done"
    return result


def cleanup_completed_intermediates(
    candidates: list[dict[str, object]],
    settings: BatchSettings,
    status: StatusLog,
) -> dict[str, object]:
    done = [index for index, state in previous_status(candidates, settings, status).items() if state == "done"]
    by_index = {int(candidate["candidate_index"]): candidate for candidate in candidates}
    cleanup_results = [
        cleanup_candidate_intermediates(by_index[index], settings, status, reason="completed_only")
        for index in sorted(done)
    ]
    summary = {
        "status": "done",
        "candidate_count": len(cleanup_results),
        "removed_files": sum(int(result["removed_files"]) for result in cleanup_results),
        "removed_bytes": sum(int(result["removed_bytes"]) for result in cleanup_results),
        "finished_at": utc_now(),
    }
    status.event(event="cleanup_completed_finish", **summary)
    return summary


def filter_candidates(
    candidates: list[dict[str, object]], settings: BatchSettings, status: StatusLog
) -> list[dict[str, object]]:
    """Apply ``start_index``/``limit`` and, with ``resume``, skip finished grids."""
    selected = candidates
    if settings.start_index is not None:
        selected = [c for c in selected if int(c["candidate_index"]) >= settings.start_index]
    if settings.limit is not None:
        selected = selected[: settings.limit]
    if not settings.resume:
        return selected
    previous = previous_status(candidates, settings, status)
    runnable = []
    for candidate in selected:
        index = int(candidate["candidate_index"])
        state = previous.get(index)
        if state == "done":
            status.event(event="candidate_skipped_resume_done", candidate_index=index, job=job_key(candidate))
            continue
        if state == "failed" and not settings.rerun_failed:
            status.event(event="candidate_skipped_resume_failed", candidate_index=index, job=job_key(candidate))
            continue
        runnable.append(candidate)
    return runnable


# Batch identity --------------------------------------------------------------------------------


def batch_identity(settings: BatchSettings) -> dict[str, Any]:
    """Settings that determine results and candidate numbering of a batch."""
    return {
        "ags": int(str(settings.ags).strip() or "0"),
        "pylovo_version_id": str(settings.pylovo_version_id),
        "scenario_id": settings.scenario.scenario_id,
        "scenario_hash": settings.scenario_hash,
        "region": {"plz": settings.plz, "kcid": settings.kcid, "bcid": settings.bcid},
        "min_buildings": int(settings.min_buildings),
        "demand_scope": settings.demand_scope,
        "model_case": settings.model_case,
        "profiles": settings.profiles,
        "timeframe_mode": settings.timeframe_mode,
        "profile_seed": int(settings.profile_seed),
        "case_qualified_output": bool(settings.case_qualified_output),
        "powerflow_output": settings.powerflow_output,
        "powerflow_grid_scope": settings.powerflow_grid_scope,
    }


def check_batch_identity(settings: BatchSettings) -> None:
    """Record the batch identity; on resume refuse a run directory of another batch.

    Raises:
        ValueError: ``resume`` and the recorded identity differs.
    """
    path = settings.run_dir / _BATCH_IDENTITY_FILE
    identity = batch_identity(settings)
    if settings.resume and path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {key: (recorded.get(key), value) for key, value in identity.items() if recorded.get(key) != value}
        if mismatches:
            raise ValueError(
                f"Refusing to resume {settings.run_dir}: it belongs to another batch. "
                f"Differences (recorded, requested): {mismatches}. Use a new --run-dir."
            )
        return
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Regional electrification assignment ------------------------------------------------------------

CANDIDATE_IDENTITY_KEYS = (
    "candidate_index",
    "ags",
    "plz",
    "kcid",
    "bcid",
    "grid_result_id",
    "version_id",
    "bridge_filename",
    "n_buildings",
    "n_residential_buildings",
)


def _candidate_manifest(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: candidate.get(key) for key in CANDIDATE_IDENTITY_KEYS} for candidate in candidates]


def _candidate_manifest_hash(candidates: list[dict[str, object]]) -> str:
    payload = json.dumps(_candidate_manifest(candidates), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_prepared_electrification_assignment(
    path: Path,
    *,
    settings: BatchSettings,
    candidates: list[dict[str, object]],
) -> None:
    """Reject a reused regional manifest that does not belong to this run."""
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"Prepared electrification assignment is missing sidecar: {metadata_path}")
    assignment = pd.read_csv(path)
    validate_electrification_assignment_config(
        assignment,
        settings.scenario.electrification,
        profile_seed=settings.profile_seed,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("assignment_hash") != assignment_manifest_hash(assignment):
        raise ValueError("Prepared electrification assignment does not match its sidecar hash.")
    expected = {
        "scenario_id": settings.scenario.scenario_id,
        "scenario_hash": settings.scenario_hash,
        "profile_seed": int(settings.profile_seed),
        "pylovo_version_id": str(settings.pylovo_version_id),
        "demand_scope": settings.demand_scope,
        "mobility_source": MOBILITY_SOURCE,
        "candidate_grid_manifest_hash": _candidate_manifest_hash(candidates),
        "candidate_grid_count": len(candidates),
        "candidate_grid_manifest": _candidate_manifest(candidates),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "Prepared electrification assignment is stale or belongs to a "
            f"different run; mismatched metadata: {mismatches}."
        )


def ensure_electrification_assignment(
    settings: BatchSettings, candidates: list[dict[str, object]], status: StatusLog
) -> Path:
    """Prepare the regional assignment of ``candidates`` unless it exists; validate it."""
    path = settings.assignment_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        status.event(event="electrification_preparation_start", output=str(path))
        run_batch_command(
            cmd=commands.electrification_preparation_command(
                settings.ags,
                plz=settings.plz,
                kcid=settings.kcid,
                bcid=settings.bcid,
                min_buildings=settings.min_buildings,
                pylovo_version_id=settings.pylovo_version_id,
                demand_scope=settings.demand_scope,
                mobility_source=MOBILITY_SOURCE,
                profile_seed=settings.profile_seed,
                scenario_config=settings.scenario_config,
                output=path,
            ),
            log_path=settings.run_dir / "logs" / "electrification_preparation.log",
            status=status,
            stage="electrification_preparation",
        )
        status.event(event="electrification_preparation_finish", output=str(path))
    else:
        status.event(event="electrification_preparation_reused", output=str(path))
    validate_prepared_electrification_assignment(path, settings=settings, candidates=candidates)
    status.event(event="electrification_preparation_validated", output=str(path))
    return path


# Expansion ------------------------------------------------------------------------------------


def materialize_expansion_analyses(
    *,
    settings: BatchSettings,
    status: StatusLog,
    failures: list[dict[str, object]] = (),
    candidate_count: int | None = None,
) -> list[dict[str, str]]:
    """Materialize the regional expansion analyses of the batch's summary runs.

    With failed grids the analyses are still materialized, but their note says
    how many grids are missing (review-orch B6).
    """
    if not settings.materialize_expansion or not settings.summary_output:
        return []
    prefix = expansion_analysis_prefix(settings)
    log_path = settings.run_dir / "expansion_materialization.log"
    incomplete = ""
    if failures:
        names = ", ".join(sorted(str(item.get("job") or item.get("candidate_index")) for item in failures))
        total = f" of {candidate_count}" if candidate_count is not None else ""
        incomplete = f" INCOMPLETE: {len(failures)}{total} grids failed and are missing ({names})."
    materialized = []

    def materialize_one(run_name: str, stage: str, key_suffix: str, detail: str = "") -> None:
        analysis_key = f"{prefix}_{key_suffix}"
        note = f"Automatically materialized by synthetic_ags_runner from {run_name} summary stage={stage}{detail}."
        run_batch_command(
            cmd=commands.expansion_command(
                run_name, stage=stage, ags=settings.ags, analysis_key=analysis_key, note=note + incomplete,
            ),
            log_path=log_path,
            status=status,
            stage=f"expansion_materialize_{key_suffix}",
        )
        materialized.append({"stage": key_suffix, "analysis_key": analysis_key})

    inflex_detail = " using fixed inflex demand with post-flex heat capacity split"
    if settings.inflex_only:
        inflex_run_name = powerflow_run_name(settings, "summary_inflex")
        materialize_one(inflex_run_name, "pre", "pre")
        materialize_one(inflex_run_name, "post", "post_inflex", inflex_detail)
        return materialized
    summary_run_name = powerflow_run_name(settings, "summary")
    status_quo = settings.profiles == "status_quo"
    for stage in ("pre",) if status_quo else ("pre", "post"):
        materialize_one(summary_run_name, stage, stage)
    if settings.include_inflex_powerflow and not status_quo:
        materialize_one(powerflow_run_name(settings, "summary_inflex"), "post", "post_inflex", inflex_detail)
    return materialized


def refresh_views(status: StatusLog) -> None:
    """Refresh the QGIS materialized views once per batch."""
    from gridexpand.db import refresh_qgis_views

    refresh_qgis_views()
    status.event(event="qgis_views_refreshed")


# Batch --------------------------------------------------------------------------------------


def _write_summary(run_dir: Path, summary: dict[str, Any], status: StatusLog) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status.event(event="batch_finish", **summary)


def execute_candidates(
    settings: BatchSettings, candidates: list[dict[str, object]], status: StatusLog
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], bool]:
    """Pilot candidate first, then the others in a worker pool.

    Returns:
        ``(completed, failures, cancelled, pilot_failed)``; with a pilot gate a
        failed pilot stops the batch before the pool starts.
    """
    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    cancelled: list[dict[str, object]] = []

    def record(result: dict[str, object]) -> None:
        {"done": completed, "cancelled": cancelled}.get(str(result.get("status")), failures).append(result)

    by_index = {int(candidate["candidate_index"]): candidate for candidate in candidates}
    pilot = by_index.get(settings.pilot_index)
    if pilot is not None:
        status.event(event="pilot_start", candidate_index=settings.pilot_index, job=job_key(pilot))
        result = run_candidate(candidate=pilot, settings=settings, status=status)
        status.event(event="pilot_finish", **result)
        record(result)
        if result["status"] == "failed" and settings.pilot_gate:
            return completed, failures, cancelled, True

    remaining = [c for c in candidates if int(c["candidate_index"]) != settings.pilot_index]
    status.event(event="batch_workers_start", remaining=len(remaining), workers=settings.workers)
    with ThreadPoolExecutor(max_workers=max(1, int(settings.workers))) as executor:
        future_map = {
            executor.submit(run_candidate, candidate=candidate, settings=settings, status=status): candidate
            for candidate in remaining
        }
        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"candidate_index": int(candidate["candidate_index"]), "job": job_key(candidate),
                          "status": "failed", "message": str(exc)}
                status.event(event="candidate_failed_unhandled", candidate_index=result["candidate_index"],
                             message=str(exc))
            record(result)
            if result.get("status") == "done":
                status.event(event="candidate_done", **result)
            elif result.get("status") != "cancelled":
                status.event(event="candidate_failed_recorded", **result)
    return completed, failures, cancelled, False


def run_batch(
    settings: BatchSettings,
    *,
    candidates: list[dict[str, object]] | None = None,
    listener: Callable[[dict[str, Any]], None] | None = None,
    refresh_qgis_views: bool = True,
    cleanup_completed_only: bool = False,
) -> int:
    """Run one synthetic batch (the body of ``gridexpand synthetic``).

    Args:
        settings: Batch settings.
        candidates: Candidate grids of the region (default: loaded from the database).
        listener: Receives every batch event (``gridexpand run`` mirrors them).
        refresh_qgis_views: Refresh the QGIS views after the expansion analyses.
        cleanup_completed_only: Only delete intermediates of finished grids.

    Returns:
        Exit code: 0 done, 1 pilot failed / no candidates, 2 failed grids,
        143 cancelled.
    """
    check_settings(settings)
    run_dir = settings.run_dir
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    status = StatusLog(run_dir, resume=settings.resume, listener=listener)
    check_batch_identity(settings)
    started_wall = time.monotonic()
    status.event(
        event="batch_start",
        project_dir=str(PROJECT_DIR),
        work_dir=str(WORK_DIR),
        ags=settings.ags,
        pylovo_version_id=settings.pylovo_version_id,
        region={"plz": settings.plz, "kcid": settings.kcid, "bcid": settings.bcid},
        min_buildings=settings.min_buildings,
        workers=settings.workers,
        step2_cpus=settings.step2_cpus,
        step2_timeseries_storage=settings.step2_timeseries_storage,
        model_case=settings.model_case,
        profiles=settings.profiles,
        demand_scope=settings.demand_scope,
        timeframe_mode=settings.timeframe_mode,
        step3_cpus=settings.step3_cpus,
        step3_max_cpus=settings.step3_max_cpus,
        step3_cluster_concurrency=settings.step3_cluster_concurrency,
        step4_cpus=settings.step4_cpus,
        solver=settings.solver or os.environ.get(SOLVER_ENV) or "gurobi",
        powerflow_output=settings.powerflow_output,
        powerflow_grid_scope=settings.powerflow_grid_scope,
        materialize_expansion=settings.materialize_expansion and settings.summary_output,
        expansion_analysis_prefix=expansion_analysis_prefix(settings),
        run_dir=str(run_dir),
        resume=settings.resume,
        rerun_failed=settings.rerun_failed,
        cleanup_intermediates=settings.cleanup_intermediates,
        cleanup_completed_only=cleanup_completed_only,
    )

    if candidates is None:
        candidates = load_candidates(settings)
    (run_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    status.event(event="candidates_loaded", count=len(candidates))
    if not candidates:
        status.event(event="batch_finish", status="failed", message="No candidates found")
        return 1

    if cleanup_completed_only:
        summary = cleanup_completed_intermediates(candidates, settings, status)
        (run_dir / "cleanup_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return 0

    if settings.profiles != "status_quo":
        ensure_electrification_assignment(settings, candidates, status)

    selected = filter_candidates(candidates, settings, status)
    status.event(event="candidates_selected", count=len(selected))
    if not selected:
        _write_summary(run_dir, {
            "status": "done",
            "candidate_count": 0,
            "failure_count": 0,
            "total_seconds": round(time.monotonic() - started_wall, 1),
            "finished_at": utc_now(),
            "message": "No runnable candidates after filtering/resume.",
        }, status)
        return 0

    completed, failures, cancelled, pilot_failed = execute_candidates(settings, selected, status)
    if pilot_failed:
        _write_summary(run_dir, {
            "status": "failed",
            "failed_at": "pilot",
            "candidate_count": len(selected),
            "failure_count": len(failures),
            "failures": failures,
            "total_seconds": round(time.monotonic() - started_wall, 1),
            "finished_at": utc_now(),
        }, status)
        return 1

    materialized_expansion: list[dict[str, str]] = []
    expansion_failure = None
    if not (cancelled or CANCEL.is_set()):
        try:
            materialized_expansion = materialize_expansion_analyses(
                settings=settings, status=status, failures=failures, candidate_count=len(selected)
            )
            if materialized_expansion and refresh_qgis_views:
                refresh_views(status)
        except Cancelled:
            cancelled.append({"status": "cancelled", "stage": "expansion"})
        except Exception as exc:
            expansion_failure = str(exc)
            status.event(event="expansion_materialization_failed", message=expansion_failure)

    if cancelled or CANCEL.is_set():
        batch_status = "cancelled"
    elif failures:
        batch_status = "completed_with_failures"
    elif expansion_failure:
        batch_status = "completed_with_expansion_failure"
    else:
        batch_status = "done"
    summary = {
        "status": batch_status,
        "candidate_count": len(selected),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "cancelled_count": len(cancelled),
        "failures": failures,
        "materialized_expansion": materialized_expansion,
        "expansion_failure": expansion_failure,
        "expansion_incomplete": (
            {"failed_grids": len(failures), "of": len(selected)} if failures and materialized_expansion else None
        ),
        "total_seconds": round(time.monotonic() - started_wall, 1),
        "finished_at": utc_now(),
    }
    _write_summary(run_dir, summary, status)
    if batch_status == "cancelled":
        return EXIT_CANCELLED
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    try:
        check_settings(settings)
    except ValueError as exc:
        parser.error(str(exc))
    install_cancel_handlers()
    try:
        return run_batch(settings, cleanup_completed_only=args.cleanup_completed_only)
    except Cancelled as exc:  # e.g. during the regional electrification preparation
        print(f"gridexpand synthetic: cancelled ({exc})", file=sys.stderr, flush=True)
        return EXIT_CANCELLED


if __name__ == "__main__":
    raise SystemExit(main())
