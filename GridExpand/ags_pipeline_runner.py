#!/usr/bin/env python3
"""Run the DB-backed GridExpand pipeline for AGS grid candidates.

This helper coordinates Step 2, Step 3, and Step 4 for long tmux runs. It keeps
per-candidate logs, can resume completed work, validates DB-backed power-flow
outputs, and records failed grids without aborting the full batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import h5py
import pandas as pd
from sqlalchemy import text

from timeframe import (
    TIMEFRAME_MODES,
    build_initial_metadata,
    horizon_hours_from_hdf,
    output_filename_for_timeframe,
    read_hdf_metadata,
    scenario_key_for_timeframe,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DB-backed GridExpand pipeline batch for one AGS.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ags", required=True, help="AGS identifier for the region to process, e.g. 09162000.")
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step2-cpus", type=int, default=4)
    parser.add_argument("--step3-cpus", type=int, default=16)
    parser.add_argument("--step3-max-cpus", type=int, default=32)
    parser.add_argument("--step3-target-columns", type=int, default=35)
    parser.add_argument("--step3-cluster-concurrency", type=int, default=1)
    parser.add_argument("--step4-cpus", type=int, default=4)
    parser.add_argument("--tsam", action="store_true", help="Enable TSAM type-week aggregation in Step 3.")
    parser.add_argument("--tsam-periods", type=int, default=6, help="Number of TSAM type weeks for Step 3.")
    parser.add_argument("--tsam-hours-per-period", type=int, default=168, help="Hours per TSAM period.")
    parser.add_argument(
        "--tsam-extreme-method",
        choices=["append", "new_cluster_center", "replace_cluster_center"],
        default="replace_cluster_center",
        help="How TSAM should include cold and solar extreme weeks.",
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
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-dynamic-step3", action="store_true")
    return parser.parse_args()


class StatusLog:
    def __init__(self, run_dir: Path, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.status_path = run_dir / "status.tsv"
        self.failed_path = run_dir / "failed_grids.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[int, dict[str, object]] = {}
        if resume and self.status_path.exists():
            self._load_status()

    def _load_status(self) -> None:
        with self.status_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not row.get("candidate_index"):
                    continue
                self.rows[int(row["candidate_index"])] = dict(row)

    def event(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def failed_grid(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.failed_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, candidate_index: int, **updates: object) -> None:
        with self.lock:
            row = self.rows.setdefault(candidate_index, {"candidate_index": candidate_index})
            row.update(updates)
            self._write_status_locked()

    def status_for(self, candidate_index: int) -> str | None:
        row = self.rows.get(candidate_index)
        if not row:
            return None
        value = row.get("status")
        return str(value) if value else None

    def _write_status_locked(self) -> None:
        columns = [
            "candidate_index",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "n_buildings",
            "bridge_filename",
            "timeframe_mode",
            "horizon_hours",
            "timeframe_start",
            "timeframe_end",
            "status",
            "stage",
            "started_at",
            "finished_at",
            "seconds",
            "step3_cpus",
            "urbs_cluster_concurrency",
            "log_file",
            "message",
        ]
        lines = ["\t".join(columns)]
        for index in sorted(self.rows):
            row = self.rows[index]
            lines.append("\t".join(str(row.get(column, "")) for column in columns))
        self.status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_imports(repo_root: Path) -> None:
    gridexpand_dir = repo_root / "GridExpand"
    if str(gridexpand_dir) not in sys.path:
        sys.path.insert(0, str(gridexpand_dir))


def normalize_ags(value: str) -> int:
    return int(str(value).strip().lstrip("0") or "0")


def get_candidates(repo_root: Path, ags: str, min_buildings: int) -> list[dict[str, object]]:
    configure_imports(repo_root)
    from database import SurroGridDatabase

    db = SurroGridDatabase()
    query = text(
        """
        WITH ags_plz AS (
            SELECT DISTINCT plz
            FROM pylovo.municipal_register
            WHERE ags = :ags
        ),
        building_counts AS (
            SELECT grid_result_id, version_id, COUNT(*) AS n_buildings
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
                bc.n_buildings
            FROM pylovo.grid_result gr
            JOIN ags_plz ap ON ap.plz = gr.plz
            JOIN building_counts bc
              ON bc.grid_result_id = gr.grid_result_id
             AND bc.version_id = gr.version_id
            WHERE bc.n_buildings >= :min_buildings
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
        | {"n_buildings": int(row["n_buildings"])}
        for row in rows
    ]


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    gurobi_home = "/opt/gurobi1302/linux64"
    env["GUROBI_HOME"] = gurobi_home
    env["GRB_LICENSE_FILE"] = "/home/breveron/gurobi.lic"
    env["PATH"] = f"{gurobi_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{gurobi_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    return env


def run_command(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    status: StatusLog,
    candidate_index: int,
    stage: str,
    env_extra: dict[str, str] | None = None,
) -> None:
    status.update(candidate_index, stage=stage, status="running", message="")
    status.event(candidate_index=candidate_index, stage=stage, event="start", cmd=cmd, env_extra=env_extra or {})
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] START {stage}: {' '.join(cmd)}\n")
        if env_extra:
            log_handle.write(f"[{utc_now()}] ENV {env_extra}\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=command_env(env_extra),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        seconds = round(time.monotonic() - started, 1)
        log_handle.write(f"[{utc_now()}] END {stage}: rc={completed.returncode} seconds={seconds}\n")
    status.event(
        candidate_index=candidate_index,
        stage=stage,
        event="finish",
        returncode=completed.returncode,
        seconds=seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


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
        demand = store["/urbs_in/demand"] if "/urbs_in/demand" in store else pd.DataFrame()
        supim = store["/urbs_in/supim"] if "/urbs_in/supim" in store else pd.DataFrame()
        process = store["/urbs_in/process"] if "/urbs_in/process" in store else pd.DataFrame()
        commodity = store["/urbs_in/commodity"] if "/urbs_in/commodity" in store else pd.DataFrame()

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


def choose_step3_settings(step2_output: Path, args: argparse.Namespace) -> tuple[int, int, dict[str, int]]:
    if args.no_dynamic_step3:
        return int(args.step3_cpus), int(args.step3_cluster_concurrency), {}

    stats: dict[str, int] = {}
    for key, name in (("urbs_in/demand", "demand_columns"), ("urbs_in/eff_factor", "eff_factor_columns")):
        try:
            stats[name] = hdf_column_count(step2_output, key)
        except Exception:
            stats[name] = 0
    largest_columns = max(stats.values() or [0])
    required = math.ceil(largest_columns / max(1, int(args.step3_target_columns))) if largest_columns else args.step3_cpus
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
    scenario_path = repo_root / "GridExpand" / "4.powerflow" / "Input" / scenario_filename
    expected_horizon = horizon_hours_from_hdf(scenario_path)
    expected_max_t = expected_horizon - 1
    from database import SurroGridDatabase

    db = SurroGridDatabase()
    with db.engine.connect() as conn:
        run = conn.execute(
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
            {"scenario_filename": scenario_filename, "pre_only": pre_only, "run_name": run_name},
        ).mappings().first()
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
            for table_name in ("powerflow_summary", "powerflow_cable_summary", "powerflow_bus_voltage_summary"):
                rows = conn.execute(
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
                ).mappings().all()
                by_stage = {str(row["stage"]): dict(row) for row in rows}
                validation["tables"][table_name] = by_stage
                for stage in expected_summary_stages:
                    row = by_stage.get(stage)
                    if not row or int(row["rows"]) <= 0:
                        missing.append(f"{table_name}:{stage}:missing")
            if missing:
                raise RuntimeError("Incomplete Step 4 DB summary results: " + ", ".join(missing))
            return validation

        for table_name, expected_stages in EXPECTED_POWERFLOW_TABLES.items():
            rows = conn.execute(
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
            ).mappings().all()
            by_stage = {str(row["stage"]): dict(row) for row in rows}
            validation["tables"][table_name] = by_stage
            for stage in expected_stages:
                row = by_stage.get(stage)
                if not row:
                    missing.append(f"{table_name}:{stage}:missing")
                    continue
                if int(row["rows"]) <= 0 or int(row["min_t"]) != 0 or int(row["max_t"]) != expected_max_t:
                    missing.append(f"{table_name}:{stage}:incomplete")

        reactive_rows = conn.execute(
            text(
                """
                SELECT count(*) AS rows, min(t_index) AS min_t, max(t_index) AS max_t
                FROM surrogrid.powerflow_reactive_component
                WHERE powerflow_run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).mappings().one()
        validation["tables"]["powerflow_reactive_component"] = {"all": dict(reactive_rows)}
        if int(reactive_rows["rows"]) <= 0 or int(reactive_rows["min_t"]) != 0 or int(reactive_rows["max_t"]) != expected_max_t:
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
    step2_filename = output_filename_for_timeframe(bridge_filename, args.timeframe_mode)
    prefix = step2_filename.split("_", 1)[0]
    scenario_filename = ""
    log_file = args.run_dir / "logs" / f"candidate_{candidate_index:03d}_{step2_filename}.log"
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
        run_command(
            cmd=[
                "uv",
                "run",
                "--project",
                "..",
                "python",
                "main.py",
                ags,
                "--storage",
                "db",
                "--candidate-index",
                str(candidate_index),
                "--min-buildings",
                str(args.min_buildings),
                "--profiles",
                args.profiles,
                "--mobility-source",
                "pool",
                "--timeseries-storage",
                args.step2_timeseries_storage,
                "--timeframe-mode",
                args.timeframe_mode,
                "--n_cpu",
                str(args.step2_cpus),
            ],
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
        shutil.copy2(step2_output, step3_dir / "Input" / step2_filename)

        step3_cpus, cluster_concurrency, step3_stats = choose_step3_settings(step2_output, args)
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
        if args.tsam:
            step3_cmd.extend([
                "--tsam",
                "--tsam-periods",
                str(args.tsam_periods),
                "--tsam-hours-per-period",
                str(args.tsam_hours_per_period),
                "--tsam-extreme-method",
                args.tsam_extreme_method,
            ])
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
            run_command(
                cmd=[
                    "uv",
                    "run",
                    "python",
                    "run_pwrflw.py",
                    scenario_filename,
                    "--storage",
                    "db",
                    "--n_cpu",
                    str(args.step4_cpus),
                ],
                cwd=step4_dir,
                log_path=log_file,
                status=status,
                candidate_index=candidate_index,
                stage=current_stage,
            )

            current_stage = "step4_validate_raw"
            validations.append(validate_powerflow_db(repo_root, scenario_filename, summary_only=False, pre_only=False))

        if args.powerflow_output in {"summary", "both"}:
            current_stage = "step4_powerflow_summary"
            summary_pre_only = args.profiles == "status_quo"
            expected_summary_stages = ("pre",) if summary_pre_only else ("pre", "post")
            summary_run_name = f"{scenario_key_for_timeframe(args.timeframe_mode)}_{args.profiles}_summary_powerflow"
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
            ]
            if summary_pre_only:
                summary_cmd.insert(summary_cmd.index("--summary-only"), "--pre-only")
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

        with log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] STEP4 VALIDATION OK\n")
            log_handle.write(json.dumps(validations, indent=2, sort_keys=True, default=str) + "\n")

        seconds = round(time.monotonic() - started, 1)
        status.update(
            candidate_index,
            status="done",
            stage="complete",
            finished_at=utc_now(),
            seconds=seconds,
            message="ok",
        )
        return {"candidate_index": candidate_index, "status": "done", "seconds": seconds}
    except Exception as exc:
        seconds = round(time.monotonic() - started, 1)
        with log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] FAILURE in {current_stage}: {exc}\n")
            log_handle.write(traceback.format_exc())
        payload = candidate_failed_payload(candidate, current_stage, str(exc), seconds, log_file)
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
        return {"candidate_index": candidate_index, "status": "failed", "seconds": seconds, "message": str(exc)}


def filter_candidates(candidates: list[dict[str, object]], args: argparse.Namespace, status: StatusLog) -> list[dict[str, object]]:
    selected = candidates
    if args.start_index is not None:
        selected = [candidate for candidate in selected if int(candidate["candidate_index"]) >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not args.resume:
        return selected

    runnable = []
    for candidate in selected:
        candidate_index = int(candidate["candidate_index"])
        previous = status.status_for(candidate_index)
        if previous == "done":
            status.event(event="candidate_skipped_resume_done", candidate_index=candidate_index)
            continue
        if previous == "failed" and not args.rerun_failed:
            status.event(event="candidate_skipped_resume_failed", candidate_index=candidate_index)
            continue
        runnable.append(candidate)
    return runnable


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    args.run_dir = run_dir
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    status = StatusLog(run_dir, resume=args.resume)

    started_wall = time.monotonic()
    status.event(
        event="batch_start",
        repo_root=str(repo_root),
        ags=args.ags,
        min_buildings=args.min_buildings,
        workers=args.workers,
        step2_cpus=args.step2_cpus,
        step2_timeseries_storage=args.step2_timeseries_storage,
        profiles=args.profiles,
        timeframe_mode=args.timeframe_mode,
        step3_cpus=args.step3_cpus,
        step3_max_cpus=args.step3_max_cpus,
        step3_cluster_concurrency=args.step3_cluster_concurrency,
        step4_cpus=args.step4_cpus,
        powerflow_output=args.powerflow_output,
        run_dir=str(run_dir),
        resume=args.resume,
        rerun_failed=args.rerun_failed,
    )

    candidates = get_candidates(repo_root, args.ags, args.min_buildings)
    (run_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    status.event(event="candidates_loaded", count=len(candidates))
    if not candidates:
        status.event(event="batch_finish", status="failed", message="No candidates found")
        return 1

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
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        status.event(event="batch_finish", **summary)
        return 0

    by_index = {int(candidate["candidate_index"]): candidate for candidate in candidates}
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
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
                status.event(event="batch_finish", **summary)
                return 1

    remaining = [candidate for candidate in candidates if int(candidate["candidate_index"]) != args.pilot_index]
    status.event(event="batch_workers_start", remaining=len(remaining), workers=args.workers)
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
                result = {"candidate_index": candidate_index, "status": "failed", "message": str(exc)}
                status.event(event="candidate_failed_unhandled", candidate_index=candidate_index, message=str(exc))
            if result.get("status") == "done":
                completed.append(result)
                status.event(event="candidate_done", **result)
            else:
                failures.append(result)
                status.event(event="candidate_failed_recorded", **result)

    total_seconds = round(time.monotonic() - started_wall, 1)
    summary = {
        "status": "done" if not failures else "completed_with_failures",
        "candidate_count": len(candidates),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "total_seconds": total_seconds,
        "finished_at": utc_now(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status.event(event="batch_finish", **summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
