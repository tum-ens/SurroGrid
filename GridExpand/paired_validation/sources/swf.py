"""SWF real-grid adapter for paired validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common.orchestration import StatusLog, run_command

TARGET_NETWORK = "real_swf"
ALLOCATION_PLAN_FILENAME = "paired_real_bus_allocation_plan.csv"


def load_jobs(paired_dir: Path, target_grid_id: int | None) -> list[dict[str, Any]]:
    plan = pd.read_csv(paired_dir / ALLOCATION_PLAN_FILENAME)
    grid_ids = sorted(
        pd.to_numeric(plan["target_grid_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if target_grid_id is not None:
        grid_ids = [grid_id for grid_id in grid_ids if grid_id == target_grid_id]
    return [
        {"target_network": TARGET_NETWORK, "target_grid_id": int(grid_id)}
        for grid_id in grid_ids
    ]


def input_name(job: dict[str, Any], scenario_label: str) -> str:
    grid_id = int(job["target_grid_id"])
    return f"paired_real_swf_LV_{grid_id:03d}_{scenario_label}.h5"


def run_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    step4_dir: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    job_index = int(job["job_index"])
    grid_id = int(job["target_grid_id"])
    common = [
        "uv",
        "run",
        "python",
        "run_real_swf_scenario_powerflow.py",
        "--plz",
        str(args.plz),
        "--lv-id",
        str(grid_id),
        "--urbs-result-hdf",
        str(result_hdf),
    ]
    if args.grid_data_path is not None:
        common.extend(["--grid-data-path", str(args.grid_data_path)])
    post_cases = (
        (("flexible", "post-hems-optimized", "optimized HEMS"),)
        if args.model_case == "post-hems-optimized"
        else (
            ("flexible", "post-hems-heuristic", "heuristic-assets HEMS"),
            ("no-flex", "post-inflex-heuristic", "heuristic-assets INFLEX"),
        )
    )
    for mode, case_name, label in (
        ("pre-only", "pre", "pre electricity-only"),
        *post_cases,
    ):
        run_name = f"{args.run_name_prefix}_{TARGET_NETWORK}_{case_name}"
        command = common + [
            "--post-demand-mode",
            mode,
            "--run-name",
            run_name,
            "--scenario-key",
            run_name,
            "--scenario-label",
            f"Paired SWF 2045 {TARGET_NETWORK} {label}",
        ]
        run_command(
            cmd=command,
            cwd=step4_dir,
            log_path=log_path,
            status=status,
            candidate_index=job_index,
            stage=f"step4_{TARGET_NETWORK}_{case_name}",
        )
