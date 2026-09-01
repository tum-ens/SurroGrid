"""Synthetic-grid adapter for paired validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from common.orchestration import StatusLog, run_command

TARGET_NETWORK = "synthetic"
ALLOCATION_PLAN_FILENAME = "paired_synthetic_bus_allocation_plan.csv"


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
    jobs: list[dict[str, Any]] = []
    for grid_id in grid_ids:
        grid_plan = plan[
            pd.to_numeric(plan["target_grid_id"], errors="coerce")
            .astype("Int64")
            .eq(grid_id)
        ]
        bridge_names = (
            grid_plan["synthetic_bridge_filename"].dropna().astype(str).unique()
        )
        if len(bridge_names) != 1:
            raise ValueError(f"Synthetic grid {grid_id} has ambiguous bridge names.")
        jobs.append(
            {
                "target_network": TARGET_NETWORK,
                "target_grid_id": int(grid_id),
                "bridge_filename": bridge_names[0],
            }
        )
    return jobs


def input_name(job: dict[str, Any], scenario_label: str) -> str:
    bridge_stem = Path(job["bridge_filename"]).stem
    return f"paired_synthetic_{bridge_stem}_{scenario_label}.h5"


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
    step4_input = step4_dir / "Input" / result_hdf.name
    shutil.copy2(result_hdf, step4_input)
    common = [
        "uv",
        "run",
        "python",
        "run_pwrflw.py",
        step4_input.name,
        "--grid-case-id",
        str(grid_id),
        "--storage",
        "db",
        "--summary-only",
        "--summary-nonconvergence",
        "nan",
        "--summary-grid-scope",
        args.powerflow_grid_scope,
        "--n_cpu",
        str(args.step4_cpus),
    ]
    if not args.skip_pre:
        run_command(
            cmd=common
            + [
                "--pre-only",
                "--run-name",
                f"{args.run_name_prefix}_{TARGET_NETWORK}_pre",
            ],
            cwd=step4_dir,
            log_path=log_path,
            status=status,
            candidate_index=job_index,
            stage=f"step4_{TARGET_NETWORK}_pre",
            env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
        )
    definitions = {
        "post-hems-optimized": "flexible",
        "post-hems-heuristic": "flexible",
        "post-inflex-heuristic": "no-flex",
    }
    post_cases = tuple(
        (definitions[case_name], case_name) for case_name in args.result_cases
    )
    for mode, case_name in post_cases:
        command = common + [
            "--post-demand-mode",
            mode,
            "--run-name",
            f"{args.run_name_prefix}_{TARGET_NETWORK}_{case_name}",
        ]
        run_command(
            cmd=command,
            cwd=step4_dir,
            log_path=log_path,
            status=status,
            candidate_index=job_index,
            stage=f"step4_{TARGET_NETWORK}_{case_name}",
            env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
        )
    if args.cleanup_intermediates:
        step4_input.unlink(missing_ok=True)
