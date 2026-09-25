"""Synthetic-grid adapter for paired validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from gridexpand.common.orchestration import StatusLog, run_command
from gridexpand.paths import POWERFLOW_INPUT_DIR, ensure_dir
from gridexpand.scenario import commands
from gridexpand.scenario.model_cases import MODEL_CASES

TARGET_NETWORK = "synthetic"
ALLOCATION_PLAN_FILENAME = "paired_synthetic_bus_allocation_plan.csv"


def load_jobs(paired_dir: Path, target_grid_id: int | None) -> list[dict[str, Any]]:
    plan = pd.read_csv(paired_dir / ALLOCATION_PLAN_FILENAME)
    plan["target_grid_id"] = pd.to_numeric(plan["target_grid_id"], errors="coerce")
    plan = plan.dropna(subset=["target_grid_id"])
    plan["target_grid_id"] = plan["target_grid_id"].astype(int)
    if target_grid_id is not None:
        plan = plan[plan["target_grid_id"].eq(int(target_grid_id))]
    jobs: list[dict[str, Any]] = []
    for grid_id, grid_plan in plan.groupby("target_grid_id", sort=True):
        bridge_names = grid_plan["synthetic_bridge_filename"].dropna().astype(str).unique()
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
    log_path: Path,
    status: StatusLog,
) -> None:
    """Step 4 summaries of one synthetic grid: pre (unless ``--skip-pre``), then each result case."""
    grid_id = int(job["target_grid_id"])
    step4_input = ensure_dir(POWERFLOW_INPUT_DIR) / result_hdf.name
    shutil.copy2(result_hdf, step4_input)
    cases = args.result_cases if args.skip_pre else ("pre", *args.result_cases)
    for case_name in cases:
        case = MODEL_CASES[case_name]
        command = commands.powerflow_command(
            step4_input.name,
            grid_case_id=grid_id,
            outputs=("summary",),
            summary_nonconvergence="nan",
            summary_grid_scope=args.powerflow_grid_scope,
            n_cpu=args.step4_cpus,
            max_timesteps=getattr(args, "max_timesteps", None),
            # A full-year request must never consume a representative-period
            # result. Pre-only jobs read the Step-2 input, which carries no
            # Step-3 temporal record.
            expect_temporal_method=None if args.tsam or args.pre_only else "full_year_no_tsam",
            pre_only=case.powerflow_mode is None,
            post_demand_mode=case.powerflow_mode,
            run_name=f"{args.run_name_prefix}_{TARGET_NETWORK}_{case_name}",
        )
        run_command(
            cmd=command,
            log_path=log_path,
            status=status,
            job=job["job"],
            stage=f"step4_{TARGET_NETWORK}_{case_name}",
            env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
        )
    if args.cleanup_intermediates:
        step4_input.unlink(missing_ok=True)
