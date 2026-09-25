"""SWF real-grid adapter for paired validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gridexpand.common.orchestration import StatusLog, run_command
from gridexpand.scenario import commands
from gridexpand.scenario.model_cases import MODEL_CASES

TARGET_NETWORK = "real_swf"
ALLOCATION_PLAN_FILENAME = "paired_real_bus_allocation_plan.csv"


def load_real_jobs(
    paired_dir: Path, target_grid_id: int | None, target_network: str
) -> list[dict[str, Any]]:
    """One job per real grid; aligned plans also name the grid file and PLZ."""
    plan = pd.read_csv(paired_dir / ALLOCATION_PLAN_FILENAME)
    plan["target_grid_id"] = pd.to_numeric(plan["target_grid_id"], errors="coerce")
    plan = plan.dropna(subset=["target_grid_id"])
    plan["target_grid_id"] = plan["target_grid_id"].astype(int)
    if target_grid_id is not None:
        plan = plan[plan["target_grid_id"].eq(int(target_grid_id))]
    jobs = []
    for grid_id, grid_plan in plan.groupby("target_grid_id", sort=True):
        job = {"target_network": target_network, "target_grid_id": int(grid_id)}
        if "real_grid_file" in grid_plan:
            files = grid_plan["real_grid_file"].dropna().astype(str).unique()
            if len(files) != 1:
                raise ValueError(f"Real grid {grid_id} has ambiguous grid files.")
            job["grid_file"] = files[0]
            # A real area can span postcodes; the majority names it.
            job["plz"] = int(grid_plan["postcode"].mode().iloc[0])
        jobs.append(job)
    return jobs


def load_jobs(paired_dir: Path, target_grid_id: int | None) -> list[dict[str, Any]]:
    return load_real_jobs(paired_dir, target_grid_id, TARGET_NETWORK)


def input_name(job: dict[str, Any], scenario_label: str) -> str:
    grid_id = int(job["target_grid_id"])
    return f"paired_real_swf_LV_{grid_id:03d}_{scenario_label}.h5"


def run_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    run_real_powerflows(
        job=job, args=args, result_hdf=result_hdf,
        log_path=log_path, status=status,
        target_network=TARGET_NETWORK, provider="swf",
    )


def run_real_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    log_path: Path,
    status: StatusLog,
    target_network: str,
    provider: str,
) -> None:
    """Step 4 of one real grid: the pre case (unless ``--skip-pre``), then each result case."""
    cases = args.result_cases if args.skip_pre else ("pre", *args.result_cases)
    for case_name in cases:
        case = MODEL_CASES[case_name]
        run_name = f"{args.run_name_prefix}_{target_network}_{case_name}"
        command = commands.real_powerflow_command(
            plz=job.get("plz", args.plz),
            lv_id=int(job["target_grid_id"]),
            provider=provider,
            profile_seed=args.profile_seed,
            urbs_result_hdf=result_hdf,
            summary_grid_scope=args.powerflow_grid_scope,
            # A full-year request must never consume a representative-period
            # result. Pre-only jobs read the Step-2 input, which carries no
            # Step-3 temporal record.
            expect_temporal_method=None if args.tsam or args.pre_only else "full_year_no_tsam",
            grid_file=job.get("grid_file"),
            grid_data_path=args.grid_data_path,
            max_timesteps=getattr(args, "max_timesteps", None),
            post_demand_mode=case.real_powerflow_mode,
            run_name=run_name,
            scenario_label=f"Paired 2045 {target_network} {case.label}",
        )
        run_command(
            cmd=command,
            log_path=log_path,
            status=status,
            job=job["job"],
            stage=f"step4_{target_network}_{case_name}",
        )
