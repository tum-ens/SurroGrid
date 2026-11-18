"""Real grids as runner targets: given grid files with their scenario dataset (``pipeline: real``, validation).

The network label (``target_network``) and the provider come from the dataset's bus allocation plan, so every real
dataset (own grids, the DSO grids of the validation) runs through this one target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gridexpand.common.orchestration import StatusLog, run_command
from gridexpand.scenario import commands
from gridexpand.scenario.model_cases import MODEL_CASES

TARGET_NETWORK = "real"
ALLOCATION_PLAN_FILENAME = "paired_real_bus_allocation_plan.csv"


def load_jobs(paired_dir: Path, target_grid_id: int | None) -> list[dict[str, Any]]:
    """One job per real grid of the dataset, with its network label, provider, grid file and PLZ."""
    plan = pd.read_csv(paired_dir / ALLOCATION_PLAN_FILENAME)
    plan["target_grid_id"] = pd.to_numeric(plan["target_grid_id"], errors="coerce")
    plan = plan.dropna(subset=["target_grid_id"])
    plan["target_grid_id"] = plan["target_grid_id"].astype(int)
    if target_grid_id is not None:
        plan = plan[plan["target_grid_id"].eq(int(target_grid_id))]
    networks = plan["target_network"].dropna().astype(str).unique() if "target_network" in plan else []
    if len(networks) > 1:
        raise ValueError(f"{paired_dir} mixes real networks {sorted(networks)}.")
    network = str(networks[0]) if len(networks) else TARGET_NETWORK
    provider = str(plan["provider"].dropna().iloc[0]) if "provider" in plan and plan["provider"].notna().any() else None
    jobs = []
    for grid_id, grid_plan in plan.groupby("target_grid_id", sort=True):
        job = {"target_network": network, "target_grid_id": int(grid_id), "provider": provider}
        if "real_grid_file" in grid_plan:
            files = grid_plan["real_grid_file"].dropna().astype(str).unique()
            if len(files) != 1:
                raise ValueError(f"Real grid {grid_id} has ambiguous grid files.")
            job["grid_file"] = files[0]
            # A real area can span postcodes; the majority names it.
            job["plz"] = int(grid_plan["postcode"].mode().iloc[0])
        jobs.append(job)
    return jobs


def input_name(job: dict[str, Any], scenario_label: str) -> str:
    return f"paired_{job['target_network']}_{int(job['target_grid_id'])}_{scenario_label}.h5"


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
        target_network=str(job["target_network"]), provider=job.get("provider") or args.provider,
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
    """Step 4 of one real grid: the pre case (unless ``--skip-pre``), then each result case.

    Result cases solve the post stage only: the pre stage is the same for every case
    and is written once, by the pre case of the job that emits it.
    """
    cases = args.result_cases if args.skip_pre else ("pre", *args.result_cases)
    for case_name in cases:
        case = MODEL_CASES[case_name]
        run_name = f"{args.run_name_prefix}_{target_network}_{case_name}"
        command = commands.real_powerflow_command(
            plz=job.get("plz", args.plz),
            lv_id=int(job["target_grid_id"]),
            provider=provider,
            urbs_result_hdf=result_hdf,
            summary_grid_scope=args.powerflow_grid_scope,
            # A full-year request must never consume a representative-period
            # result. Pre-only jobs read the Step-2 input, which carries no
            # Step-3 temporal record.
            expect_temporal_method=None if args.tsam or args.pre_only else "full_year_no_tsam",
            grid_file=job["grid_file"],
            max_timesteps=getattr(args, "max_timesteps", None),
            post_demand_mode=case.real_powerflow_mode,
            post_only=case.powerflow_mode is not None,
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
