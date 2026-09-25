"""SWF real-grid adapter for paired validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common.orchestration import StatusLog, run_command

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
    step4_dir: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    run_real_powerflows(
        job=job, args=args, result_hdf=result_hdf, step4_dir=step4_dir,
        log_path=log_path, status=status,
        target_network=TARGET_NETWORK, provider="swf",
    )


def run_real_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    step4_dir: Path,
    log_path: Path,
    status: StatusLog,
    target_network: str,
    provider: str,
) -> None:
    job_index = int(job["job_index"])
    grid_id = int(job["target_grid_id"])
    common = [
        "uv",
        "run",
        "python",
        "run_real_swf_scenario_powerflow.py",
        "--plz",
        str(job.get("plz", args.plz)),
        "--lv-id",
        str(grid_id),
        "--provider",
        provider,
        "--profile-seed",
        str(args.profile_seed),
        "--urbs-result-hdf",
        str(result_hdf),
        "--summary-grid-scope",
        args.powerflow_grid_scope,
    ]
    if not args.tsam and not args.pre_only:
        # A full-year request must never consume a representative-period
        # result. Pre-only jobs read the Step-2 input, which carries no
        # Step-3 temporal record.
        common.extend(["--expect-temporal-method", "full_year_no_tsam"])
    if "grid_file" in job:
        common.extend(["--grid-file", str(job["grid_file"])])
    elif args.grid_data_path is not None:
        common.extend(["--grid-data-path", str(args.grid_data_path)])
    if getattr(args, "max_timesteps", None) is not None:
        common.extend(["--max-timesteps", str(args.max_timesteps)])
    definitions = {
        "post-hems-optimized": ("flexible", "optimized HEMS"),
        "post-hems-heuristic": ("flexible", "heuristic-assets HEMS"),
        "post-inflex-heuristic": ("inflex", "heuristic-assets INFLEX"),
    }
    post_cases = tuple(
        (definitions[case_name][0], case_name, definitions[case_name][1])
        for case_name in args.result_cases
    )
    emitted_cases = (
        post_cases
        if args.skip_pre
        else (("pre-only", "pre", "pre electricity-only"), *post_cases)
    )
    for mode, case_name, label in emitted_cases:
        run_name = f"{args.run_name_prefix}_{target_network}_{case_name}"
        command = common + [
            "--post-demand-mode",
            mode,
            "--run-name",
            run_name,
            "--scenario-key",
            run_name,
            "--scenario-label",
            f"Paired 2045 {target_network} {label}",
        ]
        run_command(
            cmd=command,
            cwd=step4_dir,
            log_path=log_path,
            status=status,
            candidate_index=job_index,
            stage=f"step4_{target_network}_{case_name}",
        )
