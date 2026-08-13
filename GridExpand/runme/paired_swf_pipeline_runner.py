#!/usr/bin/env python3
"""Run the publication-oriented paired SWF scenario on both grid models.

The runner uses one stable scenario-unit profile contract for real and
synthetic targets. TSAM methodology comes from the scenario YAML. A canonical
mapping is recorded once and every real and synthetic result must reproduce it
before power flow starts.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import h5py
import pandas as pd
from dotenv import load_dotenv

from pipeline_support import (
    ENV_PATH,
    REPO_ROOT_DEFAULT,
    StatusLog,
    latest_step3_result,
    run_batch_command,
    run_command,
    utc_now,
)

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PAIRED_DIR = (
    GRIDEXPAND_DIR
    / "2.demand_allocation"
    / "gridalloc"
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_paired_v5_91301_station_hybrid_v2"
)
DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "config" / "scenarios"
    / "forchheim_2045.yaml"
)
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
from scenario_pipeline.config_loader import load_scenario_config
TARGET_CHOICES = ("real_swf", "synthetic", "both")


def _load_jobs(
    paired_dir: Path,
    target: str,
    target_grid_id: int | None,
) -> list[dict[str, Any]]:
    targets = ("real_swf", "synthetic") if target == "both" else (target,)
    jobs: list[dict[str, Any]] = []
    for target_network in targets:
        filename = (
            "paired_real_bus_allocation_plan.csv"
            if target_network == "real_swf"
            else "paired_synthetic_bus_allocation_plan.csv"
        )
        plan = pd.read_csv(paired_dir / filename)
        grid_ids = sorted(
            pd.to_numeric(plan["target_grid_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if target_grid_id is not None:
            grid_ids = [
                grid_id for grid_id in grid_ids if grid_id == int(target_grid_id)
            ]
        for grid_id in grid_ids:
            grid_plan = plan[
                pd.to_numeric(plan["target_grid_id"], errors="coerce")
                .astype("Int64")
                .eq(grid_id)
            ]
            job = {
                "target_network": target_network,
                "target_grid_id": int(grid_id),
            }
            if target_network == "synthetic":
                bridge_names = (
                    grid_plan["synthetic_bridge_filename"].dropna().astype(str).unique()
                )
                if len(bridge_names) != 1:
                    raise ValueError(
                        f"Synthetic grid {grid_id} has ambiguous bridge names."
                    )
                job["bridge_filename"] = bridge_names[0]
            jobs.append(job)
    return [{**job, "job_index": index} for index, job in enumerate(jobs)]


def _input_name(job: dict[str, Any], scenario_label: str) -> str:
    if job["target_network"] == "real_swf":
        prefix = f"paired_real_swf_LV_{job['target_grid_id']:03d}"
    else:
        prefix = f"paired_synthetic_{Path(job['bridge_filename']).stem}"
    return f"{prefix}_{scenario_label}.h5"


def _materialize_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "--project",
        "..",
        "python",
        "-m",
        "src.scenario_calibration.pipeline.paired_urbs_input",
        "--paired-dir",
        str(args.paired_dir),
        "--target-network",
        str(job["target_network"]),
        "--target-grid-id",
        str(job["target_grid_id"]),
        "--scenario-label",
        args.scenario_label,
        "--seed",
        str(args.seed),
        "--weather-source-hdf",
        str(args.weather_source_hdf),
        "--model-case",
        args.model_case,
        "--scenario-config",
        str(args.scenario_config),
    ]
    if args.heat_profile_library is not None:
        command.extend(["--heat-profile-library", str(args.heat_profile_library)])
    if args.allow_diagnostic_heat_fallback:
        command.append("--allow-diagnostic-heat-fallback")
    return command


def _tsam_arguments(
    args: argparse.Namespace, *, reduce_only: bool = False
) -> list[str]:
    arguments = ["--scenario-config", str(args.scenario_config)]
    if reduce_only:
        arguments.append("--reduce-only")
    return arguments


def _read_tsam_signature(result_hdf: Path) -> dict[str, Any]:
    prefix = "/urbs_out/tsam"

    def values(name: str) -> list[Any]:
        frame = pd.read_hdf(result_hdf, f"{prefix}/{name}")
        return frame.to_numpy().reshape(-1).tolist()

    return {
        "selection_variables": ["Tamb", "Irradiation"],
        "cluster_center_indices": [
            int(value) for value in values("clusterCenterIndices")
        ],
        "cluster_order": [int(value) for value in values("clusterOrder")],
        "hours_per_period": int(values("hoursPerPeriod")[0]),
        "number_of_typical_periods": int(values("noTypicalPeriods")[0]),
        "extreme_period_method": str(values("extremePeriodMethod")[0]),
    }


def _prepare_shared_tsam_reference(
    *,
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    repo_root: Path,
    status: StatusLog,
) -> dict[str, Any] | None:
    if not args.tsam:
        return None

    reference_path = args.run_dir / "shared_tsam_reference.json"
    if args.resume and reference_path.exists():
        signature = json.loads(reference_path.read_text(encoding="utf-8"))
        requested = {
            "selection_variables": ["Tamb", "Irradiation"],
            "hours_per_period": args.tsam_hours_per_period,
            "number_of_typical_periods": args.tsam_periods,
            "extreme_period_method": args.tsam_extreme_method,
        }
        mismatches = {
            key: (signature.get(key), value)
            for key, value in requested.items()
            if signature.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "The saved shared TSAM reference does not match the resumed "
                f"run settings: {mismatches}"
            )
        return signature

    reference_job = jobs[0]
    gridalloc_dir = repo_root / "GridExpand" / "2.demand_allocation" / "gridalloc"
    step3_dir = repo_root / "GridExpand" / "3.urbs"
    input_hdf = step3_dir / "Input" / _input_name(reference_job, args.scenario_label)
    log_path = args.run_dir / "logs" / "shared_tsam_reference.log"
    run_batch_command(
        cmd=_materialize_command(reference_job, args),
        cwd=gridalloc_dir,
        log_path=log_path,
        status=status,
        stage="shared_tsam_materialize_reference",
    )
    run_batch_command(
        cmd=[
            "uv",
            "run",
            "python",
            "run_urbs_cluster.py",
            input_hdf.name,
            "--n_cpu",
            "1",
            *_tsam_arguments(args, reduce_only=True),
        ],
        cwd=step3_dir,
        log_path=log_path,
        status=status,
        stage="shared_tsam_select_periods",
    )
    result_hdf = latest_step3_result(step3_dir, input_hdf)
    signature = _read_tsam_signature(result_hdf)
    signature.update(
        {
            "reference_target": str(reference_job["target_network"]),
            "reference_grid_id": int(reference_job["target_grid_id"]),
        }
    )
    reference_path.write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_hdf.unlink(missing_ok=True)
    result_hdf.unlink(missing_ok=True)
    status.event(event="shared_tsam_reference_ready", **signature)
    return signature


def _prepare_shared_pv_profiles(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    status: StatusLog,
) -> Path:
    """Build the angle-binned pvlib profiles once before parallel grid jobs."""
    gridalloc_dir = repo_root / "GridExpand" / "2.demand_allocation" / "gridalloc"
    output = args.paired_dir / "paired_pv_profile_library.h5"
    run_batch_command(
        cmd=[
            "uv",
            "run",
            "--project",
            "..",
            "python",
            "-m",
            "src.scenario_calibration.profiles.pv_profile_library",
            "--roof-catalog",
            str(args.paired_dir / "paired_roof_sections.csv"),
            "--allocation-plan",
            str(args.paired_dir / "paired_real_bus_allocation_plan.csv"),
            "--weather-source-hdf",
            str(args.weather_source_hdf),
            "--output",
            str(output),
        ],
        cwd=gridalloc_dir,
        log_path=args.run_dir / "logs" / "shared_pv_profile_library.log",
        status=status,
        stage="shared_pv_profile_library",
    )
    return output


def _validate_shared_tsam(result_hdf: Path, expected: dict[str, Any] | None) -> None:
    if expected is None:
        return
    actual = _read_tsam_signature(result_hdf)
    expected_core = {key: expected[key] for key in actual}
    if actual != expected_core:
        raise ValueError(
            "TSAM mapping differs from the shared weather-based reference; "
            "power flow was not started for this grid."
        )


def _run_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    step4_dir: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    job_index = int(job["job_index"])
    target = str(job["target_network"])
    grid_id = int(job["target_grid_id"])
    if target == "real_swf":
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
        for mode, suffix, label in (
            ("pre-only", "pre", "pre electricity-only"),
            ("flexible", "flex", "post-flex"),
            ("no-flex", "no_flex", "post-no-flex"),
        ):
            run_name = f"{args.run_name_prefix}_{target}_{suffix}"
            command = common + [
                "--post-demand-mode",
                mode,
                "--run-name",
                run_name,
                "--scenario-key",
                run_name,
                "--scenario-label",
                f"Paired SWF 2045 {target} {label}",
            ]
            run_command(
                cmd=command,
                cwd=step4_dir,
                log_path=log_path,
                status=status,
                job_index=job_index,
                stage=f"step4_{target}_{suffix}",
            )
        return

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
        "--n_cpu",
        str(args.step4_cpus),
    ]
    run_command(
        cmd=common
        + [
            "--pre-only",
            "--run-name",
            f"{args.run_name_prefix}_{target}_pre",
        ],
        cwd=step4_dir,
        log_path=log_path,
        status=status,
        job_index=job_index,
        stage=f"step4_{target}_pre",
    )
    for mode, suffix in (("flexible", "flex"), ("no-flex", "no_flex")):
        command = common + [
            "--post-demand-mode",
            mode,
            "--run-name",
            f"{args.run_name_prefix}_{target}_{suffix}",
        ]
        run_command(
            cmd=command,
            cwd=step4_dir,
            log_path=log_path,
            status=status,
            job_index=job_index,
            stage=f"step4_{target}_{suffix}",
        )
    if args.cleanup_intermediates:
        step4_input.unlink(missing_ok=True)


def _run_one(
    job: dict[str, Any],
    args: argparse.Namespace,
    status: StatusLog,
    repo_root: Path,
) -> dict[str, Any]:
    job_index = int(job["job_index"])
    target = str(job["target_network"])
    grid_id = int(job["target_grid_id"])
    if args.resume and status.status_for(job_index) == "done":
        return {**job, "status": "skipped"}

    started = time.monotonic()
    log_path = args.run_dir / "logs" / f"{target}_{grid_id}.log"
    status.update(
        job_index,
        lv_id=f"{target}:{grid_id}",
        status="running",
        stage="start",
        started_at=utc_now(),
        log_file=str(log_path),
    )
    gridalloc_dir = repo_root / "GridExpand" / "2.demand_allocation" / "gridalloc"
    step3_dir = repo_root / "GridExpand" / "3.urbs"
    step4_dir = repo_root / "GridExpand" / "4.powerflow"
    input_hdf = step3_dir / "Input" / _input_name(job, args.scenario_label)
    try:
        run_command(
            cmd=_materialize_command(job, args),
            cwd=gridalloc_dir,
            log_path=log_path,
            status=status,
            job_index=job_index,
            stage=f"step2_materialize_{target}",
        )
        if not input_hdf.exists():
            raise FileNotFoundError(f"Expected paired input {input_hdf}.")

        run_command(
            cmd=[
                "uv",
                "run",
                "python",
                "run_urbs_cluster.py",
                input_hdf.name,
                "--n_cpu",
                str(args.step3_cpus),
                *_tsam_arguments(args),
            ],
            cwd=step3_dir,
            log_path=log_path,
            status=status,
            job_index=job_index,
            stage=(
                f"step3_{target}_shared_tsam"
                if args.tsam
                else f"step3_{target}_full_year"
            ),
            env_extra={"URBS_CLUSTER_CONCURRENCY": str(args.step3_cluster_concurrency)},
        )
        result_hdf = latest_step3_result(step3_dir, input_hdf)
        _validate_shared_tsam(result_hdf, args.shared_tsam_signature)
        _run_powerflows(
            job=job,
            args=args,
            result_hdf=result_hdf,
            step4_dir=step4_dir,
            log_path=log_path,
            status=status,
        )
        if args.cleanup_intermediates:
            input_hdf.unlink(missing_ok=True)
            result_hdf.unlink(missing_ok=True)
        seconds = round(time.monotonic() - started, 1)
        status.update(
            job_index,
            status="done",
            stage="complete",
            finished_at=utc_now(),
            seconds=seconds,
            message="ok",
        )
        return {**job, "status": "done", "seconds": seconds}
    except Exception as exc:
        seconds = round(time.monotonic() - started, 1)
        status.update(
            job_index,
            status="failed",
            stage="failed",
            finished_at=utc_now(),
            seconds=seconds,
            message=str(exc),
        )
        status.failed_grid(
            target_network=target,
            target_grid_id=grid_id,
            error=str(exc),
            log_file=str(log_path),
        )
        return {
            **job,
            "status": "failed",
            "seconds": seconds,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--paired-dir", type=Path, default=DEFAULT_PAIRED_DIR)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--target", choices=TARGET_CHOICES, default="both")
    parser.add_argument("--target-grid-id", type=int, default=None)
    parser.add_argument("--weather-source-hdf", type=Path, required=True)
    parser.add_argument("--heat-profile-library", type=Path)
    parser.add_argument("--scenario-label", default="swf_2045_paired_full_local")
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument(
        "--model-case",
        choices=(
            "post-inflex-heuristic", "post-hems-optimized", "post-hems-heuristic",
        ),
        default="post-hems-heuristic",
    )
    parser.add_argument("--run-name-prefix", default="paired_swf_2045_full_local")
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step3-cpus", type=int, default=8)
    parser.add_argument("--step3-cluster-concurrency", type=int, default=1)
    parser.add_argument("--step4-cpus", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-heat-fallback",
        action="store_true",
        help=(
            "Allow non-publication area-scaled heat profiles. Omit this flag "
            "for the strict comparison run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ENV_PATH, override=True)
    repo_root = args.repo_root.resolve()
    args.scenario_config = args.scenario_config.resolve()
    scenario, scenario_hash = load_scenario_config(args.scenario_config)
    if args.model_case not in scenario.model_cases:
        raise ValueError(
            f"Model case {args.model_case!r} is not enabled by {scenario.scenario_id!r}."
        )
    # Scientific TSAM choices come exclusively from the scenario YAML.
    args.tsam = scenario.time_aggregation.enabled
    args.tsam_periods = scenario.time_aggregation.number_of_typical_periods
    args.tsam_hours_per_period = scenario.time_aggregation.hours_per_period
    args.tsam_extreme_method = scenario.time_aggregation.extreme_period_method
    args.scenario_label = f"{scenario.scenario_id}_{args.model_case}"
    args.scenario_hash = scenario_hash
    args.paired_dir = args.paired_dir.resolve()
    if args.grid_data_path is not None:
        args.grid_data_path = args.grid_data_path.resolve()
    args.weather_source_hdf = args.weather_source_hdf.resolve()
    if args.heat_profile_library is not None:
        args.heat_profile_library = args.heat_profile_library.resolve()
        if not args.heat_profile_library.exists():
            raise FileNotFoundError(
                f"Physical heat-profile library not found: {args.heat_profile_library}"
            )
    args.run_dir = (
        (repo_root / args.run_dir).resolve()
        if not args.run_dir.is_absolute()
        else args.run_dir.resolve()
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    catalog_path = args.paired_dir / "paired_heat_profile_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    library_sources = (
        catalog.get("profile_source_kind", pd.Series(dtype=str))
        .astype(str)
        .eq("physical_heat_library")
    )
    if library_sources.any() and args.heat_profile_library is None:
        raise ValueError(
            "The paired heat catalog uses the physical heat-profile library. "
            "Pass --heat-profile-library with the library used by readiness."
        )
    if library_sources.any():
        expected_profile_sets = (
            catalog.loc[library_sources, "profile_set_id"].dropna().astype(str).unique()
        )
        if len(expected_profile_sets) != 1:
            raise ValueError(
                "The paired heat catalog must reference exactly one profile set; "
                f"found {expected_profile_sets.tolist()}."
            )
        with h5py.File(args.heat_profile_library, "r") as store:
            actual_profile_set = str(store.attrs.get("profile_set_id", ""))
        if actual_profile_set != expected_profile_sets[0]:
            raise ValueError(
                "Heat-profile library mismatch: paired catalog expects "
                f"{expected_profile_sets[0]!r}, got {actual_profile_set!r}."
            )
    diagnostic_profiles = int((~catalog["publication_ready"].astype(bool)).sum())
    if diagnostic_profiles and not args.allow_diagnostic_heat_fallback:
        raise ValueError(
            f"Strict paired run blocked: {diagnostic_profiles} heat-pump "
            "buildings lack exact physical heat profiles. Regenerate full-local "
            "Step 2 sources or pass --allow-diagnostic-heat-fallback for a "
            "non-publication diagnostic run."
        )

    jobs = _load_jobs(
        args.paired_dir,
        args.target,
        args.target_grid_id,
    )
    if not jobs:
        raise ValueError("No paired target grids matched the requested scope.")
    status = StatusLog(args.run_dir, resume=args.resume)
    args.pv_profile_library = _prepare_shared_pv_profiles(
        args=args,
        repo_root=repo_root,
        status=status,
    )
    args.shared_tsam_signature = _prepare_shared_tsam_reference(
        args=args, jobs=jobs, repo_root=repo_root, status=status
    )
    status.event(
        event="batch_start",
        jobs=len(jobs),
        target=args.target,
        workers=args.workers,
        temporal_method=("shared_weather_tsam" if args.tsam else "full_year_no_tsam"),
        tsam_periods=args.tsam_periods if args.tsam else None,
        tsam_hours_per_period=(args.tsam_hours_per_period if args.tsam else None),
        tsam_extreme_method=(args.tsam_extreme_method if args.tsam else None),
        paired_dir=str(args.paired_dir),
        grid_data_path=(str(args.grid_data_path) if args.grid_data_path else None),
        diagnostic_heat_profiles=diagnostic_profiles,
        publication_ready=diagnostic_profiles == 0,
    )
    started = time.monotonic()
    results = []
    if args.workers == 1:
        for job in jobs:
            results.append(_run_one(job, args, status, repo_root))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, job, args, status, repo_root): job for job in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
    result = pd.DataFrame(results).sort_values("job_index")
    result.to_csv(args.run_dir / "results.csv", index=False)
    failures = int(result["status"].eq("failed").sum())
    status.event(
        event="batch_finish",
        status="ok" if failures == 0 else "partial_failure",
        failures=failures,
        seconds=round(time.monotonic() - started, 1),
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
