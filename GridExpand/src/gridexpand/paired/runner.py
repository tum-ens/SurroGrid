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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any

import h5py
import pandas as pd
from dotenv import load_dotenv

from gridexpand.common.electrification import assignment_manifest_hash
from gridexpand.common.orchestration import (
    Cancelled,
    StatusLog,
    install_cancel_handlers,
    latest_step3_result,
    run_batch_command,
    run_command,
    utc_now,
)
from gridexpand.common.timeframe import FULL_YEAR_HOURS, read_hdf_metadata
from gridexpand.paired.comparison import (
    read_tsam_signature,
    validate_full_year_result,
    validate_shared_tsam,
)
from gridexpand.paired.datasets import resolve_paired_dataset
from gridexpand.paired.sources import (
    TARGET_ADAPTERS,
    adapters_for_scope,
)
from gridexpand.paths import (
    ENV_FILE,
    OPTIMIZATION_INPUT_DIR,
    PROJECT_DIR,
    STATISTICS_DIR,
)
from gridexpand.scenario import commands
from gridexpand.scenario.config_loader import load_scenario_config
from gridexpand.scenario.model_cases import (
    MATERIALIZATION_CASE,
    POST_MODEL_CASES,
    compatible_result_cases,
    get_model_case,
)

ENV_PATH = ENV_FILE
TARGET_CHOICES = (*TARGET_ADAPTERS, "both")
# status.tsv of a paired run: one row per target grid, keyed by "<target>:<grid id>".
STATUS_COLUMNS = (
    "job",
    "target_network",
    "target_grid_id",
    "job_index",
    "status",
    "stage",
    "started_at",
    "finished_at",
    "seconds",
    "log_file",
    "message",
)
_LEGACY_LOG_NAME = re.compile(r"^(?P<target>.+)_(?P<grid>\d+)\.log$")


def job_key(target_network: str, target_grid_id: int) -> str:
    """Resume key of one paired job (review-orch B4: never the list position)."""
    return f"{target_network}:{int(target_grid_id)}"


def _legacy_job_key(row: dict[str, str]) -> str | None:
    """Job key of a status.tsv row written before job keys (from its log file name)."""
    match = _LEGACY_LOG_NAME.match(Path(row.get("log_file") or "").name)
    return job_key(match["target"], int(match["grid"])) if match else None


def open_status(run_dir: Path, *, resume: bool) -> StatusLog:
    """The status log of a paired run directory (rows keyed by :func:`job_key`)."""
    return StatusLog(run_dir, resume=resume, key_column="job", columns=STATUS_COLUMNS, legacy_key=_legacy_job_key)


def _load_jobs(
    paired_dir: Path,
    target: str,
    target_grid_id: int | None,
    provider: str = "swf",
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for adapter in adapters_for_scope(target, provider):
        jobs.extend(adapter.load_jobs(paired_dir, target_grid_id))
    return _number_jobs(jobs)


def _number_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**job, "job_index": index, "job": job_key(job["target_network"], job["target_grid_id"])}
        for index, job in enumerate(jobs)
    ]


def _input_name(job: dict[str, Any], scenario_label: str) -> str:
    return TARGET_ADAPTERS[str(job["target_network"])].input_name(
        job, scenario_label
    )


def _materialize_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return commands.paired_materialize_command(
        paired_dir=args.paired_dir,
        target_network=str(job["target_network"]),
        target_grid_id=int(job["target_grid_id"]),
        scenario_label=args.scenario_label,
        profile_seed=args.profile_seed,
        weather_source_hdf=args.weather_source_hdf,
        model_case=args.model_case,
        scenario_config=args.scenario_config,
        heat_profile_library=args.heat_profile_library,
        allow_diagnostic_heat_fallback=args.allow_diagnostic_heat_fallback,
    )


def _prepare_shared_tsam_reference(
    *,
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
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
    input_hdf = OPTIMIZATION_INPUT_DIR / _input_name(reference_job, args.scenario_label)
    log_path = args.run_dir / "logs" / "shared_tsam_reference.log"
    run_batch_command(
        cmd=_materialize_command(reference_job, args),
        log_path=log_path,
        status=status,
        stage="shared_tsam_materialize_reference",
        env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
    )
    run_batch_command(
        cmd=commands.optimization_command(
            input_hdf.name, n_cpu=1, scenario_config=args.scenario_config, reduce_only=True
        ),
        log_path=log_path,
        status=status,
        stage="shared_tsam_select_periods",
        env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
    )
    result_hdf = latest_step3_result(input_hdf)
    signature = read_tsam_signature(result_hdf)
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
    status: StatusLog,
) -> Path:
    """Build the angle-binned pvlib profiles once before parallel grid jobs."""
    output = args.paired_dir / "paired_pv_profile_library.h5"
    run_batch_command(
        cmd=commands.pv_profile_library_command(
            roof_catalog=args.paired_dir / "paired_roof_sections.csv",
            weather_source_hdf=args.weather_source_hdf,
            output=output,
            reference_year=args.reference_year,
        ),
        log_path=args.run_dir / "logs" / "shared_pv_profile_library.log",
        status=status,
        stage="shared_pv_profile_library",
        env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
    )
    return output


def _assignment_hash_from_hdf(path: Path) -> str:
    """Validate the assignment rows against their Step-2 metadata hash."""
    metadata = read_hdf_metadata(path)
    expected = metadata.get("electrification_assignment_hash")
    if not expected:
        raise ValueError(
            f"Paired Step-2 input is missing electrification_assignment_hash: {path}"
        )
    try:
        assignment = pd.read_hdf(
            path, key="raw_data/electrification_assignment"
        )
    except (FileNotFoundError, KeyError) as exc:
        raise ValueError(
            f"Paired Step-2 input is missing raw_data/electrification_assignment: {path}"
        ) from exc
    actual = assignment_manifest_hash(assignment)
    if actual != expected:
        raise ValueError(
            "Paired Step-2 assignment rows do not match metadata: "
            f"expected={expected}, actual={actual}, path={path}"
        )
    return actual


def _run_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Identity a resumed run must match before any prior result is reused."""
    return {
        "run_id": str(args.run_dir.name),
        "scenario_id": str(args.scenario_id),
        "scenario_hash": str(args.scenario_hash),
        "temporal_method": (
            "shared_weather_tsam" if args.tsam else "full_year_no_tsam"
        ),
        "operating_hours": (None if args.tsam else int(args.operating_hours)),
        "tsam_periods": (int(args.tsam_periods) if args.tsam else None),
        "tsam_hours_per_period": (
            int(args.tsam_hours_per_period) if args.tsam else None
        ),
        "tsam_extreme_method": (
            str(args.tsam_extreme_method) if args.tsam else None
        ),
        "storage_boundary_policy": (
            "typeperiod_common_initial_state" if args.tsam else "annual_equality"
        ),
        "ev_boundary_policy": (
            "legacy_mobility_buffer" if args.tsam else "dedicated_sessions_annual_wrap"
        ),
        "paired_dataset_id": str(args.paired_dataset_id),
        "ev_pool_id": str(getattr(args, "ev_pool_id", "") or ""),
        "pylovo_version_id": str(args.pylovo_version_id),
        "profile_seed": int(args.profile_seed),
        "powerflow_grid_scope": str(args.powerflow_grid_scope),
        "target": str(args.target),
        "target_grid_id": args.target_grid_id,
        "provider": str(args.provider),
        "powerflow_max_timesteps": args.max_timesteps,
        "job_subset": (
            None if args.job_subset is None
            else json.loads(args.job_subset.read_text(encoding="utf-8"))
        ),
    }


def _assert_resume_compatible(args: argparse.Namespace) -> None:
    """Refuse to resume a run whose recorded identity differs from this request.

    A status log alone cannot show *how* an earlier job was produced. Without
    this guard a full-year request could silently reuse jobs that were completed
    under representative-period aggregation, a different scenario hash or a
    different EV service model.
    """
    identity_path = args.run_dir / "run_identity.json"
    status_path = args.run_dir / "status.tsv"
    identity = _run_identity(args)
    if not identity_path.exists():
        # Prior completed work with no recorded identity cannot be shown to have
        # been produced the requested way. Writing the new identity over it would
        # replace missing provenance with an assumption instead of checking it.
        if args.resume and status_path.exists():
            raise ValueError(
                f"{args.run_dir} contains prior job status but no "
                "run_identity.json, so how those jobs were produced cannot be "
                "established. Refusing to resume: use a new run id, or delete "
                "the stale status file if the prior work is known to be "
                "discardable."
            )
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    recorded = json.loads(identity_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (recorded.get(key), value)
        for key, value in identity.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Refusing to reuse results from an incompatible earlier run in "
            f"{args.run_dir}. Differences (recorded, requested): {mismatches}. "
            "Use a new run id instead of resuming."
        )


def _run_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    TARGET_ADAPTERS[str(job["target_network"])].run_powerflows(
        job=job,
        args=args,
        result_hdf=result_hdf,
        log_path=log_path,
        status=status,
    )


def _run_one(
    job: dict[str, Any],
    args: argparse.Namespace,
    status: StatusLog,
) -> dict[str, Any]:
    key = str(job["job"])
    target = str(job["target_network"])
    grid_id = int(job["target_grid_id"])
    if args.resume and status.status_for(key) == "done":
        return {**job, "status": "skipped"}

    started = time.monotonic()
    log_path = args.run_dir / "logs" / f"{target}_{grid_id}.log"
    status.update(
        key,
        target_network=target,
        target_grid_id=grid_id,
        job_index=int(job["job_index"]),
        status="running",
        stage="start",
        started_at=utc_now(),
        log_file=str(log_path),
    )
    input_hdf = OPTIMIZATION_INPUT_DIR / _input_name(job, args.scenario_label)
    try:
        run_command(
            cmd=_materialize_command(job, args),
            log_path=log_path,
            status=status,
            job=key,
            stage=f"step2_materialize_{target}",
            env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
        )
        if not input_hdf.exists():
            raise FileNotFoundError(f"Expected paired input {input_hdf}.")
        assignment_hash = _assignment_hash_from_hdf(input_hdf)

        if args.pre_only:
            result_hdf = input_hdf
        else:
            run_command(
                cmd=commands.optimization_command(
                    input_hdf.name,
                    n_cpu=args.step3_cpus,
                    scenario_config=args.scenario_config,
                    cluster_concurrency=args.step3_cluster_concurrency,
                ),
                log_path=log_path,
                status=status,
                job=key,
                stage=(
                    f"step3_{target}_shared_tsam"
                    if args.tsam
                    else f"step3_{target}_full_year"
                ),
                env_extra={"PYLOVO_VERSION_ID": str(args.pylovo_version_id)},
            )
            result_hdf = latest_step3_result(input_hdf)
            if args.tsam:
                validate_shared_tsam(result_hdf, args.shared_tsam_signature)
            else:
                # A full-year run must never consume a representative-period
                # result, whatever its file name looks like.
                validate_full_year_result(
                    result_hdf,
                    expected_operating_hours=args.operating_hours,
                    expected_scenario_hash=args.scenario_hash,
                    expected_delta_t_hours=1.0,
                    expected_source_year=args.reference_year,
                )
        _run_powerflows(
            job=job,
            args=args,
            result_hdf=result_hdf,
            log_path=log_path,
            status=status,
        )
        if args.cleanup_intermediates:
            input_hdf.unlink(missing_ok=True)
            result_hdf.unlink(missing_ok=True)
        seconds = round(time.monotonic() - started, 1)
        status.update(
            key,
            status="done",
            stage="complete",
            finished_at=utc_now(),
            seconds=seconds,
            message="ok",
        )
        return {
            **job,
            "status": "done",
            "seconds": seconds,
            "electrification_assignment_hash": assignment_hash,
        }
    except Cancelled as exc:
        seconds = round(time.monotonic() - started, 1)
        status.update(key, status="cancelled", finished_at=utc_now(), seconds=seconds, message=str(exc))
        return {**job, "status": "cancelled", "seconds": seconds, "error": str(exc)}
    except Exception as exc:
        seconds = round(time.monotonic() - started, 1)
        status.update(
            key,
            status="failed",
            stage="failed",
            finished_at=utc_now(),
            seconds=seconds,
            message=str(exc),
        )
        status.failed_grid(
            job=key,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Deprecated and ignored; directories come from gridexpand.paths.",
    )
    parser.add_argument(
        "--paired-dataset-id",
        help="Resolve paired artifacts by repository-local dataset convention.",
    )
    parser.add_argument(
        "--pylovo-version-id",
        help=(
            "Required topology version. With --paired-dataset-id it must match "
            "the version recorded in paired_scenario_metadata.json."
        ),
    )
    parser.add_argument("--plz", type=int, default=None)
    parser.add_argument("--paired-dir", type=Path, default=None)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--target", choices=TARGET_CHOICES, default="both")
    parser.add_argument(
        "--provider",
        choices=("swf", "uzw"),
        default="swf",
        help="DSO whose real grids 'both' pairs with the synthetic grids.",
    )
    parser.add_argument(
        "--job-subset",
        type=Path,
        default=None,
        help="JSON {target_network: [grid ids]} restricting the jobs (diagnostic subsets).",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Smoke-test cap on Step-4 power-flow timesteps; results are not for publication.",
    )
    parser.add_argument("--target-grid-id", type=int, default=None)
    parser.add_argument("--weather-source-hdf", type=Path)
    parser.add_argument("--heat-profile-library", type=Path)
    parser.add_argument("--scenario-label", default=None)
    parser.add_argument("--scenario-config", type=Path, required=True, help="Scenario YAML (config/scenarios).")
    parser.add_argument(
        "--model-case",
        choices=POST_MODEL_CASES,
        default="post-hems-heuristic",
    )
    parser.add_argument(
        "--result-cases",
        nargs="+",
        choices=POST_MODEL_CASES,
        default=None,
        help="Power-flow cases emitted from this materialized asset plan.",
    )
    parser.add_argument("--skip-pre", action="store_true")
    parser.add_argument(
        "--pre-only",
        action="store_true",
        help="Materialize paired demand and run only pre power flow; skip URBS and all post cases.",
    )
    parser.add_argument("--run-name-prefix", default="paired_swf_2045_full_local")
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=481527,
        help="Arbitrary fixed seed for reproducible stochastic input profiles.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step3-cpus", type=int, default=8)
    parser.add_argument("--step3-cluster-concurrency", type=int, default=1)
    parser.add_argument("--step4-cpus", type=int, default=1)
    parser.add_argument(
        "--powerflow-grid-scope",
        choices=("full", "backbone"),
        default="full",
    )
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    install_cancel_handlers()
    load_dotenv(ENV_PATH, override=True)
    if args.paired_dataset_id is not None:
        dataset = resolve_paired_dataset(
            args.paired_dataset_id,
            expected_pylovo_version_id=args.pylovo_version_id,
        )
        args.pylovo_version_id = dataset.pylovo_version_id
        args.plz = args.plz if args.plz is not None else dataset.plz
        args.paired_dir = args.paired_dir or dataset.paired_dir
        args.weather_source_hdf = (
            args.weather_source_hdf or dataset.weather_source_hdf
        )
        args.heat_profile_library = (
            args.heat_profile_library or dataset.heat_profile_library
        )
    missing = [
        name
        for name in (
            "plz",
            "pylovo_version_id",
            "paired_dir",
            "weather_source_hdf",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Provide --paired-dataset-id or the explicit paired inputs; "
            f"missing: {', '.join(missing)}."
        )
    args.scenario_config = args.scenario_config.resolve()
    scenario, scenario_hash = load_scenario_config(args.scenario_config)
    requested_model_case = args.model_case
    if args.pre_only:
        if args.skip_pre:
            raise ValueError("--pre-only and --skip-pre are mutually exclusive.")
        result_cases = []
        allowed_results: set[str] = set()
    else:
        result_cases = args.result_cases or [requested_model_case]
        if len(result_cases) != len(set(result_cases)):
            raise ValueError("--result-cases contains duplicates.")
        # Heuristic cases share one asset plan, materialized as post-hems-heuristic.
        allowed_results = set(compatible_result_cases(requested_model_case))
        args.model_case = MATERIALIZATION_CASE[get_model_case(requested_model_case).asset_plan]
    incompatible = set(result_cases).difference(allowed_results)
    if incompatible:
        raise ValueError(
            f"Result cases {sorted(incompatible)} are incompatible with the "
            f"{requested_model_case!r} asset plan."
        )
    args.result_cases = tuple(result_cases)
    # Scientific TSAM choices come exclusively from the scenario YAML.
    args.tsam = scenario.time_aggregation.enabled
    # Operating hours are the reference-year horizon; the full-year path requires
    # every original chronological hour, modeled once, plus the URBS
    # initialization row that carries no energy.
    args.operating_hours = FULL_YEAR_HOURS
    args.tsam_periods = scenario.time_aggregation.number_of_typical_periods
    args.tsam_hours_per_period = scenario.time_aggregation.hours_per_period
    args.tsam_extreme_method = scenario.time_aggregation.extreme_period_method
    args.reference_year = scenario.mobility.reference_year
    scenario_label_base = args.scenario_label or scenario.scenario_id
    args.scenario_label = f"{scenario_label_base}_{'pre' if args.pre_only else args.model_case}"
    args.scenario_hash = scenario_hash
    args.scenario_id = scenario.scenario_id
    # The EV session pool's pinned content identity is part of the run identity:
    # a pool that gained profiles would otherwise re-pair buildings silently.
    pool_manifest_path = (
        STATISTICS_DIR / "general" / "mobility_profile_pool" / "mobility_pool_manifest.json"
    )
    args.ev_pool_id = (
        str(json.loads(pool_manifest_path.read_text(encoding="utf-8")).get("pool_id", ""))
        if pool_manifest_path.exists()
        else ""
    )
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
        (PROJECT_DIR.parent / args.run_dir).resolve()
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
        args.provider,
    )
    if args.job_subset is not None:
        subset = json.loads(args.job_subset.read_text(encoding="utf-8"))
        jobs = [
            job for job in jobs
            if int(job["target_grid_id"]) in {int(value) for value in subset.get(job["target_network"], [])}
        ]
        jobs = _number_jobs(jobs)
    if not jobs:
        raise ValueError("No paired target grids matched the requested scope.")
    _assert_resume_compatible(args)
    status = open_status(args.run_dir, resume=args.resume)
    _prepare_shared_pv_profiles(args=args, status=status)
    args.shared_tsam_signature = (
        None
        if args.pre_only
        else _prepare_shared_tsam_reference(
            args=args, jobs=jobs, status=status
        )
    )
    status.event(
        event="batch_start",
        jobs=len(jobs),
        target=args.target,
        workers=args.workers,
        temporal_method=("shared_weather_tsam" if args.tsam else "full_year_no_tsam"),
        operating_hours=(None if args.tsam else args.operating_hours),
        storage_boundary_policy=(None if args.tsam else "annual_equality"),
        ev_boundary_policy=(
            None if args.tsam else "dedicated_sessions_annual_wrap"
        ),
        tsam_periods=args.tsam_periods if args.tsam else None,
        tsam_hours_per_period=(args.tsam_hours_per_period if args.tsam else None),
        tsam_extreme_method=(args.tsam_extreme_method if args.tsam else None),
        paired_dir=str(args.paired_dir),
        grid_data_path=(str(args.grid_data_path) if args.grid_data_path else None),
        diagnostic_heat_profiles=diagnostic_profiles,
        publication_ready=diagnostic_profiles == 0,
        paired_dataset_id=args.paired_dataset_id,
        pylovo_version_id=args.pylovo_version_id,
        materialization_case=args.model_case,
        result_cases=list(args.result_cases),
        pre_case_emitted=not args.skip_pre,
        pre_only=args.pre_only,
        provider=args.provider,
        powerflow_max_timesteps=args.max_timesteps,
    )
    started = time.monotonic()
    results = []
    if args.workers == 1:
        for job in jobs:
            results.append(_run_one(job, args, status))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, job, args, status): job for job in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
    result = pd.DataFrame(results).sort_values("job_index")
    hashes = (
        set(result["electrification_assignment_hash"].dropna().astype(str))
        if "electrification_assignment_hash" in result
        else set()
    )
    if len(hashes) > 1:
        raise ValueError(
            "Paired real/synthetic targets carry different electrification "
            f"assignment hashes: {sorted(hashes)}"
        )
    if hashes:
        status.event(
            event="paired_electrification_assignment_equivalence",
            assignment_hash=sorted(hashes)[0],
            targets=sorted(result.loc[
                result["electrification_assignment_hash"].notna(),
                "target_network",
            ].astype(str).unique()),
        )
    result.to_csv(args.run_dir / "results.csv", index=False)
    failures = int(result["status"].eq("failed").sum())
    cancelled = int(result["status"].eq("cancelled").sum())
    status.event(
        event="batch_finish",
        status="cancelled" if cancelled else "ok" if failures == 0 else "partial_failure",
        failures=failures,
        cancelled=cancelled,
        seconds=round(time.monotonic() - started, 1),
    )
    if cancelled:
        raise SystemExit(143)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
