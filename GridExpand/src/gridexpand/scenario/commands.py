"""Command lines of the pipeline steps (pure functions, no I/O).

Every orchestrator (synthetic runner, paired runner and its adapters,
``gridexpand run``) builds its step commands here, so the argument
conventions of Steps 2-5 live in one place. Each function returns an argv
list that starts with ``python -m <module>``; ``python`` defaults to the
running interpreter.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

ALLOCATION = "gridexpand.allocation.main"
ELECTRIFICATION_PREPARATION = "gridexpand.allocation.electrification_preparation"
OPTIMIZATION = "gridexpand.optimization.run_urbs_cluster"
POWERFLOW = "gridexpand.powerflow.run_pwrflw"
REAL_POWERFLOW = "gridexpand.powerflow.run_real_swf_scenario_powerflow"
GRID_EXPANSION = "gridexpand.analysis.expansion.grid_expansion"
ALIGNED_EXPANSION = "gridexpand.analysis.expansion.aligned_expansion"
PAIRED_RUNNER = "gridexpand.paired.runner"
PAIRED_URBS_INPUT = "gridexpand.allocation.scenario_calibration.pipeline.paired_urbs_input"
PAIRED_ALLOCATION = "gridexpand.allocation.scenario_calibration.allocation.paired_allocation"
ALIGNED_ALLOCATION = "gridexpand.allocation.scenario_calibration.allocation.aligned_allocation"
PROFILES = "gridexpand.allocation.scenario_calibration.profiles"


def module_command(module: str, *args: object, python: str | None = None) -> list[str]:
    """``[python, -m, module, *args]`` with every argument as a string."""
    return [python or sys.executable, "-m", module, *(str(arg) for arg in args)]


def _flag(argv: list[str], name: str, value: object) -> None:
    """Append ``name value`` unless ``value`` is None."""
    if value is not None:
        argv += [name, str(value)]


# Step 2 ---------------------------------------------------------------------------


def allocation_command(
    grid_id: str,
    *,
    pylovo_version_id: str,
    profiles: str,
    demand_scope: str,
    timeframe_mode: str,
    model_case: str,
    profile_seed: int,
    scenario_config: Path,
    n_cpu: int,
    candidate_index: int | None = None,
    min_buildings: int | None = None,
    mobility_source: str = "pool",
    timeseries_storage: str | None = None,
    electrification_assignment: Path | None = None,
    case_qualified_output: bool = False,
    python: str | None = None,
) -> list[str]:
    """Step 2 (``gridexpand allocate``) of one DB grid."""
    argv = module_command(ALLOCATION, grid_id, "--storage", "db", "--pylovo-version-id", pylovo_version_id,
                          python=python)
    _flag(argv, "--candidate-index", candidate_index)
    _flag(argv, "--min-buildings", min_buildings)
    argv += ["--profiles", profiles, "--demand-scope", demand_scope, "--mobility-source", mobility_source]
    _flag(argv, "--timeseries-storage", timeseries_storage)
    argv += [
        "--timeframe-mode", timeframe_mode,
        "--model-case", model_case,
        "--profile-seed", str(profile_seed),
        "--scenario-config", str(scenario_config),
    ]
    _flag(argv, "--electrification-assignment", electrification_assignment)
    argv += ["--n_cpu", str(n_cpu)]
    if case_qualified_output:
        argv.append("--case-qualified-output")
    return argv


def electrification_preparation_command(
    ags: str | int,
    *,
    min_buildings: int,
    pylovo_version_id: str,
    demand_scope: str,
    mobility_source: str,
    profile_seed: int,
    scenario_config: Path,
    output: Path,
    plz: int | None = None,
    python: str | None = None,
) -> list[str]:
    """Regional electrification assignment of the candidate grids of an AGS (or one PLZ)."""
    argv = module_command(
        ELECTRIFICATION_PREPARATION,
        "--ags", ags,
        "--min-buildings", min_buildings,
        "--pylovo-version-id", pylovo_version_id,
        "--demand-scope", demand_scope,
        "--mobility-source", mobility_source,
        "--profile-seed", profile_seed,
        "--scenario-config", scenario_config,
        "--output", output,
        python=python,
    )
    _flag(argv, "--plz", plz)
    return argv


# Step 3 ---------------------------------------------------------------------------


def optimization_command(
    input_name: str,
    *,
    n_cpu: int,
    scenario_config: Path,
    cluster_concurrency: int | None = None,
    solver: str | None = None,
    reduce_only: bool = False,
    python: str | None = None,
) -> list[str]:
    """Step 3 (``gridexpand optimize``) of one Step 2 input; ``n_cpu`` = building clusters."""
    argv = module_command(OPTIMIZATION, input_name, "--n_cpu", n_cpu, "--scenario-config", scenario_config,
                          python=python)
    _flag(argv, "--cluster-concurrency", cluster_concurrency)
    _flag(argv, "--solver", solver)
    if reduce_only:
        argv.append("--reduce-only")
    return argv


# Step 4 ---------------------------------------------------------------------------


def powerflow_command(
    input_name: str,
    *,
    n_cpu: int,
    run_name: str | None = None,
    outputs: Sequence[str] = ("raw",),
    summary_run_name: str | None = None,
    pre_only: bool = False,
    post_demand_mode: str | None = None,
    inflex_ev_charger_kw: float | None = None,
    pylovo_version_id: str | None = None,
    grid_case_id: int | None = None,
    hh_only: bool = False,
    summary_nonconvergence: str | None = None,
    summary_grid_scope: str | None = None,
    max_timesteps: int | None = None,
    expect_temporal_method: str | None = None,
    python: str | None = None,
) -> list[str]:
    """Step 4 (``gridexpand powerflow``) of one synthetic-grid scenario file (DB storage).

    ``outputs`` is ``raw``, ``summary`` or both (one power-flow pass); with both,
    ``summary_run_name`` names the summary run.
    """
    argv = module_command(POWERFLOW, input_name, python=python)
    _flag(argv, "--grid-case-id", grid_case_id)
    argv += ["--storage", "db"]
    if pre_only:
        argv.append("--pre-only")
    argv += ["--outputs", ",".join(outputs)]
    _flag(argv, "--summary-nonconvergence", summary_nonconvergence)
    _flag(argv, "--summary-grid-scope", summary_grid_scope)
    _flag(argv, "--run-name", run_name)
    _flag(argv, "--summary-run-name", summary_run_name)
    _flag(argv, "--post-demand-mode", post_demand_mode)
    _flag(argv, "--inflex-ev-charger-kw", inflex_ev_charger_kw)
    argv += ["--n_cpu", str(n_cpu)]
    _flag(argv, "--pylovo-version-id", pylovo_version_id)
    _flag(argv, "--max-timesteps", max_timesteps)
    _flag(argv, "--expect-temporal-method", expect_temporal_method)
    if hh_only:
        argv.append("--hh-only")
    return argv


def real_powerflow_command(
    *,
    plz: int,
    lv_id: int,
    provider: str,
    profile_seed: int,
    urbs_result_hdf: Path,
    summary_grid_scope: str,
    post_demand_mode: str,
    run_name: str,
    scenario_label: str,
    expect_temporal_method: str | None = None,
    grid_file: str | None = None,
    grid_data_path: Path | None = None,
    max_timesteps: int | None = None,
    python: str | None = None,
) -> list[str]:
    """Step 4 of one real DSO grid; the run name doubles as scenario key."""
    argv = module_command(
        REAL_POWERFLOW,
        "--plz", plz,
        "--lv-id", lv_id,
        "--provider", provider,
        "--profile-seed", profile_seed,
        "--urbs-result-hdf", urbs_result_hdf,
        "--summary-grid-scope", summary_grid_scope,
        python=python,
    )
    _flag(argv, "--expect-temporal-method", expect_temporal_method)
    if grid_file is not None:
        argv += ["--grid-file", str(grid_file)]
    elif grid_data_path is not None:
        argv += ["--grid-data-path", str(grid_data_path)]
    _flag(argv, "--max-timesteps", max_timesteps)
    argv += [
        "--post-demand-mode", post_demand_mode,
        "--run-name", run_name,
        "--scenario-key", run_name,
        "--scenario-label", scenario_label,
    ]
    return argv


# Step 5 ---------------------------------------------------------------------------


def expansion_command(
    run_name: str,
    *,
    stage: str,
    analysis_key: str,
    ags: str | int | None = None,
    plz: int | None = None,
    data_source: str | None = None,
    exclude_real_lv_ids: Iterable[int] = (),
    note: str | None = None,
    replace: bool = True,
    python: str | None = None,
) -> list[str]:
    """Materialize one expansion analysis from the summaries of ``run_name``."""
    argv = module_command(GRID_EXPANSION, "--run-name", run_name, python=python)
    _flag(argv, "--data-source", data_source)
    argv += ["--stage", stage]
    _flag(argv, "--ags", ags)
    _flag(argv, "--plz", plz)
    for lv_id in exclude_real_lv_ids:
        argv += ["--exclude-real-lv-id", str(lv_id)]
    argv += ["--analysis-key", analysis_key]
    _flag(argv, "--note", note)
    if replace:
        argv.append("--replace")
    return argv


def aligned_expansion_command(
    run_id: str,
    *,
    providers: Iterable[str],
    cases: Iterable[str],
    pylovo_version_id: str,
    python: str | None = None,
) -> list[str]:
    """Expansion analyses of every provider and case of an aligned run."""
    return module_command(
        ALIGNED_EXPANSION,
        "--run-id", run_id,
        "--providers", *providers,
        "--cases", *cases,
        "--pylovo-version-id", pylovo_version_id,
        python=python,
    )


# Paired runner and its preparation --------------------------------------------------


def paired_runner_command(
    *,
    paired_dataset_id: str,
    pylovo_version_id: str,
    scenario_config: Path,
    target: str,
    workers: int,
    step3_cpus: int,
    step3_cluster_concurrency: int,
    step4_cpus: int,
    powerflow_grid_scope: str,
    profile_seed: int,
    scenario_label: str,
    run_name_prefix: str,
    run_dir: Path,
    materialization_case: str | None = None,
    result_cases: Sequence[str] = (),
    provider: str | None = None,
    max_timesteps: int | None = None,
    job_subset: Path | None = None,
    target_grid_id: int | None = None,
    cleanup_intermediates: bool = False,
    resume: bool = False,
    skip_pre: bool = False,
    pre_only: bool = False,
    python: str | None = None,
) -> list[str]:
    """One paired-runner invocation (one execution group, one provider)."""
    argv = module_command(
        PAIRED_RUNNER,
        "--paired-dataset-id", paired_dataset_id,
        "--pylovo-version-id", pylovo_version_id,
        python=python,
    )
    _flag(argv, "--provider", provider)
    argv += [
        "--scenario-config", str(scenario_config),
        "--target", target,
        "--workers", str(workers),
        "--step3-cpus", str(step3_cpus),
        "--step3-cluster-concurrency", str(step3_cluster_concurrency),
        "--step4-cpus", str(step4_cpus),
        "--powerflow-grid-scope", powerflow_grid_scope,
        "--profile-seed", str(profile_seed),
        "--scenario-label", scenario_label,
        "--run-name-prefix", run_name_prefix,
    ]
    _flag(argv, "--max-timesteps", max_timesteps)
    _flag(argv, "--job-subset", job_subset)
    _flag(argv, "--target-grid-id", target_grid_id)
    if cleanup_intermediates:
        argv.append("--cleanup-intermediates")
    if resume:
        argv.append("--resume")
    if pre_only:
        argv.append("--pre-only")
    else:
        if materialization_case is None or not result_cases:
            raise ValueError("A paired post run needs a materialization case and result cases.")
        argv += ["--model-case", materialization_case, "--result-cases", *result_cases]
        if skip_pre:
            argv.append("--skip-pre")
    argv += ["--run-dir", str(run_dir)]
    return argv


def paired_materialize_command(
    *,
    paired_dir: Path,
    target_network: str,
    target_grid_id: int,
    scenario_label: str,
    profile_seed: int,
    weather_source_hdf: Path,
    model_case: str,
    scenario_config: Path,
    heat_profile_library: Path | None = None,
    allow_diagnostic_heat_fallback: bool = False,
    python: str | None = None,
) -> list[str]:
    """Step 2 of one paired target grid (``paired_urbs_input``)."""
    argv = module_command(
        PAIRED_URBS_INPUT,
        "--paired-dir", paired_dir,
        "--target-network", target_network,
        "--target-grid-id", target_grid_id,
        "--scenario-label", scenario_label,
        "--profile-seed", profile_seed,
        "--weather-source-hdf", weather_source_hdf,
        "--model-case", model_case,
        "--scenario-config", scenario_config,
        python=python,
    )
    _flag(argv, "--heat-profile-library", heat_profile_library)
    if allow_diagnostic_heat_fallback:
        argv.append("--allow-diagnostic-heat-fallback")
    return argv


def pv_profile_library_command(
    *,
    roof_catalog: Path,
    weather_source_hdf: Path,
    output: Path,
    reference_year: int,
    python: str | None = None,
) -> list[str]:
    """Angle-binned PV profile library of a paired dataset."""
    return module_command(
        f"{PROFILES}.pv_profile_library",
        "--roof-catalog", roof_catalog,
        "--weather-source-hdf", weather_source_hdf,
        "--output", output,
        "--reference-year", reference_year,
        python=python,
    )


def paired_preparation_commands(
    *,
    ags: int,
    plz: int,
    milestone_year: int,
    pylovo_version_id: str,
    min_buildings: int,
    scenario_config: Path,
    profile_seed: int,
    paired_dir: Path,
    heat_library: Path,
    weather_hdf: Path,
    reference_year: int,
    python: str | None = None,
) -> list[tuple[str, list[str]]]:
    """``(stage, argv)`` that build a SWF paired dataset (``pipeline: paired_validation``)."""
    return [
        ("prepare_allocation", module_command(
            PAIRED_ALLOCATION,
            "--ags", ags,
            "--plz", plz,
            "--final-year", milestone_year,
            "--pylovo-version-id", pylovo_version_id,
            "--min-buildings", min_buildings,
            "--scenario-config", scenario_config,
            "--profile-seed", profile_seed,
            "--output-dir", paired_dir,
            python=python,
        )),
        ("prepare_heat_profiles", module_command(
            f"{PROFILES}.paired_profile_readiness",
            "--paired-dir", paired_dir,
            "--heat-profile-library", heat_library,
            python=python,
        )),
        ("prepare_pv_profiles", pv_profile_library_command(
            roof_catalog=paired_dir / "paired_roof_sections.csv",
            weather_source_hdf=weather_hdf,
            output=paired_dir / "paired_pv_profile_library.h5",
            reference_year=reference_year,
            python=python,
        )),
    ]


def aligned_preparation_commands(
    *,
    provider: str,
    alignment_dir: Path,
    population: Path,
    uzw_grids_dir: Path | None,
    pylovo_version_id: str,
    scenario_config: Path,
    profile_seed: int,
    paired_dir: Path,
    weather_hdf: Path,
    heat_sources: Path,
    heat_library: Path,
    heat_profile_set_id: str,
    heat_workers: int,
    reference_year: int,
    python: str | None = None,
) -> list[tuple[str, list[str]]]:
    """``(stage, argv)`` that build one provider's aligned paired dataset."""
    allocation = module_command(
        ALIGNED_ALLOCATION,
        "--provider", provider,
        "--alignment-dir", alignment_dir,
        "--population", population,
        "--pylovo-version-id", pylovo_version_id,
        "--scenario-config", scenario_config,
        "--profile-seed", profile_seed,
        "--output-dir", paired_dir,
        python=python,
    )
    if provider == "uzw":
        allocation += ["--uzw-grids-dir", str(uzw_grids_dir)]
    readiness = f"{PROFILES}.paired_profile_readiness"
    return [
        ("allocation", allocation),
        ("weather", module_command(
            f"{PROFILES}.aligned_weather", "--paired-dir", paired_dir, "--output", weather_hdf, python=python,
        )),
        ("heat_regeneration", module_command(
            f"{PROFILES}.paired_heat_profile_regeneration",
            "--paired-dir", paired_dir, "--refresh-catalog", "--resume",
            "--workers", heat_workers, "--n-cpu", 1,
            "--scenario-config", scenario_config,
            "--output-directory", paired_dir / "heat_regeneration",
            "--synthetic-library", heat_sources,
            python=python,
        )),
        ("heat_readiness_sources", module_command(
            readiness, "--paired-dir", paired_dir, "--synthetic-input-dir", heat_sources, python=python,
        )),
        ("heat_library", module_command(
            f"{PROFILES}.physical_heat_profile_library",
            "--source-catalog", paired_dir / "paired_heat_profile_catalog.csv",
            "--source-hdf-dir", heat_sources, "--source-mode", "exact",
            "--output", heat_library,
            "--profile-set-id", heat_profile_set_id,
            python=python,
        )),
        ("heat_readiness_library", module_command(
            readiness, "--paired-dir", paired_dir, "--synthetic-input-dir", heat_sources,
            "--heat-profile-library", heat_library,
            python=python,
        )),
        ("pv_library", pv_profile_library_command(
            roof_catalog=paired_dir / "paired_roof_sections.csv",
            weather_source_hdf=weather_hdf,
            output=paired_dir / "paired_pv_profile_library.h5",
            reference_year=reference_year,
            python=python,
        )),
    ]
