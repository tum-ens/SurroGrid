"""Command lines of the aligned validation (pure functions, no I/O).

The step commands of GridExpand itself (paired runner, profile libraries) come from
:mod:`gridexpand.scenario.commands`; this module adds the two validation steps: the aligned allocation of one
provider and the expansion analyses of every provider group.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from gridexpand.scenario.commands import PROFILES, module_command, pv_profile_library_command
from gridexpand.scenario.config_loader import load_scenario_config

ALIGNED_ALLOCATION = "validation.aligned_allocation"
ALIGNED_EXPANSION = "validation.aligned_expansion"


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


def aligned_preparation_commands(
    *,
    provider: str,
    alignment_dir: Path,
    population: Path,
    dso_b_grids_dir: Path | None,
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
    weather_year: int | None = None,
    bundle_name: str | None = None,
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
    if provider == "dso_b":
        allocation += ["--dso-b-grids-dir", str(dso_b_grids_dir)]
    if bundle_name and bundle_name != provider:
        allocation += ["--bundle-name", bundle_name]
    readiness = f"{PROFILES}.paired_profile_readiness"
    weather = module_command(
        f"{PROFILES}.aligned_weather", "--paired-dir", paired_dir, "--output", weather_hdf, python=python,
    )
    if weather_year is not None:
        weather += ["--weather-year", str(weather_year)]
    stages = [
        ("allocation", allocation),
        ("weather", weather),
        ("heat_regeneration", module_command(
            f"{PROFILES}.paired_heat_profile_regeneration",
            "--paired-dir", paired_dir, "--refresh-catalog", "--resume",
            "--workers", heat_workers, "--n-cpu", 1,
            "--scenario-config", scenario_config,
            "--output-directory", paired_dir / "heat_regeneration",
            "--synthetic-library", heat_sources,
            "--weather-hdf", weather_hdf,
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
    if Path(scenario_config).exists():
        scenario, _ = load_scenario_config(scenario_config)
        if scenario.heat.space_heat_source == "internal":
            stages = [stage for stage in stages if stage[0] in {"allocation", "weather", "pv_library"}]
            stages.insert(2, ("internal_heat_readiness", module_command(
                readiness, "--paired-dir", paired_dir, "--scenario-config", scenario_config, python=python)))
    return stages
