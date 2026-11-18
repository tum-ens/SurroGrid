"""Command lines of the aligned validation (pure functions, no I/O).

The step commands of GridExpand itself (paired runner, profile libraries) come from
:mod:`gridexpand.scenario.commands`; this module adds the two validation steps: the aligned allocation of one
provider and the expansion analyses of every provider group.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from gridexpand.scenario.commands import dataset_preparation_commands, module_command

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
    return [("allocation", allocation), *dataset_preparation_commands(
        paired_dir=paired_dir, scenario_config=scenario_config, weather_hdf=weather_hdf, heat_sources=heat_sources,
        heat_library=heat_library, heat_profile_set_id=heat_profile_set_id, heat_workers=heat_workers,
        reference_year=reference_year, weather_year=weather_year, python=python,
    )]
