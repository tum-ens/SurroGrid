#!/usr/bin/env python3
"""Step 2 entry point: allocate demands for one grid and write the urbs input HDF5.

``gridexpand allocate`` parses the command line into Grid settings
(:func:`build_settings`) and runs one profile (:func:`run_allocation`): the
stages of :data:`STAGES` in order, then the HDF keys of :data:`OUTPUT_KEYS`.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gridexpand.allocation.config import config
from gridexpand.db.database import SurroGridDatabase
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key
from gridexpand.scenario.model_cases import MODEL_CASES
from gridexpand.common.timeframe import (
    TIMEFRAME_MODES,
    build_initial_metadata,
    scenario_key_for_timeframe,
)
import gridexpand.allocation.classes.grid as grd
from gridexpand.common.resource_report import resource_report

DEFAULT_SCENARIO_CONFIG = SCENARIO_CONFIG_DIR / "forchheim_2045_synthetic.yaml"


PROFILE_CHOICES = [
    "status_quo",
    "heat_library",
    "electricity_heat",
    "electricity_mobility",
    "electricity_heat_mobility",
    "all",
]
DEMAND_SCOPE_CHOICES = ["all", "residential"]


def profile_flags(profile):
    """Return the stage switches of a ``--profiles`` value."""
    return {
        "is_status_quo": profile == "status_quo",
        "is_heat_library": profile == "heat_library",
        "include_heat": profile in {
            "heat_library", "electricity_heat", "electricity_heat_mobility", "all"
        },
        "include_mobility": profile in {"electricity_mobility", "electricity_heat_mobility", "all"},
    }


def profile_kind(settings: dict[str, Any]) -> str:
    """Return the stage and output table of a run: status_quo, heat_library or full."""
    if settings["is_status_quo"]:
        return "status_quo"
    if settings["is_heat_library"]:
        return "heat_library"
    return "full"


def scenario_assumptions(timeframe_metadata, scenario_key, demand_scope):
    """Return the initial run assumptions of one timeframe and demand scope."""
    assumptions = dict(timeframe_metadata)
    assumptions["scenario_key"] = scenario_key
    assumptions["demand_scope"] = demand_scope
    if demand_scope == "residential":
        assumptions["demand_scope_filter"] = "included Residential component (residential_floor_area > 0)"
    return assumptions


@dataclass(frozen=True)
class Stage:
    """One ``Grid`` method of a profile, optionally inside a resource report."""

    method: str
    report: str | None = None
    when: Callable[[dict[str, Any]], bool] | None = None


def _with_heat(settings):
    return settings["include_heat"]


def _without_heat(settings):
    return not settings["include_heat"]


def _with_mobility(settings):
    return settings["include_mobility"]


# The order is part of the method: weather -> base electricity -> PV -> battery
# -> heat -> mobility. status_quo maps the civil-time electricity back to UTC+1
# before selecting the week, the other profiles after it (generate_heat does
# it for heat profiles, align_electricity_output_time otherwise).
STAGES: dict[str, tuple[Stage, ...]] = {
    "status_quo": (
        Stage("generate_electricity", "Electricity Generation"),
        Stage("align_electricity_output_time"),
        Stage("select_timeframe_after_electricity"),
        Stage("apply_timeframe_slice"),
        Stage("create_demand"),
    ),
    # A physical heat library needs weather, base electricity, heat demand and
    # COP only; PV and batteries must not block heat-profile regeneration.
    "heat_library": (
        Stage("retrieve_weather"),
        Stage("select_timeframe_from_weather"),
        Stage("generate_electricity", "Electricity Generation"),
        Stage("select_timeframe_after_electricity"),
        Stage("generate_heat", "Heat Generation"),
        Stage("apply_timeframe_slice"),
        Stage("create_demand"),
        Stage("create_tve"),
    ),
    "full": (
        Stage("retrieve_weather"),
        Stage("select_timeframe_from_weather"),
        Stage("generate_electricity", "Electricity Generation"),
        Stage("generate_solar", "Solar Generation"),
        Stage("generate_battery", "Battery Sizing"),
        Stage("select_timeframe_after_electricity"),
        Stage("generate_heat", "Heat Generation", when=_with_heat),
        Stage("align_electricity_output_time", when=_without_heat),
        Stage("generate_mobility", "Mobility Generation", when=_with_mobility),
        Stage("apply_timeframe_slice"),
        Stage("create_weather_urbs"),
        Stage("create_supim"),
        Stage("create_demand"),
        Stage("create_tve"),
        Stage("create_bsp"),
        Stage("create_processes"),
        Stage("create_commodities"),
        Stage("create_process_commodity"),
        Stage("create_storages"),
    ),
}

# (HDF key, Grid attribute, skip if empty). DB-mode runs drop the bulky
# raw_data keys in SaveFile.save_df. The key sets are part of the hand-off
# contract read by Steps 3/4 and GridForecast.
OUTPUT_KEYS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "status_quo": (
        ("raw_data/buildings", "df_buildings", False),
        ("raw_data/building_components", "df_building_components", False),
        ("raw_data/demand_component_audit", "df_demand_component_audit", False),
        ("raw_data/weather", "df_weather_raw", True),
        ("urbs_in/demand", "df_demand", False),
    ),
    "heat_library": (
        ("raw_data/buildings", "df_buildings", False),
        ("raw_data/building_components", "df_building_components", False),
        ("raw_data/demand_component_audit", "df_demand_component_audit", False),
        ("raw_data/weather", "df_weather_raw", False),
        ("raw_data/heat_asset_plan", "df_heat_asset_plan", False),
        ("raw_data/heat_asset_audit", "df_heat_audit", False),
        ("urbs_in/demand", "df_demand", False),
        ("urbs_in/eff_factor", "df_tve", False),
    ),
    "full": (
        ("raw_data/building_components", "df_building_components", False),
        ("raw_data/weather", "df_weather_raw", False),
        ("raw_data/buildings", "df_buildings", False),
        ("raw_data/electrification_assignment", "df_electrification_assignment", True),
        ("raw_data/electrification_assignment_summary", "df_electrification_summary", True),
        ("raw_data/pv_roof_sections", "df_pv_roof_catalog", False),
        ("raw_data/asset_plan", "df_pv_asset_plan", False),
        ("raw_data/pv_selected_sections", "df_pv_selected_sections", False),
        ("raw_data/pv_asset_audit", "df_pv_audit", False),
        ("raw_data/battery_asset_plan", "df_battery_asset_plan", False),
        ("raw_data/battery_asset_audit", "df_battery_audit", False),
        ("raw_data/heat_asset_plan", "df_heat_asset_plan", False),
        ("raw_data/heat_asset_audit", "df_heat_audit", False),
        ("raw_data/demand_component_audit", "df_demand_component_audit", False),
        ("urbs_in/demand", "df_demand", False),
        ("urbs_in/supim", "df_supim", False),
        ("urbs_in/eff_factor", "df_tve", False),
        ("urbs_in/buy_sell_price", "df_bsp", False),
        ("urbs_in/weather", "df_weather_urbs", False),
        ("urbs_in/process", "df_pro", False),
        ("urbs_in/commodity", "df_com", False),
        ("urbs_in/process_commodity", "df_pro_com", False),
        ("urbs_in/storage", "df_sto", False),
    ),
}

COMPLETION_MESSAGES = {
    "status_quo": "Status-quo profile generation complete. Run Step 4 with --pre-only.",
    "heat_library": "Physical heat-library profile generation complete.",
}


def build_parser() -> argparse.ArgumentParser:
    """Return the ``gridexpand allocate`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="gridexpand allocate", description="Low voltage grid DER allocation."
    )
    parser.add_argument("inputfile_id", help="Input file name (no path)")
    parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
    parser.add_argument(
        "--storage",
        choices=["h5", "db"],
        default="h5",
        help=(
            "Read raw grid input from HDF5 or database. DB mode keeps urbs_in as HDF5. "
            "HDF5 inputs of post model cases need raw_data/pv_roof_sections, which "
            "the Step 1 export does not write yet."
        ),
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="DB mode: 0-based candidate grid index for the given AGS.",
    )
    parser.add_argument("--plz", type=int, help="DB mode: pin one PLZ.")
    parser.add_argument("--kcid", type=int, help="DB mode: pin one KCID.")
    parser.add_argument("--bcid", type=int, help="DB mode: pin one BCID.")
    parser.add_argument(
        "--pylovo-version-id",
        help=(
            "DB mode: explicitly pin the pylovo topology version. "
            "Scenario runs receive this value from their run YAML."
        ),
    )
    parser.add_argument(
        "--min-buildings",
        type=int,
        default=5,
        help="DB mode: minimum buildings required when selecting AGS candidates.",
    )
    parser.add_argument(
        "--profiles",
        choices=PROFILE_CHOICES,
        default="all",
        help=(
            "Demand profile scope to generate. Use status_quo for electricity-only "
            "pre-expansion powerflow; heat_library regenerates physical heat and COP "
            "profiles without PV/battery assets; 'all' aliases electricity_heat_mobility."
        ),
    )
    parser.add_argument(
        "--demand-scope",
        choices=DEMAND_SCOPE_CHOICES,
        default="all",
        help=(
            "Building scope for demand allocation and URBS input generation. "
            "Use residential for a household-only pipeline run."
        ),
    )
    parser.add_argument(
        "--mobility-source",
        choices=["emobpy", "pool"],
        default="emobpy",
        help="Generate mobility with emobpy or assign pregenerated mobility profile pool entries.",
    )
    parser.add_argument(
        "--timeseries-storage",
        choices=["db", "temp", "both"],
        default="db",
        help=(
            "DB mode: choose whether large allocated urbs_in demand/efficiency time series "
            "are persisted to PostgreSQL. 'temp' writes only the HDF5 handoff file; 'db' "
            "and 'both' preserve the current DB persistence plus HDF5 handoff behavior."
        ),
    )
    parser.add_argument(
        "--timeframe-mode",
        choices=TIMEFRAME_MODES,
        default="full_year",
        help="Simulation timeframe. One-week modes produce 168-hour operational stress runs.",
    )
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=481527,
        help=(
            "Run-level seed for topology-independent stochastic profile realization."
        ),
    )
    parser.add_argument(
        "--model-case",
        choices=tuple(MODEL_CASES),
        default="post-hems-heuristic",
        help="Stable scenario case; determines upstream PV sizing and dispatch contract.",
    )
    parser.add_argument(
        "--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG,
        help="Scientific scenario YAML.",
    )
    parser.add_argument(
        "--electrification-assignment",
        type=Path,
        help=(
            "Optional precomputed CSV or HDF5 assignment manifest. Required "
            "for source_inventory scenarios when the input has no source evidence."
        ),
    )
    parser.add_argument(
        "--case-qualified-output",
        action="store_true",
        help="Append the model-case name to the Step-2 HDF output.",
    )
    parser.add_argument("--output-directory", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and cross-check the ``gridexpand allocate`` arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeframe_mode != "full_year" and args.mobility_source != "pool":
        parser.error("Timeslice modes require --mobility-source pool.")
    if args.model_case == "pre" and args.profiles != "status_quo":
        parser.error("The pre model case requires --profiles status_quo.")
    if args.model_case != "pre" and args.profiles == "status_quo":
        parser.error("Post model cases require an electrification profile selection.")
    return args


def _resolve_input(args: argparse.Namespace) -> tuple[str, dict[str, Any] | None]:
    """Return the input file name and, in DB mode, the resolved pylovo grid."""
    if args.storage == "h5":
        # The input is the grid file whose name starts with "<inputfile_id>_".
        h5_files = [name for name in os.listdir(config.DATA_GRID_DIR) if name.endswith(".h5")]
        matched = [name for name in h5_files if name.split("_", 1)[0] == str(args.inputfile_id)]
        return matched[0], None
    db = SurroGridDatabase()
    if args.pylovo_version_id is not None:
        db.pylovo_version_id = str(args.pylovo_version_id)
    grid_ref = db.resolve_grid_identifier(
        args.inputfile_id,
        plz=args.plz,
        kcid=args.kcid,
        bcid=args.bcid,
        candidate_index=args.candidate_index,
        min_buildings=args.min_buildings,
        demand_scope=args.demand_scope,
    )
    return grid_ref["bridge_filename"], grid_ref


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Load the scenario, resolve the input grid and return the Grid settings.

    Also applies the scenario to the module configuration that the heat and
    emobpy generators read.
    """
    scenario_config, scenario_hash = load_scenario_config(args.scenario_config)
    config.apply_scenario(scenario_config)

    timeframe_metadata = build_initial_metadata(args.timeframe_mode)
    scenario_key = scenario_key_for_timeframe(
        args.timeframe_mode,
        base_key=scenario_identity_key(scenario_config.scenario_id, scenario_hash),
    )
    assumptions = scenario_assumptions(timeframe_metadata, scenario_key, args.demand_scope)
    assumptions.update({
        "scenario_id": scenario_config.scenario_id,
        "scenario_hash": scenario_hash,
        "electrification_assignment_path": (
            str(args.electrification_assignment.resolve())
            if args.electrification_assignment is not None
            else None
        ),
        "model_case": args.model_case,
        "profile_seed": args.profile_seed,
        "pv_feed_in_tariff_eur_per_kwh": (
            scenario_config.economics.pv_feed_in_tariff_eur_per_kwh
        ),
        "battery_sizing_method": scenario_config.battery_sizing_method(args.model_case),
        "battery_energy_to_power_hours": scenario_config.battery.energy_to_power_hours,
        "heat_sizing_method": scenario_config.heat_sizing_method(args.model_case),
        "heat_scope": "residential_buildings",
    })

    inputfile, grid_ref = _resolve_input(args)
    return {
        "grid_filename": inputfile,         # Name of input file
        "grid_ref": grid_ref,               # DB-mode resolved pylovo grid metadata
        "storage": args.storage,            # h5 or db raw-grid storage
        "weather_data_exists": args.storage == "h5" or args.profiles == "status_quo",  # DB mode has no raw weather cache.
        "parallel": (int(args.n_cpu) > 1),  # Parallelized run?
        "n_cpu": int(args.n_cpu),           # cpus if parallel
        "profiles": args.profiles,
        "demand_scope": args.demand_scope,
        "mobility_source": args.mobility_source,
        "timeseries_storage": args.timeseries_storage,
        "timeframe_mode": args.timeframe_mode,
        "timeframe_metadata": timeframe_metadata,
        "scenario_key": scenario_key,
        "scenario_assumptions": assumptions,
        "scenario_config": scenario_config,
        "scenario_hash": scenario_hash,
        "electrification_assignment_path": (
            args.electrification_assignment.resolve()
            if args.electrification_assignment is not None
            else None
        ),
        "model_case": args.model_case,
        "profile_seed": args.profile_seed,
        "case_qualified_output": args.case_qualified_output,
        "output_directory": args.output_directory,
        **profile_flags(args.profiles),
    }


def run_stages(grid: grd.Grid, stages: tuple[Stage, ...]) -> None:
    """Run the stages of one profile on ``grid`` in order."""
    for stage in stages:
        if stage.when is not None and not stage.when(grid.settings):
            continue
        method = getattr(grid, stage.method)
        if stage.report is None:
            method()
        else:
            with resource_report(include_children=True, name=stage.report):
                method()


def write_outputs(grid: grd.Grid, kind: str) -> None:
    """Write the HDF keys and database rows of one profile.

    The run metadata is final here. It is flushed to the database before any
    row is written, because the database derives the time stamps of allocated
    time series from the run assumptions.
    """
    if kind == "full":
        grid.record_asset_plan_summary()
    save_file = grid.SF
    save_file.copy_save_file()
    save_file.save_timeframe_metadata()
    save_file.flush_metadata()
    with save_file.output_store():
        for key, attribute, skip_if_empty in OUTPUT_KEYS[kind]:
            frame = getattr(grid, attribute)
            if skip_if_empty and (frame is None or frame.empty):
                continue
            if kind == "full" and key == "urbs_in/demand":
                save_file.save_allocated_vehicles(grid.df_buildings, grid.battery_dict)
            save_file.save_df(frame, key)


def run_allocation(settings: dict[str, Any]) -> Path:
    """Run Step 2 for one grid and return the written HDF5 file.

    Args:
        settings: Grid settings from :func:`build_settings`.
    """
    grid = grd.Grid(settings)
    kind = profile_kind(settings)
    run_stages(grid, STAGES[kind])
    write_outputs(grid, kind)
    if kind in COMPLETION_MESSAGES:
        print(COMPLETION_MESSAGES[kind])
    return Path(grid.SF.output_path)


def main(argv: list[str] | None = None) -> None:
    """Run Step 2 for one grid; see ``gridexpand allocate --help``."""
    with resource_report(include_children=True, name="Main Script"):
        args = parse_args(argv)
        settings = build_settings(args)
        print(
            f"Running input file {settings['grid_filename']} (ID {args.inputfile_id}, "
            f"storage {args.storage}) with {settings['n_cpu']} CPUs, timeframe "
            f"{args.timeframe_mode}, and demand scope {args.demand_scope}!"
        )
        run_allocation(settings)


if __name__ == "__main__":
    main()
