#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "config" / "scenarios"
    / "forchheim_2045.yaml"
)

from config import config
from common.database import SurroGridDatabase
from scenario_pipeline.config_loader import load_scenario_config
from scenario_pipeline.model_cases import MODEL_CASES
from common.timeframe import (
    TIMEFRAME_MODES,
    build_initial_metadata,
    scenario_key_for_timeframe,
)
import src.classes.grid as grd
from src.classes.resource_report import resource_report


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
    return {
        "is_status_quo": profile == "status_quo",
        "is_heat_library": profile == "heat_library",
        "include_heat": profile in {
            "heat_library", "electricity_heat", "electricity_heat_mobility", "all"
        },
        "include_mobility": profile in {"electricity_mobility", "electricity_heat_mobility", "all"},
    }


def scenario_base_key_for_scope(demand_scope):
    if demand_scope == "all":
        return "baseline_static"
    if demand_scope == "residential":
        return "baseline_static_hh_only"
    raise ValueError(f"Unknown demand scope: {demand_scope}")


def scenario_assumptions(timeframe_metadata, scenario_key, demand_scope):
    assumptions = dict(timeframe_metadata)
    assumptions["scenario_key"] = scenario_key
    assumptions["demand_scope"] = demand_scope
    if demand_scope == "residential":
        assumptions["demand_scope_filter"] = "building_use == Residential, with residential building-type fallback for HDF inputs"
    return assumptions


if __name__ == '__main__':
    with resource_report(include_children=True, name="Main Script") as rr_main:
        ####### Input arguments: #######
        parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
        parser.add_argument("inputfile_id", help="Input file name (no path)")
        parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
        parser.add_argument(
            "--storage",
            choices=["h5", "db"],
            default="h5",
            help="Read raw grid input from HDF5 or database. DB mode keeps urbs_in as HDF5.",
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
            "--case-qualified-output",
            action="store_true",
            help="Append the model-case name to the Step-2 HDF output.",
        )
        parser.add_argument("--output-directory", type=Path)
        args = parser.parse_args()
        if args.timeframe_mode != "full_year" and args.mobility_source != "pool":
            parser.error("Timeslice modes require --mobility-source pool.")
        if args.model_case == "pre" and args.profiles != "status_quo":
            parser.error("The pre model case requires --profiles status_quo.")
        if args.model_case != "pre" and args.profiles == "status_quo":
            parser.error("Post model cases require an electrification profile selection.")
        scenario_config, scenario_hash = load_scenario_config(args.scenario_config)
        config.apply_scenario(scenario_config)

        timeframe_metadata = build_initial_metadata(args.timeframe_mode)
        scenario_key = scenario_key_for_timeframe(
            args.timeframe_mode,
            base_key=scenario_base_key_for_scope(args.demand_scope),
        )
        assumptions = scenario_assumptions(timeframe_metadata, scenario_key, args.demand_scope)
        assumptions.update({
            "scenario_id": scenario_config.scenario_id,
            "scenario_hash": scenario_hash,
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

        #### Obtain relevant input file ####
        grid_ref = None
        if args.storage == "h5":
            # list all .h5 files in your directory
            all_entries = os.listdir("data/grids")
            h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
            # find file with correct id prefix
            input_id_str = str(args.inputfile_id)
            matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
            inputfile = matched_files[0]
        else:
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
            inputfile = grid_ref["bridge_filename"]

        ####### Run Settings: #######
        profile_settings = profile_flags(args.profiles)
        settings = {
            # "grid_filename": "N2775500E4431500_86154_1_-6.h5"
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
            "model_case": args.model_case,
            "profile_seed": args.profile_seed,
            "case_qualified_output": args.case_qualified_output,
            "output_directory": args.output_directory,
            **profile_settings,
        }

        print(
            f"Running input file {inputfile} (ID {args.inputfile_id}, storage {args.storage}) "
            f"with {settings['n_cpu']} CPUs, timeframe {args.timeframe_mode}, "
            f"and demand scope {args.demand_scope}!"
        )
        #----------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------#

        ### Setup grid which stores all relevant data for assigning demands
        GRD = grd.Grid(settings)
        # GRD.df_buildings = GRD.df_buildings.iloc[0:5].reset_index(drop=True)

        ### Data and Demand Generation
        # Order is important: Weather -> base electricity -> PV -> battery -> Heat -> Mobility.
        if settings["is_status_quo"]:
            with resource_report(include_children=True, name="Electricity Generation") as rr:
                GRD.generate_electricity()
            GRD.align_electricity_output_time()
            GRD.select_timeframe_after_electricity()
            GRD.apply_timeframe_slice()
            GRD.create_demand()
            GRD.SF.copy_save_file()
            GRD.SF.save_timeframe_metadata()
            GRD.SF.save_df(GRD.df_buildings, "raw_data/buildings")
            if GRD.df_weather_raw is not None and not GRD.df_weather_raw.empty:
                GRD.SF.save_df(GRD.df_weather_raw, "raw_data/weather")
            GRD.SF.save_df(GRD.df_demand, "urbs_in/demand")
            print("Status-quo profile generation complete. Run Step 4 with --pre-only.")
        elif settings["is_heat_library"]:
            # A physical heat library needs weather, base electricity, heat demand,
            # and COP only. PV and batteries are independent physical assumptions
            # and must not block heat-profile regeneration.
            GRD.retrieve_weather()
            GRD.select_timeframe_from_weather()
            with resource_report(include_children=True, name="Electricity Generation") as rr:
                GRD.generate_electricity()
            GRD.select_timeframe_after_electricity()
            with resource_report(include_children=True, name="Heat Generation") as rr:
                GRD.generate_heat()
            GRD.apply_timeframe_slice()
            GRD.create_demand()
            GRD.create_tve()

            GRD.SF.copy_save_file()
            GRD.SF.save_timeframe_metadata()
            GRD.SF.save_df(GRD.df_buildings, "raw_data/buildings")
            GRD.SF.save_df(GRD.df_weather_raw, "raw_data/weather")
            GRD.SF.save_df(GRD.df_heat_asset_plan, "raw_data/heat_asset_plan")
            GRD.SF.save_df(GRD.df_heat_audit, "raw_data/heat_asset_audit")
            GRD.SF.save_df(GRD.df_demand, "urbs_in/demand")
            GRD.SF.save_df(GRD.df_tve, "urbs_in/eff_factor")
            print("Physical heat-library profile generation complete.")
        else:
            GRD.retrieve_weather()          # Weather
            GRD.select_timeframe_from_weather()

            with resource_report(include_children=True, name="Electricity Generation") as rr:
                GRD.generate_electricity()  # Electricity
            with resource_report(include_children=True, name="Solar Generation") as rr:
                GRD.generate_solar()        # LoD2 potential and PV sizing use base electricity
            with resource_report(include_children=True, name="Battery Sizing") as rr:
                GRD.generate_battery()      # Uses PV capacity and base electricity only
            GRD.select_timeframe_after_electricity()
            if settings["include_heat"]:
                with resource_report(include_children=True, name="Heat Generation") as rr:
                    GRD.generate_heat()     # Heat
            else:
                GRD.align_electricity_output_time()
            if settings["include_mobility"]:
                with resource_report(include_children=True, name="Mobility Generation") as rr:
                    GRD.generate_mobility() # Mobility
            GRD.apply_timeframe_slice()

            ### Conversion of generated data to urbs outputs
            GRD.create_weather_urbs()       # Weather
            GRD.create_supim()              # SupIm
            GRD.create_demand()             # Demands
            GRD.create_tve()                # Eff Factor
            GRD.create_bsp()                # Buy-Sell-Price
            GRD.create_processes()          # Process
            GRD.create_commodities()        # Commoditites
            GRD.create_process_commodity()  # Process Commodity
            GRD.create_storages()           # Storage

            ### Saving Grid Data
            GRD.save_grid_data()
