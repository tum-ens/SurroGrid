#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase
from timeframe import (
    TIMEFRAME_MODES,
    build_initial_metadata,
    scenario_key_for_timeframe,
)
import src.classes.grid as grd
from src.classes.resource_report import resource_report


PROFILE_CHOICES = [
    "status_quo",
    "electricity_heat",
    "electricity_mobility",
    "electricity_heat_mobility",
    "all",
]


def profile_flags(profile):
    return {
        "is_status_quo": profile == "status_quo",
        "include_heat": profile in {"electricity_heat", "electricity_heat_mobility", "all"},
        "include_mobility": profile in {"electricity_mobility", "electricity_heat_mobility", "all"},
    }


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
                "pre-expansion powerflow; 'all' is an alias for electricity_heat_mobility."
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
        args = parser.parse_args()
        if args.timeframe_mode != "full_year" and args.mobility_source != "pool":
            parser.error("Timeslice modes require --mobility-source pool.")

        timeframe_metadata = build_initial_metadata(args.timeframe_mode)
        scenario_key = scenario_key_for_timeframe(args.timeframe_mode)

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
            grid_ref = db.resolve_grid_identifier(
                args.inputfile_id,
                plz=args.plz,
                kcid=args.kcid,
                bcid=args.bcid,
                candidate_index=args.candidate_index,
                min_buildings=args.min_buildings,
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
            "mobility_source": args.mobility_source,
            "timeseries_storage": args.timeseries_storage,
            "timeframe_mode": args.timeframe_mode,
            "timeframe_metadata": timeframe_metadata,
            "scenario_key": scenario_key,
            "scenario_assumptions": timeframe_metadata,
            **profile_settings,
        }

        print(
            f"Running input file {inputfile} (ID {args.inputfile_id}, storage {args.storage}) "
            f"with {settings['n_cpu']} CPUs and timeframe {args.timeframe_mode}!"
        )
        #----------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------#

        ### Setup grid which stores all relevant data for assigning demands
        GRD = grd.Grid(settings)
        # GRD.df_buildings = GRD.df_buildings.iloc[0:5].reset_index(drop=True)

        ### Data and Demand Generation
        # Order of operations is important: Weather -> Solar -> Electricity -> Heat -> Mobility
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
        else:
            GRD.retrieve_weather()          # Weather
            GRD.select_timeframe_from_weather()

            with resource_report(include_children=True, name="Solar Generation") as rr:
                GRD.generate_solar()        # Solar data
            with resource_report(include_children=True, name="Electricity Generation") as rr:
                GRD.generate_electricity()  # Electricity
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
