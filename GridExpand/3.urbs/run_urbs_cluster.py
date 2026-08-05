# -*- coding: utf-8 -*-
import urbs
import os
import shutil
import time
import argparse
import sys
from pathlib import Path
from urbs.resource_report import resource_report

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "configurations" / "scenarios"
    / "forchheim_2045.yaml"
)

from scenario_pipeline.configuration.loader import load_scenario_config

# Note - this urbs version is deviating in the following ways from urbs-lvds (04 Feb 2025):
# :: removed grid optimization, 14a/bui-react, uhp, coordination, curtailment
# :: keeping tsam, flexibility, power_price, different shares of electrification, vartariff
# :: removed microgrid inputs
# :: removed LP file and excel generation
# :: removed CO2 limit/environmental commodities
# :: removed inputs from Global Excel sheet
# :: removed multiple input scenarios
# :: removed several support_timeframes
# :: removed reactive power support


if __name__ == '__main__':
    with resource_report(include_children=True, name="Urbs Script") as rr_main:
        ### Read args:
        parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
        parser.add_argument("inputfile_id", help="Input file name (no path)")
        parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
        parser.add_argument("--tsam", action="store_true", help="Optional enable override; scenario YAML is the default.")
        parser.add_argument("--tsam-periods", type=int, default=None, help="Optional run override for TSAM type weeks.")
        parser.add_argument("--tsam-hours-per-period", type=int, default=None, help="Optional run override for hours per period.")
        parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
        parser.add_argument(
            "--tsam-extreme-method",
            choices=["append", "new_cluster_center", "replace_cluster_center"],
            default=None,
            help="How TSAM should include cold and solar extreme weeks.",
        )
        parser.add_argument(
            "--reduce-only",
            action="store_true",
            help="Run preprocessing and TSAM reduction, write reduced_data/tsam outputs, and skip the URBS optimization solve.",
        )
        args = parser.parse_args()
        scenario, scenario_hash = load_scenario_config(args.scenario_config)
        time_aggregation = scenario.time_aggregation
        tsam_enabled = bool(args.tsam or time_aggregation.enabled)
        if args.reduce_only and not tsam_enabled:
            parser.error("--reduce-only requires time_aggregation.enabled in the scenario YAML.")

        ### Obtain relevant input_files
        # list all .h5 files in your directory
        all_entries = os.listdir("Input/")
        h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
        # find file with correct id prefix
        input_id_str = str(args.inputfile_id)
        if input_id_str.endswith(".h5"):
            matched_files = [fname for fname in h5_files if fname == input_id_str]
        else:
            matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
        if not matched_files:
            raise FileNotFoundError(f"No Step 3 input file matches {input_id_str} in Input/.")
        input_file = matched_files[0]


        ### Give global run settings
        tsam_periods = args.tsam_periods or time_aggregation.number_of_typical_periods
        tsam_hours = args.tsam_hours_per_period or time_aggregation.hours_per_period
        tsam_extreme_method = (
            args.tsam_extreme_method or time_aggregation.extreme_period_method
        )
        tsam_method_settings = {
            "clustering_method": time_aggregation.clustering_method,
            "cluster_representation": time_aggregation.cluster_representation,
            "segmentation": time_aggregation.segmentation,
            "rescale_cluster_periods": time_aggregation.rescale_cluster_periods,
            "feature_weights": time_aggregation.feature_weights,
            "extreme_features": list(time_aggregation.extreme_features),
        }
        global_settings = {
            "input_file": input_file,
            # "input_file": 'N2775500E4431500_86154_1_-6.h5',    # input file name in dir "Input" 
            # "input_file": 'N2827500E4503500_93426_5_41.h5',    # input file name in dir "Input"
            "tsam": tsam_enabled,
            "noTypicalPeriods": tsam_periods,
            "hoursPerPeriod": tsam_hours,
            "tsamExtremePeriodMethod": tsam_extreme_method,
            "tsamMethodSettings": tsam_method_settings,
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario_hash,
            "reduce_only": args.reduce_only,

            # Electrification
            "PV_electr": 100,       # 100           # % of building nodes adopting PV (0-100)
            "HP_electr": 100,       # 100           # % of building nodes adopting HP (0-100)
            "EV_electr": 100,       # 100           # % of building nodes adopting EV (0-100)

            # CPUs
            "n_cpu": int(args.n_cpu)
        }

        print("Following global settings are applied:")
        for key, value in global_settings.items():
            print(f"{key:<24} {value}")
        print("\n")


        ### Input and result handling
        # Extract input path
        input_file = global_settings['input_file']
        input_dir = 'Input'
        input_path = os.path.join(input_dir, input_file)

        # Create result directory (format: datetime-inputfile-resultname), copy input and runfile into it 
        script_name = os.path.basename(__file__)
        result_dir = urbs.prepare_result_directory(input_file=input_file.replace('.h5', ''), script_name=script_name[:-3])  # time stamp + filename + script name
        result_path = os.path.join(result_dir, input_file) 
        shutil.copyfile(input_path, result_path)


        ### Run defined scenario through pyomo model setup and solver
        urbs.run_lvds_opt(input_path,      # path to input files
                        result_path,     # path to store results
                        result_dir,
                        global_settings) # all input settings  