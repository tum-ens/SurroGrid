# -*- coding: utf-8 -*-
import urbs
import os
import shutil
import time
import argparse
from urbs.resource_report import resource_report

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
        parser.add_argument("--pv", default=100, type=int, help="PV electrification %% (0-100)")
        parser.add_argument("--hp", default=100, type=int, help="HP electrification %% (0-100)")
        parser.add_argument("--ev", default=100, type=int, help="EV electrification %% (0-100)")
        parser.add_argument("--inflex", action="store_true",
                            help="inflexible case: no battery and heat storage, uncontrolled charging")
        parser.add_argument("--p14a", action="store_true",
                            help="section 14a control on the inflexible case (rule-based, applied to the demand)")
        parser.add_argument("--p14a-flex", action="store_true", dest="p14a_flex",
                            help="section 14a control on the flexible case (setpoints through availability)")
        parser.add_argument("--p14a-ref", dest="p14a_ref", default=None,
                            help="result h5 of the same run without control (capacities, first-pass load)")
        parser.add_argument("--p14a-lam-in", dest="p14a_lam_in", default=None,
                            help="npy with the ratios of the previous pass, taken as a floor (--p14a-flex)")
        parser.add_argument("--p14a-lam-out", dest="p14a_lam_out", default=None,
                            help="npy to write the ratios of this pass to (--p14a-flex)")
        args = parser.parse_args()

        ### Obtain relevant input_files
        # list all .h5 files in your directory
        all_entries = os.listdir("Input/")
        h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
        # find file with correct id prefix
        input_id_str = str(args.inputfile_id)
        matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
        input_file = matched_files[0]


        ### Give global run settings
        global_settings = {
            "input_file": input_file,
            # "input_file": 'N2775500E4431500_86154_1_-6.h5',    # input file name in dir "Input" 
            # "input_file": 'N2827500E4503500_93426_5_41.h5',    # input file name in dir "Input"
            "tsam": False,                           # apply time series aggregation ("True", "False")
            "noTypicalPeriods": 6,                  # tsam: number of aggregated typical periods (int, max 52) 
            "hoursPerPeriod": 168,                  # tsam: length of typical period (int)

            # Electrification
            "PV_electr": int(args.pv),       # % of building nodes adopting PV (0-100)
            "HP_electr": int(args.hp),       # % of building nodes adopting HP (0-100)
            "EV_electr": int(args.ev),       # % of building nodes adopting EV (0-100)
            "inflex": bool(args.inflex),   # inflex scenario flag
            "p14a": bool(args.p14a),
            "p14a_flex": bool(args.p14a_flex),
            "p14a_ref": args.p14a_ref,
            "p14a_lam_in": args.p14a_lam_in,
            "p14a_lam_out": args.p14a_lam_out,

            # CPUs
            "n_cpu": int(args.n_cpu)
        }

        print("Following global settings are applied:")
        for key, value in global_settings.items():
            print(f"{key:<16} {value!s:>1}")
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