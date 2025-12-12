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
            "PV_electr": 100,       # 100           # % of building nodes adopting PV (0-100)
            "HP_electr": 100,       # 100           # % of building nodes adopting HP (0-100)
            "EV_electr": 100,       # 100           # % of building nodes adopting EV (0-100)

            # CPUs
            "n_cpu": int(args.n_cpu)
        }

        print("Following global settings are applied:")
        for key, value in global_settings.items():
            print(f"{key:<16} {value:>1}")
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