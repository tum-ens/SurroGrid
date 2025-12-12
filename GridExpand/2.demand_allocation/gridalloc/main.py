#!/usr/bin/env python3
import argparse
import os
import time

import src.classes.grid as grd
from src.classes.resource_report import resource_report


if __name__ == '__main__':
    with resource_report(include_children=True, name="Main Script") as rr_main:
        ####### Input arguments: #######
        parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
        parser.add_argument("inputfile_id", help="Input file name (no path)")
        parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
        args = parser.parse_args()

        #### Obtain relevant input file ####
        # list all .h5 files in your directory
        all_entries = os.listdir("data/grids")
        h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
        # find file with correct id prefix
        input_id_str = str(args.inputfile_id)
        matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
        inputfile = matched_files[0]

        ####### Run Settings: #######
        settings = {
            # "grid_filename": "N2775500E4431500_86154_1_-6.h5"
            "grid_filename": inputfile,         # Name of input file
            "weather_data_exists": True,        # Is weather data already included in input grid file's raw data? (recommended, as on HPC cluster no outside API access)
            "parallel": (int(args.n_cpu) > 1),  # Parallelized run?
            "n_cpu": int(args.n_cpu)            # cpus if parallel 
        }                    

        print(f"Running input file {inputfile} (ID {args.inputfile_id}) with {settings['n_cpu']} CPUs!")
        #----------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------#

        ### Setup grid which stores all relevant data for assigning demands
        GRD = grd.Grid(settings)
        # GRD.df_buildings = GRD.df_buildings.iloc[0:5].reset_index(drop=True)

        ### Data and Demand Generation
        # Order of operations is important: Weather -> Solar -> Electricity -> Heat -> Mobility
        GRD.retrieve_weather()          # Weather

        with resource_report(include_children=True, name="Solar Generation") as rr:
            GRD.generate_solar()        # Solar data
        with resource_report(include_children=True, name="Electricity Generation") as rr:
            GRD.generate_electricity()  # Electricity
        with resource_report(include_children=True, name="Heat Generation") as rr:
            GRD.generate_heat()         # Heat
        with resource_report(include_children=True, name="Mobility Generation") as rr:
            GRD.generate_mobility()     # Mobility

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

        ### Saving Grid Data to .h5
        df = GRD.save_grid_data()