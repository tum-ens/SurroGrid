"""Run time-series power-flow for a single scenario file.

This is the entrypoint for GridExpand step 4 (powerflow).

It selects one input `.h5` file from `Input/` based on the provided `inputfile_id`:
the script matches the prefix before the first underscore, e.g. `0_... .h5`.

The selected file is copied to `Output/` and augmented with:

- `/pwrflw/input/*` demand tables (pre/post expansion)
- `/pwrflw/output/pre/*` and `/pwrflw/output/post/*` power-flow results

See `README.md` in this folder for required HDF5 keys and expected outputs.
"""

import src.save_grid as svgrd
import src.demands as dmnds
import src.powerflow as pwrflw

import argparse
import os
from src.resource_report import resource_report

if __name__ == "__main__":
    ##### Read args + Obtain relevant input_files #####:
    parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
    parser.add_argument("inputfile_id", help="Input file name (no path)")
    parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
    args = parser.parse_args()

    # list all .h5 files in your directory
    all_entries = os.listdir("Input/")
    h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
    # find file with correct id prefix
    input_id_str = str(args.inputfile_id)
    matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
    input_file = matched_files[0]


    ##### Input Settings + Setup #####
    settings = {
        "file": input_file,
        "parallel": True,
        "n_cpu": int(args.n_cpu)
    }
    print(f"Running input file {settings['file']} (ID {args.inputfile_id}) with {settings['n_cpu']} CPUs!")

    # Save file handler
    SF = svgrd.SaveFile(settings["file"])


    ##### Obtaining Power Demands #####
    # Read-out and preprocess demand before and after DER expansion
    df_pre_demand, df_post_demand = dmnds.obtain_demand(SF)

    # Save to be retrieved later by ML model
    SF.save_df(df_pre_demand, "/pwrflw/input/demand_pre")
    SF.save_df(df_post_demand, "/pwrflw/input/demand_post")


    ##### Powerflow #####
    # Readout grid from file
    grid = SF.get_input_grid()
    # Remove any load restrictions and replace transformer with switch
    grid = pwrflw.prepare_grid(grid)

    # Run powerflow pre DER expansion
    with resource_report(name="Pre-Expansion Powerflow Run", include_children=True):
        ext_import_pre, vm_pre, line_loads_pre = pwrflw.pf(grid, df_pre_demand, settings["parallel"], settings["n_cpu"])
        # Save results
        SF.save_df(ext_import_pre, "/pwrflw/output/pre/demand_import")
        SF.save_df(vm_pre, "/pwrflw/output/pre/vm")
        SF.save_df(line_loads_pre, "/pwrflw/output/pre/line_loads")

    # Run powerflow post DER expansion
    with resource_report(name="Post-Expansion Powerflow Run", include_children=True):
        ext_import_post, vm_post, line_loads_post = pwrflw.pf(grid, df_post_demand, settings["parallel"], settings["n_cpu"])
        ##### Save results #####
        SF.save_df(ext_import_post, "/pwrflw/output/post/demand_import")
        SF.save_df(vm_post, "/pwrflw/output/post/vm")
        SF.save_df(line_loads_post, "/pwrflw/output/post/line_loads")

    print("Done!")