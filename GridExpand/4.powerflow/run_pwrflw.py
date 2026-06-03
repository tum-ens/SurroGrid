"""Run time-series power-flow for a single scenario file.

This is the entrypoint for GridExpand step 4 (powerflow).

It selects one input `.h5` file from `Input/` based on the provided `inputfile_id`:
the script matches the prefix before the first underscore, e.g. `0_... .h5`.

The selected file is copied to `Output/` and augmented with:

- `/pwrflw/input/*` demand tables (pre/post expansion)
- `/pwrflw/output/pre/*` and `/pwrflw/output/post/*` power-flow results

See `README.md` in this folder for required HDF5 keys and expected outputs.
"""

import sys
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

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
    parser.add_argument(
        "--storage",
        choices=["h5", "db"],
        default="h5",
        help="Write powerflow results to HDF5 or database. DB mode still reads urbs_in/urbs_out from HDF5.",
    )
    parser.add_argument(
        "--pre-only",
        action="store_true",
        help="Run only pre-expansion powerflow from urbs_in/demand; does not require urbs_out/MILP/tau_pro.",
    )
    args = parser.parse_args()

    # list all .h5 files in your directory
    all_entries = os.listdir("Input/")
    h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
    # find file with correct id prefix
    input_id_str = str(args.inputfile_id)
    if input_id_str.endswith(".h5"):
        matched_files = [fname for fname in h5_files if fname == input_id_str]
    else:
        matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
    input_file = matched_files[0]


    ##### Input Settings + Setup #####
    settings = {
        "file": input_file,
        "parallel": True,
        "n_cpu": int(args.n_cpu),
        "storage": args.storage,
    }
    print(
        f"Running input file {settings['file']} (ID {args.inputfile_id}, storage {args.storage}) "
        f"with {settings['n_cpu']} CPUs!"
    )

    # Save file handler
    SF = svgrd.SaveFile(
        settings["file"],
        storage=args.storage,
        pre_only=args.pre_only,
    )


    ##### Obtaining Power Demands #####
    # Read-out and preprocess demand before and after DER expansion
    if args.pre_only:
        df_pre_demand = dmnds.obtain_pre_demand(SF)
        df_post_demand = None
    else:
        df_pre_demand, df_post_demand = dmnds.obtain_demand(SF)

    # Save to be retrieved later by ML model
    SF.save_df(df_pre_demand, "/pwrflw/input/demand_pre")
    if df_post_demand is not None:
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

    if not args.pre_only:
        # Run powerflow post DER expansion
        with resource_report(name="Post-Expansion Powerflow Run", include_children=True):
            ext_import_post, vm_post, line_loads_post = pwrflw.pf(grid, df_post_demand, settings["parallel"], settings["n_cpu"])
            ##### Save results #####
            SF.save_df(ext_import_post, "/pwrflw/output/post/demand_import")
            SF.save_df(vm_post, "/pwrflw/output/post/vm")
            SF.save_df(line_loads_post, "/pwrflw/output/post/line_loads")

    print("Done!")
