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
from config import config as pf_config
import argparse
import os
from sqlalchemy import text
from src.resource_report import resource_report



def _synthetic_residential_buses(save_file: svgrd.SaveFile) -> set[int]:
    """Return pandapower buses linked to synthetic residential buildings."""
    if save_file.storage != "db" or save_file.db is None or save_file.powerflow_run_id is None:
        raise ValueError("--hh-only requires --storage db so residential buses can be read from grid_building_bus.")

    query = text(
        """
        SELECT DISTINCT gbb.bus
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_building_bus gbb ON gbb.grid_case_id = pr.grid_case_id
        WHERE pr.powerflow_run_id = :powerflow_run_id
          AND gbb.bus IS NOT NULL
          AND lower(COALESCE(gbb.building_use, '')) = 'residential'
        ORDER BY gbb.bus
        """
    )
    with save_file.db.engine.connect() as conn:
        buses = [int(bus) for bus in conn.execute(query, {"powerflow_run_id": save_file.powerflow_run_id}).scalars()]
    if not buses:
        raise ValueError("--hh-only found no residential buses in surrogrid.grid_building_bus for this grid case.")
    return set(buses)


def _filter_demand_to_buses(df, buses: set[int], label: str):
    if df is None:
        return None
    if getattr(df.columns, "nlevels", 1) < 2:
        raise ValueError(f"Cannot apply --hh-only to {label}: expected MultiIndex columns (bus, component).")

    keep_columns = []
    for column in df.columns:
        try:
            keep_columns.append(int(column[0]) in buses)
        except (TypeError, ValueError):
            keep_columns.append(False)
    filtered = df.loc[:, keep_columns].copy()
    if filtered.empty:
        raise ValueError(f"--hh-only removed all {label} demand columns; residential bus mapping and demand table do not match.")
    print(f"HH-only {label}: kept {filtered.shape[1]} of {df.shape[1]} demand columns.", flush=True)
    return filtered


def _filter_grid_loads_to_buses(grid, buses: set[int]):
    if "bus" not in grid.load.columns:
        raise ValueError("Cannot apply --hh-only: pandapower grid.load has no bus column.")
    before_rows = len(grid.load)
    before_buses = grid.load["bus"].dropna().astype(int).nunique()
    mask = [False if bus != bus else int(bus) in buses for bus in grid.load["bus"]]
    grid.load = grid.load.loc[mask].copy().reset_index(drop=True)
    after_rows = len(grid.load)
    after_buses = grid.load["bus"].dropna().astype(int).nunique()
    if grid.load.empty:
        raise ValueError("--hh-only removed all pandapower load rows; residential bus mapping and grid.load do not match.")
    print(
        "HH-only grid.load: "
        f"kept {after_rows}/{before_rows} load rows on {after_buses}/{before_buses} load buses.",
        flush=True,
    )
    return grid

def _scale_hh_annual_demand(df, scale: float, label: str):
    if df is None:
        return None
    scale = float(scale)
    if scale <= 0:
        raise ValueError("--hh-annual-demand-scale must be greater than zero.")
    if scale == 1.0:
        return df
    if getattr(df.columns, "nlevels", 1) < 2:
        raise ValueError(f"Cannot scale {label}: expected MultiIndex columns (bus, component).")
    scaled = df.copy()
    component_level = scaled.columns.get_level_values(1)
    mask = component_level.isin(["electricity", "electricity-reactive"])
    if not mask.any():
        raise ValueError(f"Cannot scale {label}: no electricity or electricity-reactive columns found.")
    before_kwh = float(scaled.loc[:, component_level == "electricity"].sum().sum())
    scaled.loc[:, mask] = scaled.loc[:, mask] * scale
    after_kwh = float(scaled.loc[:, component_level == "electricity"].sum().sum())
    print(
        f"HH annual demand scaling for {label}: factor={scale:.6g}, "
        f"active energy {before_kwh:.1f} -> {after_kwh:.1f} kWh.",
        flush=True,
    )
    return scaled

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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "DB mode: run all timesteps but save only compact headline metrics "
            "to surrogrid.powerflow_summary instead of full time-series tables. "
            "With --pre-only only the pre stage is summarized; otherwise pre and post are summarized."
        ),
    )
    parser.add_argument(
        "--grid-case-id",
        type=int,
        default=None,
        help=(
            "Explicit synthetic surrogrid.grid_case_id. Required when a paired "
            "scenario HDF filename does not encode the pylovo grid identifier."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional DB powerflow run name. Useful for storing backbone-only summaries separately.",
    )
    parser.add_argument(
        "--hh-only",
        action="store_true",
        help=(
            "Synthetic-grid DB mode: restrict demand and pandapower load rows to "
            "residential building_use buses from surrogrid.grid_building_bus."
        ),
    )
    parser.add_argument(
        "--hh-annual-demand-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplier for HH-only pre-expansion electricity and reactive demand. "
            "This is intended for aggregate SWF annual-demand sensitivity checks and requires --hh-only --pre-only."
        ),
    )
    parser.add_argument(
        "--post-demand-mode",
        choices=["flexible", "no-flex"],
        default="flexible",
        help=(
            "Post-electrification demand reconstruction. 'flexible' uses optimized URBS net import; "
            "'no-flex' derives fixed heat, PV, and capped EV charging while splitting heat via optimized post-flex heatpump_air/heatpump_booster capacities."
        ),
    )
    parser.add_argument(
        "--no-flex-ev-charger-kw",
        type=float,
        default=None,
        help="EV home charger cap for --post-demand-mode no-flex. Defaults to powerflow config EV_HOME_CHARGER_KW.",
    )
    parser.add_argument(
        "--summary-nonconvergence",
        choices=["auto", "raise", "nan"],
        default="auto",
        help=(
            "Power-flow non-convergence handling for --summary-only. 'raise' aborts the grid, "
            "'nan' records failed timesteps and continues, and 'auto' uses 'nan' for no-flex "
            "summary runs and 'raise' otherwise."
        ),
    )
    args = parser.parse_args()
    if args.summary_only and args.storage != "db":
        parser.error("--summary-only requires --storage db.")
    if args.hh_only and args.storage != "db":
        parser.error("--hh-only requires --storage db.")
    if args.hh_annual_demand_scale != 1.0 and not args.hh_only:
        parser.error("--hh-annual-demand-scale requires --hh-only.")
    if args.hh_annual_demand_scale != 1.0 and not args.pre_only:
        parser.error("--hh-annual-demand-scale is only supported for --pre-only HH demand runs.")
    if args.post_demand_mode == "no-flex" and args.pre_only:
        parser.error("--post-demand-mode no-flex requires a post-electrification run, not --pre-only.")
    if args.no_flex_ev_charger_kw is not None and args.post_demand_mode != "no-flex":
        parser.error("--no-flex-ev-charger-kw requires --post-demand-mode no-flex.")
    if args.summary_nonconvergence != "auto" and not args.summary_only:
        parser.error("--summary-nonconvergence only applies with --summary-only.")

    summary_nonconvergence = args.summary_nonconvergence
    if summary_nonconvergence == "auto":
        summary_nonconvergence = "nan" if args.summary_only and args.post_demand_mode == "no-flex" else "raise"
    protect_summary_grid_state = summary_nonconvergence == "nan"

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

    no_flex_ev_charger_kw = (
        args.no_flex_ev_charger_kw
        if args.no_flex_ev_charger_kw is not None
        else pf_config.EV_HOME_CHARGER_KW
    )
    assumptions_extra = {
        "post_demand_mode": args.post_demand_mode,
    }
    if args.post_demand_mode == "no-flex":
        assumptions_extra.update({
            "no_flex_assumption": "fixed heat, rooftop PV, and EV charging profiles; heat split uses optimized post-flex heatpump_air and heatpump_booster capacities; no URBS optimized dispatch for post demand",
            "no_flex_ev_charger_kw": float(no_flex_ev_charger_kw),
            "no_flex_capacity_source": "post-flex cap_pro",
        })
    if args.hh_only:
        assumptions_extra.update({
            "demand_scope": "synthetic_hh_only",
            "hh_only_filter": "grid_building_bus.building_use == Residential",
            "hh_annual_demand_scale": float(args.hh_annual_demand_scale),
        })

    # Save file handler
    SF = svgrd.SaveFile(
        settings["file"],
        storage=args.storage,
        pre_only=args.pre_only,
        run_name=args.run_name,
        assumptions_extra=assumptions_extra,
        grid_case_id=args.grid_case_id,
    )
    residential_buses = _synthetic_residential_buses(SF) if args.hh_only else None


    ##### Obtaining Power Demands #####
    # Read-out and preprocess demand before and after DER expansion
    if args.pre_only:
        df_pre_demand = dmnds.obtain_pre_demand(SF)
        df_post_demand = None
    else:
        df_pre_demand, df_post_demand = dmnds.obtain_demand(
            SF,
            save_reactive=not args.summary_only,
            post_demand_mode=args.post_demand_mode,
            ev_charger_kw=no_flex_ev_charger_kw,
        )

    if SF.timeframe_metadata.get("optimization_space") == "scenario_unit":
        allocation = SF.get_allocation_plan()
        df_pre_demand = dmnds.project_scenario_units_to_buses(
            df_pre_demand, allocation
        )
        if df_post_demand is not None:
            df_post_demand = dmnds.project_scenario_units_to_buses(
                df_post_demand, allocation
            )

    if residential_buses is not None:
        df_pre_demand = _filter_demand_to_buses(df_pre_demand, residential_buses, "pre")
        df_post_demand = _filter_demand_to_buses(df_post_demand, residential_buses, "post")

    if args.hh_annual_demand_scale != 1.0:
        df_pre_demand = _scale_hh_annual_demand(df_pre_demand, args.hh_annual_demand_scale, "pre")

    # Save to be retrieved later by ML model unless this run is intentionally summary-only.
    if not args.summary_only:
        SF.save_df(df_pre_demand, "/pwrflw/input/demand_pre")
        if df_post_demand is not None:
            SF.save_df(df_post_demand, "/pwrflw/input/demand_post")


    ##### Powerflow #####
    # Readout grid from file
    grid = SF.get_input_grid()
    if residential_buses is not None:
        grid = _filter_grid_loads_to_buses(grid, residential_buses)
    transformer_s_rated_mva = float(grid.trafo["sn_mva"].sum()) if "sn_mva" in grid.trafo.columns else float("nan")
    cable_max_i_ka = grid.line.get("max_i_ka")
    if cable_max_i_ka is None:
        cable_max_i_ka = grid.line.assign(max_i_ka=float("nan"))["max_i_ka"]
    load_buses = grid.load["bus"].dropna().astype(int).unique().tolist() if "bus" in grid.load.columns else []
    voltage_buses = load_buses or grid.bus.index.tolist()
    # Remove any load restrictions and replace transformer with switch
    grid = pwrflw.prepare_grid(grid)
    backbone_cable_ids, voltage_buses = pwrflw.comparison_backbone_scope(grid, load_buses)
    if not voltage_buses:
        voltage_buses = load_buses or grid.bus.index.tolist()

    # Run powerflow pre DER expansion
    with resource_report(name="Pre-Expansion Powerflow Run", include_children=True):
        if args.summary_only:
            summary_pre = pwrflw.pf_summary(
                grid,
                df_pre_demand,
                transformer_s_rated_mva=transformer_s_rated_mva,
                cable_max_i_ka=cable_max_i_ka,
                voltage_buses=voltage_buses,
                cable_ids=backbone_cable_ids,
                on_nonconvergence=summary_nonconvergence,
                protect_grid_state=protect_summary_grid_state,
            )
            SF.save_summary(summary_pre, "pre")
        else:
            ext_import_pre, vm_pre, line_loads_pre = pwrflw.pf(grid, df_pre_demand, settings["parallel"], settings["n_cpu"])
            # Save results
            SF.save_df(ext_import_pre, "/pwrflw/output/pre/demand_import")
            SF.save_df(vm_pre, "/pwrflw/output/pre/vm")
            SF.save_df(line_loads_pre, "/pwrflw/output/pre/line_loads")

    if not args.pre_only:
        # Run powerflow post DER expansion
        with resource_report(name="Post-Expansion Powerflow Run", include_children=True):
            if args.summary_only:
                summary_post = pwrflw.pf_summary(
                    grid,
                    df_post_demand,
                    transformer_s_rated_mva=transformer_s_rated_mva,
                    cable_max_i_ka=cable_max_i_ka,
                    voltage_buses=voltage_buses,
                    cable_ids=backbone_cable_ids,
                    on_nonconvergence=summary_nonconvergence,
                    protect_grid_state=protect_summary_grid_state,
                )
                SF.save_summary(summary_post, "post")
            else:
                ext_import_post, vm_post, line_loads_post = pwrflw.pf(grid, df_post_demand, settings["parallel"], settings["n_cpu"])
                ##### Save results #####
                SF.save_df(ext_import_post, "/pwrflw/output/post/demand_import")
                SF.save_df(vm_post, "/pwrflw/output/post/vm")
                SF.save_df(line_loads_post, "/pwrflw/output/post/line_loads")

    print("Done!")
