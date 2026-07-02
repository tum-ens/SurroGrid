"""Run compact pre-expansion power-flow summaries for real SWF LV grids.

The demand allocation intentionally mirrors the synthetic electric-only baseline:
one real DSO load row is treated as one residential household, assigned a
stochastic household electricity profile, then profiles are summed per real
pandapower load bus before running the compact p99/p01 power-flow summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import os
import re
import sys
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as pp_top
from dotenv import load_dotenv

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
STEP4_DIR = Path(__file__).resolve().parent
DEMAND_DIR = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
ENV_PATH = GRIDEXPAND_DIR / "1.grid_sampling" / ".env"

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))

from database import DEFAULT_SCENARIO_KEY, SurroGridDatabase
import src.powerflow as pwrflw

PF_ELC = 0.959
RUN_NAME = "baseline_real"
MIN_HH_ANNUAL_DEMAND_KWH = 500.0
ASSUMPTION_TEXT = (
    "Real SWF load rows are filtered to active low-voltage household loads "
    "with type=HH, names matching NS_Last or NS_ErLast, and parsed annual demand "
    f"of at least {MIN_HH_ANNUAL_DEMAND_KWH:.0f} kWh/a where annual demand metadata "
    "is available. HH rows on buses that pandapower marks as unsupplied are "
    "excluded before profile generation. Existing DSO p_mw/q_mvar values are ignored except for household "
    "load-bus placement. Cable metrics keep only lines on paths from the root "
    "bus to selected HH load buses and exclude terminal service connections. "
    "Voltage metrics are evaluated at the nearest upstream retained backbone bus."
)
ANNUAL_DEMAND_PATTERN = re.compile(r"2022:\s*([0-9]+(?:[.,][0-9]+)?)\s*kWh", re.IGNORECASE)


def _load_electricity_module():
    demand_dir = DEMAND_DIR.resolve()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(demand_dir))
    previous_config = sys.modules.pop("config", None)
    try:
        os.chdir(demand_dir)
        spec = importlib.util.spec_from_file_location(
            "gridalloc_electricity",
            demand_dir / "src" / "functions" / "electricity.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load gridalloc electricity module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(demand_dir))
        except ValueError:
            pass
        if previous_config is not None:
            sys.modules["config"] = previous_config

    stat_dir = demand_dir / "data" / "statistics"
    module.config.ELEC_LPS_PATH = str(stat_dir / "inhabited_buildings" / "elec_lps.h5")
    module.config.ELEC_GHD_PATH = str(stat_dir / "uninhabited_buildings" / "elec_ghd_per_m2.csv")
    return module


def _select_manifest_rows(root: Path, plz: int, limit: int | None, lv_id: str | None) -> list[dict[str, Any]]:
    manifest = pd.read_csv(root / "split_manifest.csv")
    selected = manifest[
        (manifest["variant"] == "radialized")
        & (manifest["status"] == "exported")
        & (manifest["category"] == "regular")
        & (manifest["load_status"] == "lvload")
    ].copy()
    if lv_id:
        lv_id_text = str(lv_id).strip()
        normalized_int = int(lv_id_text.removeprefix("LV_"))
        if pd.api.types.is_numeric_dtype(selected["lv_id"]):
            selected = selected[selected["lv_id"] == normalized_int]
        else:
            selected = selected[selected["lv_id"].astype(str).isin({str(normalized_int), f"LV_{normalized_int:03d}"})]
    selected = selected.sort_values(["load_count", "lv_id"], ascending=[True, True])
    if limit is not None:
        selected = selected.head(limit)

    rows = []
    for row in selected.to_dict("records"):
        source_file = Path(str(row["file"]))
        if not source_file.is_absolute():
            source_file = root / source_file
        row["source_file"] = str(source_file)
        row["plz"] = int(plz)
        rows.append(row)
    return rows


def _sample_household_occupants(electricity_module, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = electricity_module.config.HH_SIZE_DISTRIBUTION
    return rng.choice(
        dist["size"].to_numpy(dtype=int),
        size=n,
        p=dist["probability"].to_numpy(dtype=float),
    )


def _add_output_data_daylight_saving_shift(df_ts: pd.DataFrame) -> pd.DataFrame:
    if df_ts.empty:
        return df_ts.copy()
    ts_hour1 = 2090
    ts_hour2 = 7130
    df_ts = df_ts.copy()
    new_row = df_ts.iloc[ts_hour2].copy()
    new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
    df_ts = pd.concat([df_ts.iloc[: ts_hour2 + 1], new_row_df, df_ts.iloc[ts_hour2 + 1 :]]).reset_index(drop=True)
    return df_ts.drop(index=ts_hour1).reset_index(drop=True)


def _parse_annual_demand_kwh(load: pd.DataFrame) -> pd.Series:
    """Parse yearly measured HH demand from SWF load descriptions."""
    if "description" not in load.columns:
        return pd.Series(np.nan, index=load.index, dtype=float)
    values = []
    for description in load["description"].astype(str):
        match = ANNUAL_DEMAND_PATTERN.search(description)
        values.append(float(match.group(1).replace(",", ".")) if match else np.nan)
    return pd.Series(values, index=load.index, dtype=float)


def _household_load_rows(net: pp.pandapowerNet, allowed_buses: set[int] | None = None) -> pd.DataFrame:
    """Return active low-voltage household load rows used for profile assignment."""
    load = net.load.copy()
    if "in_service" in load.columns:
        load = load[load["in_service"].fillna(True).astype(bool)]
    load = load.dropna(subset=["bus"]).copy()
    if "type" not in load.columns or "name" not in load.columns:
        raise ValueError("The real grid load table must contain 'type' and 'name' columns for HH filtering.")

    type_mask = load["type"].astype(str).str.strip().eq("HH")
    name_mask = load["name"].astype(str).str.contains(r"NS_(?:Er)?Last", case=False, regex=True, na=False)
    load = load[type_mask & name_mask].copy()
    annual_demand_kwh = _parse_annual_demand_kwh(load)
    load = load[annual_demand_kwh.isna() | (annual_demand_kwh >= MIN_HH_ANNUAL_DEMAND_KWH)].copy()
    if allowed_buses is not None:
        allowed_buses = {int(bus) for bus in allowed_buses}
        load = load[load["bus"].astype(int).isin(allowed_buses)].copy()
    if load.empty:
        raise ValueError(
            "The real grid contains no active low-voltage household loads "
            "with type=HH, name matching NS_Last or NS_ErLast, and annual demand "
            f"of at least {MIN_HH_ANNUAL_DEMAND_KWH:.0f} kWh/a."
        )
    return load


def _supplied_buses(grid: pp.pandapowerNet) -> set[int]:
    unsupplied = {int(bus) for bus in pp_top.unsupplied_buses(grid)}
    return {int(bus) for bus in grid.bus.index}.difference(unsupplied)


def _build_real_electric_demand(
    net: pp.pandapowerNet,
    seed: int,
    load_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    electricity_module = _load_electricity_module()
    load = _household_load_rows(net) if load_rows is None else load_rows.copy()
    if load.empty:
        raise ValueError("No supplied household load rows available for profile generation.")

    rng = np.random.default_rng(seed)
    profile_ids = np.arange(len(load), dtype=int)
    real_buses = load["bus"].astype(int).to_numpy()
    pseudo_buildings = pd.DataFrame(
        {
            "bus": profile_ids,
            "real_bus": real_buses,
            "households": 1,
            "occupants": _sample_household_occupants(electricity_module, len(load), rng),
            "building_use": "Residential",
            "building_type": "single_family_house",
            "floor_area": np.nan,
        }
    )

    np_random_state = np.random.get_state()
    py_random_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        pseudo_buildings = electricity_module.sample_statistics(pseudo_buildings)
    finally:
        np.random.set_state(np_random_state)
        random.setstate(py_random_state)

    _, df_elec = electricity_module.get_elec_demand(pseudo_buildings)
    df_elec = _add_output_data_daylight_saving_shift(df_elec)

    profile_to_bus = dict(zip(profile_ids, real_buses, strict=True))
    df_elec.columns = pd.MultiIndex.from_tuples(
        [(int(profile_to_bus[int(profile_id)]), component) for profile_id, component in df_elec.columns],
        names=["bus", "component"],
    )
    df_elec = df_elec.T.groupby(level=[0, 1]).sum().T.sort_index(axis=1, level=[0, 1])

    q_factor = -math.tan(math.acos(PF_ELC))
    df_reactive = df_elec.copy() * q_factor
    df_reactive.columns = pd.MultiIndex.from_tuples(
        [(bus, "electricity-reactive") for bus, _ in df_reactive.columns],
        names=["bus", "component"],
    )
    return pd.concat([df_elec, df_reactive], axis=1).sort_index(axis=1, level=[0, 1])


def _prepare_real_grid(
    net: pp.pandapowerNet,
) -> tuple[pp.pandapowerNet, float, pd.Series, list[int], list[int], pd.DataFrame, dict[str, int]]:
    grid = deepcopy(net)
    pp.replace_zero_branches_with_switches(
        grid,
        elements=("line", "impedance"),
        zero_length=True,
        zero_impedance=True,
        in_service_only=True,
        drop_affected=False,
    )
    transformer_s_rated_mva = float("nan")
    if hasattr(grid, "trafo") and not grid.trafo.empty and "sn_mva" in grid.trafo.columns:
        transformer_s_rated_mva = float(pd.to_numeric(grid.trafo["sn_mva"], errors="coerce").sum())

    if "max_i_ka" in grid.line.columns:
        cable_max_i_ka = pd.to_numeric(grid.line["max_i_ka"], errors="coerce")
    else:
        cable_max_i_ka = pd.Series(np.nan, index=grid.line.index, name="max_i_ka")

    if "in_service" in grid.line.columns:
        active_lines = grid.line["in_service"].fillna(True).astype(bool)
        cable_max_i_ka = cable_max_i_ka.where(active_lines)

    if not grid.line.empty:
        grid.line["max_i_ka"] = 1000.0

    all_selected_loads = _household_load_rows(grid)
    all_selected_buses = all_selected_loads["bus"].dropna().astype(int).drop_duplicates()
    supplied_buses = _supplied_buses(grid)
    selected_loads = _household_load_rows(grid, allowed_buses=supplied_buses)
    if selected_loads.empty:
        raise ValueError(
            "The real grid contains no selected household loads on pandapower-supplied buses. "
            "Check open switches, radialization, and the transformer/root component."
        )
    selected_buses = selected_loads["bus"].dropna().astype(int).drop_duplicates()
    load_scope = {
        "household_load_rows_before_supply_filter": int(len(all_selected_loads)),
        "household_load_buses_before_supply_filter": int(len(all_selected_buses)),
        "dropped_unsupplied_household_load_rows": int(len(all_selected_loads) - len(selected_loads)),
        "dropped_unsupplied_household_load_buses": int(len(set(all_selected_buses).difference(set(selected_buses)))),
    }
    load_buses = selected_buses.tolist()
    grid.load = selected_loads.drop_duplicates(subset=["bus"]).copy().reset_index(drop=True)
    if not grid.load.empty:
        grid.load["name"] = "HH_Profile_" + grid.load["bus"].astype(int).astype(str)
        grid.load["p_mw"] = 0.0
        grid.load["q_mvar"] = 0.0
        grid.load["max_p_mw"] = 1000.0
        for column, value in {
            "const_z_percent": 0.0,
            "const_i_percent": 0.0,
            "const_z_p_percent": 0.0,
            "const_z_q_percent": 0.0,
            "const_i_p_percent": 0.0,
            "const_i_q_percent": 0.0,
            "scaling": 1.0,
            "in_service": True,
        }.items():
            grid.load[column] = value
    if hasattr(grid, "sgen") and not grid.sgen.empty:
        grid.sgen["in_service"] = False
        for column in ("p_mw", "q_mvar"):
            if column in grid.sgen.columns:
                grid.sgen[column] = grid.sgen[column].fillna(0.0)
        if "scaling" in grid.sgen.columns:
            grid.sgen["scaling"] = grid.sgen["scaling"].fillna(1.0)
    if not grid.bus.empty:
        grid.bus[["min_vm_pu", "max_vm_pu"]] = (0.0, 10.0)

    backbone_cable_ids, voltage_buses = pwrflw.comparison_backbone_scope(grid, load_buses)
    if not voltage_buses:
        voltage_buses = load_buses
    return grid, transformer_s_rated_mva, cable_max_i_ka, voltage_buses, backbone_cable_ids, selected_loads, load_scope


def _grid_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "swf",
        "plz": int(row["plz"]),
        "lv_id": str(row["lv_id"]),
        "variant": row.get("variant"),
        "category": row.get("category"),
        "load_status": row.get("load_status"),
        "status": row.get("status"),
        "source_file": str(row["source_file"]),
        "bus_count": _optional_int(row.get("bus_count")),
        "line_count": _optional_int(row.get("line_count")),
        "load_count": _optional_int(row.get("load_count")),
        "assumptions": {"demand_allocation": ASSUMPTION_TEXT},
    }


def _optional_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def run_one(row: dict[str, Any], run_name: str, scenario_key: str, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    source_file = Path(row["source_file"])
    net = pp.from_excel(source_file)
    (
        grid,
        transformer_s_rated_mva,
        cable_max_i_ka,
        voltage_buses,
        backbone_cable_ids,
        selected_household_loads,
        load_scope,
    ) = _prepare_real_grid(net)
    df_demand = _build_real_electric_demand(net, seed=seed, load_rows=selected_household_loads)
    summary = pwrflw.pf_summary(
        grid,
        df_demand,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_max_i_ka=cable_max_i_ka,
        voltage_buses=voltage_buses,
        algorithm=["nr", "iwamoto_nr"],
        cable_ids=backbone_cable_ids,
        on_nonconvergence="nan",
        protect_grid_state=True,
    )

    db = SurroGridDatabase()
    db.ensure_schema()
    run_id = db.create_real_powerflow_run(
        _grid_ref(row),
        run_name=run_name,
        scenario_key=scenario_key,
        assumptions={
            "timeframe_mode": "full_year",
            "demand_allocation": ASSUMPTION_TEXT,
            "profile_seed": int(seed),
            **load_scope,
            "selected_household_load_rows": int(len(selected_household_loads)),
            "selected_household_load_buses": int(len(selected_household_loads["bus"].dropna().astype(int).drop_duplicates())),
            "backbone_voltage_buses": int(len(voltage_buses)),
            "backbone_cables": int(len(backbone_cable_ids)),
            "nonconverged_timesteps": int(summary["grid_summary"].get("n_failed_timesteps", 0)),
        },
    )
    db.write_real_powerflow_summary(run_id, "pre", summary)
    elapsed = time.perf_counter() - start
    grid_summary = summary["grid_summary"]
    return {
        "lv_id": row["lv_id"],
        "run_id": run_id,
        "source_file": str(source_file),
        "elapsed_s": elapsed,
        "n_timesteps": grid_summary["n_timesteps"],
        "n_converged_timesteps": grid_summary.get("n_converged_timesteps"),
        "n_failed_timesteps": grid_summary.get("n_failed_timesteps"),
        "n_voltage_buses": grid_summary["n_voltage_buses"],
        "n_cables": grid_summary["n_cables"],
        **load_scope,
        "selected_household_load_rows": int(len(selected_household_loads)),
        "selected_household_load_buses": int(len(selected_household_loads["bus"].dropna().astype(int).drop_duplicates())),
        "backbone_voltage_buses": int(len(voltage_buses)),
        "backbone_cables": int(len(backbone_cable_ids)),
        "tail_rows": len(summary.get("tail_summary", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact SWF real-grid power-flow summaries.")
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lv-id", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--scenario-key", default=DEFAULT_SCENARIO_KEY)
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--skip-existing", action="store_true", help="Skip grids that already have a compact real summary for this run/stage.")
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=True)
    SurroGridDatabase().ensure_schema()
    root = args.grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    rows = _select_manifest_rows(root, args.plz, args.limit, args.lv_id)
    if args.skip_existing and rows:
        db = SurroGridDatabase()
        db.ensure_schema()
        query = """
            SELECT rgc.source_file
            FROM surrogrid.real_powerflow_summary rps
            JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
            JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
            WHERE rpr.run_name = :run_name
              AND rps.stage = 'pre'
              AND rgc.plz = :plz
        """
        from sqlalchemy import text as sql_text
        with db.engine.connect() as conn:
            existing = set(pd.read_sql_query(sql_text(query), conn, params={"run_name": args.run_name, "plz": args.plz})["source_file"].astype(str))
        before = len(rows)
        rows = [row for row in rows if str(row["source_file"]) not in existing]
        print(f"Skipping {before - len(rows)} existing real-grid summary job(s).", flush=True)
    if not rows:
        raise ValueError("No SWF real-grid rows matched the requested selection or all selected grids already exist.")

    print(
        f"Running {len(rows)} SWF real-grid powerflow summary job(s) "
        f"from {root} with {args.workers} worker(s).",
        flush=True,
    )
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for i, row in enumerate(rows):
            print(f"[{i + 1}/{len(rows)}] {row['lv_id']} {row['source_file']}", flush=True)
            try:
                result = run_one(row, args.run_name, args.scenario_key, args.seed + i)
            except Exception as exc:
                result = {
                    "lv_id": row["lv_id"],
                    "run_id": None,
                    "source_file": row["source_file"],
                    "elapsed_s": np.nan,
                    "n_timesteps": 0,
                    "n_voltage_buses": 0,
                    "n_cables": 0,
                    "tail_rows": 0,
                    "status": "failed",
                    "error": str(exc),
                }
                print(f"  -> FAILED: {exc}", flush=True)
            else:
                result["status"] = "ok"
                result["error"] = ""
                print(f"  -> run_id={result['run_id']} elapsed={result['elapsed_s']:.1f}s tail_rows={result['tail_rows']}", flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, row, args.run_name, args.scenario_key, args.seed + i): row
                for i, row in enumerate(rows)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "lv_id": row["lv_id"],
                        "run_id": None,
                        "source_file": row["source_file"],
                        "elapsed_s": np.nan,
                        "n_timesteps": 0,
                        "n_voltage_buses": 0,
                        "n_cables": 0,
                        "tail_rows": 0,
                        "status": "failed",
                        "error": str(exc),
                    }
                    print(f"[{done}/{len(rows)}] {row['lv_id']} FAILED: {exc}", flush=True)
                else:
                    result["status"] = "ok"
                    result["error"] = ""
                    print(
                        f"[{done}/{len(rows)}] {result['lv_id']} -> run_id={result['run_id']} "
                        f"elapsed={result['elapsed_s']:.1f}s tail_rows={result['tail_rows']}",
                        flush=True,
                    )
                results.append(result)

    elapsed = time.perf_counter() - start
    result_df = pd.DataFrame(results).sort_values("lv_id")
    print(result_df.to_string(index=False), flush=True)
    ok_count = int((result_df.get("status", "ok") == "ok").sum()) if "status" in result_df else len(result_df)
    failed_count = len(result_df) - ok_count
    print(f"Finished {ok_count}/{len(results)} real-grid job(s) successfully in {elapsed:.1f}s; failed={failed_count}.", flush=True)


if __name__ == "__main__":
    main()
