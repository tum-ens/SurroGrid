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
ENV_PATH = GRIDEXPAND_DIR / ".env"

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))

from common.database import DEFAULT_SCENARIO_KEY, SurroGridDatabase
import src.powerflow as pwrflw

PF_ELC = 0.959
RUN_NAME = "baseline_real"
ANNUAL_DEMAND_MODE_SYNTHETIC = "synthetic"
ANNUAL_DEMAND_MODE_MEASURED = "measured"
ANNUAL_DEMAND_MODE_CHOICES = (ANNUAL_DEMAND_MODE_SYNTHETIC, ANNUAL_DEMAND_MODE_MEASURED)
MEASURED_PROFILE_SELECTION_CLOSEST = "closest"
MEASURED_PROFILE_SELECTION_RANDOM_BAND = "random_band"
MEASURED_PROFILE_SELECTION_CHOICES = (MEASURED_PROFILE_SELECTION_CLOSEST, MEASURED_PROFILE_SELECTION_RANDOM_BAND)
DEFAULT_MEASURED_PROFILE_BAND_PCT = 10.0
DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES = 10
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


def _comparison_manifest(root: Path) -> pd.DataFrame:
    station_manifest_path = root / "station_split_manifest.csv"
    radial_manifest_path = root / "station_radialization_manifest.csv"
    if station_manifest_path.exists() and radial_manifest_path.exists():
        stations = pd.read_csv(station_manifest_path)
        radial = pd.read_csv(radial_manifest_path).rename(
            columns={
                "grid": "station_id",
                "file": "radial_file",
                "status": "radial_status",
            }
        )
        manifest = stations.merge(
            radial[["station_id", "radial_file", "radial_status"]],
            on="station_id",
            how="inner",
            validate="one_to_one",
        )
        manifest["lv_id"] = (
            manifest["station_id"]
            .astype(str)
            .str.extract(r"^LV_(\d+)")[0]
            .astype(int)
        )
        manifest["variant"] = "station_radialized"
        manifest["file"] = manifest["radial_file"]
        manifest["status"] = (
            manifest["radial_status"]
            .map({"ok": "exported"})
            .fillna(manifest["radial_status"])
        )
        return manifest

    return pd.read_csv(root / "split_manifest.csv")


def _select_manifest_rows(root: Path, plz: int, limit: int | None, lv_id: str | None) -> list[dict[str, Any]]:
    manifest = _comparison_manifest(root)
    selected = manifest[
        manifest["variant"].isin({"radialized", "station_radialized"})
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
    load["annual_demand_kwh"] = _parse_annual_demand_kwh(load)
    load = load[load["annual_demand_kwh"].isna() | (load["annual_demand_kwh"] >= MIN_HH_ANNUAL_DEMAND_KWH)].copy()
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


def _sum_demand_list(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = pd.to_numeric(pd.Series(list(value)), errors="coerce")
        return float(values.sum())
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _apply_measured_annual_demands(pseudo_buildings: pd.DataFrame) -> pd.Series:
    measured = pd.to_numeric(pseudo_buildings["swf_annual_demand_kwh"], errors="coerce")
    has_measured = measured.notna()
    pseudo_buildings.loc[has_measured, "demand_tot_list"] = measured.loc[has_measured].map(lambda value: [float(value)])
    return has_measured


def _select_residential_profile(
    lps_total_demands: pd.DataFrame,
    demand: float,
    rng: np.random.Generator,
    measured_profile_selection: str,
    measured_profile_band_pct: float,
    measured_profile_min_candidates: int,
) -> dict[str, Any]:
    profile_kwh = pd.to_numeric(lps_total_demands["kWh"], errors="coerce")
    distances = (profile_kwh - float(demand)).abs().dropna()
    if distances.empty:
        raise ValueError("No residential electricity load profiles with valid annual kWh values are available.")

    if measured_profile_selection == MEASURED_PROFILE_SELECTION_CLOSEST:
        candidate_index = pd.Index([distances.idxmin()])
        method = "closest"
    elif measured_profile_selection == MEASURED_PROFILE_SELECTION_RANDOM_BAND:
        band_abs = abs(float(demand)) * float(measured_profile_band_pct) / 100.0
        candidate_index = distances[distances <= band_abs].index
        if len(candidate_index) < int(measured_profile_min_candidates):
            n_candidates = min(int(measured_profile_min_candidates), len(distances))
            candidate_index = distances.nsmallest(n_candidates).index
            method = "nearest_fallback"
        else:
            method = "band"
    else:
        raise ValueError(
            f"measured_profile_selection must be one of {MEASURED_PROFILE_SELECTION_CHOICES}, "
            f"got {measured_profile_selection!r}."
        )

    chosen_index = rng.choice(candidate_index.to_numpy())
    return {
        "chosen_index": chosen_index,
        "chosen_profile_device": lps_total_demands.loc[chosen_index, "devicenumber"],
        "chosen_profile_kwh": float(profile_kwh.loc[chosen_index]),
        "candidate_count": int(len(candidate_index)),
        "candidate_method": method,
        "candidate_min_kwh": float(profile_kwh.loc[candidate_index].min()),
        "candidate_max_kwh": float(profile_kwh.loc[candidate_index].max()),
    }


def _get_elec_demand_with_profile_selection(
    pseudo_buildings: pd.DataFrame,
    electricity_module,
    seed: int,
    measured_profile_selection: str,
    measured_profile_band_pct: float,
    measured_profile_min_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normalized_lps_res = pd.read_hdf(electricity_module.config.ELEC_LPS_PATH, key="df_normalized_scaled")
    lps_res_total_demand = pd.read_hdf(electricity_module.config.ELEC_LPS_PATH, key="df_sums")
    rng = np.random.default_rng(int(seed) + 1_000_003)

    data_dict_res = {}
    selection_rows: list[dict[str, Any]] = []
    for _, row in pseudo_buildings.iterrows():
        if row["building_use"] != "Residential":
            continue
        profile_id = int(row["bus"])
        ts_list = []
        for sequence, demand in enumerate(row["demand_tot_list"]):
            selected = _select_residential_profile(
                lps_res_total_demand,
                float(demand),
                rng,
                measured_profile_selection,
                measured_profile_band_pct,
                measured_profile_min_candidates,
            )
            chosen_device = selected["chosen_profile_device"]
            ts_list.append(df_normalized_lps_res[chosen_device] * float(demand))
            selection_rows.append(
                {
                    "profile_id": profile_id,
                    "profile_sequence": int(sequence),
                    "selected_profile_annual_demand_kwh": float(demand),
                    **{key: value for key, value in selected.items() if key != "chosen_index"},
                }
            )
        if ts_list:
            data_dict_res[profile_id] = pd.concat(ts_list, axis=1).sum(axis=1)

    df_elec = pd.DataFrame(data_dict_res).reset_index(drop=True)
    df_elec.columns = pd.MultiIndex.from_product([df_elec.columns, ["electricity"]])
    return df_elec, pd.DataFrame(selection_rows)


def _profile_selection_summary(demand_audit: pd.DataFrame) -> dict[str, Any]:
    if "chosen_profile_device" not in demand_audit.columns:
        return {}
    chosen = demand_audit["chosen_profile_device"].dropna().astype(str)
    if chosen.empty:
        return {}
    counts = chosen.value_counts()
    method_counts = demand_audit.get("candidate_method", pd.Series(dtype=object)).dropna().astype(str).value_counts()
    return {
        "measured_profile_unique_devices": int(counts.size),
        "measured_profile_largest_reuse_count": int(counts.iloc[0]),
        "measured_profile_largest_reuse_share": float(counts.iloc[0] / len(chosen)),
        "measured_profile_top5_reuse_share": float(counts.head(5).sum() / len(chosen)),
        "measured_profile_band_candidate_rows": int(method_counts.get("band", 0)),
        "measured_profile_nearest_fallback_rows": int(method_counts.get("nearest_fallback", 0)),
    }


def _build_real_electric_demand(
    net: pp.pandapowerNet,
    seed: int,
    load_rows: pd.DataFrame | None = None,
    annual_demand_mode: str = ANNUAL_DEMAND_MODE_SYNTHETIC,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if annual_demand_mode not in ANNUAL_DEMAND_MODE_CHOICES:
        raise ValueError(f"annual_demand_mode must be one of {ANNUAL_DEMAND_MODE_CHOICES}, got {annual_demand_mode!r}.")
    if measured_profile_selection not in MEASURED_PROFILE_SELECTION_CHOICES:
        raise ValueError(
            f"measured_profile_selection must be one of {MEASURED_PROFILE_SELECTION_CHOICES}, "
            f"got {measured_profile_selection!r}."
        )
    if measured_profile_band_pct <= 0:
        raise ValueError("measured_profile_band_pct must be greater than zero.")
    if measured_profile_min_candidates <= 0:
        raise ValueError("measured_profile_min_candidates must be greater than zero.")

    electricity_module = _load_electricity_module()
    load = _household_load_rows(net) if load_rows is None else load_rows.copy()
    if load.empty:
        raise ValueError("No supplied household load rows available for profile generation.")
    if "annual_demand_kwh" not in load.columns:
        load["annual_demand_kwh"] = _parse_annual_demand_kwh(load)

    rng = np.random.default_rng(seed)
    profile_ids = np.arange(len(load), dtype=int)
    real_buses = load["bus"].astype(int).to_numpy()
    pseudo_buildings = pd.DataFrame(
        {
            "bus": profile_ids,
            "real_bus": real_buses,
            "source_load_index": load.index.to_numpy(),
            "households": 1,
            "occupants": _sample_household_occupants(electricity_module, len(load), rng),
            "building_use": "Residential",
            "building_type": "single_family_house",
            "floor_area": np.nan,
            "swf_annual_demand_kwh": pd.to_numeric(load["annual_demand_kwh"], errors="coerce").to_numpy(),
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

    synthetic_profile_annual_kwh = pseudo_buildings["demand_tot_list"].map(_sum_demand_list)
    measured_mask = pd.Series(False, index=pseudo_buildings.index)
    if annual_demand_mode == ANNUAL_DEMAND_MODE_MEASURED:
        measured_mask = _apply_measured_annual_demands(pseudo_buildings)

    selection_audit = pd.DataFrame()
    if annual_demand_mode == ANNUAL_DEMAND_MODE_MEASURED:
        df_elec, selection_audit = _get_elec_demand_with_profile_selection(
            pseudo_buildings,
            electricity_module,
            seed,
            measured_profile_selection,
            measured_profile_band_pct,
            measured_profile_min_candidates,
        )
    else:
        _, df_elec = electricity_module.get_elec_demand(pseudo_buildings)
    df_elec = _add_output_data_daylight_saving_shift(df_elec)

    generated_energy = df_elec.loc[:, df_elec.columns.get_level_values(1) == "electricity"].sum(axis=0)
    generated_energy_by_profile = {int(profile_id): float(value) for (profile_id, _), value in generated_energy.items()}
    audit = pd.DataFrame(
        {
            "profile_id": profile_ids,
            "real_bus": real_buses,
            "source_load_index": pseudo_buildings["source_load_index"].to_numpy(),
            "occupants": pseudo_buildings["occupants"].to_numpy(),
            "swf_annual_demand_kwh": pseudo_buildings["swf_annual_demand_kwh"].to_numpy(),
            "synthetic_sampled_annual_kwh": synthetic_profile_annual_kwh.to_numpy(),
            "profile_annual_demand_kwh": pseudo_buildings["demand_tot_list"].map(_sum_demand_list).to_numpy(),
            "generated_profile_energy_kwh": [generated_energy_by_profile.get(int(profile_id), float("nan")) for profile_id in profile_ids],
            "annual_demand_source": np.where(measured_mask.to_numpy(), "swf_annual_kwh", "synthetic_sample"),
        }
    )
    if not selection_audit.empty:
        audit = audit.merge(selection_audit, on="profile_id", how="left")

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
    demand = pd.concat([df_elec, df_reactive], axis=1).sort_index(axis=1, level=[0, 1])
    if return_audit:
        return demand, audit
    return demand


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


def run_one(
    row: dict[str, Any],
    run_name: str,
    scenario_key: str,
    seed: int,
    annual_demand_mode: str = ANNUAL_DEMAND_MODE_SYNTHETIC,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
) -> dict[str, Any]:
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
    df_demand, demand_audit = _build_real_electric_demand(
        net,
        seed=seed,
        load_rows=selected_household_loads,
        annual_demand_mode=annual_demand_mode,
        measured_profile_selection=measured_profile_selection,
        measured_profile_band_pct=measured_profile_band_pct,
        measured_profile_min_candidates=measured_profile_min_candidates,
        return_audit=True,
    )
    profile_selection_summary = _profile_selection_summary(demand_audit)
    demand_audit_totals = {
        "annual_demand_mode": annual_demand_mode,
        "swf_measured_annual_kwh": float(pd.to_numeric(demand_audit["swf_annual_demand_kwh"], errors="coerce").sum()),
        "synthetic_sampled_annual_kwh": float(pd.to_numeric(demand_audit["synthetic_sampled_annual_kwh"], errors="coerce").sum()),
        "profile_annual_demand_kwh": float(pd.to_numeric(demand_audit["profile_annual_demand_kwh"], errors="coerce").sum()),
        "generated_profile_energy_kwh": float(pd.to_numeric(demand_audit["generated_profile_energy_kwh"], errors="coerce").sum()),
        "swf_annual_demand_rows": int(pd.to_numeric(demand_audit["swf_annual_demand_kwh"], errors="coerce").notna().sum()),
        "measured_profile_selection": measured_profile_selection,
        "measured_profile_band_pct": float(measured_profile_band_pct),
        "measured_profile_min_candidates": int(measured_profile_min_candidates),
        **profile_selection_summary,
    }
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
            **demand_audit_totals,
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
        **demand_audit_totals,
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
    parser.add_argument(
        "--annual-demand-mode",
        choices=ANNUAL_DEMAND_MODE_CHOICES,
        default=ANNUAL_DEMAND_MODE_SYNTHETIC,
        help=(
            "How to set annual HH profile energy. 'synthetic' samples annual demand from the synthetic "
            "household-size distribution. 'measured' uses parsed SWF 2022 kWh values where available "
            "and falls back to the synthetic sample for rows without annual metadata."
        ),
    )
    parser.add_argument(
        "--measured-profile-selection",
        choices=MEASURED_PROFILE_SELECTION_CHOICES,
        default=MEASURED_PROFILE_SELECTION_RANDOM_BAND,
        help=(
            "Profile-shape selection for --annual-demand-mode measured. 'random_band' randomly samples "
            "profiles within the configured annual-kWh band and falls back to nearest candidates; "
            "'closest' reproduces deterministic nearest-profile selection."
        ),
    )
    parser.add_argument(
        "--measured-profile-band-pct",
        type=float,
        default=DEFAULT_MEASURED_PROFILE_BAND_PCT,
        help="Relative annual-kWh band for random measured profile selection, in percent.",
    )
    parser.add_argument(
        "--measured-profile-min-candidates",
        type=int,
        default=DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
        help="Minimum candidate profiles before falling back to nearest measured profiles.",
    )
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
                result = run_one(
                    row,
                    args.run_name,
                    args.scenario_key,
                    args.seed + i,
                    args.annual_demand_mode,
                    args.measured_profile_selection,
                    args.measured_profile_band_pct,
                    args.measured_profile_min_candidates,
                )
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
                pool.submit(
                    run_one,
                    row,
                    args.run_name,
                    args.scenario_key,
                    args.seed + i,
                    args.annual_demand_mode,
                    args.measured_profile_selection,
                    args.measured_profile_band_pct,
                    args.measured_profile_min_candidates,
                ): row
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
