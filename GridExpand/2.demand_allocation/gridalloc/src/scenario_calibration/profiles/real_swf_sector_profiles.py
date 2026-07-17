"""Synthetic-profile sector asset translation for real SWF URBS inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import GRIDALLOC_DIR

MOBILITY_POOL_DIR = (
    GRIDALLOC_DIR / "data" / "statistics" / "general" / "mobility_profile_pool"
)
DEFAULT_MOBILITY_WEATHER_KEY = "central_germany_tmy"
DEFAULT_EV_CHARGER_KW = 11.0
DEFAULT_PV_TARGET_TILT = 45.0
DEFAULT_PV_TARGET_AZIMUTH = 180.0


@dataclass(frozen=True)
class SectorUrbsInputs:
    demand: pd.DataFrame
    supim: pd.DataFrame
    eff_factor: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame
    metadata: dict[str, Any]


def build_real_swf_sector_urbs_inputs(
    allocation: pd.DataFrame,
    *,
    hours: int,
    seed: int,
    source_hdf: Path | None,
    include_pv: bool = True,
    include_mobility: bool = True,
    include_heat: bool = True,
) -> SectorUrbsInputs:
    """Build optional sector-coupling URBS sheets from SWF placement/capacity and shared profiles.

    This intentionally does not create heat-pump heat demand yet. SWF WP rows give
    placement/capacity evidence, but the heat-demand time series need an explicit
    shared thermal-demand model before they can be compared fairly.
    """
    empty_ts = _empty_timeseries(hours)
    frames = []
    audit_frames = []
    process_frames = []
    commodity_frames = []
    process_commodity_frames = []
    storage_frames = []
    supim = empty_ts.copy()
    eff_factor = empty_ts.copy()
    demand_frames = []

    if include_pv:
        pv = _build_pv_inputs(allocation, hours=hours, source_hdf=source_hdf)
        if pv is not None:
            supim = pv.supim
            process_frames.append(pv.process)
            commodity_frames.append(pv.commodity)
            process_commodity_frames.append(pv.process_commodity)
            audit_frames.append(pv.audit)
            frames.append("pv")

    if include_mobility:
        ev = _build_mobility_inputs(allocation, hours=hours, seed=seed)
        if ev is not None:
            frames.append("mobility")
            audit_frames.append(ev.audit)
            process_frames.append(ev.process)
            commodity_frames.append(ev.commodity)
            process_commodity_frames.append(ev.process_commodity)
            storage_frames.append(ev.storage)
            demand_frames.append(ev.demand)
            eff_factor = _concat_timeseries([eff_factor, ev.eff_factor], hours)

    if include_heat:
        heat = _build_heat_inputs(
            allocation, hours=hours, seed=seed, source_hdf=source_hdf
        )
        if heat is not None:
            frames.append("heat")
            audit_frames.append(heat.audit)
            process_frames.append(heat.process)
            commodity_frames.append(heat.commodity)
            process_commodity_frames.append(heat.process_commodity)
            storage_frames.append(heat.storage)
            demand_frames.append(heat.demand)
            eff_factor = _concat_timeseries([eff_factor, heat.eff_factor], hours)

    sector_demand = _concat_timeseries(demand_frames, hours)

    metadata = {
        "sector_assets_simulated": bool(frames),
        "sector_assets_simulated_components": sorted(frames),
        "heat_assets_simulated": "heat" in frames,
        "heat_asset_note": "SWF WP rows are translated with sampled synthetic heat-demand/COP profiles from the selected source HDF; this is a profile-library approximation, not a fresh building-physics run.",
    }

    return SectorUrbsInputs(
        demand=sector_demand,
        supim=supim,
        eff_factor=eff_factor,
        process=_concat_static(process_frames),
        commodity=_concat_static(commodity_frames),
        process_commodity=_concat_static(process_commodity_frames),
        storage=_concat_static(storage_frames),
        audit=_concat_static(audit_frames),
        metadata=metadata,
    )


@dataclass(frozen=True)
class _PvInputs:
    supim: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class _MobilityInputs:
    demand: pd.DataFrame
    eff_factor: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class _HeatInputs:
    demand: pd.DataFrame
    eff_factor: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame


def _empty_timeseries(hours: int) -> pd.DataFrame:
    df = pd.DataFrame(index=pd.RangeIndex(hours, name="t"))
    df.columns = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["Site", "Commodity"]
    )
    return df


def _concat_timeseries(frames: list[pd.DataFrame], hours: int) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return _empty_timeseries(hours)
    out = pd.concat(non_empty, axis=1)
    out.index = pd.RangeIndex(len(out), name="t")
    return out


def _concat_static(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _sector_capacity_by_bus(allocation: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [f"residential_{suffix}", f"ghd_{suffix}"]
    values = pd.Series(0.0, index=allocation.index)
    for column in columns:
        if column in allocation.columns:
            values = values + pd.to_numeric(allocation[column], errors="coerce").fillna(
                0.0
            )
    grouped = values.groupby(allocation["allocation_bus"].astype(int)).sum()
    return grouped[grouped.gt(0.0)]


def _sector_rows_by_bus(allocation: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [f"residential_{suffix}", f"ghd_{suffix}"]
    values = pd.Series(0.0, index=allocation.index)
    for column in columns:
        if column in allocation.columns:
            values = values + pd.to_numeric(allocation[column], errors="coerce").fillna(
                0.0
            )
    grouped = (
        values.groupby(allocation["allocation_bus"].astype(int))
        .sum()
        .round()
        .astype(int)
    )
    return grouped[grouped.gt(0)]


def _build_pv_inputs(
    allocation: pd.DataFrame, *, hours: int, source_hdf: Path | None
) -> _PvInputs | None:
    pv_kw_by_bus = _sector_capacity_by_bus(allocation, "pv_kw")
    if pv_kw_by_bus.empty:
        return None
    if source_hdf is None:
        raise ValueError(
            "PV sector materialization requires --weather-source-hdf pointing to a synthetic HDF with urbs_in/supim."
        )
    source_supim = pd.read_hdf(source_hdf, key="urbs_in/supim")
    if source_supim.empty:
        raise ValueError(
            f"PV sector materialization requires non-empty urbs_in/supim in {source_hdf}."
        )
    profile_label, source_column = _choose_source_pv_profile(source_supim)
    base_profile = (
        pd.to_numeric(source_supim[source_column], errors="coerce")
        .fillna(0.0)
        .reset_index(drop=True)
    )
    if len(base_profile) < hours:
        raise ValueError(
            f"PV source profile has {len(base_profile)} rows, expected at least {hours}."
        )
    base_profile = base_profile.iloc[:hours]

    supim_frames = []
    process_rows = []
    commodity_rows = []
    audit_rows = []
    process_name = profile_label.replace("solar", "Rooftop PV", 1)
    for bus, capacity_kw in pv_kw_by_bus.items():
        bus = int(bus)
        supim_frames.append(base_profile.rename((bus, profile_label)))
        process_rows.append(
            {
                "Site": bus,
                "Process": process_name,
                "inst-cap": 0,
                "cap-up": float(capacity_kw),
                "inv-cost-fix": 6565,
                "inv-cost": 533.7,
                "fix-cost": 0,
                "var-cost": 0,
                "wacc": 0.022,
                "depreciation": 15,
                "pf-min": np.nan,
            }
        )
        commodity_rows.append(
            {"Site": bus, "Commodity": profile_label, "Type": "SupIm", "price": np.nan}
        )
        audit_rows.append(
            {
                "sector": "pv",
                "allocation_bus": bus,
                "asset_count": np.nan,
                "capacity_kw": float(capacity_kw),
                "profile_label": profile_label,
                "source_hdf": str(source_hdf),
                "source_column": str(source_column),
            }
        )

    supim = pd.concat(supim_frames, axis=1)
    supim.columns = pd.MultiIndex.from_tuples(supim.columns)
    supim.index.name = "t"
    process_commodity = pd.DataFrame(
        [
            {
                "Process": process_name,
                "Commodity": profile_label,
                "Direction": "In",
                "ratio": 1,
            },
            {
                "Process": process_name,
                "Commodity": "electricity",
                "Direction": "Out",
                "ratio": 1,
            },
        ]
    )
    return _PvInputs(
        supim=supim,
        process=pd.DataFrame(process_rows),
        commodity=pd.DataFrame(commodity_rows),
        process_commodity=process_commodity,
        audit=pd.DataFrame(audit_rows),
    )


def _choose_source_pv_profile(
    source_supim: pd.DataFrame,
) -> tuple[str, tuple[Any, Any]]:
    best = None
    for column in source_supim.columns:
        label = str(column[1]) if isinstance(column, tuple) else str(column)
        parsed = _parse_solar_label(label)
        if parsed is None:
            continue
        tilt, azimuth = parsed
        score = (tilt - DEFAULT_PV_TARGET_TILT) ** 2 + min(
            abs(azimuth - DEFAULT_PV_TARGET_AZIMUTH),
            360 - abs(azimuth - DEFAULT_PV_TARGET_AZIMUTH),
        ) ** 2
        if best is None or score < best[0]:
            best = (score, label, column)
    if best is None:
        column = source_supim.columns[0]
        label = str(column[1]) if isinstance(column, tuple) else str(column)
        return label, column
    return best[1], best[2]


def _parse_solar_label(label: str) -> tuple[float, float] | None:
    match = re.search(r"solar_([0-9.]+)_([0-9.]+)", label)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _build_heat_inputs(
    allocation: pd.DataFrame, *, hours: int, seed: int, source_hdf: Path | None
) -> _HeatInputs | None:
    wp_rows_by_bus = _sector_rows_by_bus(allocation, "wp_rows")
    if wp_rows_by_bus.empty:
        return None
    if source_hdf is None:
        raise ValueError(
            "Heat sector materialization requires --weather-source-hdf pointing to a synthetic HDF with heat demand and COP profiles."
        )
    source_demand = pd.read_hdf(source_hdf, key="urbs_in/demand")
    source_eff = pd.read_hdf(source_hdf, key="urbs_in/eff_factor")
    heat_buses = _source_heat_buses(source_demand, source_eff)
    if not heat_buses:
        raise ValueError(
            f"Heat sector materialization requires space_heat, water_heat, and heatpump_air columns in {source_hdf}."
        )

    rng = np.random.default_rng(int(seed) + 4_000_003)
    demand_by_bus: dict[int, list[pd.DataFrame]] = {}
    cop_parts_by_bus: dict[int, list[pd.DataFrame]] = {}
    audit_rows = []
    for bus, wp_count in wp_rows_by_bus.items():
        bus = int(bus)
        for _ in range(int(wp_count)):
            source_bus = int(rng.choice(heat_buses))
            space = (
                pd.to_numeric(
                    source_demand[(source_bus, "space_heat")], errors="coerce"
                )
                .fillna(0.0)
                .iloc[:hours]
                .reset_index(drop=True)
            )
            water = (
                pd.to_numeric(
                    source_demand[(source_bus, "water_heat")], errors="coerce"
                )
                .fillna(0.0)
                .iloc[:hours]
                .reset_index(drop=True)
            )
            cop = (
                pd.to_numeric(source_eff[(source_bus, "heatpump_air")], errors="coerce")
                .fillna(1.0)
                .iloc[:hours]
                .reset_index(drop=True)
                .clip(lower=0.1)
            )
            heat = space + water
            demand_by_bus.setdefault(bus, []).append(
                pd.DataFrame({"space_heat": space, "water_heat": water})
            )
            cop_parts_by_bus.setdefault(bus, []).append(
                pd.DataFrame({"heat": heat, "cop_weighted": heat * cop})
            )
            audit_rows.append(
                {
                    "sector": "heat",
                    "allocation_bus": bus,
                    "asset_count": 1,
                    "capacity_kw": np.nan,
                    "profile_label": f"source_bus_{source_bus}",
                    "source_hdf": str(source_hdf),
                    "source_column": str(
                        (source_bus, "space_heat/water_heat/heatpump_air")
                    ),
                    "annual_space_heat_kwh": float(space.sum()),
                    "annual_water_heat_kwh": float(water.sum()),
                    "mean_cop": float(cop.mean()),
                }
            )

    demand_frames = []
    eff_frames = []
    for bus in sorted(demand_by_bus):
        summed = sum(demand_by_bus[bus])
        demand_frames.append(summed["space_heat"].rename((bus, "space_heat")))
        demand_frames.append(summed["water_heat"].rename((bus, "water_heat")))
        cop_sum = sum(cop_parts_by_bus[bus])
        cop = (cop_sum["cop_weighted"] / cop_sum["heat"].replace(0.0, np.nan)).fillna(
            1.0
        )
        eff_frames.append(cop.rename((bus, "heatpump_air")))

    demand = pd.concat(demand_frames, axis=1)
    demand.columns = pd.MultiIndex.from_tuples(demand.columns)
    demand.index.name = "t"
    eff_factor = pd.concat(eff_frames, axis=1)
    eff_factor.columns = pd.MultiIndex.from_tuples(eff_factor.columns)
    eff_factor.index.name = "t"
    heat_buses_target = sorted(demand.columns.get_level_values(0).unique())
    return _HeatInputs(
        demand=demand,
        eff_factor=eff_factor,
        process=_create_pro_heat(heat_buses_target),
        commodity=_create_com_heat(heat_buses_target),
        process_commodity=_create_pro_com_heat(),
        storage=_create_sto_heat(heat_buses_target),
        audit=pd.DataFrame(audit_rows),
    )


def _source_heat_buses(
    source_demand: pd.DataFrame, source_eff: pd.DataFrame
) -> list[int]:
    buses = []
    if source_demand.empty or source_eff.empty:
        return buses
    for bus in source_demand.columns.get_level_values(0).unique():
        if (
            (bus, "space_heat") in source_demand.columns
            and (bus, "water_heat") in source_demand.columns
            and (bus, "heatpump_air") in source_eff.columns
        ):
            buses.append(int(bus))
    return buses


def _create_pro_heat(consumer_list):
    rows = []
    for bus in consumer_list:
        for (
            process,
            inst_cap,
            cap_up,
            inv_cost_fix,
            inv_cost,
            fix_cost,
            var_cost,
            wacc,
            depreciation,
        ) in [
            ("heatpump_air", 0, 2000, 6600, 750, 0, 0, 0.0216, 20),
            ("heatpump_booster", 0, 2000, 100, 83.3, 0, 0, 0.0216, 20),
            ("Heat_dummy_space", 2000, 2000, np.nan, 0, 0, 0, 0.07, 1),
            ("Heat_dummy_water", 2000, 2000, np.nan, 0, 0, 0, 0.07, 1),
        ]:
            rows.append(
                {
                    "Site": bus,
                    "Process": process,
                    "inst-cap": inst_cap,
                    "cap-up": cap_up,
                    "inv-cost-fix": inv_cost_fix,
                    "inv-cost": inv_cost,
                    "fix-cost": fix_cost,
                    "var-cost": var_cost,
                    "wacc": wacc,
                    "depreciation": depreciation,
                    "pf-min": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _create_com_heat(consumer_list):
    rows = []
    for bus in consumer_list:
        rows.extend(
            [
                {
                    "Site": bus,
                    "Commodity": "common_heat",
                    "Type": "Stock",
                    "price": np.nan,
                },
                {
                    "Site": bus,
                    "Commodity": "space_heat",
                    "Type": "Demand",
                    "price": np.nan,
                },
                {
                    "Site": bus,
                    "Commodity": "water_heat",
                    "Type": "Demand",
                    "price": np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def _create_pro_com_heat():
    return pd.DataFrame(
        {
            "Process": [
                "Heat_dummy_space",
                "Heat_dummy_space",
                "Heat_dummy_water",
                "Heat_dummy_water",
                "heatpump_air",
                "heatpump_air",
                "heatpump_booster",
                "heatpump_booster",
            ],
            "Commodity": [
                "common_heat",
                "space_heat",
                "common_heat",
                "water_heat",
                "electricity",
                "common_heat",
                "electricity",
                "common_heat",
            ],
            "Direction": ["In", "Out", "In", "Out", "In", "Out", "In", "Out"],
            "ratio": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


def _create_sto_heat(consumer_list):
    rows = []
    for bus in consumer_list:
        rows.append(
            {
                "Site": bus,
                "Storage": "heat_storage",
                "Commodity": "common_heat",
                "inst-cap-c": 0,
                "cap-up-c": 10000,
                "inst-cap-p": 0,
                "cap-up-p": 1500,
                "eff-in": 0.932,
                "eff-out": 1,
                "discharge": 0,
                "ep-ratio": 0.15,
                "inv-cost-p": 0,
                "inv-cost-c": 58,
                "fix-cost-p": 0,
                "fix-cost-c": 0,
                "var-cost-p": 0.001,
                "wacc": 0.0216,
                "depreciation": 20,
            }
        )
    return pd.DataFrame(rows)


def _build_mobility_inputs(
    allocation: pd.DataFrame, *, hours: int, seed: int
) -> _MobilityInputs | None:
    charger_rows_by_bus = _sector_rows_by_bus(allocation, "ev_charger_rows")
    charger_kw_by_bus = _sector_capacity_by_bus(allocation, "ev_charger_kw")
    if charger_rows_by_bus.empty:
        return None

    metadata = pd.read_csv(MOBILITY_POOL_DIR / "mobility_profile_pool_metadata.csv")
    metadata = metadata[
        metadata["weather_key"].astype(str).eq(DEFAULT_MOBILITY_WEATHER_KEY)
    ].copy()
    if metadata.empty:
        raise ValueError(
            f"No mobility pool metadata for weather_key={DEFAULT_MOBILITY_WEATHER_KEY}."
        )
    rng = np.random.default_rng(int(seed) + 3_000_003)
    total_chargers = int(charger_rows_by_bus.sum())
    selected_indices = rng.choice(
        metadata.index.to_numpy(), size=total_chargers, replace=True
    )
    selected = metadata.loc[selected_indices].reset_index(drop=True)
    profile_ids = selected["profile_id"].tolist()
    demand_by_profile = _read_pool_timeseries(
        MOBILITY_POOL_DIR / "mobility_demand_pool.csv", profile_ids, "demand_kwh"
    )
    availability_by_profile = _read_pool_timeseries(
        MOBILITY_POOL_DIR / "mobility_availability_pool.csv",
        profile_ids,
        "availability",
    )

    demand_frames = []
    availability_frames = []
    process_rows = []
    commodity_rows = []
    storage_rows = []
    process_commodity_rows = []
    audit_rows = []
    global_id = 0
    for bus, charger_count in charger_rows_by_bus.items():
        bus = int(bus)
        total_kw = float(
            charger_kw_by_bus.get(bus, DEFAULT_EV_CHARGER_KW * charger_count)
        )
        charger_kw = (
            total_kw / charger_count if charger_count else DEFAULT_EV_CHARGER_KW
        )
        for _ in range(int(charger_count)):
            profile = selected.iloc[global_id]
            profile_id = profile["profile_id"]
            demand = demand_by_profile[profile_id].iloc[:hours].reset_index(drop=True)
            availability = (
                availability_by_profile[profile_id].iloc[:hours].reset_index(drop=True)
            )
            mobility_label = f"mobility{global_id}"
            charger_label = f"charging_station{global_id}"
            storage_label = f"mobility_storage{global_id}"
            demand_frames.append(demand.rename((bus, mobility_label)))
            availability_frames.append(availability.rename((bus, charger_label)))
            process_rows.append(
                {
                    "Site": bus,
                    "Process": charger_label,
                    "inst-cap": float(charger_kw),
                    "cap-up": float(charger_kw),
                    "inv-cost-fix": np.nan,
                    "inv-cost": 0,
                    "fix-cost": 0,
                    "var-cost": 0,
                    "wacc": 0.07,
                    "depreciation": 1,
                    "pf-min": np.nan,
                }
            )
            commodity_rows.append(
                {
                    "Site": bus,
                    "Commodity": mobility_label,
                    "Type": "Demand",
                    "price": np.nan,
                }
            )
            process_commodity_rows.extend(
                [
                    {
                        "Process": charger_label,
                        "Commodity": "electricity",
                        "Direction": "In",
                        "ratio": 1,
                    },
                    {
                        "Process": charger_label,
                        "Commodity": mobility_label,
                        "Direction": "Out",
                        "ratio": 1,
                    },
                ]
            )
            battery_cap = float(profile["battery_cap_kwh"])
            storage_rows.append(
                {
                    "Site": bus,
                    "Storage": storage_label,
                    "Commodity": mobility_label,
                    "inst-cap-c": battery_cap,
                    "cap-up-c": battery_cap,
                    "inst-cap-p": battery_cap,
                    "cap-up-p": battery_cap,
                    "eff-in": 1,
                    "eff-out": 1,
                    "discharge": 0,
                    "ep-ratio": np.nan,
                    "inv-cost-p": 0,
                    "inv-cost-c": 0,
                    "fix-cost-p": 0,
                    "fix-cost-c": 0,
                    "var-cost-p": 0.001,
                    "wacc": 0.07,
                    "depreciation": 20,
                }
            )
            audit_rows.append(
                {
                    "sector": "mobility",
                    "allocation_bus": bus,
                    "asset_count": 1,
                    "capacity_kw": float(charger_kw),
                    "profile_label": profile_id,
                    "battery_cap_kwh": battery_cap,
                    "demand_sum_kwh": float(demand.sum()),
                    "availability_hours": float(availability.sum()),
                }
            )
            global_id += 1

    demand_df = pd.concat(demand_frames, axis=1)
    demand_df.columns = pd.MultiIndex.from_tuples(demand_df.columns)
    demand_df.index.name = "t"
    eff_df = pd.concat(availability_frames, axis=1)
    eff_df.columns = pd.MultiIndex.from_tuples(eff_df.columns)
    eff_df.index.name = "t"
    return _MobilityInputs(
        demand=demand_df,
        eff_factor=eff_df,
        process=pd.DataFrame(process_rows),
        commodity=pd.DataFrame(commodity_rows),
        process_commodity=pd.DataFrame(process_commodity_rows),
        storage=pd.DataFrame(storage_rows),
        audit=pd.DataFrame(audit_rows),
    )


def _read_pool_timeseries(
    path: Path, profile_ids: list[str], value_column: str
) -> dict[str, pd.Series]:
    unique_profile_ids = set(profile_ids)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        subset = chunk[chunk["profile_id"].isin(unique_profile_ids)]
        if not subset.empty:
            chunks.append(subset)
    if not chunks:
        raise ValueError(f"No selected mobility profiles found in {path}.")
    table = pd.concat(chunks, ignore_index=True)
    result = {}
    for profile_id, group in table.groupby("profile_id"):
        result[profile_id] = (
            pd.to_numeric(group.sort_values("t")[value_column], errors="coerce")
            .fillna(0.0)
            .reset_index(drop=True)
        )
    missing = unique_profile_ids.difference(result)
    if missing:
        raise ValueError(
            f"Missing mobility profile rows in {path}: {sorted(missing)[:5]}"
        )
    return result
