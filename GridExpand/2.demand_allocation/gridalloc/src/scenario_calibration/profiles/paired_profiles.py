"""Canonical demand profiles shared by paired real and synthetic grid runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .profile_contract import (
    assert_energy_conserved,
    assert_unique_columns,
    profile_key,
    stable_seed,
    sum_series_by_key,
)
from .real_swf_electricity_profiles import (
    DEFAULT_MEASURED_PROFILE_BAND_PCT,
    DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    add_output_data_daylight_saving_shift,
    ghd_shape_by_building_use,
    load_electricity_module,
    select_residential_profile,
)
from .real_swf_sector_profiles import (
    DEFAULT_MOBILITY_WEATHER_KEY,
    MOBILITY_POOL_DIR,
    SectorUrbsInputs,
    _choose_source_pv_profile,
    _concat_static,
    _concat_timeseries,
    _create_com_heat,
    _create_pro_com_heat,
    _create_pro_heat,
    _create_sto_heat,
    _empty_timeseries,
    _read_pool_timeseries,
)

from ..paths import SYNTHETIC_INPUT_DIR


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


@dataclass(frozen=True)
class _PvInputs:
    supim: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class _BatteryInputs:
    storage: pd.DataFrame
    audit: pd.DataFrame


def _source_values(row: dict[str, Any]) -> tuple[int, int]:
    source_lv = row.get("source_lv_id", row.get("lv_id"))
    source_bus = row.get("source_allocation_bus", row.get("allocation_bus"))
    return int(source_lv), int(source_bus)


def _target_bus(row: dict[str, Any]) -> int:
    if "_profile_site_id" in row:
        return int(row["_profile_site_id"])
    return int(row["allocation_bus"])


def _sorted_allocation(allocation: pd.DataFrame) -> pd.DataFrame:
    frame = allocation.copy()
    if "source_lv_id" not in frame:
        frame["source_lv_id"] = frame["lv_id"]
    if "source_allocation_bus" not in frame:
        frame["source_allocation_bus"] = frame["allocation_bus"]
    return frame.sort_values(
        ["building_objectid", "source_lv_id", "source_allocation_bus"],
        na_position="last",
    )


def build_paired_base_electric_demand(
    allocation: pd.DataFrame,
    *,
    seed: int,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one reproducible physical demand realization on target buses."""
    electricity = load_electricity_module()
    normalized_hh = pd.read_hdf(
        electricity.config.ELEC_LPS_PATH,
        key="df_normalized_scaled",
    )
    hh_totals = pd.read_hdf(electricity.config.ELEC_LPS_PATH, key="df_sums")
    ghd_shapes = ghd_shape_by_building_use(electricity)

    entries: list[tuple[int, pd.Series]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in _sorted_allocation(allocation).to_dict("records"):
        target_bus = _target_bus(row)
        source_lv, source_bus = _source_values(row)
        physical_key = profile_key(row)
        hh_count = int(
            round(float(row.get("residential_equivalent_hh_rows", 0.0) or 0.0))
        )
        hh_kwh = float(row.get("residential_equivalent_hh_annual_kwh", 0.0) or 0.0)
        if hh_count > 0 and hh_kwh > 0.0:
            per_hh_kwh = hh_kwh / hh_count
            for sequence in range(hh_count):
                rng = np.random.default_rng(
                    stable_seed(
                        seed,
                        *physical_key,
                        source_lv,
                        source_bus,
                        "HH",
                        sequence,
                    )
                )
                selected = select_residential_profile(
                    hh_totals,
                    per_hh_kwh,
                    rng,
                    measured_profile_selection,
                    measured_profile_band_pct,
                    measured_profile_min_candidates,
                )
                series = (
                    normalized_hh[selected["chosen_profile_device"]] * per_hh_kwh
                ).reset_index(drop=True)
                entries.append((target_bus, series))
                audit_rows.append(
                    {
                        "target_bus": target_bus,
                        "source_lv_id": source_lv,
                        "source_allocation_bus": source_bus,
                        "building_objectid": row.get("building_objectid"),
                        "demand_class": "HH",
                        "profile_sequence": sequence,
                        "annual_demand_kwh": per_hh_kwh,
                        **{
                            key: value
                            for key, value in selected.items()
                            if key != "chosen_index"
                        },
                    }
                )

        ghd_kwh = float(row.get("calibrated_annual_ghd_kwh", 0.0) or 0.0)
        if ghd_kwh > 0.0:
            building_use = str(row.get("building_use") or "default")
            shape_key = building_use if building_use in ghd_shapes else "default"
            series = (ghd_shapes[shape_key] * ghd_kwh).reset_index(drop=True)
            entries.append((target_bus, series))
            audit_rows.append(
                {
                    "target_bus": target_bus,
                    "source_lv_id": source_lv,
                    "source_allocation_bus": source_bus,
                    "building_objectid": row.get("building_objectid"),
                    "demand_class": "GHD",
                    "profile_sequence": 0,
                    "annual_demand_kwh": ghd_kwh,
                    "chosen_profile_device": f"weighted_{shape_key}_ghd",
                    "chosen_profile_kwh": ghd_kwh,
                    "candidate_count": np.nan,
                    "candidate_method": "weighted_ghd_shape",
                    "candidate_min_kwh": np.nan,
                    "candidate_max_kwh": np.nan,
                }
            )

    demand = sum_series_by_key(entries)
    if demand.empty:
        raise ValueError("Paired allocation produced no active electricity demand.")
    demand = add_output_data_daylight_saving_shift(demand)
    expected_by_bus = (
        allocation.assign(
            _annual_kwh=(
                pd.to_numeric(
                    allocation["residential_equivalent_hh_annual_kwh"],
                    errors="coerce",
                ).fillna(0.0)
                + pd.to_numeric(
                    allocation["calibrated_annual_ghd_kwh"],
                    errors="coerce",
                ).fillna(0.0)
            )
        )
        .groupby(
            allocation.get("_profile_site_id", allocation["allocation_bus"]).astype(int)
        )["_annual_kwh"]
        .sum()
    )
    generated_by_bus = demand.sum(axis=0)
    demand = demand.mul(
        expected_by_bus.reindex(demand.columns)
        .div(generated_by_bus.replace(0.0, np.nan))
        .fillna(1.0),
        axis=1,
    )
    demand.columns = pd.MultiIndex.from_tuples(
        [(int(bus), "electricity") for bus in demand.columns],
        names=["Site", "Commodity"],
    )
    demand = demand.sort_index(axis=1)
    assert_unique_columns(demand, "paired base electricity")
    expected_kwh = float(
        pd.to_numeric(
            allocation["residential_equivalent_hh_annual_kwh"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
        + pd.to_numeric(
            allocation["calibrated_annual_ghd_kwh"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    assert_energy_conserved(
        demand,
        expected_kwh,
        label="paired base electricity",
    )
    return demand, pd.DataFrame(audit_rows)


def build_paired_sector_urbs_inputs(
    allocation: pd.DataFrame,
    *,
    hours: int,
    seed: int,
    weather_source_hdf: Path,
    synthetic_input_dir: Path = SYNTHETIC_INPUT_DIR,
    heat_profile_catalog: pd.DataFrame | None = None,
    allow_diagnostic_heat_fallback: bool = False,
) -> SectorUrbsInputs:
    """Build SWF-2045 sector profiles once and remap them to target buses."""
    pv = _build_paired_pv(allocation, hours=hours, source_hdf=weather_source_hdf)
    battery = _build_paired_battery(allocation)
    mobility = _build_paired_mobility(allocation, hours=hours, seed=seed)
    heat = _build_paired_heat(
        allocation,
        heat_profile_catalog=heat_profile_catalog,
        allow_diagnostic_fallback=allow_diagnostic_heat_fallback,
        hours=hours,
        synthetic_input_dir=synthetic_input_dir,
    )

    parts = [part for part in (pv, mobility, heat) if part is not None]
    audit_parts = [part for part in (pv, battery, mobility, heat) if part is not None]
    audit = _concat_static([part.audit for part in audit_parts])
    process = _concat_static([part.process for part in parts])
    commodity = _concat_static([part.commodity for part in parts])
    process_commodity = _concat_static([part.process_commodity for part in parts])
    storage = _concat_static(
        [part.storage for part in (battery, mobility, heat) if part is not None]
    )
    demand = _concat_timeseries(
        [part.demand for part in (mobility, heat) if part is not None],
        hours,
    )
    eff_factor = _concat_timeseries(
        [part.eff_factor for part in (mobility, heat) if part is not None],
        hours,
    )
    supim = _empty_timeseries(hours) if pv is None else pv.supim
    for label, frame in (
        ("paired sector demand", demand),
        ("paired sector efficiency", eff_factor),
        ("paired sector intermittent supply", supim),
    ):
        assert_unique_columns(frame, label)
    return SectorUrbsInputs(
        demand=demand,
        supim=supim,
        eff_factor=eff_factor,
        process=process,
        commodity=commodity,
        process_commodity=process_commodity,
        storage=storage,
        audit=audit,
        metadata={
            "sector_assets_simulated": bool(audit_parts),
            "sector_assets_simulated_components": [
                name
                for name, part in (
                    ("pv", pv),
                    ("stationary_battery", battery),
                    ("mobility", mobility),
                    ("heat", heat),
                )
                if part is not None
            ],
            "profile_contract": "physical_building_paired_v1",
            "heat_profile_source": (
                "matched physical building from pylovo version-specific "
                "synthetic URBS input"
            ),
            "stationary_battery_model": (
                "fixed SWF energy capacity; fixed charge/discharge power equal "
                "to half the energy capacity; optimized post-flex dispatch"
            ),
        },
    )


def _sum_numeric_columns(
    allocation: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    values = pd.Series(0.0, index=allocation.index)
    for column in columns:
        if column in allocation:
            values = values.add(
                pd.to_numeric(allocation[column], errors="coerce").fillna(0.0)
            )
    return values


def _build_paired_pv(
    allocation: pd.DataFrame,
    *,
    hours: int,
    source_hdf: Path,
) -> _PvInputs | None:
    capacity = _sum_numeric_columns(
        allocation,
        ("residential_pv_kw", "ghd_pv_kw"),
    )
    profile_sites = allocation.get(
        "_profile_site_id", allocation["allocation_bus"]
    ).astype(int)
    capacity_by_bus = capacity.groupby(profile_sites).sum()
    capacity_by_bus = capacity_by_bus[capacity_by_bus.gt(0.0)]
    if capacity_by_bus.empty:
        return None

    source_supim = pd.read_hdf(source_hdf, key="urbs_in/supim")
    profile_label, source_column = _choose_source_pv_profile(source_supim)
    base = (
        pd.to_numeric(
            source_supim[source_column],
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[:hours]
        .reset_index(drop=True)
    )
    if len(base) != hours:
        raise ValueError(f"PV source profile has {len(base)} rows, expected {hours}.")
    process_name = profile_label.replace("solar", "Rooftop PV", 1)
    supim = pd.concat(
        [base.rename((int(bus), profile_label)) for bus in capacity_by_bus.index],
        axis=1,
    )
    supim.columns = pd.MultiIndex.from_tuples(supim.columns)
    supim.index.name = "t"
    process = pd.DataFrame(
        [
            {
                "Site": int(bus),
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
            for bus, capacity_kw in capacity_by_bus.items()
        ]
    )
    commodity = pd.DataFrame(
        [
            {
                "Site": int(bus),
                "Commodity": profile_label,
                "Type": "SupIm",
                "price": np.nan,
            }
            for bus in capacity_by_bus.index
        ]
    )
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
    audit = pd.DataFrame(
        [
            {
                "sector": "pv",
                "allocation_bus": int(bus),
                "capacity_kw": float(capacity_kw),
                "profile_label": profile_label,
                "source_hdf": str(source_hdf),
                "source_column": str(source_column),
            }
            for bus, capacity_kw in capacity_by_bus.items()
        ]
    )
    return _PvInputs(
        supim=supim,
        process=process,
        commodity=commodity,
        process_commodity=process_commodity,
        audit=audit,
    )


def _build_paired_battery(
    allocation: pd.DataFrame,
) -> _BatteryInputs | None:
    """Create the fixed SWF stationary-battery inventory at scenario-unit sites."""
    capacity = _sum_numeric_columns(
        allocation,
        ("residential_battery_kwh", "ghd_battery_kwh"),
    )
    profile_sites = allocation.get(
        "_profile_site_id", allocation["allocation_bus"]
    ).astype(int)
    capacity_by_site = capacity.groupby(profile_sites).sum()
    capacity_by_site = capacity_by_site[capacity_by_site.gt(0.0)]
    if capacity_by_site.empty:
        return None

    electricity = load_electricity_module()
    config = electricity.config
    rows = []
    audit_rows = []
    for site, energy_kwh in capacity_by_site.items():
        power_kw = float(energy_kwh) / 2.0
        rows.append(
            {
                "Site": int(site),
                "Storage": "battery_private",
                "Commodity": "electricity",
                "inst-cap-c": float(energy_kwh),
                "cap-up-c": float(energy_kwh),
                "inst-cap-p": power_kw,
                "cap-up-p": power_kw,
                "eff-in": config.BS_EFF_IN,
                "eff-out": config.BS_EFF_OUT,
                "discharge": config.BS_DISCHARGE,
                "ep-ratio": 2.0,
                "inv-cost-p": 0.0,
                "inv-cost-c": 0.0,
                "fix-cost-p": 0.0,
                "fix-cost-c": 0.0,
                "var-cost-p": config.BS_VAR_COST_P,
                "wacc": config.BS_WACC,
                "depreciation": config.BS_DEPRECIATION,
            }
        )
        audit_rows.append(
            {
                "sector": "stationary_battery",
                "allocation_bus": int(site),
                "energy_capacity_kwh": float(energy_kwh),
                "power_capacity_kw": power_kw,
                "energy_to_power_hours": 2.0,
                "capacity_source": "deduplicated_swf_2045_inventory",
            }
        )
    return _BatteryInputs(
        storage=pd.DataFrame(rows),
        audit=pd.DataFrame(audit_rows),
    )

def _build_paired_mobility(
    allocation: pd.DataFrame,
    *,
    hours: int,
    seed: int,
) -> _MobilityInputs | None:
    row_counts = (
        _sum_numeric_columns(
            allocation,
            ("residential_ev_charger_rows", "ghd_ev_charger_rows"),
        )
        .round()
        .astype(int)
    )
    row_capacity = _sum_numeric_columns(
        allocation,
        ("residential_ev_charger_kw", "ghd_ev_charger_kw"),
    )
    if int(row_counts.sum()) == 0:
        return None

    metadata = pd.read_csv(MOBILITY_POOL_DIR / "mobility_profile_pool_metadata.csv")
    metadata = (
        metadata[metadata["weather_key"].astype(str).eq(DEFAULT_MOBILITY_WEATHER_KEY)]
        .sort_values("profile_id")
        .reset_index(drop=True)
    )
    if metadata.empty:
        raise ValueError(
            f"No mobility profiles are available for {DEFAULT_MOBILITY_WEATHER_KEY}."
        )

    selected_records: list[dict[str, Any]] = []
    sorted_allocation = allocation.assign(
        _asset_count=row_counts,
        _asset_capacity_kw=row_capacity,
    ).sort_values(
        ["building_objectid", "source_lv_id", "source_allocation_bus"],
        na_position="last",
    )
    for row in sorted_allocation.to_dict("records"):
        count = int(row["_asset_count"])
        if count <= 0:
            continue
        source_lv, source_bus = _source_values(row)
        per_asset_kw = float(row["_asset_capacity_kw"]) / count
        for sequence in range(count):
            selected_index = stable_seed(
                seed,
                *profile_key(row),
                source_lv,
                source_bus,
                "EV",
                sequence,
            ) % len(metadata)
            selected_records.append(
                {
                    **row,
                    "profile_sequence": sequence,
                    "charger_kw": per_asset_kw,
                    "profile_id": metadata.iloc[selected_index]["profile_id"],
                    "battery_cap_kwh": float(
                        metadata.iloc[selected_index]["battery_cap_kwh"]
                    ),
                }
            )
    profile_ids = [record["profile_id"] for record in selected_records]
    demand_pool = _read_pool_timeseries(
        MOBILITY_POOL_DIR / "mobility_demand_pool.csv",
        profile_ids,
        "demand_kwh",
    )
    availability_pool = _read_pool_timeseries(
        MOBILITY_POOL_DIR / "mobility_availability_pool.csv",
        profile_ids,
        "availability",
    )

    demand_parts = []
    availability_parts = []
    process_rows = []
    commodity_rows = []
    process_commodity_rows = []
    storage_rows = []
    audit_rows = []
    for global_id, record in enumerate(selected_records):
        bus = _target_bus(record)
        profile_id = record["profile_id"]
        mobility = f"mobility{global_id}"
        charger = f"charging_station{global_id}"
        storage = f"mobility_storage{global_id}"
        demand = demand_pool[profile_id].iloc[:hours].reset_index(drop=True)
        availability = availability_pool[profile_id].iloc[:hours].reset_index(drop=True)
        demand_parts.append(demand.rename((bus, mobility)))
        availability_parts.append(availability.rename((bus, charger)))
        process_rows.append(
            {
                "Site": bus,
                "Process": charger,
                "inst-cap": record["charger_kw"],
                "cap-up": record["charger_kw"],
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
                "Commodity": mobility,
                "Type": "Demand",
                "price": np.nan,
            }
        )
        process_commodity_rows.extend(
            [
                {
                    "Process": charger,
                    "Commodity": "electricity",
                    "Direction": "In",
                    "ratio": 1,
                },
                {
                    "Process": charger,
                    "Commodity": mobility,
                    "Direction": "Out",
                    "ratio": 1,
                },
            ]
        )
        storage_rows.append(
            {
                "Site": bus,
                "Storage": storage,
                "Commodity": mobility,
                "inst-cap-c": record["battery_cap_kwh"],
                "cap-up-c": record["battery_cap_kwh"],
                "inst-cap-p": record["battery_cap_kwh"],
                "cap-up-p": record["battery_cap_kwh"],
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
                "building_objectid": record.get("building_objectid"),
                "profile_sequence": record["profile_sequence"],
                "capacity_kw": record["charger_kw"],
                "profile_label": profile_id,
                "battery_cap_kwh": record["battery_cap_kwh"],
                "demand_sum_kwh": float(demand.sum()),
            }
        )

    demand_df = pd.concat(demand_parts, axis=1)
    demand_df.columns = pd.MultiIndex.from_tuples(demand_df.columns)
    demand_df.index.name = "t"
    eff_df = pd.concat(availability_parts, axis=1)
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


def _build_paired_heat(
    allocation: pd.DataFrame,
    *,
    hours: int,
    synthetic_input_dir: Path,
    heat_profile_catalog: pd.DataFrame | None,
    allow_diagnostic_fallback: bool,
) -> _HeatInputs | None:
    wp_count = _sum_numeric_columns(
        allocation,
        ("residential_wp_rows", "ghd_wp_rows"),
    )
    selected = allocation.assign(_wp_count=wp_count)
    selected = selected[selected["_wp_count"].gt(0.0)].copy()
    if selected.empty:
        return None

    required = {
        "building_objectid",
        "synthetic_bridge_filename",
        "synthetic_bus",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(
            "Paired heat allocation requires physical synthetic source mapping: "
            f"{sorted(missing)}"
        )

    source_cache: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}
    demand_entries: list[tuple[tuple[int, str], pd.Series]] = []
    heat_by_target: dict[int, list[tuple[pd.Series, pd.Series]]] = {}
    audit_rows: list[dict[str, Any]] = []
    catalog_by_building = {}
    if heat_profile_catalog is not None and not heat_profile_catalog.empty:
        catalog_by_building = (
            heat_profile_catalog.drop_duplicates("building_objectid")
            .set_index("building_objectid")
            .to_dict("index")
        )

    for building_id, group in selected.groupby(
        "building_objectid",
        sort=True,
        observed=True,
    ):
        source_files = group["synthetic_bridge_filename"].dropna().unique()
        source_buses = (
            pd.to_numeric(
                group["synthetic_bus"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )
        if len(source_files) != 1 or len(source_buses) != 1:
            raise ValueError(
                f"Physical building {building_id} has ambiguous synthetic heat "
                "profile sources."
            )
        source_name = str(source_files[0])
        source_bus = int(source_buses[0])
        profile_method = "exact_physical_building"
        profile_scale = 1.0
        catalog_row = catalog_by_building.get(building_id)
        if catalog_row is not None and not bool(
            catalog_row.get("exact_profile_available", True)
        ):
            if not allow_diagnostic_fallback:
                raise ValueError(
                    "No exact physical heat profile is available for "
                    f"{building_id}. Generate a full-local synthetic source "
                    "profile or explicitly enable the diagnostic fallback."
                )
            source_name = str(catalog_row["profile_source_hdf"])
            source_bus = int(catalog_row["profile_source_bus"])
            profile_method = str(catalog_row["profile_method"])
            profile_scale = float(catalog_row["profile_scale"])
        if source_name not in source_cache:
            source_path = synthetic_input_dir / source_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Missing physical-building profile source {source_path}."
                )
            source_cache[source_name] = (
                pd.read_hdf(source_path, key="urbs_in/demand"),
                pd.read_hdf(source_path, key="urbs_in/eff_factor"),
            )
        source_demand, source_eff = source_cache[source_name]
        needed_demand = [
            (source_bus, "space_heat"),
            (source_bus, "water_heat"),
        ]
        needed_eff = (source_bus, "heatpump_air")
        if any(column not in source_demand for column in needed_demand):
            raise ValueError(
                f"Missing heat demand for physical building {building_id} "
                f"at {source_name}:{source_bus}."
            )
        if needed_eff not in source_eff:
            raise ValueError(
                f"Missing heat-pump COP for physical building {building_id} "
                f"at {source_name}:{source_bus}."
            )
        space = (
            pd.to_numeric(
                source_demand[needed_demand[0]],
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[:hours]
            .reset_index(drop=True)
            * profile_scale
        )
        water = (
            pd.to_numeric(
                source_demand[needed_demand[1]],
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[:hours]
            .reset_index(drop=True)
            * profile_scale
        )
        cop = (
            pd.to_numeric(
                source_eff[needed_eff],
                errors="coerce",
            )
            .fillna(1.0)
            .iloc[:hours]
            .reset_index(drop=True)
            .clip(lower=0.1)
        )
        total_weight = float(group["_wp_count"].sum())
        for row in group.to_dict("records"):
            share = float(row["_wp_count"]) / total_weight
            target_bus = _target_bus(row)
            allocated_space = space * share
            allocated_water = water * share
            demand_entries.extend(
                [
                    ((target_bus, "space_heat"), allocated_space),
                    ((target_bus, "water_heat"), allocated_water),
                ]
            )
            heat = allocated_space + allocated_water
            heat_by_target.setdefault(target_bus, []).append((heat, heat * cop))
            audit_rows.append(
                {
                    "sector": "heat",
                    "allocation_bus": target_bus,
                    "building_objectid": building_id,
                    "asset_rows": float(row["_wp_count"]),
                    "building_profile_share": share,
                    "source_hdf": source_name,
                    "profile_method": profile_method,
                    "profile_scale": profile_scale,
                    "source_bus": source_bus,
                    "annual_space_heat_kwh": float(allocated_space.sum()),
                    "annual_water_heat_kwh": float(allocated_water.sum()),
                    "heat_weighted_mean_cop": float((heat * cop).sum() / heat.sum())
                    if float(heat.sum()) > 0.0
                    else 1.0,
                }
            )

    demand = sum_series_by_key(demand_entries)
    demand.columns = pd.MultiIndex.from_tuples(demand.columns)
    demand.index.name = "t"
    cop_parts = []
    for bus, parts in sorted(heat_by_target.items()):
        heat = sum(part[0] for part in parts)
        weighted = sum(part[1] for part in parts)
        cop = (weighted / heat.replace(0.0, np.nan)).fillna(1.0)
        cop_parts.append(cop.rename((bus, "heatpump_air")))
    eff_factor = pd.concat(cop_parts, axis=1)
    eff_factor.columns = pd.MultiIndex.from_tuples(eff_factor.columns)
    eff_factor.index.name = "t"
    buses = sorted(demand.columns.get_level_values(0).unique())
    return _HeatInputs(
        demand=demand,
        eff_factor=eff_factor,
        process=_create_pro_heat(buses),
        commodity=_create_com_heat(buses),
        process_commodity=_create_pro_com_heat(),
        storage=_create_sto_heat(buses),
        audit=pd.DataFrame(audit_rows),
    )
