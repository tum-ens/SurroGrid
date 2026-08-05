"""Canonical demand profiles shared by paired real and synthetic grid runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ...assets.battery.materialization import materialize_battery_urbs_inputs
from ...assets.battery.sizing import build_battery_asset_plan
from ...assets.pv.materialization import materialize_pv_urbs_inputs
from ...assets.pv.sizing import build_pv_asset_plan

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
from .heat_profile_source import load_physical_heat_profile
from .physical_heat_profile_library import PhysicalHeatProfileLibrary
from .pv_profile_library import read_pv_profile_library


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
    roof_catalog: pd.DataFrame,
    pv_profile_library: Path,
    pv_sizing_method: str = "optimization",
    pv_demand_multiplier: float = 2.5,
    battery_sizing_method: str = "htw_2025_upper_bound",
    battery_minimum_pv_kwp_per_annual_mwh: float = 0.5,
    battery_maximum_usable_kwh_per_pv_kwp: float = 1.5,
    battery_maximum_usable_kwh_per_annual_mwh: float = 1.5,
    battery_energy_to_power_hours: float = 2.0,
    battery_predefined_locations_when_available: bool = True,
    synthetic_input_dir: Path = SYNTHETIC_INPUT_DIR,
    heat_profile_catalog: pd.DataFrame | None = None,
    heat_profile_library: Path | None = None,
    allow_diagnostic_heat_fallback: bool = False,
) -> SectorUrbsInputs:
    """Build SWF-2045 sector profiles once and remap them to target buses."""
    pv = _build_paired_pv(
        allocation,
        roof_catalog=roof_catalog,
        hours=hours,
        profile_library=pv_profile_library,
        sizing_method=pv_sizing_method,
        demand_multiplier=pv_demand_multiplier,
    )
    if pv is None:
        battery = None
    else:
        pv_asset_plan = pv.audit[
            pv.audit["audit_record_type"].eq("asset_plan")
        ].copy()
        battery = _build_paired_battery(
            allocation,
            pv_asset_plan=pv_asset_plan,
            sizing_method=battery_sizing_method,
            minimum_pv_kwp_per_annual_mwh=battery_minimum_pv_kwp_per_annual_mwh,
            maximum_usable_kwh_per_pv_kwp=battery_maximum_usable_kwh_per_pv_kwp,
            maximum_usable_kwh_per_annual_mwh=(
                battery_maximum_usable_kwh_per_annual_mwh
            ),
            energy_to_power_hours=battery_energy_to_power_hours,
            predefined_locations_when_available=(
                battery_predefined_locations_when_available
            ),
        )
    mobility = _build_paired_mobility(allocation, hours=hours, seed=seed)
    heat = _build_paired_heat(
        allocation,
        heat_profile_catalog=heat_profile_catalog,
        heat_profile_library=heat_profile_library,
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
    roof_catalog: pd.DataFrame,
    hours: int,
    profile_library: Path,
    sizing_method: str = "optimization",
    demand_multiplier: float = 2.5,
) -> _PvInputs | None:
    if "pv_roof_eligible" not in allocation:
        raise ValueError(
            "Paired allocation is missing pv_roof_eligible. Regenerate the "
            "LoD2-aware paired allocation first."
        )
    eligible = allocation[allocation["pv_roof_eligible"].astype(bool)].copy()
    if eligible.empty:
        return None
    eligible["building_objectid"] = eligible["building_objectid"].astype(str)
    eligible["_profile_site_id"] = eligible.get(
        "_profile_site_id", eligible["allocation_bus"]
    ).astype(int)
    profiles = read_pv_profile_library(profile_library).iloc[:hours].reset_index(drop=True)
    if len(profiles) != hours:
        raise ValueError(
            f"PV profile library has {len(profiles)} rows, expected {hours}."
        )
    buildings = eligible.assign(
        Site=eligible["_profile_site_id"],
        annual_electricity_kwh=_sum_numeric_columns(
            eligible, ("residential_equivalent_hh_annual_kwh", "calibrated_annual_ghd_kwh")
        ),
    )
    asset_plan, selected_sections = build_pv_asset_plan(
        buildings,
        roof_catalog,
        profiles,
        sizing_method=sizing_method,
        demand_multiplier=demand_multiplier,
    )
    shared = materialize_pv_urbs_inputs(
        asset_plan,
        selected_sections,
        profiles,
        sizing_method=sizing_method,
    )
    audit = shared.audit.copy()
    audit["sector"] = "pv"
    audit["profile_method"] = "lod2_roof_angles_pvlib"
    audit["audit_record_type"] = "pv_materialization"
    plan_audit = asset_plan.copy()
    plan_audit["sector"] = "pv"
    plan_audit["profile_method"] = "lod2_roof_angles_pvlib"
    plan_audit["audit_record_type"] = "asset_plan"
    audit = pd.concat(
        [plan_audit, audit],
        ignore_index=True,
        sort=False,
    )
    return _PvInputs(
        shared.supim,
        shared.process,
        shared.commodity,
        shared.process_commodity,
        audit,
    )



def _build_paired_battery(
    allocation: pd.DataFrame,
    *,
    pv_asset_plan: pd.DataFrame,
    sizing_method: str,
    minimum_pv_kwp_per_annual_mwh: float,
    maximum_usable_kwh_per_pv_kwp: float,
    maximum_usable_kwh_per_annual_mwh: float,
    energy_to_power_hours: float,
    predefined_locations_when_available: bool,
) -> _BatteryInputs | None:
    """Size batteries consistently while using SWF rows only as location evidence."""
    inventory_capacity = _sum_numeric_columns(
        allocation, ("residential_battery_kwh", "ghd_battery_kwh")
    )
    inventory = allocation.assign(_battery_inventory_kwh=inventory_capacity)
    inventory = _sorted_allocation(
        inventory[inventory["_battery_inventory_kwh"].gt(0.0)]
    )
    use_inventory = predefined_locations_when_available and not inventory.empty
    if use_inventory:
        eligible_buildings = set(inventory["building_objectid"].astype(str))
        site_by_building = (
            inventory.assign(
                _battery_site=inventory.get(
                    "_profile_site_id", inventory["allocation_bus"]
                ).astype(int)
            )
            .drop_duplicates("building_objectid")
            .set_index(inventory["building_objectid"].astype(str))["_battery_site"]
            .to_dict()
        )
        location_source = "predefined_swf_battery_locations"
    else:
        eligible_buildings = None
        site_by_building = None
        location_source = "all_pv_buildings"

    plan = build_battery_asset_plan(
        pv_asset_plan,
        sizing_method=sizing_method,
        minimum_pv_kwp_per_annual_mwh=minimum_pv_kwp_per_annual_mwh,
        maximum_usable_kwh_per_pv_kwp=maximum_usable_kwh_per_pv_kwp,
        maximum_usable_kwh_per_annual_mwh=maximum_usable_kwh_per_annual_mwh,
        eligible_buildings=eligible_buildings,
        site_by_building=site_by_building,
        location_source=location_source,
    )
    electricity = load_electricity_module()
    materialized = materialize_battery_urbs_inputs(
        plan,
        sizing_method=sizing_method,
        energy_to_power_hours=energy_to_power_hours,
        technical_parameters=electricity.config,
    )
    if materialized.storage.empty:
        return None
    plan_audit = plan.copy()
    plan_audit["sector"] = "stationary_battery"
    plan_audit["audit_record_type"] = "battery_asset_plan"
    return _BatteryInputs(
        storage=materialized.storage,
        audit=pd.concat(
            [plan_audit, materialized.audit], ignore_index=True, sort=False
        ),
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
    heat_profile_library: Path | None,
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

    library = (
        PhysicalHeatProfileLibrary(heat_profile_library)
        if heat_profile_library is not None
        else None
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
        (
            space,
            water,
            cop,
            source_name,
            source_bus,
            profile_method,
            profile_scale,
        ) = load_physical_heat_profile(
            building_objectid=building_id,
            group=group,
            hours=hours,
            synthetic_input_dir=synthetic_input_dir,
            library=library,
            catalog_by_building=catalog_by_building,
            source_cache=source_cache,
            allow_diagnostic_fallback=allow_diagnostic_fallback,
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
