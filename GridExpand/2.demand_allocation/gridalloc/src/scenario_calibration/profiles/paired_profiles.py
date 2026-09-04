"""Canonical demand profiles shared by paired real and synthetic grid runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from ...assets.battery.materialization import materialize_battery_urbs_inputs
from ...assets.battery.sizing import build_battery_asset_plan
from ...assets.heat.materialization import materialize_heat_urbs_inputs
from ...assets.heat.sizing import build_heat_asset_plan
from ...functions.heat import get_norm_outside_temperature
from ...assets.pv.materialization import materialize_pv_urbs_inputs
from ...assets.pv.sizing import build_pv_asset_plan
from common.electrification import validate_electrification_assignment

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
    _empty_timeseries,
    _read_pool_timeseries,
)

from ..paths import SYNTHETIC_INPUT_DIR
from .heat_profile_source import load_physical_heat_profile
from .physical_heat_profile_library import PhysicalHeatProfileLibrary
from .pv_profile_library import read_pv_profile_library


def source_match_buildings(
    assignment: pd.DataFrame, evidence_key: str
) -> set[str]:
    """Return selected source-study buildings with one exact evidence type."""
    rows = assignment.loc[
        assignment["technology"].eq("pv_battery")
        & assignment["selected"].astype(bool)
    ]
    selected: set[str] = set()
    for row in rows.to_dict("records"):
        evidence = row.get("source_evidence")
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid pv_battery source evidence in electrification assignment."
                ) from exc
        if isinstance(evidence, dict) and int(evidence.get(evidence_key, 0)) > 0:
            selected.add(str(row["building_objectid"]))
    return selected


def source_asset_sites(
    allocation: pd.DataFrame,
    selected_buildings: set[str],
    count_columns: tuple[str, ...],
) -> dict[str, int]:
    """Resolve one exact source-study connection per selected building."""
    selected_buildings = {str(value) for value in selected_buildings}
    selected_buildings &= set(
        allocation["building_objectid"].astype(str)
    )
    if not selected_buildings:
        return {}
    rows = allocation.loc[
        allocation["building_objectid"].astype(str).isin(selected_buildings)
    ].copy()
    rows["_source_asset_count"] = _sum_numeric_columns(rows, count_columns)
    rows = rows.loc[rows["_source_asset_count"].gt(0.0)].copy()
    site_column = (
        "_profile_site_id" if "_profile_site_id" in rows else "allocation_bus"
    )
    sort_columns = ["building_objectid", "_source_asset_count"]
    ascending = [True, False]
    for column in (
        "source_lv_id",
        "source_allocation_bus",
        "scenario_unit_id",
        site_column,
    ):
        if column in rows and column not in sort_columns:
            sort_columns.append(column)
            ascending.append(True)
    chosen = rows.sort_values(
        sort_columns,
        ascending=ascending,
        kind="stable",
    ).drop_duplicates("building_objectid", keep="first")
    result = dict(
        zip(
            chosen["building_objectid"].astype(str),
            pd.to_numeric(chosen[site_column], errors="raise").astype(int),
        )
    )
    missing = sorted(selected_buildings - set(result))
    if missing:
        raise ValueError(
            "Selected source-study buildings lack an exact asset connection: "
            f"{missing[:10]}"
        )
    return result


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


def _build_paired_component_electric_demand(
    component_plan: pd.DataFrame,
    *,
    seed: int,
    measured_profile_selection: str,
    measured_profile_band_pct: float,
    measured_profile_min_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate shared electricity profiles from the paired component plan."""
    electricity = load_electricity_module()
    normalized_hh = pd.read_hdf(electricity.config.ELEC_LPS_PATH, key="df_normalized_scaled")
    hh_totals = pd.read_hdf(electricity.config.ELEC_LPS_PATH, key="df_sums")
    ghd_shapes = ghd_shape_by_building_use(electricity)
    entries: list[tuple[int, pd.Series]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in component_plan.sort_values(["building_objectid", "component_id"]).to_dict("records"):
        if not bool(row.get("included_in_lv", False)):
            continue
        annual = float(pd.to_numeric(pd.Series([row.get("annual_energy_kwh")]), errors="coerce").fillna(0.0).iloc[0])
        if annual <= 0.0:
            continue
        bus = int(row["target_bus"])
        category = str(row["component_category"])
        asset_count = int(round(float(pd.to_numeric(pd.Series([row.get("source_asset_count", 0)]), errors="coerce").fillna(0.0).iloc[0])))
        if category == "Residential":
            if asset_count <= 0:
                raise ValueError(f"Residential paired component {row['component_id']} has energy but no HH evidence.")
            per_hh_kwh = annual / asset_count
            for sequence in range(asset_count):
                rng = np.random.default_rng(stable_seed(int(row["stable_seed"]), "HH", sequence))
                selected = select_residential_profile(
                    hh_totals, per_hh_kwh, rng, measured_profile_selection,
                    measured_profile_band_pct, measured_profile_min_candidates,
                )
                series = (normalized_hh[selected["chosen_profile_device"]] * per_hh_kwh).reset_index(drop=True)
                entries.append((bus, series))
                audit_rows.append({
                    "component_id": row["component_id"], "building_objectid": row["building_objectid"],
                    "target_bus": bus, "category": category, "demand_class": "HH",
                    "profile_sequence": sequence, "annual_demand_kwh": per_hh_kwh,
                    "profile_hash": row.get("profile_hash"), "stable_seed": int(row["stable_seed"]),
                    **{key: value for key, value in selected.items() if key != "chosen_index"},
                })
        elif category in {"Commercial", "Public"}:
            shape_key = category if category in ghd_shapes else "default"
            entries.append((bus, (ghd_shapes[shape_key] * annual).reset_index(drop=True)))
            audit_rows.append({
                "component_id": row["component_id"], "building_objectid": row["building_objectid"],
                "target_bus": bus, "category": category, "demand_class": "GHD",
                "profile_sequence": 0, "annual_demand_kwh": annual,
                "profile_hash": row.get("profile_hash"), "stable_seed": int(row["stable_seed"]),
                "chosen_profile_device": f"weighted_{shape_key}_ghd",
                "chosen_profile_kwh": annual, "candidate_count": np.nan,
                "candidate_method": "weighted_ghd_shape",
                "candidate_min_kwh": np.nan, "candidate_max_kwh": np.nan,
            })
        else:
            raise ValueError(f"Unsupported paired component category {category!r}.")

    demand = sum_series_by_key(entries)
    if demand.empty:
        raise ValueError("Paired component plan produced no active electricity demand.")
    demand = add_output_data_daylight_saving_shift(demand)
    demand.columns = pd.MultiIndex.from_tuples(
        [(int(bus), "electricity") for bus in demand.columns], names=["Site", "Commodity"]
    )
    demand = demand.sort_index(axis=1)
    assert_unique_columns(demand, "paired component electricity")
    assert_energy_conserved(
        demand,
        float(pd.to_numeric(component_plan.loc[component_plan["included_in_lv"].astype(bool), "annual_energy_kwh"], errors="coerce").fillna(0.0).sum()),
        label="paired component electricity",
    )
    return demand, pd.DataFrame(audit_rows)


def build_paired_base_electric_demand(
    allocation: pd.DataFrame,
    *,
    seed: int,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    component_plan: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one reproducible physical demand realization on target buses."""
    if component_plan is not None:
        return _build_paired_component_electric_demand(
            component_plan,
            seed=seed,
            measured_profile_selection=measured_profile_selection,
            measured_profile_band_pct=measured_profile_band_pct,
            measured_profile_min_candidates=measured_profile_min_candidates,
        )
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
    pv_demand_multiplier: float = 2.0,
    battery_sizing_method: str = "htw_2025_scaled_rule",
    battery_minimum_pv_kwp_per_annual_mwh: float = 0.5,
    battery_usable_kwh_per_pv_kwp: float = 1.0,
    battery_usable_kwh_per_annual_mwh: float = 1.0,
    battery_energy_to_power_hours: float = 2.0,
    synthetic_input_dir: Path = SYNTHETIC_INPUT_DIR,
    technology_parameters=None,
    heat_sizing_method: str = "full_load_hours_rule",
    heat_config=None,
    heat_profile_catalog: pd.DataFrame | None = None,
    heat_profile_library: Path | None = None,
    allow_diagnostic_heat_fallback: bool = False,
    electrification_assignment: pd.DataFrame | None = None,
) -> SectorUrbsInputs:
    """Build SWF-2045 sector profiles once and remap them to target buses."""
    if electrification_assignment is None:
        raise ValueError(
            "Paired sector materialization requires the shared electrification assignment."
        )
    validate_electrification_assignment(electrification_assignment)
    selected_by_technology = {
        technology: set(
            electrification_assignment.loc[
                electrification_assignment["technology"].eq(technology)
                & electrification_assignment["selected"].astype(bool),
                "building_objectid",
            ].astype(str)
        )
        for technology in ("heat", "mobility", "pv_battery")
    }
    pv_battery_rows = electrification_assignment.loc[
        electrification_assignment["technology"].eq("pv_battery")
    ]
    pv_battery_mode = str(pv_battery_rows["adoption_mode"].iloc[0])
    if pv_battery_mode == "source_inventory":
        selected_pv_buildings = source_match_buildings(
            electrification_assignment, "pv_match_count"
        )
        selected_battery_buildings = source_match_buildings(
            electrification_assignment, "battery_match_count"
        )
    else:
        selected_pv_buildings = selected_by_technology["pv_battery"]
        selected_battery_buildings = selected_by_technology["pv_battery"]
    battery_site_by_building = (
        source_asset_sites(
            allocation,
            selected_battery_buildings,
            (
                "residential_battery_rows",
                "ghd_battery_rows",
            ),
        )
        if pv_battery_mode == "source_inventory"
        else None
    )

    pv = _build_paired_pv(
        allocation,
        roof_catalog=roof_catalog,
        hours=hours,
        profile_library=pv_profile_library,
        sizing_method=pv_sizing_method,
        demand_multiplier=pv_demand_multiplier,
        technical_parameters=technology_parameters.processes["rooftop_pv"],
        selected_buildings=selected_pv_buildings,
        battery_candidate_buildings=selected_battery_buildings,
    )
    if pv is None:
        battery = None
    else:
        pv_asset_plan = pv.audit[
            pv.audit["audit_record_type"].isin(
                ["asset_plan", "battery_pv_reference_plan"]
            )
        ].copy()
        battery = _build_paired_battery(
            pv_asset_plan=pv_asset_plan,
            sizing_method=battery_sizing_method,
            minimum_pv_kwp_per_annual_mwh=battery_minimum_pv_kwp_per_annual_mwh,
            usable_kwh_per_pv_kwp=battery_usable_kwh_per_pv_kwp,
            usable_kwh_per_annual_mwh=(
                battery_usable_kwh_per_annual_mwh
            ),
            energy_to_power_hours=battery_energy_to_power_hours,
            eligible_buildings=selected_battery_buildings,
            site_by_building=battery_site_by_building,
            technical_parameters=technology_parameters.storages["stationary_battery"],
        )
    mobility = _build_paired_mobility(
        allocation,
        hours=hours,
        seed=seed,
        process_parameters=technology_parameters.processes["home_charger"],
        storage_parameters=technology_parameters.storages["mobility_storage"],
        selected_buildings=selected_by_technology["mobility"],
        adoption_mode=str(
            electrification_assignment.loc[
                electrification_assignment["technology"].eq("mobility"),
                "adoption_mode",
            ].iloc[0]
        ),
    )
    heat = _build_paired_heat(
        allocation,
        heat_profile_catalog=heat_profile_catalog,
        heat_profile_library=heat_profile_library,
        allow_diagnostic_fallback=allow_diagnostic_heat_fallback,
        hours=hours,
        synthetic_input_dir=synthetic_input_dir,
        weather_source_hdf=weather_source_hdf,
        sizing_method=heat_sizing_method,
        heat_config=heat_config,
        technology_parameters=technology_parameters,
        selected_buildings=selected_by_technology["heat"],
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
            "profile_contract": "physical_building_component_paired_v2",
            "heat_profile_source": (
                "matched physical building from pylovo version-specific "
                "synthetic URBS input"
            ),
            "stationary_battery_model": (
                "SWF rows provide location evidence; capacity follows the shared "
                "HTW bound and the selected heuristic or optimization sizing mode"
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
    demand_multiplier: float = 2.0,
    technical_parameters=None,
    selected_buildings: set[str] | None = None,
    battery_candidate_buildings: set[str] | None = None,
) -> _PvInputs | None:
    if "pv_roof_capacity_kw" not in allocation:
        raise ValueError(
            "Paired allocation is missing pv_roof_capacity_kw. Regenerate the "
            "LoD2-aware paired allocation first."
        )
    roof_capacity = pd.to_numeric(
        allocation["pv_roof_capacity_kw"], errors="coerce"
    ).fillna(0.0)
    eligible = allocation.loc[roof_capacity.gt(0.0)].copy()
    selected_buildings = (
        {str(value) for value in selected_buildings}
        if selected_buildings is not None
        else set(eligible["building_objectid"].astype(str))
    )
    battery_candidate_buildings = {
        str(value) for value in (battery_candidate_buildings or set())
    }
    planning_buildings = selected_buildings | battery_candidate_buildings
    eligible = eligible[
        eligible["building_objectid"].astype(str).isin(planning_buildings)
    ].copy()
    if eligible.empty:
        return None
    eligible["building_objectid"] = eligible["building_objectid"].astype(str)
    eligible["_profile_site_id"] = eligible.get(
        "_profile_site_id", eligible["allocation_bus"]
    ).astype(int)
    local_selected_pv = selected_buildings & set(
        eligible["building_objectid"]
    )
    if local_selected_pv:
        if "pv_roof_eligible" not in eligible:
            raise ValueError(
                "Paired allocation is missing the physical PV source connection."
            )
        pv_locations = eligible.loc[
            eligible["building_objectid"].isin(local_selected_pv)
            & eligible["pv_roof_eligible"].astype(bool)
        ]
        ambiguous = (
            pv_locations.groupby("building_objectid")["_profile_site_id"]
            .nunique()
            .gt(1)
        )
        if ambiguous.any():
            raise ValueError(
                "Paired allocation contains multiple PV source connections for "
                f"buildings: {sorted(ambiguous[ambiguous].index)[:10]}"
            )
        pv_sites = (
            pv_locations.drop_duplicates("building_objectid")
            .set_index("building_objectid")["_profile_site_id"]
            .astype(int)
            .to_dict()
        )
        missing = sorted(local_selected_pv - set(pv_sites))
        if missing:
            raise ValueError(
                "Selected PV buildings lack an exact assigned connection: "
                f"{missing[:10]}"
            )
    else:
        pv_sites = {}
    eligible["_pv_asset_site_id"] = (
        eligible["building_objectid"].map(pv_sites)
        .fillna(eligible["_profile_site_id"])
        .astype(int)
    )
    profiles = read_pv_profile_library(profile_library).iloc[:hours].reset_index(drop=True)
    if len(profiles) != hours:
        raise ValueError(
            f"PV profile library has {len(profiles)} rows, expected {hours}."
        )
    household_column = (
        "building_households"
        if "building_households" in eligible
        else "residential_equivalent_hh_rows"
    )
    buildings = eligible.assign(
        Site=eligible["_pv_asset_site_id"],
        annual_electricity_kwh=_sum_numeric_columns(
            eligible, ("residential_equivalent_hh_annual_kwh", "calibrated_annual_ghd_kwh")
        ),
        number_of_households=pd.to_numeric(
            eligible[household_column], errors="coerce"
        ).fillna(0.0),
    )
    asset_plan, selected_sections = build_pv_asset_plan(
        buildings,
        roof_catalog,
        profiles,
        sizing_method=sizing_method,
        demand_multiplier=demand_multiplier,
    )
    pv_asset_plan = asset_plan.loc[
        asset_plan["building_objectid"].astype(str).isin(selected_buildings)
    ].copy()
    pv_sections = selected_sections.loc[
        selected_sections["building_objectid"].astype(str).isin(selected_buildings)
    ].copy()
    shared = materialize_pv_urbs_inputs(
        pv_asset_plan,
        pv_sections,
        profiles,
        sizing_method=sizing_method,
        technical_parameters=technical_parameters,
    )
    audit = shared.audit.copy()
    audit["sector"] = "pv"
    audit["profile_method"] = "lod2_roof_angles_pvlib"
    audit["audit_record_type"] = "pv_materialization"
    plan_audit = asset_plan.copy()
    plan_audit["sector"] = "pv"
    plan_audit["profile_method"] = "lod2_roof_angles_pvlib"
    plan_audit["audit_record_type"] = np.where(
        plan_audit["building_objectid"].astype(str).isin(selected_buildings),
        "asset_plan",
        "battery_pv_reference_plan",
    )
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
    *,
    pv_asset_plan: pd.DataFrame,
    sizing_method: str,
    minimum_pv_kwp_per_annual_mwh: float,
    usable_kwh_per_pv_kwp: float,
    usable_kwh_per_annual_mwh: float,
    energy_to_power_hours: float,
    eligible_buildings: set[str] | None,
    site_by_building: dict[str, int] | None,
    technical_parameters: dict,
) -> _BatteryInputs | None:
    """Size batteries consistently while using SWF rows only as location evidence."""
    eligible_buildings = (
        {str(value) for value in eligible_buildings}
        if eligible_buildings is not None
        else None
    )
    plan = build_battery_asset_plan(
        pv_asset_plan,
        sizing_method=sizing_method,
        minimum_pv_kwp_per_annual_mwh=minimum_pv_kwp_per_annual_mwh,
        usable_kwh_per_pv_kwp=usable_kwh_per_pv_kwp,
        usable_kwh_per_annual_mwh=usable_kwh_per_annual_mwh,
        eligible_buildings=eligible_buildings,
        site_by_building=site_by_building,
        location_source="electrification_assignment",
    )
    materialized = materialize_battery_urbs_inputs(
        plan,
        sizing_method=sizing_method,
        energy_to_power_hours=energy_to_power_hours,
        technical_parameters=technical_parameters,
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
    process_parameters: dict,
    storage_parameters: dict,
    hours: int,
    seed: int,
    selected_buildings: set[str] | None = None,
    adoption_mode: str = "source_inventory",
) -> _MobilityInputs | None:
    # Mobility is a physical household asset. Non-residential EV evidence is
    # intentionally outside this v1 component scope and must not create a
    # second vehicle population, or vehicles on a pure non-residential row.
    residential_area = pd.to_numeric(
        allocation.get(
            "residential_effective_floor_area_m2",
            pd.Series(0.0, index=allocation.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    if adoption_mode == "source_inventory":
        row_counts = _sum_numeric_columns(
            allocation, ("residential_ev_charger_rows",)
        )
        row_capacity = _sum_numeric_columns(
            allocation, ("residential_ev_charger_kw",)
        )
    elif adoption_mode == "deterministic_share":
        row_counts = _sum_numeric_columns(
            allocation, ("deterministic_vehicle_count",)
        )
        row_capacity = row_counts * float(
            process_parameters["installed_capacity_kw"]
        )
    else:
        raise ValueError(f"Unknown mobility adoption mode {adoption_mode!r}.")
    row_counts = row_counts.where(residential_area.gt(0.0), 0.0).round().astype(int)
    if selected_buildings is not None:
        row_counts = row_counts.where(
            allocation["building_objectid"].astype(str).isin(selected_buildings),
            0,
        )
    row_capacity = row_capacity.where(residential_area.gt(0.0), 0.0)
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
                "inv-cost-fix": process_parameters["fixed_investment_cost_eur"],
                "inv-cost": process_parameters["investment_cost_eur_per_kw"],
                "fix-cost": process_parameters["fixed_cost_eur_per_hour"],
                "var-cost": process_parameters["variable_cost_eur_per_kwh"],
                "wacc": process_parameters["wacc"],
                "depreciation": process_parameters["depreciation_years"],
                "pf-min": process_parameters["minimum_power_factor"],
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
                "eff-in": storage_parameters["charge_efficiency"],
                "eff-out": storage_parameters["discharge_efficiency"],
                "discharge": storage_parameters["self_discharge_per_timestep"],
                "ep-ratio": storage_parameters["energy_to_power_hours"],
                "inv-cost-p": storage_parameters["investment_cost_eur_per_kw"],
                "inv-cost-c": storage_parameters["investment_cost_eur_per_kwh"],
                "fix-cost-p": storage_parameters["fixed_investment_cost_power_eur"],
                "fix-cost-c": storage_parameters["fixed_investment_cost_energy_eur"],
                "var-cost-p": storage_parameters["variable_cost_eur_per_kwh"],
                "wacc": storage_parameters["wacc"],
                "depreciation": storage_parameters["depreciation_years"],
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


def _weather_and_postcode(weather_source_hdf: Path, hours: int):
    try:
        raw = pd.read_hdf(weather_source_hdf, key="raw_data/weather")
        ambient = pd.to_numeric(raw["temp_air"], errors="coerce")
    except (KeyError, FileNotFoundError):
        weather = pd.read_hdf(weather_source_hdf, key="urbs_in/weather")
        key = ("ambient", "Tamb") if ("ambient", "Tamb") in weather.columns else "Tamb"
        ambient = pd.to_numeric(weather[key], errors="coerce")
    ambient = ambient.iloc[:hours].reset_index(drop=True)
    try:
        region = pd.read_hdf(weather_source_hdf, key="raw_data/region")
        postcode = str(region.iloc[0]["plz"]).zfill(5)
    except KeyError:
        match = re.search(r"_(\d{5})_", weather_source_hdf.name)
        if match is None:
            raise ValueError(
                "Paired heat sizing requires postcode metadata or a postcode-qualified weather filename."
            )
        postcode = match.group(1)
    return ambient, postcode


def _build_paired_heat(
    allocation: pd.DataFrame,
    *,
    hours: int,
    synthetic_input_dir: Path,
    heat_profile_catalog: pd.DataFrame | None,
    heat_profile_library: Path | None,
    allow_diagnostic_fallback: bool,
    weather_source_hdf: Path,
    sizing_method: str,
    heat_config,
    technology_parameters,
    selected_buildings: set[str] | None = None,
) -> _HeatInputs | None:
    # The first heat heuristic is residential only. Commercial HP rows remain
    # outside scope until a separate commercial sizing method is documented.
    selected = allocation.copy()
    if selected_buildings is not None:
        selected = selected[
            selected["building_objectid"].astype(str).isin(selected_buildings)
        ].copy()
    selected = selected[
        pd.to_numeric(
            selected.get(
                "residential_effective_floor_area_m2",
                pd.Series(0.0, index=selected.index),
            ),
            errors="coerce",
        ).fillna(0.0).gt(0.0)
    ].copy()
    if selected.empty:
        return None
    required = {"building_objectid", "synthetic_bridge_filename", "synthetic_bus"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(
            "Paired heat allocation requires physical synthetic source mapping: "
            f"{sorted(missing)}"
        )
    if heat_config is None:
        raise ValueError("Paired heat materialization requires scenario heat configuration.")

    library = PhysicalHeatProfileLibrary(heat_profile_library) if heat_profile_library is not None else None
    source_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    demand_entries: list[tuple[tuple[int, str], pd.Series]] = []
    heat_by_target: dict[int, list[tuple[pd.Series, pd.Series]]] = {}
    profile_audit = []
    building_rows = []
    catalog_by_building = {}
    if heat_profile_catalog is not None and not heat_profile_catalog.empty:
        catalog_by_building = heat_profile_catalog.drop_duplicates("building_objectid").set_index("building_objectid").to_dict("index")

    for building_id, group in selected.groupby("building_objectid", sort=True, observed=True):
        space, water, cop, source_name, source_bus, profile_method, profile_scale = load_physical_heat_profile(
            building_objectid=building_id,
            group=group,
            hours=hours,
            synthetic_input_dir=synthetic_input_dir,
            library=library,
            catalog_by_building=catalog_by_building,
            source_cache=source_cache,
            allow_diagnostic_fallback=allow_diagnostic_fallback,
        )
        water = water.reset_index(drop=True)
        primary = group.sort_values("_profile_site_id").iloc[0]
        target_bus = _target_bus(primary)
        space = space.reset_index(drop=True)
        cop = cop.reset_index(drop=True).clip(lower=0.1)
        demand_entries.extend([
            ((target_bus, "space_heat"), space),
            ((target_bus, "water_heat"), water),
        ])
        heat = space + water
        heat_by_target.setdefault(target_bus, []).append((heat, heat * cop))
        building_rows.append({
            "building_objectid": str(building_id),
            "Site": target_bus,
            "building_type": primary.get("building_type"),
            "building_use": primary.get("building_use"),
            "number_of_households": primary.get("building_households"),
            "floor_area": primary.get("residential_effective_floor_area_m2", primary.get("building_floor_area")),
        })
        profile_audit.append({
            "sector": "heat", "audit_record_type": "heat_profile",
            "allocation_bus": target_bus, "building_objectid": str(building_id),
            "source_hdf": source_name, "profile_method": profile_method,
            "profile_scale": profile_scale, "source_bus": source_bus,
            "annual_space_heat_kwh": float(space.sum()),
            "annual_water_heat_kwh": float(water.sum()),
            "heat_weighted_mean_cop": float((heat * cop).sum() / heat.sum()) if float(heat.sum()) > 0.0 else 1.0,
        })

    demand = sum_series_by_key(demand_entries)
    demand.columns = pd.MultiIndex.from_tuples(demand.columns)
    demand.index.name = "t"
    cop_parts = []
    for bus, parts in sorted(heat_by_target.items()):
        heat = sum(part[0] for part in parts)
        weighted = sum(part[1] for part in parts)
        cop_parts.append((weighted / heat.replace(0.0, np.nan)).fillna(1.0).rename((bus, "heatpump_air")))
    eff_factor = pd.concat(cop_parts, axis=1)
    eff_factor.columns = pd.MultiIndex.from_tuples(eff_factor.columns)
    eff_factor.index.name = "t"
    ambient, postcode = _weather_and_postcode(weather_source_hdf, hours)
    plan, _climate = build_heat_asset_plan(
        pd.DataFrame(building_rows), demand.loc[:, pd.IndexSlice[:, ["space_heat"]]],
        demand.loc[:, pd.IndexSlice[:, ["water_heat"]]], eff_factor, ambient,
        sizing_method=sizing_method,
        norm_outside_temperature_c=get_norm_outside_temperature(postcode),
        indoor_design_temperature_c=heat_config.indoor_design_temperature_c,
        heating_limit_temperature_c=heat_config.heating_limit_temperature_c,
        heat_pump_design_share=heat_config.heat_pump_design_share,
        buffer_volume_l_per_kw_th=heat_config.buffer_volume_l_per_kw_th,
        buffer_usable_temperature_spread_k=heat_config.buffer_usable_temperature_spread_k,
    )
    materialized = materialize_heat_urbs_inputs(
        plan, sizing_method=sizing_method,
        process_parameters=technology_parameters.processes,
        storage_parameters=technology_parameters.storages["thermal_storage"],
    )
    audit = pd.concat([pd.DataFrame(profile_audit), materialized.audit], ignore_index=True, sort=False)
    return _HeatInputs(
        demand=demand, eff_factor=eff_factor,
        process=materialized.process, commodity=materialized.commodity,
        process_commodity=materialized.process_commodity,
        storage=materialized.storage, audit=audit,
    )
