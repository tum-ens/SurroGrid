"""Physical-building PV asset-plan compilation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import profile_label
from .roof_catalog import building_roof_capacity


def heuristic_pv_capacity(annual_electricity_kwh, maximum_pv_kwp, demand_multiplier=2.5):
    annual = np.nan_to_num(np.asarray(annual_electricity_kwh, dtype=float), nan=0.0)
    maximum = np.nan_to_num(np.asarray(maximum_pv_kwp, dtype=float), nan=0.0)
    result = np.minimum(
        np.maximum(annual, 0.0) * float(demand_multiplier) / 1000.0,
        np.maximum(maximum, 0.0),
    )
    return float(result) if result.ndim == 0 else result


def allocate_capacity_best_yield_first(
    roof_catalog: pd.DataFrame,
    targets_kwp: pd.Series,
    profile_library: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate each building target to its highest annual-yield roof bins."""
    usable = roof_catalog[roof_catalog["profile_usable"]].copy()
    annual_yield = profile_library.sum(axis=0)
    usable["profile_label"] = [profile_label(t, a) for t, a in usable[["profile_tilt_deg", "profile_azimuth_deg"]].itertuples(index=False, name=None)]
    missing = set(usable["profile_label"]).difference(annual_yield.index)
    if missing:
        raise ValueError(f"PV profile library lacks profiles: {sorted(missing)[:10]}")
    usable["annual_specific_yield_kwh_per_kwp"] = usable["profile_label"].map(annual_yield)
    usable = usable.sort_values(
        ["building_objectid", "annual_specific_yield_kwh_per_kwp", "roof_surface_id"],
        ascending=[True, False, True],
    )
    selected = []
    for building_id, sections in usable.groupby("building_objectid", sort=True):
        remaining = float(targets_kwp.get(str(building_id), 0.0))
        for row in sections.to_dict("records"):
            capacity = min(max(remaining, 0.0), float(row["available_pv_kw"]))
            row["selected_pv_kw"] = capacity
            selected.append(row)
            remaining -= capacity
    return pd.DataFrame(selected)


def build_pv_asset_plan(
    buildings: pd.DataFrame,
    roof_catalog: pd.DataFrame,
    profile_library: pd.DataFrame,
    *,
    sizing_method: str,
    demand_multiplier: float = 2.5,
    eligibility_column: str = "pv_roof_eligible",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one building asset row and selected roof sections."""
    required = {"building_objectid", "Site", "annual_electricity_kwh"}
    missing = required.difference(buildings.columns)
    if missing:
        raise ValueError(f"PV asset-plan buildings lack columns: {sorted(missing)}")
    source = buildings.copy()
    source["building_objectid"] = source["building_objectid"].astype(str)
    if eligibility_column not in source:
        source[eligibility_column] = True
    source["annual_electricity_kwh"] = pd.to_numeric(
        source["annual_electricity_kwh"], errors="coerce"
    ).fillna(0.0)
    # Paired inputs can contain several scenario units for one building. Size
    # the physical asset once from their summed base-electricity demand.
    aggregation = {
        "Site": ("Site", "first"),
        "annual_electricity_kwh": ("annual_electricity_kwh", "sum"),
        eligibility_column: (eligibility_column, "any"),
    }
    # Retain classification as audit context. It does not restrict the sizing
    # rule: the current HTW extrapolation intentionally applies to every type.
    for column in ("building_use", "building_type", "number_of_households", "floor_area"):
        if column in source:
            aggregation[column] = (column, "first")
    plan = source.groupby(
        "building_objectid", sort=True, as_index=False
    ).agg(**aggregation)
    # A regional LoD2 catalog can contain roofs from many target grids. Keep
    # this asset plan strictly local before validating profile coverage or
    # allocating capacity. This also prevents optimized sections from being
    # assigned to buildings that have no Site in the current plan.
    plan_buildings = set(plan["building_objectid"])
    roof_catalog = roof_catalog[
        roof_catalog["building_objectid"].astype(str).isin(plan_buildings)
    ].copy()
    capacities = building_roof_capacity(roof_catalog)
    capacities.index = capacities.index.astype(str)
    plan["pv_max_kwp"] = plan["building_objectid"].map(capacities).fillna(0.0)
    plan.loc[~plan[eligibility_column].astype(bool), "pv_max_kwp"] = 0.0
    if sizing_method == "annual_electricity_rule":
        plan["pv_demand_target_kwp"] = pd.to_numeric(plan["annual_electricity_kwh"], errors="coerce").fillna(0.0) * demand_multiplier / 1000.0
        plan["pv_installed_kwp"] = heuristic_pv_capacity(plan["annual_electricity_kwh"], plan["pv_max_kwp"], demand_multiplier)
        targets = plan.set_index("building_objectid")["pv_installed_kwp"]
    elif sizing_method == "optimization":
        plan["pv_demand_target_kwp"] = np.nan
        plan["pv_installed_kwp"] = np.nan
        targets = plan.set_index("building_objectid")["pv_max_kwp"]
    else:
        raise ValueError(f"Unknown PV sizing method {sizing_method!r}.")
    selected = allocate_capacity_best_yield_first(roof_catalog, targets, profile_library)
    selected["pv_sizing_method"] = sizing_method
    plan["pv_sizing_method"] = sizing_method
    plan["pv_fallback_used"] = plan["building_objectid"].isin(
        set(roof_catalog.loc[roof_catalog["quality_flag"].eq("fallback_14_5_kw"), "building_objectid"].astype(str))
    )
    return plan, selected
