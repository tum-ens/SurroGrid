"""Building-level stationary-battery sizing rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

HTW_2025_MAX_USABLE_KWH_PER_PV_KWP = 1.5
HTW_2025_MAX_USABLE_KWH_PER_ANNUAL_MWH = 1.5


def htw_usable_capacity_kwh(
    annual_electricity_kwh,
    pv_capacity_kwp,
    *,
    minimum_pv_kwp_per_annual_mwh: float = 0.5,
    usable_kwh_per_pv_kwp: float = 1.0,
    usable_kwh_per_annual_mwh: float = 1.0,
):
    """Return usable energy from coefficients within the HTW 2025 envelope."""
    annual_kwh = np.maximum(
        np.nan_to_num(np.asarray(annual_electricity_kwh, dtype=float), nan=0.0), 0.0
    )
    pv_kwp = np.maximum(
        np.nan_to_num(np.asarray(pv_capacity_kwp, dtype=float), nan=0.0), 0.0
    )
    annual_mwh = annual_kwh / 1000.0
    sufficient_surplus = pv_kwp > minimum_pv_kwp_per_annual_mwh * annual_mwh
    capacity = np.where(
        sufficient_surplus & (pv_kwp > 0.0) & (annual_mwh > 0.0),
        np.minimum(
            usable_kwh_per_pv_kwp * pv_kwp,
            usable_kwh_per_annual_mwh * annual_mwh,
        ),
        0.0,
    )
    return float(capacity) if capacity.ndim == 0 else capacity


def build_battery_asset_plan(
    pv_asset_plan: pd.DataFrame,
    *,
    sizing_method: str,
    minimum_pv_kwp_per_annual_mwh: float,
    usable_kwh_per_pv_kwp: float,
    usable_kwh_per_annual_mwh: float,
    eligible_buildings: set[str] | None = None,
    site_by_building: dict[str, int] | None = None,
    location_source: str = "all_pv_buildings",
) -> pd.DataFrame:
    """Compile one stationary-battery decision row per physical building."""
    required = {
        "building_objectid", "Site", "annual_electricity_kwh",
        "pv_installed_kwp", "pv_max_kwp",
    }
    missing = required.difference(pv_asset_plan.columns)
    if missing:
        raise ValueError(f"Battery sizing requires PV asset-plan columns: {sorted(missing)}")
    plan = pv_asset_plan.copy()
    plan["building_objectid"] = plan["building_objectid"].astype(str)
    if plan["building_objectid"].duplicated().any():
        raise ValueError("Battery sizing requires one PV asset row per physical building.")
    if sizing_method == "htw_2025_scaled_rule":
        plan["battery_reference_pv_kwp"] = pd.to_numeric(
            plan["pv_installed_kwp"], errors="coerce"
        ).fillna(0.0)
    elif sizing_method == "optimization":
        plan["battery_reference_pv_kwp"] = pd.to_numeric(
            plan["pv_max_kwp"], errors="coerce"
        ).fillna(0.0)
    else:
        raise ValueError(f"Unknown battery sizing method {sizing_method!r}.")
    plan["annual_electricity_mwh"] = pd.to_numeric(
        plan["annual_electricity_kwh"], errors="coerce"
    ).fillna(0.0) / 1000.0
    plan["battery_pv_threshold_kwp"] = (
        minimum_pv_kwp_per_annual_mwh * plan["annual_electricity_mwh"]
    )
    plan["battery_pv_bound_kwh"] = (
        usable_kwh_per_pv_kwp * plan["battery_reference_pv_kwp"]
    )
    plan["battery_demand_bound_kwh"] = (
        usable_kwh_per_annual_mwh * plan["annual_electricity_mwh"]
    )
    plan["battery_capacity_upper_kwh"] = htw_usable_capacity_kwh(
        plan["annual_electricity_kwh"],
        plan["battery_reference_pv_kwp"],
        minimum_pv_kwp_per_annual_mwh=minimum_pv_kwp_per_annual_mwh,
        usable_kwh_per_pv_kwp=usable_kwh_per_pv_kwp,
        usable_kwh_per_annual_mwh=usable_kwh_per_annual_mwh,
    )
    if eligible_buildings is not None:
        plan["battery_location_eligible"] = plan["building_objectid"].isin(
            {str(value) for value in eligible_buildings}
        )
        plan.loc[~plan["battery_location_eligible"], "battery_capacity_upper_kwh"] = 0.0
    else:
        plan["battery_location_eligible"] = True
    if site_by_building:
        mapped = plan["building_objectid"].map(
            {str(key): int(value) for key, value in site_by_building.items()}
        )
        plan.loc[mapped.notna(), "Site"] = mapped[mapped.notna()].astype(int)
    plan["battery_installed_kwh"] = (
        plan["battery_capacity_upper_kwh"]
        if sizing_method == "htw_2025_scaled_rule" else 0.0
    )
    plan["battery_sizing_method"] = sizing_method
    plan["battery_pv_coefficient_kwh_per_kwp"] = float(usable_kwh_per_pv_kwp)
    plan["battery_demand_coefficient_kwh_per_mwh"] = float(usable_kwh_per_annual_mwh)
    plan["battery_htw_upper_kwh"] = htw_usable_capacity_kwh(
        plan["annual_electricity_kwh"],
        plan["battery_reference_pv_kwp"],
        minimum_pv_kwp_per_annual_mwh=minimum_pv_kwp_per_annual_mwh,
        usable_kwh_per_pv_kwp=HTW_2025_MAX_USABLE_KWH_PER_PV_KWP,
        usable_kwh_per_annual_mwh=HTW_2025_MAX_USABLE_KWH_PER_ANNUAL_MWH,
    )
    plan["battery_within_htw_upper_bound"] = (
        plan["battery_capacity_upper_kwh"] <= plan["battery_htw_upper_kwh"] + 1e-9
    )
    plan["battery_capacity_kwh_per_reference_pv_kwp"] = np.divide(
        plan["battery_capacity_upper_kwh"],
        plan["battery_reference_pv_kwp"],
        out=np.zeros(len(plan), dtype=float),
        where=plan["battery_reference_pv_kwp"].to_numpy(dtype=float) > 0.0,
    )
    plan["battery_capacity_kwh_per_annual_mwh"] = np.divide(
        plan["battery_capacity_upper_kwh"],
        plan["annual_electricity_mwh"],
        out=np.zeros(len(plan), dtype=float),
        where=plan["annual_electricity_mwh"].to_numpy(dtype=float) > 0.0,
    )
    if "number_of_households" in plan:
        households = pd.to_numeric(plan["number_of_households"], errors="coerce")
        plan["battery_capacity_kwh_per_household"] = (
            plan["battery_capacity_upper_kwh"] / households.where(households > 0.0)
        )
    plan["battery_location_source"] = location_source
    plan["battery_limiting_bound"] = np.where(
        plan["battery_pv_bound_kwh"] <= plan["battery_demand_bound_kwh"],
        "pv_capacity", "annual_electricity",
    )
    plan["battery_exclusion_reason"] = np.select(
        [
            ~plan["battery_location_eligible"],
            plan["battery_reference_pv_kwp"].le(0.0),
            plan["annual_electricity_mwh"].le(0.0),
            plan["battery_reference_pv_kwp"].le(plan["battery_pv_threshold_kwp"]),
        ],
        ["not_predefined", "no_pv", "no_base_electricity", "insufficient_pv_surplus"],
        default="selected",
    )
    return plan
