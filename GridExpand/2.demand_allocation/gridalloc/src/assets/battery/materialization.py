"""Materialize stationary-battery asset plans as urbs storage rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BatteryUrbsInputs:
    storage: pd.DataFrame
    audit: pd.DataFrame


def materialize_battery_urbs_inputs(
    asset_plan: pd.DataFrame,
    *,
    sizing_method: str,
    energy_to_power_hours: float,
    technical_parameters,
) -> BatteryUrbsInputs:
    """Create fixed heuristic or bounded endogenous stationary batteries."""
    if energy_to_power_hours <= 0.0:
        raise ValueError("Battery energy-to-power ratio must be positive.")
    active = asset_plan[asset_plan["battery_capacity_upper_kwh"].gt(0.0)].copy()
    fixed = sizing_method == "htw_2025_upper_bound"
    if not fixed and sizing_method != "optimization":
        raise ValueError(f"Unknown battery sizing method {sizing_method!r}.")
    audit_rows = []
    for row in active.itertuples(index=False):
        upper_energy = float(row.battery_capacity_upper_kwh)
        installed_energy = float(row.battery_installed_kwh) if fixed else 0.0
        audit_rows.append({
            "sector": "stationary_battery",
            "audit_record_type": "battery_materialization",
            "building_objectid": str(row.building_objectid),
            "allocation_bus": int(row.Site),
            "energy_capacity_kwh": installed_energy if fixed else None,
            "energy_capacity_upper_kwh": upper_energy,
            "power_capacity_kw": installed_energy / energy_to_power_hours if fixed else None,
            "power_capacity_upper_kw": upper_energy / energy_to_power_hours,
            "energy_to_power_hours": float(energy_to_power_hours),
            "capacity_source": sizing_method,
            "location_source": row.battery_location_source,
        })

    # urbs requires one (Site, Storage, Commodity) row. Buildings sharing an
    # electrical site are equivalent under one common E/P ratio, so sum them.
    by_site = active.groupby("Site", as_index=False).agg(
        battery_capacity_upper_kwh=("battery_capacity_upper_kwh", "sum"),
        battery_installed_kwh=("battery_installed_kwh", "sum"),
    )
    rows = []
    for row in by_site.itertuples(index=False):
        upper_energy = float(row.battery_capacity_upper_kwh)
        installed_energy = float(row.battery_installed_kwh) if fixed else 0.0
        rows.append({
            "Site": int(row.Site), "Storage": "battery_private",
            "Commodity": "electricity", "inst-cap-c": installed_energy,
            "cap-up-c": upper_energy,
            "inst-cap-p": installed_energy / energy_to_power_hours,
            "cap-up-p": upper_energy / energy_to_power_hours,
            "eff-in": technical_parameters.BS_EFF_IN,
            "eff-out": technical_parameters.BS_EFF_OUT,
            "discharge": technical_parameters.BS_DISCHARGE,
            "ep-ratio": float(energy_to_power_hours),
            "inv-cost-p": 0.0 if fixed else technical_parameters.BS_INV_COST_P,
            "inv-cost-c": 0.0 if fixed else technical_parameters.BS_INV_COST_C,
            "fix-cost-p": 0.0 if fixed else technical_parameters.BS_FIX_COST_P,
            "fix-cost-c": 0.0 if fixed else technical_parameters.BS_FIX_COST_C,
            "var-cost-p": technical_parameters.BS_VAR_COST_P,
            "wacc": technical_parameters.BS_WACC,
            "depreciation": technical_parameters.BS_DEPRECIATION,
        })
    return BatteryUrbsInputs(pd.DataFrame(rows), pd.DataFrame(audit_rows))
