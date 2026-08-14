"""Materialize building heat plans as compact urbs input tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeatUrbsInputs:
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame


def _process_row(site, name, installed, upper, *, fixed, parameters):
    return {
        "Site": int(site), "Process": name,
        "inst-cap": float(installed), "cap-up": float(upper),
        "inv-cost-fix": 0.0 if fixed else parameters["fixed_investment_cost_eur"],
        "inv-cost": 0.0 if fixed else parameters["investment_cost_eur_per_kw"],
        "fix-cost": parameters["fixed_cost_eur_per_hour"],
        "var-cost": parameters["variable_cost_eur_per_kwh"],
        "wacc": parameters["wacc"], "depreciation": parameters["depreciation_years"],
        "pf-min": parameters["minimum_power_factor"],
    }


def materialize_heat_urbs_inputs(asset_plan, *, sizing_method, process_parameters, storage_parameters):
    fixed = sizing_method == "full_load_hours_rule"
    if not fixed and sizing_method != "optimization":
        raise ValueError(f"Unknown heat sizing method {sizing_method!r}.")
    if asset_plan.empty:
        empty = pd.DataFrame()
        return HeatUrbsInputs(empty, empty, empty, empty, empty)
    sums = {
        name: (name, "sum") for name in (
            "heat_pump_installed_kw_el", "heat_pump_capacity_upper_kw_el",
            "auxiliary_installed_kw_el", "auxiliary_capacity_upper_kw_el",
            "buffer_installed_kwh_th", "buffer_capacity_upper_kwh_th",
            "buffer_installed_power_kw_th", "buffer_power_upper_kw_th",
            "heat_conversion_capacity_kw_th",
        )
    }
    by_site = asset_plan.groupby("Site", as_index=False).agg(**sums)
    process_rows = []
    storage_rows = []
    for row in by_site.to_dict("records"):
        site = int(row["Site"])
        process_rows.extend([
            _process_row(site, "heatpump_air", row["heat_pump_installed_kw_el"], row["heat_pump_capacity_upper_kw_el"], fixed=fixed, parameters=process_parameters["heatpump_air"]),
            _process_row(site, "heatpump_booster", row["auxiliary_installed_kw_el"], row["auxiliary_capacity_upper_kw_el"], fixed=fixed, parameters=process_parameters["heatpump_booster"]),
            _process_row(site, "Heat_dummy_space", row["heat_conversion_capacity_kw_th"], row["heat_conversion_capacity_kw_th"], fixed=True, parameters=process_parameters["heat_dummy"]),
            _process_row(site, "Heat_dummy_water", row["heat_conversion_capacity_kw_th"], row["heat_conversion_capacity_kw_th"], fixed=True, parameters=process_parameters["heat_dummy"]),
        ])
        energy_upper = float(row["buffer_capacity_upper_kwh_th"])
        power_upper = float(row["buffer_power_upper_kw_th"])
        if energy_upper > 0.0 and power_upper > 0.0:
            storage_rows.append({
                "Site": site, "Storage": "heat_storage", "Commodity": "space_heat",
                "inst-cap-c": float(row["buffer_installed_kwh_th"]),
                "cap-up-c": energy_upper,
                "inst-cap-p": float(row["buffer_installed_power_kw_th"]),
                "cap-up-p": power_upper,
                "eff-in": storage_parameters["charge_efficiency"],
                "eff-out": storage_parameters["discharge_efficiency"],
                "discharge": storage_parameters["self_discharge_per_timestep"],
                "ep-ratio": energy_upper / power_upper,
                "inv-cost-p": 0.0 if fixed else storage_parameters["investment_cost_eur_per_kw"],
                "inv-cost-c": 0.0 if fixed else storage_parameters["investment_cost_eur_per_kwh"],
                "fix-cost-p": 0.0 if fixed else storage_parameters["fixed_investment_cost_power_eur"],
                "fix-cost-c": 0.0 if fixed else storage_parameters["fixed_investment_cost_energy_eur"],
                "var-cost-p": storage_parameters["variable_cost_eur_per_kwh"],
                "wacc": storage_parameters["wacc"],
                "depreciation": storage_parameters["depreciation_years"],
            })
    sites = sorted(int(value) for value in by_site["Site"].unique())
    commodity = pd.DataFrame([
        {"Site": site, "Commodity": commodity, "Type": kind, "price": np.nan}
        for site in sites
        for commodity, kind in (("common_heat", "Stock"), ("space_heat", "Demand"), ("water_heat", "Demand"))
    ])
    process_commodity = pd.DataFrame({
        "Process": ["Heat_dummy_space", "Heat_dummy_space", "Heat_dummy_water", "Heat_dummy_water", "heatpump_air", "heatpump_air", "heatpump_booster", "heatpump_booster"],
        "Commodity": ["common_heat", "space_heat", "common_heat", "water_heat", "electricity", "common_heat", "electricity", "common_heat"],
        "Direction": ["In", "Out", "In", "Out", "In", "Out", "In", "Out"],
        "ratio": [1] * 8,
    })
    audit = asset_plan.copy()
    audit["sector"] = "heat"
    audit["audit_record_type"] = "heat_asset_plan"
    return HeatUrbsInputs(pd.DataFrame(process_rows), commodity, process_commodity, pd.DataFrame(storage_rows), audit)
