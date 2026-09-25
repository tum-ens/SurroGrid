"""Materialize building heat plans as compact urbs input tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..urbs_rows import process_row, storage_parameter_fields


@dataclass(frozen=True)
class HeatUrbsInputs:
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame


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
            "heat_pump_reference_kw_th",
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
            process_row(site, "heatpump_air", row["heat_pump_installed_kw_el"], row["heat_pump_capacity_upper_kw_el"], fixed=fixed, parameters=process_parameters["heatpump_air"]),
            process_row(site, "heatpump_booster", row["auxiliary_installed_kw_el"], row["auxiliary_capacity_upper_kw_el"], fixed=fixed, parameters=process_parameters["heatpump_booster"]),
            process_row(site, "Heat_dummy_space", row["heat_conversion_capacity_kw_th"], row["heat_conversion_capacity_kw_th"], fixed=True, parameters=process_parameters["heat_dummy"]),
            process_row(site, "Heat_dummy_water", row["heat_conversion_capacity_kw_th"], row["heat_conversion_capacity_kw_th"], fixed=True, parameters=process_parameters["heat_dummy"]),
        ])
        energy_upper = float(row["buffer_capacity_upper_kwh_th"])
        power_upper = float(row["buffer_power_upper_kw_th"])
        if energy_upper > 0.0 and power_upper > 0.0:
            storage_row = {
                "Site": site, "Storage": "heat_storage", "Commodity": "space_heat",
                "inst-cap-c": float(row["buffer_installed_kwh_th"]),
                "cap-up-c": energy_upper,
                "inst-cap-p": float(row["buffer_installed_power_kw_th"]),
                "cap-up-p": power_upper,
                **storage_parameter_fields(
                    storage_parameters, fixed=fixed, ep_ratio=energy_upper / power_upper
                ),
            }
            if not fixed:
                hp_upper_kw_el = float(row["heat_pump_capacity_upper_kw_el"])
                if hp_upper_kw_el <= 0.0:
                    raise ValueError(
                        f"Optimized heat storage at site {site} requires a positive HP bound."
                    )
                storage_row.update({
                    "linked-process": "heatpump_air",
                    "max-energy-per-process-capacity": energy_upper / hp_upper_kw_el,
                })
            storage_rows.append(storage_row)
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
