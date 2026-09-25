"""Materialize compiled PV asset plans as urbs input tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PvUrbsInputs:
    supim: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    audit: pd.DataFrame


def _process_row(site, process, installed, upper, *, fixed, parameters):
    return {
        "Site": int(site),
        "Process": process,
        "inst-cap": float(installed),
        "cap-up": float(upper),
        "inv-cost-fix": 0.0 if fixed else parameters["fixed_investment_cost_eur"],
        "inv-cost": 0.0 if fixed else parameters["investment_cost_eur_per_kw"],
        "fix-cost": parameters["fixed_cost_eur_per_hour"],
        "var-cost": parameters["variable_cost_eur_per_kwh"],
        "wacc": parameters["wacc"],
        "depreciation": parameters["depreciation_years"],
        "pf-min": parameters["minimum_power_factor"],
    }


def _static_tables(records):
    process = pd.DataFrame([record["process"] for record in records])
    commodity = pd.DataFrame([
        {"Site": record["site"], "Commodity": record["commodity"], "Type": "SupIm", "price": np.nan}
        for record in records
    ])
    process_commodity = pd.DataFrame([
        row
        for record in records
        for row in (
            {"Process": record["process_name"], "Commodity": record["commodity"], "Direction": "In", "ratio": 1},
            {"Process": record["process_name"], "Commodity": "electricity", "Direction": "Out", "ratio": 1},
        )
    ]).drop_duplicates().reset_index(drop=True)
    return process, commodity, process_commodity


def materialize_pv_urbs_inputs(
    asset_plan: pd.DataFrame,
    selected_sections: pd.DataFrame,
    profile_library: pd.DataFrame,
    *,
    sizing_method: str,
    technical_parameters: dict,
) -> PvUrbsInputs:
    """Create one capacity-weighted LoD2 PV process per physical building."""
    active_plan = asset_plan[asset_plan["pv_max_kwp"].gt(0.0)].copy()
    if active_plan.empty:
        empty = pd.DataFrame()
        return PvUrbsInputs(empty, empty, empty, empty, empty)
    site_by_building = active_plan.drop_duplicates("building_objectid").set_index("building_objectid")["Site"]
    sections = selected_sections[selected_sections["selected_pv_kw"].gt(0.0)].copy()
    sections["building_objectid"] = sections["building_objectid"].astype(str)
    sections["Site"] = sections["building_objectid"].map(site_by_building).astype(int)
    records = []
    columns = []

    if sizing_method in {"optimization", "annual_electricity_rule"}:
        for building_id, group in sections.groupby("building_objectid", sort=True):
            capacity = float(group["selected_pv_kw"].sum())
            if capacity <= 0:
                continue
            site = int(group["Site"].iloc[0])
            safe_id = "".join(char if char.isalnum() else "_" for char in str(building_id))
            mode = "optimized" if sizing_method == "optimization" else "heuristic"
            commodity = f"solar_{mode}_{safe_id}"
            process_name = f"Rooftop PV_{mode}_{safe_id}"
            weighted = sum(
                profile_library[label] * float(section_capacity)
                for label, section_capacity in group[["profile_label", "selected_pv_kw"]].itertuples(index=False, name=None)
            ) / capacity
            columns.append(weighted.rename((site, commodity)))
            fixed = sizing_method == "annual_electricity_rule"
            records.append({
                "site": site, "commodity": commodity, "process_name": process_name,
                "process": _process_row(
                    site,
                    process_name,
                    capacity if fixed else 0.0,
                    capacity,
                    fixed=fixed,
                    parameters=technical_parameters,
                ),
                "capacity_kw": capacity,
                "fallback_used": bool(group["quality_flag"].eq("fallback_14_5_kw").any()),
                "building_objectid": str(building_id),
                "roof_profile_count": int(group["profile_label"].nunique()),
            })
    else:
        raise ValueError(f"Unknown PV sizing method {sizing_method!r}.")

    supim = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=profile_library.index)
    if columns:
        supim.columns = pd.MultiIndex.from_tuples(supim.columns, names=["Site", "Commodity"])
    supim.index.name = "t"
    process, commodity, process_commodity = _static_tables(records)
    audit = pd.DataFrame([
        {key: value for key, value in record.items() if key not in {"process"}}
        | {"pv_sizing_method": sizing_method}
        for record in records
    ])
    return PvUrbsInputs(supim, process, commodity, process_commodity, audit)
