"""Asset-plan summary of the run assumptions (moved out of Grid)."""

from __future__ import annotations

import pandas as pd

from gridexpand.allocation.summary import electrification_asset_plan_summary, selected_buildings


def _assignment():
    rows = []
    for building, selected in (("B1", True), ("B2", True), ("B3", False)):
        for technology in ("heat", "mobility", "pv_battery"):
            rows.append({"building_objectid": building, "technology": technology, "selected": selected})
    return pd.DataFrame(rows)


def test_summary_counts_and_sums():
    buildings = pd.DataFrame(
        {"objectid": ["B1", "B2", "B3"], "annual_electricity_kwh": [1000.0, 2000.0, 500.0], "n_cars_tot": [1, 2, 3]}
    )
    summary = electrification_asset_plan_summary(
        _assignment(),
        pv_plan=pd.DataFrame({"building_objectid": ["B1", "B2"], "pv_max_kwp": [5.0, 0.0], "pv_installed_kwp": [2.0, 0.0]}),
        battery_plan=pd.DataFrame({"building_objectid": ["B1"], "battery_capacity_upper_kwh": [4.0], "battery_installed_kwh": [0.0]}),
        heat_plan=pd.DataFrame({
            "building_objectid": ["B1", "B2"], "heat_pump_capacity_upper_kw_el": [3.0, 2.0],
            "heat_pump_installed_kw_el": [0.0, 0.0], "annual_space_heat_kwh": [100.0, 50.0],
            "annual_water_heat_kwh": [10.0, 5.0],
        }),
        pv_supply=pd.DataFrame({"a": [0.5, 0.25]}),
        buildings=buildings,
        mobility_demand=pd.DataFrame({"m": [1.0, 2.0]}),
        battery_dict={(1, 0): 50.0, (2, 1): 60.0, (2, 2): 40.0},
        home_charger_kw=11.0,
    )
    pv = summary["pv_battery"]
    assert pv["selected_candidate_building_count"] == 2
    assert pv["step2_materialized_asset_count"] == pv["positive_pv_capacity_upper_bound_building_count"] == 1
    assert pv["step2_input_capacity_kw"] == 5.0 and pv["pv_supply_profile_sum_hours"] == 0.75
    assert summary["heat"]["annual_heat_demand_kwh_th"] == 165.0
    assert summary["heat"]["selected_building_base_electricity_kwh"] == 3000.0
    assert summary["mobility"]["positive_ev_vehicle_count"] == 3
    assert summary["mobility"]["step2_input_capacity_kw"] == 33.0
    assert summary["mobility"]["annual_ev_charging_demand_kwh"] == 3.0
    assert selected_buildings(_assignment(), "heat") == {"B1", "B2"}
