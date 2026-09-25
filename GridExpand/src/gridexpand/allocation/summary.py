"""Step 2 asset-plan summary recorded in the demand-allocation run assumptions."""

from __future__ import annotations

import pandas as pd


def selected_buildings(assignment: pd.DataFrame, technology: str) -> set[str]:
    """Return the building ids selected for one technology."""
    if assignment.empty:
        raise ValueError("Electrification assignment has not been prepared.")
    return set(
        assignment.loc[
            assignment["technology"].eq(technology)
            & assignment["selected"].astype(bool),
            "building_objectid",
        ].astype(str)
    )


def _positive_plan_count(
    plan: pd.DataFrame,
    building_column: str,
    capacity_column: str,
) -> int:
    if plan.empty or building_column not in plan or capacity_column not in plan:
        return 0
    capacity = pd.to_numeric(plan[capacity_column], errors="coerce").fillna(0.0)
    return int(plan.loc[capacity.gt(0.0), building_column].astype(str).nunique())


def _plan_sum(plan: pd.DataFrame, column: str) -> float:
    if plan.empty or column not in plan:
        return 0.0
    return float(pd.to_numeric(plan[column], errors="coerce").fillna(0.0).sum())


def electrification_asset_plan_summary(
    assignment: pd.DataFrame,
    *,
    pv_plan: pd.DataFrame,
    battery_plan: pd.DataFrame,
    heat_plan: pd.DataFrame,
    pv_supply: pd.DataFrame,
    buildings: pd.DataFrame,
    mobility_demand: pd.DataFrame,
    battery_dict: dict,
    home_charger_kw: float,
) -> dict[str, dict[str, object]]:
    """Report selected cohorts separately from the positive materialized capacity.

    Written to the run assumptions as ``electrification_asset_plan_summary``.

    Args:
        assignment: The grid's electrification assignment.
        pv_plan: PV asset plan (``raw_data/asset_plan``).
        battery_plan: Battery asset plan.
        heat_plan: Heat asset plan.
        pv_supply: Per-unit PV supply profiles of the run horizon (``urbs_in/supim``).
        buildings: Physical buildings with ``annual_electricity_kwh`` and ``n_cars_tot``.
        mobility_demand: EV charging demand profiles.
        battery_dict: ``{(bus, vehicle_id): battery_kwh}`` of the EVs.
        home_charger_kw: Installed home-charger capacity per EV.
    """
    if assignment.empty:
        return {}

    selected_by_technology = {
        technology: selected_buildings(assignment, technology)
        for technology in ("heat", "mobility", "pv_battery")
    }
    summary: dict[str, dict[str, object]] = {
        technology: {
            "reporting_stage": "step2_urbs_input",
            "selected_candidate_building_count": int(
                len(selected_by_technology[technology])
            ),
        }
        for technology in selected_by_technology
    }

    pv_capacity = _plan_sum(pv_plan, "pv_max_kwp")
    pv_installed = _plan_sum(
        pv_plan, "pv_installed_kwp"
    )
    battery_capacity = _plan_sum(
        battery_plan, "battery_capacity_upper_kwh"
    )
    battery_installed = _plan_sum(
        battery_plan, "battery_installed_kwh"
    )
    summary["pv_battery"].update(
        {
            "step2_materialized_asset_count": _positive_plan_count(
                pv_plan,
                "building_objectid",
                "pv_max_kwp",
            ),
            "positive_pv_capacity_upper_bound_building_count": _positive_plan_count(
                pv_plan,
                "building_objectid",
                "pv_max_kwp",
            ),
            "positive_pv_input_installed_building_count": _positive_plan_count(
                pv_plan,
                "building_objectid",
                "pv_installed_kwp",
            ),
            "step2_input_capacity_kw": pv_capacity,
            "step2_input_installed_capacity_kw": pv_installed,
            "positive_battery_capacity_upper_bound_building_count": _positive_plan_count(
                battery_plan,
                "building_objectid",
                "battery_capacity_upper_kwh",
            ),
            "positive_battery_input_installed_building_count": _positive_plan_count(
                battery_plan,
                "building_objectid",
                "battery_installed_kwh",
            ),
            "step2_materialized_battery_candidate_count": _positive_plan_count(
                battery_plan,
                "building_objectid",
                "battery_capacity_upper_kwh",
            ),
            "battery_capacity_upper_bound_kwh": battery_capacity,
            "battery_input_installed_capacity_kwh": battery_installed,
            "pv_supply_profile_sum_hours": float(
                pv_supply.apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy()
                .sum()
            ),
            "pv_supply_profile_basis": "per_unit_available_pv_output",
        }
    )

    heat_capacity = _plan_sum(
        heat_plan, "heat_pump_capacity_upper_kw_el"
    )
    heat_installed = _plan_sum(
        heat_plan, "heat_pump_installed_kw_el"
    )
    summary["heat"].update(
        {
            "step2_materialized_asset_count": _positive_plan_count(
                heat_plan,
                "building_objectid",
                "heat_pump_capacity_upper_kw_el",
            ),
            "positive_heat_pump_capacity_upper_bound_building_count": _positive_plan_count(
                heat_plan,
                "building_objectid",
                "heat_pump_capacity_upper_kw_el",
            ),
            "positive_heat_pump_input_installed_building_count": _positive_plan_count(
                heat_plan,
                "building_objectid",
                "heat_pump_installed_kw_el",
            ),
            "step2_capacity_upper_kw_el": heat_capacity,
            "step2_input_installed_capacity_kw_el": heat_installed,
            "selected_building_base_electricity_kwh": float(
                buildings.loc[
                    buildings["objectid"].astype(str).isin(
                        selected_by_technology["heat"]
                    ),
                    "annual_electricity_kwh",
                ]
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .sum()
            )
            if "annual_electricity_kwh" in buildings
            else 0.0,
            "heat_electricity_outcome_basis": "solver_output_not_step2_input",
            "annual_heat_demand_kwh_th": _plan_sum(
                heat_plan, "annual_space_heat_kwh"
            )
            + _plan_sum(
                heat_plan, "annual_water_heat_kwh"
            ),
        }
    )

    if not mobility_demand.empty and "n_cars_tot" in buildings:
        selected_mobility = buildings["objectid"].astype(str).isin(
            selected_by_technology["mobility"]
        )
        cars = pd.to_numeric(
            buildings["n_cars_tot"], errors="coerce"
        ).fillna(0.0).where(selected_mobility, 0.0)
        ev_count = int(cars.sum())
        summary["mobility"].update(
            {
                "step2_materialized_asset_count": ev_count,
                "positive_ev_building_count": int(cars.gt(0.0).sum()),
                "positive_ev_vehicle_count": ev_count,
                "ev_profile_count": int(len(battery_dict)),
                "step2_input_capacity_kw": float(home_charger_kw * ev_count),
                "annual_ev_charging_demand_kwh": float(
                    mobility_demand.apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy()
                    .sum()
                ),
                "ev_energy_basis": "charging_demand_input",
            }
        )
    else:
        summary["mobility"].update(
            {
                "step2_materialized_asset_count": 0,
                "positive_ev_building_count": 0,
                "positive_ev_vehicle_count": 0,
                "ev_profile_count": 0,
                "step2_input_capacity_kw": 0.0,
                "annual_ev_charging_demand_kwh": 0.0,
                "ev_energy_basis": "charging_demand_input",
            }
        )
    return summary
