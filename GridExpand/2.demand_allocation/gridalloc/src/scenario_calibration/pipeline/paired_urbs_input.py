"""Materialize the same physical-building scenario on either target network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..paths import GRIDALLOC_DIR, GRIDEXPAND_DIR

DEFAULT_PAIRED_DIR = (
    GRIDALLOC_DIR
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_paired_v5_91301_station_hybrid_v2"
)
DEFAULT_OUTPUT_DIR = GRIDEXPAND_DIR / "3.urbs" / "Input"
DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "config" / "scenarios"
    / "forchheim_2045.yaml"
)

for path in (GRIDEXPAND_DIR, GRIDALLOC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import config  # noqa: E402
from common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    validate_electrification_assignment,
    validate_electrification_assignment_config,
)  # noqa: E402
from common.timeframe import build_full_year_metadata, write_hdf_metadata  # noqa: E402
from scenario_pipeline.config_loader import (  # noqa: E402
    load_scenario_config,
    scenario_identity_key,
)
from scenario_pipeline.model_cases import POST_MODEL_CASES  # noqa: E402
from src.assets.pv.roof_catalog import assert_fallback_share  # noqa: E402
from src.scenario_calibration.profiles.paired_profiles import (  # noqa: E402
    build_paired_base_electric_demand,
    build_paired_sector_urbs_inputs,
    load_electricity_module,
    source_match_buildings,
)
from src.scenario_calibration.profiles.physical_heat_profile_library import (  # noqa: E402
    PhysicalHeatProfileLibrary,
)
from src.scenario_calibration.profiles.paired_profile_readiness import (  # noqa: E402
    PUBLICATION_READY_HEAT_METHODS,
)
from src.scenario_calibration.pipeline.urbs_input_tables import (  # noqa: E402
    buy_sell_price,
    read_or_create_weather,
    urbs_static_tables,
)
from src.scenario_calibration.profiles.profile_contract import (  # noqa: E402
    assert_paired_component_plan_equivalence,
)


def _plan_path(paired_dir: Path, target_network: str) -> Path:
    filename = (
        "paired_real_bus_allocation_plan.csv"
        if target_network == "real_swf"
        else "paired_synthetic_bus_allocation_plan.csv"
    )
    return paired_dir / filename



def _target_component_plan(
    paired_dir: Path,
    target_network: str,
    target_grid_id: int,
    allocation: pd.DataFrame,
    *,
    profile_seed: int,
) -> pd.DataFrame:
    """Select and validate the shared component manifest for one target."""
    path = paired_dir / "paired_component_scenario_plan.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing paired component scenario plan; regenerate allocation: {path}"
        )
    combined = pd.read_csv(path)
    assert_paired_component_plan_equivalence(combined)
    if target_network == "real_swf":
        mask = pd.to_numeric(combined["real_target_grid_id"], errors="coerce").eq(int(target_grid_id))
        combined["target_bus"] = pd.to_numeric(combined["real_target_bus"], errors="raise").astype(int)
    else:
        mask = pd.to_numeric(combined["synthetic_target_grid_case_id"], errors="coerce").eq(int(target_grid_id))
        combined["target_bus"] = pd.to_numeric(combined["synthetic_target_bus"], errors="raise").astype(int)
    target = combined.loc[mask].copy()
    if target.empty:
        raise ValueError(f"No paired component rows for {target_network} target grid {target_grid_id}.")
    seeds = pd.to_numeric(target["profile_seed"], errors="coerce").dropna().astype(int).unique()
    if len(seeds) != 1 or int(seeds[0]) != int(profile_seed):
        raise ValueError(
            "Paired component profile seed differs from allocation seed; regenerate the paired allocation."
        )
    allocation_units = set(pd.to_numeric(allocation["scenario_unit_id"], errors="raise").astype(int))
    component_units = set(pd.to_numeric(target["scenario_unit_id"], errors="raise").astype(int))
    if allocation_units != component_units:
        raise ValueError("Paired component and physical allocation plans contain different scenario units.")
    return target.sort_values(["scenario_unit_id", "building_objectid", "component_id"]).reset_index(drop=True)

def _target_plan(
    paired_dir: Path,
    target_network: str,
    target_grid_id: int,
) -> pd.DataFrame:
    plan = pd.read_csv(_plan_path(paired_dir, target_network))
    target = plan[
        pd.to_numeric(plan["target_grid_id"], errors="coerce")
        .astype("Int64")
        .eq(int(target_grid_id))
    ].copy()
    if target.empty:
        raise ValueError(
            f"No {target_network} allocation rows for target grid {target_grid_id}."
        )
    return target


def _output_name(
    allocation: pd.DataFrame,
    target_network: str,
    target_grid_id: int,
    scenario_label: str,
) -> str:
    if target_network == "real_swf":
        prefix = f"paired_real_swf_LV_{int(target_grid_id):03d}"
    else:
        bridge_names = (
            allocation["synthetic_bridge_filename"].dropna().astype(str).unique()
        )
        if len(bridge_names) != 1:
            raise ValueError("A synthetic target grid must have one bridge filename.")
        prefix = f"paired_synthetic_{Path(bridge_names[0]).stem}"
    return f"{prefix}_{scenario_label}.h5"


def _combine_static_tables(
    active_buses: list[int],
    sector_inputs,
    electricity_module,
) -> dict[str, pd.DataFrame]:
    tables = urbs_static_tables(
        active_buses,
        electricity_module,
        include_generic_battery=False,
    )
    for key in ("process", "commodity", "process_commodity", "storage"):
        sector_table = getattr(sector_inputs, key)
        if not sector_table.empty:
            tables[key] = pd.concat(
                [tables[key], sector_table],
                ignore_index=True,
                sort=False,
            )
    return tables


def _paired_electrification_asset_plan_summary(
    assignment: pd.DataFrame,
    audit: pd.DataFrame,
    component_plan: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    """Separate assigned cohorts from positive capacities in paired outputs."""
    result: dict[str, dict[str, object]] = {}
    for technology in ("heat", "mobility", "pv_battery"):
        result[technology] = {
            "reporting_stage": "step2_urbs_input",
            "selected_candidate_building_count": int(
                assignment.loc[
                    assignment["technology"].eq(technology)
                    & assignment["selected"].astype(bool),
                    "building_objectid",
                ].astype(str).nunique()
            )
        }

    def subset(sector: str, record_type: str | None = None) -> pd.DataFrame:
        if audit.empty:
            return audit.iloc[0:0]
        mask = audit.get(
            "sector", pd.Series(index=audit.index, dtype=object)
        ).astype(str).eq(sector)
        if record_type is not None:
            mask &= audit.get(
                "audit_record_type",
                pd.Series(index=audit.index, dtype=object),
            ).astype(str).eq(record_type)
        return audit.loc[mask]

    def positive(rows: pd.DataFrame, column: str) -> int:
        if rows.empty or column not in rows or "building_objectid" not in rows:
            return 0
        values = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
        return int(
            rows.loc[values.gt(0.0), "building_objectid"]
            .dropna()
            .astype(str)
            .nunique()
        )

    def total(rows: pd.DataFrame, column: str) -> float:
        if rows.empty or column not in rows:
            return 0.0
        return float(pd.to_numeric(rows[column], errors="coerce").fillna(0.0).sum())

    def component_energy(selected: set[str]) -> float:
        if component_plan is None or component_plan.empty:
            return 0.0
        rows = component_plan.loc[
            component_plan["building_objectid"].astype(str).isin(selected)
            & component_plan["included_in_lv"].astype(bool)
        ]
        return total(rows, "annual_energy_kwh")

    pv_plan = subset("pv", "asset_plan")
    pv_materialized = subset("pv", "pv_materialization")
    battery_plan = subset("stationary_battery", "battery_asset_plan")
    battery_materialized = subset("stationary_battery", "battery_materialization")
    heat_plan = subset("heat", "heat_asset_plan")
    mobility_rows = subset("mobility")

    result["pv_battery"].update(
        {
            "step2_materialized_asset_count": positive(pv_materialized, "capacity_kw"),
            "positive_pv_capacity_upper_bound_building_count": positive(
                pv_plan, "pv_max_kwp"
            ),
            "positive_pv_input_building_count": positive(
                pv_materialized, "capacity_kw"
            ),
            "step2_input_capacity_kw": total(pv_materialized, "capacity_kw"),
            "pv_capacity_upper_bound_kw": total(pv_plan, "pv_max_kwp"),
            "positive_battery_capacity_upper_bound_building_count": positive(
                battery_plan, "battery_capacity_upper_kwh"
            ),
            "positive_battery_input_building_count": positive(
                battery_materialized, "energy_capacity_upper_kwh"
            ),
            "step2_materialized_battery_candidate_count": positive(
                battery_materialized, "energy_capacity_upper_kwh"
            ),
            "battery_input_capacity_upper_kwh": total(
                battery_materialized, "energy_capacity_upper_kwh"
            ),
            "battery_capacity_upper_bound_kwh": total(
                battery_plan, "battery_capacity_upper_kwh"
            ),
            "selected_building_base_electricity_kwh": total(
                pv_plan, "annual_electricity_kwh"
            ),
            "electricity_quantity_basis": "base_electricity_sizing_input",
        }
    )
    heat_selected = set(
        assignment.loc[
            assignment["technology"].eq("heat")
            & assignment["selected"].astype(bool),
            "building_objectid",
        ].astype(str)
    )
    result["heat"].update(
        {
            "step2_materialized_asset_count": positive(
                heat_plan, "heat_pump_capacity_upper_kw_el"
            ),
            "positive_heat_pump_capacity_upper_bound_building_count": positive(
                heat_plan, "heat_pump_capacity_upper_kw_el"
            ),
            "positive_heat_pump_input_installed_building_count": positive(
                heat_plan, "heat_pump_installed_kw_el"
            ),
            "step2_capacity_upper_kw_el": total(
                heat_plan, "heat_pump_capacity_upper_kw_el"
            ),
            "step2_input_installed_capacity_kw_el": total(
                heat_plan, "heat_pump_installed_kw_el"
            ),
            "selected_building_base_electricity_kwh": component_energy(heat_selected),
            "heat_electricity_outcome_basis": "solver_output_not_step2_input",
        }
    )
    if not mobility_rows.empty and "building_objectid" in mobility_rows:
        ev_capacity = pd.to_numeric(
            mobility_rows.get(
                "capacity_kw",
                pd.Series(0.0, index=mobility_rows.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        positive_ev_buildings = int(
            mobility_rows.loc[ev_capacity.gt(0.0), "building_objectid"]
            .dropna()
            .astype(str)
            .nunique()
        )
        ev_count = int(len(mobility_rows))
    else:
        positive_ev_buildings = 0
        ev_count = 0
    result["mobility"].update(
        {
            "step2_materialized_asset_count": ev_count,
            "positive_ev_building_count": positive_ev_buildings,
            "positive_ev_vehicle_count": ev_count,
            "ev_profile_count": ev_count,
            "step2_input_capacity_kw": total(mobility_rows, "capacity_kw"),
            "annual_ev_charging_demand_kwh": total(
                mobility_rows, "demand_sum_kwh"
            ),
            "electricity_quantity_basis": "charging_demand_input",
        }
    )
    return result


def materialize_paired_urbs_input(
    *,
    paired_dir: Path,
    output_dir: Path,
    target_network: str,
    target_grid_id: int,
    scenario_label: str,
    seed: int,
    weather_source_hdf: Path,
    heat_profile_library: Path | None,
    allow_diagnostic_heat_fallback: bool,
    model_case: str = "post-hems-heuristic",
    scenario_config_path: Path = DEFAULT_SCENARIO_CONFIG,
) -> Path:
    """Write one paired full-year Step-3 input HDF."""
    scenario, scenario_hash = load_scenario_config(scenario_config_path)
    config.apply_scenario(scenario)
    if model_case not in POST_MODEL_CASES:
        raise ValueError(
            f"Paired model case must be one of {POST_MODEL_CASES}; got {model_case!r}."
        )
    pv_sizing_method = scenario.pv_sizing_method(model_case)
    battery_sizing_method = scenario.battery_sizing_method(model_case)
    battery_pv_coefficient, battery_demand_coefficient = (
        scenario.battery_capacity_coefficients(model_case)
    )
    heat_sizing_method = scenario.heat_sizing_method(model_case)
    if pv_sizing_method == "none":
        raise ValueError("The pre case is materialized by the electricity-only pipeline.")
    paired_dir = paired_dir.resolve()
    allocation = _target_plan(
        paired_dir,
        target_network,
        target_grid_id,
    )
    assignment_path = paired_dir / "paired_electrification_assignment.csv"
    if not assignment_path.exists():
        raise FileNotFoundError(
            "Run paired_allocation with the selected scenario before materialization: "
            f"{assignment_path}"
        )
    electrification_assignment = pd.read_csv(assignment_path)
    validate_electrification_assignment_config(
        electrification_assignment,
        scenario.electrification,
        profile_seed=seed,
    )
    target_building_ids = set(allocation["building_objectid"].astype(str))
    assignment_building_ids = set(
        electrification_assignment["building_objectid"].astype(str)
    )
    missing_assignment_ids = sorted(target_building_ids - assignment_building_ids)
    if missing_assignment_ids:
        raise ValueError(
            "Paired electrification assignment is missing target buildings: "
            f"{missing_assignment_ids[:10]}"
        )
    electrification_assignment_hash = assignment_manifest_hash(
        electrification_assignment
    )
    component_plan = _target_component_plan(
        paired_dir,
        target_network,
        target_grid_id,
        allocation,
        profile_seed=seed,
    )
    if "scenario_unit_id" not in allocation:
        raise ValueError(
            "Paired allocation is missing scenario_unit_id. Regenerate it with "
            "paired_allocation before materializing URBS inputs."
        )
    allocation["_profile_site_id"] = allocation["scenario_unit_id"].astype(int)
    catalog_path = paired_dir / "paired_heat_profile_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Run paired_profile_readiness first: {catalog_path}")
    heat_catalog = pd.read_csv(catalog_path)
    roof_catalog_path = paired_dir / "paired_roof_sections.csv"
    if not roof_catalog_path.exists():
        raise FileNotFoundError(
            f"Regenerate the paired allocation with LoD2 PV roofs: {roof_catalog_path}"
        )
    roof_catalog = pd.read_csv(roof_catalog_path)
    pv_profile_library = paired_dir / "paired_pv_profile_library.h5"
    pv_battery_assignment = electrification_assignment.loc[
        electrification_assignment["technology"].eq("pv_battery")
    ]
    if str(pv_battery_assignment["adoption_mode"].iloc[0]) == "source_inventory":
        selected_pv_buildings = source_match_buildings(
            electrification_assignment, "pv_match_count"
        )
        selected_battery_buildings = source_match_buildings(
            electrification_assignment, "battery_match_count"
        )
        planning_buildings = selected_pv_buildings | selected_battery_buildings
    else:
        planning_buildings = set(
            pv_battery_assignment.loc[
                pv_battery_assignment["selected"].astype(bool),
                "building_objectid",
            ].astype(str)
        )
    if planning_buildings and not pv_profile_library.exists():
        raise FileNotFoundError(
            "Build the shared PV profile library before materialization: "
            f"{pv_profile_library}"
        )
    assert_fallback_share(
        roof_catalog,
        planning_buildings,
        scenario.pv.maximum_fallback_share,
    )
    library_sources = (
        heat_catalog.get("profile_source_kind", pd.Series(dtype=str))
        .astype(str)
        .eq("physical_heat_library")
    )
    if library_sources.any():
        if heat_profile_library is None:
            raise ValueError(
                "The paired heat catalog requires --heat-profile-library."
            )
        library = PhysicalHeatProfileLibrary(heat_profile_library)
        expected_profile_sets = (
            heat_catalog.loc[library_sources, "profile_set_id"]
            .dropna()
            .astype(str)
            .unique()
        )
        profile_set_matches = (
            len(expected_profile_sets) == 1
            and library.profile_set_id == expected_profile_sets[0]
        )
        if not profile_set_matches:
            raise ValueError(
                "Heat-profile library mismatch: paired catalog expects "
                f"{expected_profile_sets.tolist()}, got {library.profile_set_id!r}."
            )
    demand, demand_audit = build_paired_base_electric_demand(
        allocation,
        seed=seed,
        component_plan=component_plan,
    )
    sector_inputs = build_paired_sector_urbs_inputs(
        allocation,
        hours=len(demand),
        seed=seed,
        weather_source_hdf=weather_source_hdf.resolve(),
        roof_catalog=roof_catalog,
        pv_profile_library=pv_profile_library,
        pv_sizing_method=pv_sizing_method,
        battery_sizing_method=battery_sizing_method,
        battery_minimum_pv_kwp_per_annual_mwh=scenario.battery.minimum_pv_kwp_per_annual_mwh,
        battery_usable_kwh_per_pv_kwp=battery_pv_coefficient,
        battery_usable_kwh_per_annual_mwh=battery_demand_coefficient,
        battery_energy_to_power_hours=scenario.battery.energy_to_power_hours,
        technology_parameters=scenario.technologies,
        heat_sizing_method=heat_sizing_method,
        heat_config=scenario.heat,
        pv_demand_multiplier=scenario.pv.demand_multiplier,
        heat_profile_catalog=heat_catalog,
        heat_profile_library=heat_profile_library,
        allow_diagnostic_heat_fallback=allow_diagnostic_heat_fallback,
        electrification_assignment=electrification_assignment,
    )
    if not sector_inputs.demand.empty:
        demand = pd.concat(
            [demand, sector_inputs.demand],
            axis=1,
        ).sort_index(axis=1)
    active_buses = sorted(
        int(bus) for bus in demand.columns.get_level_values(0).unique()
    )
    electricity_module = load_electricity_module()
    electricity_module.config.apply_scenario(scenario)
    static_tables = _combine_static_tables(
        active_buses,
        sector_inputs,
        electricity_module,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / _output_name(
        allocation,
        target_network,
        target_grid_id,
        scenario_label,
    )
    if output.exists():
        output.unlink()
    heat_audit = sector_inputs.audit[
        sector_inputs.audit.get("sector", pd.Series(dtype=str)).eq("heat")
    ]
    heat_profile_audit = heat_audit[
        heat_audit.get(
            "audit_record_type",
            pd.Series(index=heat_audit.index, dtype=str),
        ).eq("heat_profile")
    ]
    asset_plan = sector_inputs.audit[
        sector_inputs.audit.get(
            "audit_record_type",
            pd.Series(index=sector_inputs.audit.index, dtype=str),
        ).isin(["asset_plan", "battery_asset_plan", "heat_asset_plan"])
    ].copy()
    metadata = {
        **build_full_year_metadata(),
        "source": "paired_swf_2045",
        "target_network": target_network,
        "target_grid_id": int(target_grid_id),
        "scenario_label": scenario_label,
        "scenario_scope": "paired_full_local_demand",
        "profile_contract": "physical_building_component_paired_v2",
        "component_contract": "physical_building_component_v1",
        "component_rows": int(len(component_plan)),
        "included_component_rows": int(component_plan["included_in_lv"].astype(bool).sum()),
        "suppressed_component_rows": int((~component_plan["included_in_lv"].astype(bool)).sum()),
        "mixed_use_physical_buildings": int(
            component_plan.groupby("building_objectid")["component_category"].nunique().gt(1).sum()
        ),
        "heat_area_method": "residential_effective_floor_area_v1",
        "optimization_space": "scenario_unit",
        "paired_dir": str(paired_dir),
        "profile_seed": int(seed),
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, scenario_hash),
        "electrification_assignment_hash": electrification_assignment_hash,
        "electrification_assignment_summary": (
            assignment_summary(electrification_assignment).to_dict("records")
        ),
        "electrification_asset_plan_summary": _paired_electrification_asset_plan_summary(
            electrification_assignment,
            sector_inputs.audit,
            component_plan=component_plan,
        ),
        "model_case": model_case,
        "pv_sizing_method": pv_sizing_method,
        "pv_feed_in_tariff_eur_per_kwh": scenario.economics.pv_feed_in_tariff_eur_per_kwh,
        "battery_sizing_method": battery_sizing_method,
        "battery_energy_to_power_hours": scenario.battery.energy_to_power_hours,
        "heat_sizing_method": heat_sizing_method,
        "heat_scope": "residential_buildings",
        "physical_buildings": int(allocation["building_objectid"].nunique()),
        "hh_rows": float(allocation["residential_equivalent_hh_rows"].sum()),
        "hh_annual_kwh": float(
            allocation["residential_equivalent_hh_annual_kwh"].sum()
        ),
        "ghd_annual_kwh": float(allocation["calibrated_annual_ghd_kwh"].sum()),
        "heat_profile_fallback_buildings": int(
            heat_profile_audit.loc[
                heat_profile_audit.get("profile_method", "").ne("exact_physical_building"),
                "building_objectid",
            ].nunique()
        )
        if not heat_profile_audit.empty
        else 0,
        "publication_ready": bool(
            heat_profile_audit.empty
            or heat_profile_audit.get(
                "profile_method",
                pd.Series(dtype=str),
            )
            .isin(PUBLICATION_READY_HEAT_METHODS)
            .all()
        ),
        **sector_inputs.metadata,
    }
    write_hdf_metadata(output, metadata)
    with pd.HDFStore(
        output,
        mode="a",
        complib="blosc",
        complevel=9,
    ) as store:
        store.put(
            "raw_data/electrification_assignment",
            electrification_assignment.reset_index(drop=True),
        )
        store.put("raw_data/component_scenario_plan", component_plan.reset_index(drop=True))
        store.put("raw_data/allocation_plan", allocation.reset_index(drop=True))
        store.put("raw_data/asset_plan", asset_plan.reset_index(drop=True))
        store.put(
            "raw_data/demand_profile_audit",
            demand_audit.reset_index(drop=True),
        )
        store.put(
            "raw_data/sector_profile_audit",
            sector_inputs.audit.reset_index(drop=True),
        )
        store.put("urbs_in/demand", demand)
        store.put("urbs_in/supim", sector_inputs.supim)
        store.put("urbs_in/eff_factor", sector_inputs.eff_factor)
        store.put(
            "urbs_in/buy_sell_price",
            buy_sell_price(
                len(demand),
                electricity_module,
                import_price_eur_per_kwh=scenario.economics.import_price_eur_per_kwh,
                pv_feed_in_tariff_eur_per_kwh=scenario.economics.pv_feed_in_tariff_eur_per_kwh,
            ),
        )
        store.put(
            "urbs_in/weather",
            read_or_create_weather(
                weather_source_hdf.resolve(),
                len(demand),
            ),
        )
        for key, table in static_tables.items():
            store.put(f"urbs_in/{key}", table)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=DEFAULT_PAIRED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-network",
        choices=["real_swf", "synthetic"],
        required=True,
    )
    parser.add_argument("--target-grid-id", type=int, required=True)
    parser.add_argument(
        "--scenario-label",
        default="swf_2045_paired_full_local",
    )
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=481527,
        help="Arbitrary fixed seed for reproducible stochastic input profiles.",
    )
    parser.add_argument("--weather-source-hdf", type=Path, required=True)
    parser.add_argument("--heat-profile-library", type=Path)
    parser.add_argument(
        "--model-case",
        choices=POST_MODEL_CASES,
        default="post-hems-heuristic",
    )
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument(
        "--allow-diagnostic-heat-fallback",
        action="store_true",
        help=(
            "Permit area-scaled proxy heat profiles for buildings whose "
            "current residential-only synthetic source HDF has no exact heat "
            "profile. Such outputs are marked publication_ready=false."
        ),
    )
    args = parser.parse_args()
    output = materialize_paired_urbs_input(
        paired_dir=args.paired_dir,
        output_dir=args.output_dir,
        target_network=args.target_network,
        target_grid_id=args.target_grid_id,
        scenario_label=args.scenario_label,
        seed=args.profile_seed,
        weather_source_hdf=args.weather_source_hdf,
        heat_profile_library=(
            args.heat_profile_library.resolve()
            if args.heat_profile_library is not None
            else None
        ),
        allow_diagnostic_heat_fallback=(args.allow_diagnostic_heat_fallback),
        model_case=args.model_case,
        scenario_config_path=args.scenario_config,
    )
    print(output)


if __name__ == "__main__":
    main()
