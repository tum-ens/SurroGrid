"""Shared synthetic electrification inventory (alloc D1/B4) and component aggregation (D2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import gridexpand.allocation.functions.electricity as electricity
from gridexpand.allocation.electrification import (
    INVENTORY_COLUMNS,
    electrification_inventory,
    has_household,
)
from gridexpand.common.electrification import build_electrification_assignment

ADOPTION = {
    technology: {"adoption_mode": "deterministic_share", "building_share": 0.5}
    for technology in ("heat", "mobility", "pv_battery")
}


def _physical(n=40, seed=1):
    rng = np.random.default_rng(seed)
    occupancy = [list(rng.integers(1, 4, size=rng.integers(0, 3))) for _ in range(n)]
    return pd.DataFrame(
        {
            "objectid": [f"DEBY{i:05d}" for i in range(n)],
            "residential": rng.random(n) < 0.7,
            "occ_list": occupancy,
            "n_cars_tot": rng.integers(0, 3, size=n).astype(float),
            "roof_kw": np.where(rng.random(n) < 0.6, rng.random(n) * 20.0, 0.0),
            "annual_electricity_kwh": np.where(rng.random(n) < 0.8, rng.random(n) * 5000.0, 0.0),
        },
        index=rng.permutation(n) + 100,
    )


def _old_grid_inventory(physical):
    """Verbatim logic of the former Grid.prepare_electrification_assignment."""
    frame = physical.copy()
    frame["building_objectid"] = frame["objectid"].astype(str)
    residential = frame["residential"]
    frame["heat_eligible"] = residential
    frame["heat_exclusion_reason"] = np.select(
        [~residential, ~frame["heat_eligible"]],
        ["no_residential_component", "no_valid_heat_profile_source"],
        default=None,
    )
    household = frame["occ_list"].apply(
        lambda value: isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
    )
    vehicles = pd.to_numeric(frame["n_cars_tot"], errors="coerce").fillna(0.0)
    frame["mobility_eligible"] = residential & household & vehicles.gt(0.0)
    frame["mobility_exclusion_reason"] = np.select(
        [~residential, ~household, vehicles.le(0.0)],
        ["no_residential_component", "no_household", "no_vehicle_inventory"],
        default=None,
    )
    roof = frame["roof_kw"]
    annual = pd.to_numeric(frame["annual_electricity_kwh"], errors="coerce").fillna(0.0)
    frame["pv_battery_eligible"] = roof.gt(0.0) & annual.gt(0.0)
    frame["pv_battery_exclusion_reason"] = np.select(
        [roof.le(0.0), annual.le(0.0)], ["no_usable_lod2_roof", "no_base_electricity"], default=None
    )
    return frame


def _old_regional_pv_reason(physical):
    """The former electrification_preparation PV/battery reason (not roof-first)."""
    eligible = physical["roof_kw"].gt(0.0) & physical["annual_electricity_kwh"].gt(0.0)
    reason = pd.Series("no_usable_lod2_roof", index=physical.index, dtype=object)
    reason[eligible] = None
    reason[~eligible & physical["annual_electricity_kwh"].le(0.0)] = "no_base_electricity"
    return reason


def _new_inventory(physical):
    frame = physical.copy()
    frame["building_objectid"] = frame["objectid"].astype(str)
    inventory = electrification_inventory(
        frame["building_objectid"],
        residential=frame["residential"],
        has_household=has_household(frame["occ_list"]),
        vehicle_count=frame["n_cars_tot"],
        roof_capacity_kw=frame["roof_kw"],
        annual_electricity_kwh=frame["annual_electricity_kwh"],
    )
    for column in INVENTORY_COLUMNS:
        frame[column] = inventory[column]
    return frame


def _assignment(frame):
    return build_electrification_assignment(
        frame, ADOPTION, selection_scope_id="test|scope", profile_seed=481527
    )


def test_inventory_reproduces_the_in_grid_assignment_exactly():
    physical = _physical()
    pd.testing.assert_frame_equal(
        _assignment(_new_inventory(physical)), _assignment(_old_grid_inventory(physical)), check_exact=True
    )


def test_regional_labels_change_only_for_roofless_zero_demand_buildings():
    """B4: roof-first reason order also in the regional preparation."""
    physical = _physical(n=200, seed=4)
    new = _new_inventory(physical)["pv_battery_exclusion_reason"]
    old = _old_regional_pv_reason(physical)
    changed = new.fillna("<none>") != old.fillna("<none>")
    expected = physical["roof_kw"].le(0.0) & physical["annual_electricity_kwh"].le(0.0)
    assert changed.equals(expected)
    assert expected.any()
    assert set(new[changed]) == {"no_usable_lod2_roof"}
    assert set(old[changed]) == {"no_base_electricity"}


def test_inventory_reasons():
    ids = pd.Series(["a", "b", "c", "d"])
    inventory = electrification_inventory(
        ids,
        residential=pd.Series([True, False, True, True]),
        has_household=pd.Series([True, False, False, True]),
        vehicle_count=pd.Series([1, 2, 3, np.nan]),
        roof_capacity_kw=pd.Series([5.0, 0.0, np.nan, 1.0]),
        annual_electricity_kwh=pd.Series([10.0, 0.0, 5.0, 0.0]),
    )
    assert list(inventory["heat_eligible"]) == [True, False, True, True]
    assert list(inventory["heat_exclusion_reason"]) == [None, "no_residential_component", None, None]
    assert list(inventory["mobility_exclusion_reason"]) == [
        None, "no_residential_component", "no_household", "no_vehicle_inventory"]
    assert list(inventory["pv_battery_exclusion_reason"]) == [
        None, "no_usable_lod2_roof", "no_usable_lod2_roof", "no_base_electricity"]


def _components():
    return pd.DataFrame(
        {
            "objectid": ["B1", "B1", "B2", "B3", "B5"],
            "component_category": ["Residential", "Commercial", "Public", "Residential", "Residential"],
            "annual_electricity_kwh": [1000.0, 250.5, 400.0, 0.0, 3000.0],
            "effective_floor_area_m2": [120.0, 30.0, 500.0, 80.0, 60.0],
            "occ_list": [[2, 1], pd.NA, pd.NA, [], [3]],
        }
    )


def test_component_aggregation_matches_former_grid_and_regional_code():
    physical = pd.DataFrame({"objectid": ["B1", "B2", "B3", "B4"], "bus": [1, 2, 3, 4]})
    components = _components()

    # former Grid.generate_electricity
    residential = components.loc[components["component_category"].eq("Residential")]
    occupancy = dict(zip(residential["objectid"].astype(str), residential["occ_list"]))
    annual = components.assign(_id=components["objectid"].astype(str)).groupby("_id")["annual_electricity_kwh"].sum()
    grid = physical.copy()
    ids = grid["objectid"].astype(str)
    grid["occ_list"] = ids.map(occupancy).apply(
        lambda value: value if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    grid["annual_electricity_kwh"] = ids.map(annual).fillna(0.0)
    pd.testing.assert_frame_equal(
        electricity.aggregate_components_to_buildings(physical, components), grid, check_exact=True
    )

    # former electrification_preparation._prepare_grid_inventory
    regional = electricity.aggregate_components_to_buildings(physical, components, residential_area=True)
    area = residential.groupby("objectid")["effective_floor_area_m2"].sum()
    assert list(regional["residential_effective_floor_area_m2"]) == [120.0, 0.0, 80.0, 0.0]
    assert regional["residential_effective_floor_area_m2"].equals(
        physical["objectid"].map(area).fillna(0.0).rename("residential_effective_floor_area_m2")
    )
    assert list(regional["occ_list"]) == [[2, 1], [], [], []]
    assert list(regional["annual_electricity_kwh"]) == [1250.5, 400.0, 0.0, 0.0]


def test_profile_components_samples_then_profiles(monkeypatch):
    calls = []
    monkeypatch.setattr(electricity, "sample_statistics", lambda frame, seed: calls.append(("sample", seed)) or frame)
    monkeypatch.setattr(
        electricity,
        "get_elec_demand",
        lambda frame, base_seed, return_component_profiles: calls.append(("profile", base_seed, return_component_profiles)) or (frame, "bus", "components"),
    )
    assert electricity.profile_components(pd.DataFrame(), 7)[1:] == ("bus", "components")
    assert calls == [("sample", 7), ("profile", 7, True)]
