"""Scenario dataset of given grids, with the database and the large data files replaced by small tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import gridexpand.allocation.functions.electricity as electricity
import gridexpand.allocation.functions.mobility as mobility
from tests.fixture_paths import FIXTURE_SCENARIOS
from gridexpand.allocation.config import config
from gridexpand.common import ghd
from gridexpand.grids import attributes, dataset

SCENARIO = FIXTURE_SCENARIOS / "joint_2045_full_year.yaml"


def buildings():
    rows = pd.DataFrame([
        ("SFH", 100.0, 2, "Residential", "31001_1000", "SFH", 200.0, 0.0, None, None, None, "SFH"),
        ("MFH", 200.0, 3, "Residential", "31001_1000", "MFH", 600.0, 0.0, 9, None, None, "MFH"),
        ("SHOP", 150.0, 2, "Commercial", "31001_2000", None, 0.0, 300.0, None, None, "Commercial", "Commercial"),
        ("HALL", 1000.0, 2, "Commercial", "31001_2000", None, 0.0, 2000.0, None, None, "Commercial", "Commercial"),
    ], columns=["objectid", "floor_area", "floor_number", "building_use", "building_use_id", "building_type",
                "residential_floor_area", "nonresidential_floor_area", "occupants", "households",
                "nonresidential_use", "type"])
    rows["mix_rule"] = None
    frame = attributes.pylovo_equivalent_buildings(rows)
    frame["postcode"], frame["gemeindeschluessel"] = 85653, "09184137"
    for column in attributes.PYLOVO_ONLY_COLUMNS:
        frame[column] = pd.NA
    return frame


@pytest.fixture
def offline(monkeypatch):
    """Small load-profile and GHD tables, no database, one vehicle per household building."""
    hours = 24
    rng = np.random.default_rng(0)
    normalized = pd.DataFrame(rng.random((hours, 3)), columns=["d1", "d2", "d3"])
    normalized = normalized / normalized.sum()
    sums = pd.DataFrame({"devicenumber": ["d1", "d2", "d3"], "kWh": [1500.0, 3000.0, 4500.0]})
    types = ["public_office", "office", "trade_retail"]
    per_m2 = pd.DataFrame({name: np.full(hours, (index + 1) / 1000.0) for index, name in enumerate(types)})
    distribution = pd.DataFrame({"type": types, "public_prob": [1.0, 0.0, 0.0], "commercial_prob": [0.0, 0.5, 0.5]})
    monkeypatch.setattr(electricity, "residential_load_profiles", lambda: (normalized, sums))
    monkeypatch.setattr(electricity, "commercial_load_profiles", lambda: per_m2)
    monkeypatch.setattr(type(config), "TYPE_GHD_DISTRIBUTION", property(lambda self: distribution))
    monkeypatch.setattr(attributes, "building_attributes",
                        lambda engine, ids: buildings().loc[lambda f: f["objectid"].isin(ids)])
    monkeypatch.setattr(ghd, "load_evidence", lambda config, ids, engine=None: None)
    monkeypatch.setattr(dataset, "_regio7", lambda engine, frame: pd.Series(77, index=frame.index))
    monkeypatch.setattr(dataset, "load_lod2_roof_catalog", lambda engine, ids, **options: pd.DataFrame(
        {"building_objectid": ["SFH"], "quality_flag": ["lod2"]}))
    monkeypatch.setattr(dataset, "building_lod2_capacity", lambda catalog: pd.Series({"SFH": 9.5}))
    monkeypatch.setattr(dataset, "session_pool_supported_models", lambda: ("car",))
    monkeypatch.setattr(mobility, "sample_statistics", lambda frame, region, allowed_models=None, base_seed=0: frame.assign(
        n_cars_tot=[1 if occupants else 0 for occupants in frame["occ_list"]]))


def test_dataset_of_two_given_grids(offline, tmp_path):
    supply = pd.DataFrame({"building_objectid": ["SFH", "MFH", "SHOP", "HALL"], "grid_id": ["g1", "g1", "g1", "g2"],
                           "bus": [3, 1, 2, 5]})
    files = {"g1": tmp_path / "g1.json", "g2": tmp_path / "g2.json"}
    metadata = dataset.build_dataset(supply, files, engine=None,
                                     scenario_config_path=SCENARIO, profile_seed=481527,
                                     output_dir=tmp_path / "out")
    plan = pd.read_csv(tmp_path / "out" / "paired_real_bus_allocation_plan.csv")
    assert plan[["building_objectid", "target_grid_id", "allocation_bus", "scenario_unit_id"]].values.tolist() == [
        ["MFH", 1, 1, 0], ["SHOP", 1, 2, 1], ["SFH", 1, 3, 2], ["HALL", 2, 5, 3]]
    assert plan["real_grid_name"].tolist() == ["g1", "g1", "g1", "g2"]
    assert dataset.grid_numbers(["7", "12"]) == {"7": 7, "12": 12}
    assert plan["target_network"].eq("real").all() and plan["real_grid_file"].iloc[0].endswith("g1.json")
    components = pd.read_csv(tmp_path / "out" / "paired_component_scenario_plan.csv").set_index("component_id")
    hall = components.loc["HALL::commercial"]
    assert not hall["included_in_lv"] and hall["annual_energy_kwh"] == 0 and hall["suppression_reason"] == \
        "outside_paired_lv_scope"  # MV-direct: grid g2 carries no LV load, and that is no error
    assert components.loc["MFH::residential", "real_target_bus"] == 1
    assert components.loc["SHOP::commercial", "annual_energy_kwh"] > 0
    pv = plan.set_index("building_objectid")["pv_roof_eligible"]
    assert pv["SFH"] and not pv["MFH"]
    assignment = pd.read_csv(tmp_path / "out" / "paired_electrification_assignment.csv")
    assert len(assignment) == 12
    assert metadata["grids"] == {"g1": 1, "g2": 2} and metadata["real_grids"] == 2 and metadata["attribute_source"] == "infdb_basedata_pylovo_rules"
    assert metadata["selection_scope_id"].endswith("|real|grids=g1,g2")
    stored = json.loads((tmp_path / "out" / "paired_scenario_metadata.json").read_text())
    assert stored["electrification_assignment_hash"] == metadata["electrification_assignment_hash"]


def test_supply_checks(tmp_path):
    files = {"g1": tmp_path / "g1.json"}
    with pytest.raises(ValueError, match="lacks the columns"):
        dataset._supply(pd.DataFrame({"building_objectid": ["a"], "bus": [1]}), files)
    with pytest.raises(ValueError, match="two supplies"):
        dataset._supply(pd.DataFrame({"building_objectid": ["a", "a"], "grid_id": ["g1", "g1"], "bus": [1, 2]}), files)
    with pytest.raises(ValueError, match="without a grid file"):
        dataset._supply(pd.DataFrame({"building_objectid": ["a"], "grid_id": ["g9"], "bus": [1]}), files)


def test_one_pv_unit_per_roof():
    plan = pd.DataFrame({"building_objectid": ["a", "a", "b"], "scenario_unit_id": [0, 1, 2],
                         "residential_equivalent_hh_annual_kwh": [1000.0, 3000.0, 500.0],
                         "calibrated_annual_ghd_kwh": [0.0, 0.0, 0.0]})
    chosen = dataset.pv_scenario_unit_assignments(plan)
    assert chosen[["building_objectid", "scenario_unit_id"]].values.tolist() == [["a", 1], ["b", 2]]
