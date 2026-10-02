"""Status-quo test: flags on a tiny feeder, peak hours, pylovo-equivalent inputs, exclusion list."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
import pytest

import gridexpand.allocation.functions.electricity as electricity
from gridexpand.allocation.config import config
from gridexpand.common import ghd
from gridexpand.paired import status_quo


def _feeder(length_km, std_type="NAYY 4x150 SE"):
    """External grid on the LV busbar -- one line -- one load bus."""
    net = pp.create_empty_network()
    busbar = pp.create_bus(net, vn_kv=0.4)
    end = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, busbar, vm_pu=1.0)
    pp.create_line(net, busbar, end, length_km=length_km, std_type=std_type)
    pp.create_load(net, end, p_mw=0.0)
    return net, end


@pytest.mark.parametrize(
    ("length_km", "std_type", "rating_mva", "kw", "expected"),
    [
        (0.1, "NAYY 4x150 SE", 0.4, [30.0, 10.0], []),
        (0.1, "NAYY 4x150 SE", 0.05, [60.0, 10.0], ["transformer_overload"]),
        (0.05, "NAYY 4x50 SE", 1.0, [120.0, 10.0], ["cable_overload"]),
        (0.6, "NAYY 4x150 SE", 1.0, [150.0, 20.0], ["undervoltage"]),  # 0.876 p.u. with both tap steps
    ],
)
def test_flags_on_a_tiny_feeder(length_km, std_type, rating_mva, kw, expected):
    net, end = _feeder(length_km, std_type)
    metrics = status_quo.evaluate_grid(net, rating_mva, pd.DataFrame({end: kw}))
    assert status_quo.flag_reasons(metrics) == expected
    assert metrics["evaluated_hours"] == 2 and metrics["failed_hours"] == 0
    assert metrics["transformer_rated_kva"] == pytest.approx(rating_mva * 1000.0)
    if expected == ["undervoltage"]:
        assert metrics["tap_steps"] == 2 and metrics["min_voltage_pu"] < 0.90


def test_a_diverging_hour_is_flagged_as_nonconvergence():
    net, end = _feeder(0.6)
    metrics = status_quo.evaluate_grid(net, 1.0, pd.DataFrame({end: [3000.0, 10.0]}))
    assert metrics["failed_hours"] == 1
    assert "nonconvergence" in status_quo.flag_reasons(metrics)


def test_a_missing_transformer_rating_skips_only_the_transformer_check():
    metrics = {"transformer_max_percent": float("nan"), "cables_over_100": 0, "buses_below_limit": 0, "failed_hours": 0}
    assert status_quo.flag_reasons(metrics) == []


def test_a_grid_without_lv_demand_passes():
    job = {"row": {"provider": "swf", "real_grid_id": "LV_154", "real_grid_number": 154, "buildings": 9},
           "plan": pd.DataFrame(columns=status_quo.PLAN_COLUMNS), "grid_file": "unused.xlsx", "profile_seed": 1,
           "peak_hours": 60, "all_hours": False}
    row = status_quo._evaluate_job(job)
    assert row["flagged"] is False and row["reason"] == "" and row["evaluated_hours"] == 0


def test_peak_hours_are_the_largest_total_household_and_ghd_hours():
    hours = 200
    total = pd.DataFrame({1: np.arange(hours, dtype=float)})
    ghd_part = pd.DataFrame({1: np.zeros(hours)})
    ghd_part.loc[10, 1] = 50.0  # a GHD-only peak in an otherwise low hour
    selected = status_quo.peak_hours(total, ghd_part, 5)
    assert selected[-5:] == [195, 196, 197, 198, 199]
    assert 10 in selected


def test_exclusion_list_and_grid_ids():
    grids = pd.DataFrame({
        "provider": ["swf", "swf", "uzw"], "real_grid_id": ["LV_059", "LV_060", "7"],
        "flagged": [True, False, True], "reason": ["transformer_overload;undervoltage", "", "nonconvergence"],
    })
    exclusions = status_quo.exclusion_list(grids)
    assert list(exclusions.columns) == ["provider", "real_grid_id", "reason"]
    assert exclusions["real_grid_id"].tolist() == ["LV_059", "7"]
    assert status_quo.bundle_grid_id("swf", 59) == "LV_059" and status_quo.bundle_grid_id("uzw", 7) == "7"


def _basedata_rows():
    return pd.DataFrame([
        # objectid, floor_area, floors, use, use_id, building_type, res, nonres, mix_rule, occupants, households, nonres_use, type
        ("SFH", 100.0, 2, "Residential", "31001_1000", "SFH", 200.0, 0.0, "full_residential", None, None, None, "SFH"),
        ("MFH", 200.0, 3, "Residential", "31001_1000", "MFH", 600.0, 0.0, "full_residential", 9, None, None, "MFH"),
        ("SHOP", 150.0, 2, "Commercial", "31001_2000", None, 0.0, 300.0, "full_nonresidential", None, None, "Commercial", "Commercial"),
        ("HALL", 1000.0, 2, "Commercial", "31001_2000", None, 0.0, 2000.0, "full_nonresidential", None, None, "Commercial", "Commercial"),
        ("UNK", 120.0, 2, "Unknown", "31001_9998", None, 0.0, 240.0, "full_nonresidential", None, None, "Unknown", "Unknown"),
        ("MIX", 100.0, 3, "Mixed", "31001_9998", None, 200.0, 100.0, "standard", None, 2, "Unknown", "Mixed"),
    ], columns=["objectid", "floor_area", "floor_number", "building_use", "building_use_id", "building_type",
                "residential_floor_area", "nonresidential_floor_area", "mix_rule", "occupants", "households",
                "nonresidential_use", "type"])


def test_pylovo_equivalent_buildings():
    b = status_quo.pylovo_equivalent_buildings(_basedata_rows()).set_index("objectid")
    assert b.loc["SFH", "households"] == 1 and b.loc["MFH", "households"] == 3  # max(2, round(600 / 181))
    assert b.loc["SFH", "occupants"] == pytest.approx(2.03) and b.loc["MFH", "occupants"] == 9
    assert b.loc["SHOP", "nonresidential_peak_load_in_kw"] == pytest.approx(300 * 0.079)
    assert b.loc["UNK", "nonresidential_peak_load_in_kw"] == pytest.approx(240 * 0.029)
    assert b.loc["HALL", "nonresidential_mv_direct"] is True  # 158 kW > 100 kW
    assert b.loc["SHOP", "nonresidential_mv_direct"] is False and b.loc["SFH", "nonresidential_mv_direct"] is None
    assert b.loc["MFH", "residential_peak_load_in_kw"] == pytest.approx(3 * 16.825)


@pytest.fixture
def small_tables(monkeypatch):
    """Synthetic load-profile and use-type tables (the real ones are data files outside git)."""
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
    return per_m2


def test_component_plan_follows_the_aligned_model(small_tables):
    physical = status_quo.pylovo_equivalent_buildings(_basedata_rows())
    physical["bus"] = [1, 1, 2, 3, 2, 4]
    plan, audit = status_quo.component_plan(physical, ghd.GhdConfig(), None, profile_seed=481527)
    assert set(plan["component_id"]) == {"SFH::residential", "MFH::residential", "SHOP::commercial",
                                         "HALL::commercial", "MIX::residential"}
    rows = plan.set_index("component_id")
    assert not rows.loc["HALL::commercial", "included_in_lv"] and rows.loc["HALL::commercial", "annual_energy_kwh"] == 0
    assert rows.loc["MFH::residential", "source_asset_count"] == 3
    shop_kwh = rows.loc["SHOP::commercial", "annual_energy_kwh"]
    assert shop_kwh == pytest.approx(300.0 * 24 * 0.002) or shop_kwh == pytest.approx(300.0 * 24 * 0.003)
    assert rows.loc["SHOP::commercial", "target_bus"] == 2
    decisions = dict(zip(audit["objectid"], audit["decision"]))
    assert decisions == {"SHOP": "active", "HALL": "mv_direct", "UNK": "non_demand_unknown", "MIX": "non_demand_unknown"}


def test_jobs_from_a_prepared_dataset(tmp_path):
    pd.DataFrame({
        "component_id": ["A::residential", "B::commercial", "C::residential"],
        "building_objectid": ["A", "B", "C"], "component_category": ["Residential", "Commercial", "Residential"],
        "included_in_lv": [True, True, True], "annual_energy_kwh": [3000.0, 20000.0, 2500.0],
        "source_asset_count": [1, 1, 1], "stable_seed": [1, 2, 3], "profile_hash": ["h", "h", "h"],
        "profile_seed": [481527] * 3, "real_target_grid_id": [59, 59, 60], "real_target_bus": [10, 11, 20],
    }).to_csv(tmp_path / "paired_component_scenario_plan.csv", index=False)
    pd.DataFrame({
        "provider": ["swf"] * 3, "building_objectid": ["A", "B", "C"], "real_grid_id": [59, 59, 60],
        "real_grid_file": ["/grids/LV_059.xlsx", "/grids/LV_059.xlsx", "/grids/LV_060.xlsx"],
    }).to_csv(tmp_path / "paired_real_bus_allocation_plan.csv", index=False)
    jobs = status_quo.jobs_from_dataset(tmp_path, "swf", profile_seed=481527, peak_hours_count=60, all_hours=False)
    assert [job["row"]["real_grid_id"] for job in jobs] == ["LV_059", "LV_060"]
    assert jobs[0]["plan"]["target_bus"].tolist() == [10, 11] and jobs[0]["grid_file"] == "/grids/LV_059.xlsx"
    with pytest.raises(ValueError, match="profile seed"):
        status_quo.jobs_from_dataset(tmp_path, "swf", profile_seed=1, peak_hours_count=60, all_hours=False)


JOINT_SWF = os.environ.get("GRIDEXPAND_STATUS_QUO_JOINT_SWF_DATASET")


@pytest.mark.skipif(
    not JOINT_SWF or not Path(config.ELEC_LPS_PATH).is_file(),
    reason="set GRIDEXPAND_STATUS_QUO_JOINT_SWF_DATASET (and GRIDEXPAND_DATA_DIR with elec_lps.h5)",
)
def test_joint_run_flags_are_reproduced(tmp_path):
    """Regression: the pre stage of joint_2045_v1_full_year flagged these SWF grids (analysis run 68)."""
    assert status_quo.main(["--provider", "swf", "--dataset", JOINT_SWF, "--output-dir", str(tmp_path), "--jobs", "4"]) == 0
    exclusions = pd.read_csv(tmp_path / "status_quo_swf_exclusions.csv")
    assert exclusions["real_grid_id"].tolist() == [
        "LV_032", "LV_035", "LV_038", "LV_059", "LV_080", "LV_099", "LV_113", "LV_137",
    ]


def test_households_only_drops_the_ghd_components():
    plan = pd.DataFrame({
        "component_category": ["Residential", "Commercial", "Public", "Residential"],
        "target_bus": [1, 1, 2, 3],
    })
    jobs = status_quo.households_only([{"plan": plan}])
    assert list(jobs[0]["plan"]["component_category"]) == ["Residential", "Residential"]
