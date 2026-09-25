"""Pool CSV reader, emobpy retry and pool task planning (alloc D7, D9)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

import gridexpand.allocation.functions.mobility as mobility
import gridexpand.allocation.generate_mobility_profile_pool as pool
from gridexpand.allocation.assets.pv.roof_catalog import roof_catalog_options
from gridexpand.allocation.scenario_calibration.profiles import paired_profiles


def _pool_csv(tmp_path):
    frame = pd.DataFrame(
        {
            "profile_id": ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3,
            "t": [0, 1, 2] * 3,
            "demand_kwh": [0.0, 1.5, 0.0, 2.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        }
    )
    path = tmp_path / "mobility_demand_pool.csv"
    frame.to_csv(path, index=False)
    return path, frame


@pytest.mark.parametrize("chunksize", [2, 4, 200_000])
def test_read_rows_for_profiles(tmp_path, chunksize):
    path, frame = _pool_csv(tmp_path)
    rows = mobility.read_rows_for_profiles(path, ["p3", "p1"], chunksize=chunksize)
    expected = frame[frame["profile_id"].isin({"p1", "p3"})].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows, expected)
    assert mobility.read_rows_for_profiles(path, ["nope"], chunksize=chunksize) is None


def test_callers_keep_their_empty_semantics(tmp_path):
    path, _ = _pool_csv(tmp_path)
    with pytest.raises(ValueError, match="No selected profile rows"):
        mobility._read_pool_timeseries(path, ["nope"], "demand_kwh")
    empty = paired_profiles._read_session_pool(path, {"nope"}, columns=["profile_id", "t"])
    assert empty.empty and list(empty.columns) == ["profile_id", "t"]
    assert len(paired_profiles._read_session_pool(path, {"p2"})) == 3


def test_pool_identity_constants_are_shared():
    assert pool.SESSION_GENERATION_VERSION == paired_profiles.SESSION_GENERATION_VERSION == "emobpy_pool_v2_sessions"
    assert pool.POOL_MANIFEST_FILENAME == paired_profiles.POOL_MANIFEST_FILENAME == "mobility_pool_manifest.json"


def test_emobpy_retry_bumps_seeds(monkeypatch):
    attempts = []

    def flaky(vehicles, weather):
        attempts.append({key: car["seed"] for key, car in vehicles.items()})
        if len(attempts) < 3:
            raise RuntimeError("infeasible trip")
        return {"series": 1}, {"battery": 2}

    monkeypatch.setattr(mobility, "_simulate_vehicles", flaky)
    vehicles = {(1, 0): {"seed": 10}}
    assert mobility._simulate_vehicles_with_retry(vehicles, weather=None) == ({"series": 1}, {"battery": 2})
    assert [a[(1, 0)] for a in attempts] == [10, 11, 12]
    monkeypatch.setattr(mobility, "_simulate_vehicles", lambda v, w: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        mobility._simulate_vehicles_with_retry({(1, 0): {"seed": 1}}, weather=None)


def test_plan_tasks_matches_the_former_generate_pool_loop():
    args = argparse.Namespace(market_share_threshold=0.5, models=None, schedules=None, profiles_per_stratum=2)
    existing = pd.DataFrame()
    planned = pool._plan_tasks(args, existing, "central_germany_tmy")
    expected = []
    for model_index, model in pool._select_models(0.5, None):
        for schedule_index, schedule in enumerate(pool.SCHEDULES):
            for sample_index in range(2):
                expected.append({
                    "profile_id": pool._profile_id("central_germany_tmy", model_index, model, schedule, sample_index),
                    "model_index": model_index, "model": model, "schedule_index": schedule_index,
                    "schedule": schedule, "sample_index": sample_index,
                    "pool_seed": pool._pool_seed(model_index, schedule_index, sample_index),
                    "weather_key": "central_germany_tmy",
                })
    assert planned == expected and planned


def test_roof_catalog_options_keeps_the_former_dict():
    pv = SimpleNamespace(
        tilt_bin_degrees=5.0, azimuth_bin_degrees=15.0, module_capacity_kw_per_m2=0.202,
        flat_roof_utilization=0.27, slanted_roof_utilization=0.58, fallback_capacity_kwp=14.5,
    )
    options = roof_catalog_options(pv)
    assert list(options.items()) == [
        ("tilt_bin_deg", 5.0), ("azimuth_bin_deg", 15.0), ("module_capacity_kw_per_m2", 0.202),
        ("flat_roof_utilization", 0.27), ("slanted_roof_utilization", 0.58), ("fallback_capacity_kw", 14.5),
    ]
