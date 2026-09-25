"""Electricity profiling with small synthetic load-profile tables (no 113 MB file)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gridexpand.allocation.functions.electricity as electricity
from gridexpand.allocation.config import config
from gridexpand.common.reproducibility import stable_seed

HOURS = 24


@pytest.fixture
def small_tables(monkeypatch):
    rng = np.random.default_rng(0)
    normalized = pd.DataFrame(rng.random((HOURS, 3)), columns=["d1", "d2", "d3"])
    normalized = normalized / normalized.sum()
    sums = pd.DataFrame({"devicenumber": ["d1", "d2", "d3"], "kWh": [1500.0, 3000.0, 4500.0]})
    ghd = pd.DataFrame(rng.random((HOURS, 2)) / 1000.0, columns=["office", "shop"])
    monkeypatch.setattr(electricity, "residential_load_profiles", lambda: (normalized, sums))
    monkeypatch.setattr(electricity, "commercial_load_profiles", lambda: ghd)
    return normalized, sums, ghd


def _components():
    return pd.DataFrame(
        {
            "component_id": ["B1::residential", "B1::commercial", "B2::residential", "B3::public"],
            "objectid": ["B1", "B1", "B2", "B3"],
            "component_category": ["Residential", "Commercial", "Residential", "Public"],
            "effective_floor_area_m2": [100.0, 40.0, 80.0, 300.0],
            "bus": [1, 1, 2, 3],
            "included_in_lv": [True, True, True, False],
            "demand_tot_list": [[1600.0, 2900.0], pd.NA, [4400.0], pd.NA],
            "profile_type": ["SFH", "office", "MFH", "shop"],
        }
    )


def test_component_seeds_match_the_former_row_lookup(small_tables):
    result, bus_demand, profiles = electricity.get_elec_demand(
        _components(), base_seed=481527, return_component_profiles=True
    )
    expected = result["component_id"].map(
        lambda component_id: stable_seed(
            481527,
            result.loc[result["component_id"].eq(component_id), "objectid"].iloc[0],
            result.loc[result["component_id"].eq(component_id), "component_category"].iloc[0],
            "electricity",
            "profile",
        )
    )
    pd.testing.assert_series_equal(result["stable_seed"], expected, check_exact=True, check_names=False)
    # MV-direct components are not profiled; bus 3 carries no demand.
    assert list(bus_demand.columns) == [(1, "electricity"), (2, "electricity")]
    assert set(profiles.columns.get_level_values(0)) == {"B1::residential", "B1::commercial", "B2::residential"}
    np.testing.assert_allclose(bus_demand.to_numpy().sum(), result["annual_electricity_kwh"].sum())


def _old_audit(base, profiled_components, component_profiles):
    """Verbatim logic of the former Grid._build_demand_component_audit."""
    base = base.copy()
    profiled = profiled_components.set_index("component_id")
    profile_ids = [str(column[0]) for column in component_profiles.columns]
    profile_max = component_profiles.max(axis=0)
    profile_max.index = profile_ids
    audit = base.rename(columns={"component_category": "category"})
    audit["scenario_unit_id"] = audit["objectid"].astype(str)
    audit["commodity"] = "electricity"
    audit["annual_energy_kwh"] = audit["component_id"].map(profiled["annual_electricity_kwh"]).fillna(0.0)
    audit["max_profile_value"] = audit["component_id"].map(profile_max).fillna(0.0)
    audit["profile_hash"] = audit["component_id"].map(profiled["profile_hash"])
    audit["profile_method"] = audit["component_id"].map(profiled["profile_method"]).fillna("not_allocated")
    audit["stable_seed"] = audit["component_id"].map(profiled["stable_seed"])
    selected_ids = set(profiled_components["component_id"].astype(str))
    audit["suppression_reason"] = audit.apply(
        lambda row: (
            "outside_lv_scope" if not bool(row["included_in_lv"])
            else None if str(row["component_id"]) in selected_ids
            else "outside_demand_scope"
        ),
        axis=1,
    )
    audit["source_asset_count"] = pd.NA
    audit["matched_swf_asset_count"] = pd.NA
    audit["mv_direct"] = audit["mv_direct"].astype(bool)
    return audit


@pytest.mark.parametrize("scope", ["all", "residential"])
def test_demand_component_audit_matches_former_code(small_tables, scope):
    base = _components().assign(
        pylovo_version_id="1", mix_score=0.5, mix_rule="r", mix_confidence="high",
        mv_direct=[False, False, False, True],
    )
    selected = base.loc[base["included_in_lv"]]
    if scope == "residential":
        selected = selected.loc[selected["component_category"].eq("Residential")]
    profiled, _, profiles = electricity.get_elec_demand(selected, base_seed=1, return_component_profiles=True)
    new = electricity.demand_component_audit(base, profiled, profiles)
    old = _old_audit(base, profiled, profiles)[new.columns]
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    expected = [None, None if scope == "all" else "outside_demand_scope", None, "outside_lv_scope"]
    assert list(new["suppression_reason"]) == expected


def test_load_profile_tables_are_read_once(tmp_path, monkeypatch):
    path = tmp_path / "lps.h5"
    pd.DataFrame({"d1": [0.5, 0.5]}).to_hdf(path, key="df_normalized_scaled")
    pd.DataFrame({"devicenumber": ["d1"], "kWh": [1.0]}).to_hdf(path, key="df_sums")
    monkeypatch.setattr(config, "ELEC_LPS_PATH", str(path))
    electricity.residential_load_profiles.cache_clear()
    try:
        first = electricity.residential_load_profiles()
        assert electricity.residential_load_profiles() is first
        assert list(first[1]["devicenumber"]) == ["d1"]
    finally:
        electricity.residential_load_profiles.cache_clear()
