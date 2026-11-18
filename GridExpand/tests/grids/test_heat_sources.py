"""Per-building heat profiles of own grids (the ro_heat source with stand-in series; no database)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridexpand.grids import heat_sources
from gridexpand.scenario.commands import dataset_preparation_commands
from tests.fixture_paths import FIXTURE_SCENARIOS

HOURS = 8760


def residential():
    return pd.DataFrame({
        "objectid": ["a", "b"], "bus": [0, 1], "construction_year": [1970, None], "building_type": ["SFH", "MFH"],
        "households": [1, 4], "floor_area": [100.0, 200.0], "floor_number": [2, 3],
        "residential_effective_floor_area_m2": [180.0, 520.0], "residential_area_share": [0.9, 0.87],
    })


@pytest.fixture
def ro_heat(monkeypatch):
    import gridexpand.allocation.functions.infdb_ro_heat as source

    def space(buildings):
        return (pd.DataFrame({(int(bus), "space_heat"): np.full(HOURS, 1.0 + bus) for bus in buildings["bus"]}),
                {"space_heat_source": "infdb_ro_heat"})

    def water(buildings, base_seed=0):
        return pd.DataFrame({(int(bus), "water_heat"): np.full(HOURS, 0.25) for bus in buildings["bus"]})

    monkeypatch.setattr(source, "load_space_heat", space)
    monkeypatch.setattr(source, "generate_opendhw", water)


def test_ro_heat_profiles_per_building(ro_heat):
    weather = pd.DataFrame({"temp_air": np.linspace(-10.0, 30.0, HOURS)})
    demand, cop, audit = heat_sources.grid_heat_profiles(residential(), weather, source="infdb_ro_heat", plz=85653,
                                                          profile_seed=481527)
    assert set(demand.columns) == {(0, "space_heat"), (1, "space_heat"), (0, "water_heat"), (1, "water_heat")}
    assert len(demand) == HOURS and demand[(1, "space_heat")].sum() == pytest.approx(2.0 * HOURS, rel=1e-3)
    assert set(cop.columns) == {(0, "heatpump_air"), (1, "heatpump_air")} and (cop.to_numpy() > 0).all()
    assert audit == {"space_heat_source": "infdb_ro_heat"}


def test_internal_heat_needs_no_heat_sources(tmp_path):
    from gridexpand.paths import SCENARIO_CONFIG_DIR

    with pytest.raises(ValueError, match="Per-building heat sources"):
        heat_sources.build_heat_sources(tmp_path, scenario_config=SCENARIO_CONFIG_DIR / "default_2045.yaml",
                                        weather_hdf=tmp_path / "w.h5", sources_dir=tmp_path / "s",
                                        library=tmp_path / "l.h5", profile_set_id="x")


def test_own_grids_build_heat_per_building(tmp_path):
    common = dict(paired_dir=tmp_path / "d", weather_hdf=tmp_path / "w.h5", heat_sources=tmp_path / "s",
                  heat_library=tmp_path / "l.h5", heat_profile_set_id="set", heat_workers=1, reference_year=2009)
    teaser = FIXTURE_SCENARIOS / "teaser_2045.yaml"
    own = [stage for stage, _ in dataset_preparation_commands(scenario_config=teaser, per_building_heat=True, **common)]
    validation = [stage for stage, _ in dataset_preparation_commands(scenario_config=teaser, **common)]
    assert own == ["weather", "heat_sources", "pv_library"]
    assert validation == ["weather", "heat_regeneration", "heat_readiness_sources", "heat_library",
                          "heat_readiness_library", "pv_library"]
