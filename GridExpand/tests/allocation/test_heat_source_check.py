"""Space-heat source coverage check before adopter selection (ro_heat path)."""

from __future__ import annotations

import pandas as pd
import pytest

import gridexpand.allocation.functions.infdb_ro_heat as infdb_ro_heat
from gridexpand.allocation.electrification import check_heat_profile_source


def _buildings():
    return pd.DataFrame(
        {
            "building_objectid": ["B1", "B2"],
            "floor_area": [100.0, 50.0],
            "floor_number": [2, 1],
            "residential_effective_floor_area_m2": [150.0, 50.0],
            "bus": [1, 2],
        }
    )


def test_ro_heat_is_loaded_with_the_residential_area_share(monkeypatch):
    seen = []
    monkeypatch.setattr(
        infdb_ro_heat, "load_space_heat", lambda buildings, engine=None: seen.append((buildings, engine))
    )
    check_heat_profile_source("infdb_ro_heat", _buildings(), engine="engine")
    (buildings, engine), = seen
    assert engine == "engine"
    assert list(buildings["residential_area_share"]) == [0.75, 1.0]


def test_teaser_and_empty_inputs_skip_the_database(monkeypatch):
    monkeypatch.setattr(infdb_ro_heat, "load_space_heat", lambda *a, **k: pytest.fail("no DB access expected"))
    check_heat_profile_source("teaser", _buildings())
    check_heat_profile_source("infdb_ro_heat", _buildings().iloc[0:0])
    with pytest.raises(ValueError, match="Unknown space heat source"):
        check_heat_profile_source("other", _buildings())
