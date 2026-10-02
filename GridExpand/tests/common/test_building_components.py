"""Component manifest: an Unknown non-residential part is accepted and carries no demand."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.common.building_components import build_building_components, validate_physical_buildings


def _building(objectid, *, use, res=0.0, nonres=0.0, floor_area=100.0, floors=None, nonres_peak=None, direct=None):
    floors = floors if floors is not None else (res + nonres) / floor_area
    return {
        "objectid": objectid,
        "floor_area": floor_area,
        "floor_number": floors,
        "residential_floor_area": res,
        "nonresidential_floor_area": nonres,
        "nonresidential_use": use,
        "households": 2 if res > 0 else 1,
        "occupants": 4.0 if res > 0 else None,
        "building_use": "Mixed" if res > 0 and nonres > 0 else ("Residential" if res > 0 else "Commercial"),
        "building_use_id": "31001_9998" if use == "Unknown" else "31001_2000",
        "building_type": None,
        "mix_score": None,
        "mix_rule": "standard" if res > 0 and nonres > 0 else "full_nonresidential",
        "mix_confidence": None,
        "residential_peak_load_in_kw": 33.65 if res > 0 else 0.0,
        "nonresidential_peak_load_in_kw": nonres * 0.029 if nonres_peak is None else nonres_peak,
        "nonresidential_mv_direct": direct if direct is not None else (False if nonres > 0 else None),
        "peak_load_in_kw": 10.0,
        "bus": 1,
    }


def test_unknown_building_is_valid_and_has_no_component():
    physical = pd.DataFrame([
        _building("U1", use="Unknown", nonres=200.0),
        _building("C1", use="Commercial", nonres=150.0),
    ])
    validate_physical_buildings(physical)
    components = build_building_components(physical)
    assert components["component_id"].tolist() == ["C1::commercial"]


def test_mixed_building_with_unknown_part_keeps_only_its_residential_component():
    physical = pd.DataFrame([_building("M1", use="Unknown", res=200.0, nonres=100.0)])
    components = build_building_components(physical)
    assert components["component_id"].tolist() == ["M1::residential"]
    assert components["effective_floor_area_m2"].tolist() == [200.0]


def test_unknown_part_needs_no_peak_but_a_demand_part_does():
    build_building_components(pd.DataFrame([
        _building("U1", use="Unknown", nonres=200.0, nonres_peak=0.0),
        _building("R1", use=None, res=100.0),
    ]))
    with pytest.raises(ValueError, match="positive non-residential peak"):
        validate_physical_buildings(pd.DataFrame([_building("C1", use="Commercial", nonres=150.0, nonres_peak=0.0)]))


def test_other_non_residential_uses_are_still_rejected():
    with pytest.raises(ValueError, match="Commercial, Public or Unknown"):
        validate_physical_buildings(pd.DataFrame([_building("X1", use="Industrial", nonres=150.0)]))


def test_mv_direct_unknown_part_is_accepted_without_component():
    physical = pd.DataFrame([
        _building("U1", use="Unknown", nonres=5000.0, direct=True),
        _building("R1", use=None, res=100.0),
    ])
    assert build_building_components(physical)["component_id"].tolist() == ["R1::residential"]
