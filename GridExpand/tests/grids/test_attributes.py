"""Building attributes from InfDB with pylovo's rules, and the whole-building MV-direct flag."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.grids import attributes


def infdb_rows(ids):
    rows = {
        "SFH": (100.0, 2, "Residential", "31001_1000", "SFH", 200.0, 0.0, None, None, None, "SFH"),
        "HALL": (1000.0, 2, "Commercial", "31001_2000", None, 0.0, 2000.0, None, None, "Commercial", "Commercial"),
        "MIXED": (400.0, 3, "Mixed", "31001_1000", None, 400.0, 1600.0, None, 3, "Commercial", "Mixed"),
    }
    return pd.DataFrame([(key, *rows[key]) for key in ids], columns=[
        "objectid", "floor_area", "floor_number", "building_use", "building_use_id", "building_type",
        "residential_floor_area", "nonresidential_floor_area", "occupants", "households", "nonresidential_use",
        "type"])


def test_attributes_come_from_infdb_only(monkeypatch):
    calls = []

    def infdb(engine, ids, schema="basedata"):
        calls.append((schema, list(ids)))
        return infdb_rows(ids)

    monkeypatch.setattr(attributes, "read_infdb_buildings", infdb)
    frame = attributes.building_attributes(None, ["SFH", "HALL", "SFH", "MIXED"])
    assert calls == [("basedata", ["HALL", "MIXED", "SFH"])]
    rows = frame.set_index("objectid")
    assert rows.loc["SFH", "households"] == 1 and rows.loc["SFH", "occupants"] == pytest.approx(2.03)
    assert rows.loc["HALL", "nonresidential_mv_direct"] is True
    assert frame["pylovo_grid_result_id"].isna().all()


def test_mv_direct_buildings_have_no_residential_share():
    derived = attributes.pylovo_equivalent_buildings(infdb_rows(["SFH", "HALL", "MIXED"]))
    flags = dict(zip(derived["objectid"], attributes.mv_direct_buildings(derived)))
    # MIXED: 1600 m2 x 79 W/m2 = 126 kW goes to MV, its flats stay on LV.
    assert flags == {"SFH": False, "HALL": True, "MIXED": False}
