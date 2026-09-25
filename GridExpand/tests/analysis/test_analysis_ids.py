"""Shared identifiers and labels (gridexpand.analysis.ids)."""

from __future__ import annotations

import pytest

from gridexpand.analysis import ids


@pytest.mark.parametrize(
    "value, expected",
    [("LV_007", "7"), ("007", "7"), (7, "7"), (7.0, "7"), ("7.0", "7"), (" LV_113 ", "113"),
     ("area-12", "12"), ("area_12", "12"), ("north-3", "north-3"), ("3.5", "3.5")],
)
def test_canonical_real_grid_id(value, expected):
    assert ids.canonical_real_grid_id(value) == expected


def test_labels():
    assert ids.ags_string(9184137) == "09184137" and ids.ags_string("09184137") == "09184137"
    assert ids.optional_ags(None) is None and ids.optional_ags("0918") == 918
    assert ids.synthetic_grid_label({"ags": 9184137, "plz": 85653, "kcid": 1, "bcid": -1}) == "09184137-85653_1_-1"
    assert ids.real_grid_label("uzw", "12") == "ÜZW area-0012"
    assert ids.real_grid_label("swf", "LV_7") == "SWF LV_007"
