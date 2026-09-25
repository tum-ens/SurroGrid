"""Retained-asset cutoff data preparation (gridexpand.analysis.plotting.cutoff_data)."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.analysis.plotting import cutoff_data as cd


def _frame():
    return pd.DataFrame(
        {
            "grid": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "powerflow_run_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "value": [10.0, 50.0, 20.0, 30.0, 90.0, 5.0, 40.0, None],
        }
    )


def test_normalizers():
    assert cd.normalize_filter_scope(" Grids ") == "grid"
    assert cd.normalize_quantiles((95, 0.5, 1), "q") == (0.95, 0.5, 1.0)
    assert cd.select_metrics(("voltage", "Cables", "cables")) == ["Voltage", "Cables"]
    with pytest.raises(ValueError):
        cd.normalize_quantiles((0,), "q")
    with pytest.raises(ValueError):
        cd.select_metrics(("power",))
    assert cd.color_defaults({"Real ÜZW": "#000000"})["Real ÜZW"] == "#000000"


def test_asset_and_grid_cutoffs():
    frame = _frame().dropna()
    assert cd.retained_frame(frame, "Cables", 1.0, "asset")["value"].tolist() == frame["value"].tolist()
    kept = cd.retained_frame(frame, "Cables", 0.5, "asset")["value"].tolist()
    assert kept == [10.0, 20.0, 30.0, 5.0]
    # Grid scope ranks grids by their maximum: a=50, b=30, c=90, d=40 -> P50 keeps b and d.
    assert cd.retained_frame(frame, "Cables", 0.5, "grid")["grid"].unique().tolist() == ["b", "d"]
    # Voltage is critical when low: grids ranked by minimum.
    assert cd.retained_frame(frame, "Voltage", 0.5, "grid")["grid"].unique().tolist() == ["b", "d"]


def test_curve_and_worst_asset():
    curve = cd.retained_curve(_frame(), "Cables", (1.0, 0.5), filter_scope="grid", center_stat="median", counts=True)
    assert curve["retained_assets"].tolist() == [4, 2] and curve["total_assets"].tolist() == [4, 4]
    assert curve.loc[0, "band_upper"] == 90.0
    worst = cd.select_worst_asset_per_grid(_frame(), "Cables", "comparison_group")
    assert worst.set_index("grid")["value"].to_dict() == {"a": 50.0, "b": 30.0, "c": 90.0, "d": 40.0}


def test_empty_group_in_grid_scope():
    empty = _frame().iloc[0:0]
    assert cd.grid_keys(empty).empty
    assert cd.retained_frame(empty, "Voltage", 0.95, "grid").empty
