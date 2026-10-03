"""Publication figure of the retained cutoff overview (plot_powerflow_asset_cutoff_overview_static)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gridexpand.analysis.plotting.powerflow_asset_plots import (
    _static_axis_limits,
    plot_powerflow_asset_cutoff_overview_static,
)

CASES = ("status-quo", "INFLEX", "HEMS")


def _profile() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for network in ("Real", "Synthetic"):
        for case_index, case in enumerate(CASES):
            for grid in range(8):
                name = f"{network[0]}{grid}"
                for metric, percentile, value in (
                    ("Transformer", "max", 30 + 40 * case_index + rng.normal(0, 10)),
                    ("Cables", "max", 20 + 30 * case_index + rng.normal(0, 8)),
                    ("Voltage", "min", 1.01 - 0.03 * case_index - abs(rng.normal(0, 0.01))),
                ):
                    rows.append({"grid": name, "run_name": f"{network}_{case}", "metric": metric,
                                 "percentile": percentile, "value": value, "comparison_stage": case,
                                 "network": network})
    frame = pd.DataFrame(rows)
    frame["comparison_stage"] = pd.Categorical(frame["comparison_stage"], categories=CASES, ordered=True)
    frame["network"] = pd.Categorical(frame["network"], categories=("Synthetic", "Real"), ordered=True)
    return frame


def _figure(**kwargs):
    return plot_powerflow_asset_cutoff_overview_static(
        _profile(), group_col="comparison_stage", source_col="network", asset_cutoff_percentile=0.95,
        filter_scope="grid", group_labels={"status-quo": "Status quo", "INFLEX": "No HEMS"},
        reference_group="status-quo", **kwargs,
    )


def test_pooled_layout_legend_and_labels():
    font_size = plt.rcParams["font.size"]
    fig = _figure()
    assert plt.rcParams["font.size"] == font_size  # no global style change
    assert len(fig.axes) == 6 and fig._suptitle is None
    legend = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend == ["Status quo", "No HEMS", "HEMS", "Synthetic", "Real"]
    bottom = fig.axes[3]
    assert [label.get_text() for label in bottom.get_xticklabels()] == ["Status quo", "No HEMS", "HEMS"]
    width, _ = fig.get_size_inches()
    assert np.isclose(width, 180 / 25.4)
    plt.close(fig)


def test_real_violins_have_a_visible_dashed_outline():
    fig = _figure()
    bottom = fig.axes[3]
    synthetic, real = [collection for collection in bottom.collections if isinstance(collection, PolyCollection)][:2]
    assert real.get_linestyle() != synthetic.get_linestyle()
    assert real.get_edgecolor()[0][3] == 1.0 and real.get_facecolor()[0][3] < 1.0  # dashed outline stays visible
    plt.close(fig)


def test_voltage_above_one_is_not_clipped():
    fig = _figure(distribution_y_axis_limits={"Voltage": (0.8, None)})
    voltage_dist = fig.axes[5]
    low, high = voltage_dist.get_ylim()
    assert np.isclose(low, 0.8) and high > 1.0
    plt.close(fig)


def test_curve_and_distribution_limits_are_set_per_row():
    limits = {"transformer": (0, 150), "Cables": (0, 60), "Voltage": (0.90, 1.00)}
    fig = _figure(curve_y_axis_limits=limits, distribution_y_axis_limits={"Cables": (0, 400)})
    assert [fig.axes[i].get_ylim() for i in range(3)] == [(0.0, 150.0), (0.0, 60.0), (0.9, 1.0)]
    assert fig.axes[4].get_ylim() == (0.0, 400.0)
    plt.close(fig)
    with pytest.raises(ValueError, match="low < high"):
        _figure(curve_y_axis_limits={"Voltage": (1.0, 0.9)})
    with pytest.raises(ValueError, match="Unsupported metric"):
        _static_axis_limits({"Power": (0, 1)}, "curve_y_axis_limits")


def test_saves_every_format(tmp_path):
    fig = _figure(save_path=tmp_path / "overview", save_formats=("pdf", "svg"), title="Test")
    assert fig._suptitle.get_text().startswith("Test")
    assert (tmp_path / "overview.pdf").stat().st_size > 0
    assert "<text" in (tmp_path / "overview.svg").read_text()  # text stays text
    plt.close(fig)
