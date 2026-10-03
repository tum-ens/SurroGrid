"""Publication figure of the expansion cost (plot_expansion_cost_overview_static)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridexpand.analysis.plotting.expansion_cost_plots import (
    plot_expansion_cost_overview_static,
    plot_expansion_cost_stacked_static,
)

CASES = ("status-quo", "INFLEX", "HEMS")
COLORS = {"status-quo": "#2E7D32", "INFLEX": "#D62728", "HEMS": "#1F77B4"}


def _costs() -> pd.DataFrame:
    rows = []
    for network, scale in (("Real", 1.0), ("Synthetic", 1.2)):
        for case, factor in zip(CASES, (0.0, 1.0, 0.5)):
            for component, cost in (("Cables", 2e6), ("Transformer exchange", 1e6), ("Load transfer", 0.1e6),
                                    ("New substations", 1.8e6), ("Voltage measures", 0.1e6)):
                rows.append({"stage": case, "data_source": network, "component": component,
                             "cost_eur": cost * factor * scale})
    return pd.DataFrame(rows)


def _figure(**kwargs):
    return plot_expansion_cost_overview_static(
        _costs(), stage_order=CASES, color_map=COLORS, stage_labels={"status-quo": "Status quo", "INFLEX": "No HEMS"},
        reduction_stages=("INFLEX", "HEMS"), **kwargs,
    )


def test_layout_legend_and_labels():
    font_size = plt.rcParams["font.size"]
    fig = _figure()
    assert plt.rcParams["font.size"] == font_size  # no global style change
    total, assets = fig.axes
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "Status quo", "No HEMS", "HEMS", "Synthetic", "Real",
    ]
    assert [label.get_text() for label in total.get_xticklabels()] == ["Status quo", "No HEMS", "HEMS"]
    assert [label.get_text() for label in total.get_xticklabels(minor=True)] == ["Syn.", "Real"] * 3
    assert [label.get_text() for label in assets.get_xticklabels()] == [
        "Cables", "Transformer\nexchange", "New\nsubstations", "Other",  # load transfer + voltage measures
    ]
    assert len(total.patches) == 6 and len(assets.patches) == 24
    assert np.isclose(assets.patches[-3].get_height(), 0.2)  # Other of real in INFLEX
    # Synthetic first: 1.2 x (2 + 1 + 2) M€ in INFLEX, half of it with HEMS.
    assert np.isclose(total.patches[2].get_height(), 6.0) and np.isclose(total.patches[3].get_height(), 5.0)
    labels = [text.get_text() for text in total.texts]
    assert "3.00\n−50 %" in labels and "2.50\n−50 %" in labels
    assert np.isclose(fig.get_size_inches()[0], 180 / 25.4)
    plt.close(fig)


def test_asset_panel_case_subset():
    fig = _figure(asset_stage_order=("INFLEX", "HEMS"), value_labels=False)
    total, assets = fig.axes
    assert len(total.patches) == 6 and len(assets.patches) == 16 and not assets.texts
    plt.close(fig)


def test_saves_every_format(tmp_path):
    fig = _figure(save_path=tmp_path / "cost", save_formats=("pdf", "svg"), title="Test")
    assert fig._suptitle.get_text() == "Test"
    assert (tmp_path / "cost.pdf").stat().st_size > 0
    assert "<text" in (tmp_path / "cost.svg").read_text()  # text stays text
    plt.close(fig)


def test_stacked_variant_stacks_the_assets_and_marks_the_reduction():
    fig = plot_expansion_cost_stacked_static(
        _costs(), stage_order=("INFLEX", "HEMS"), color_map=COLORS, reduction_stages=("INFLEX", "HEMS"),
    )
    (ax,) = fig.axes
    legend = fig.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "Cables", "Transformer exchange", "New substations", "Other",
    ]
    # Each swatch is split into the No HEMS (left) and HEMS shade (right).
    swatches = [patch for patch in legend.findobj(Rectangle) if patch is not legend.legendPatch]
    assert len(swatches) == 8 and swatches[0].get_facecolor()[:3] == to_rgb(COLORS["INFLEX"])
    assert [label.get_text() for label in ax.get_xticklabels(minor=True)] == ["Syn.", "Real"] * 2
    # Synthetic without HEMS: four segments up to 6 M€.
    synthetic_inflex = [patch for patch in ax.patches if np.isclose(patch.get_x() + patch.get_width() / 2, -0.2)]
    assert len(synthetic_inflex) == 4
    assert np.isclose(max(patch.get_y() + patch.get_height() for patch in synthetic_inflex), 6.0)
    # The saved cost as dotted boxes on the HEMS bars: synthetic 3 -> 6 M€, real 2.5 -> 5 M€.
    boxes = [patch for patch in ax.patches if not patch.get_fill()]
    assert [(patch.get_y(), patch.get_height()) for patch in boxes] == [(3.0, 3.0), (2.5, 2.5)]
    texts = [text.get_text() for text in ax.texts]
    assert texts.count("\u221250 %") == 2 and "Reduction through HEMS" in texts
    plt.close(fig)


def test_empty_input_returns_none():
    assert plot_expansion_cost_overview_static(pd.DataFrame(), stage_order=CASES, color_map=COLORS) is None
    assert plot_expansion_cost_stacked_static(pd.DataFrame(), stage_order=CASES, color_map=COLORS) is None
