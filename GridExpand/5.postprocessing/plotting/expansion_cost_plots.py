"""Expansion-cost plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_expansion_cost_comparison_bar(
    expansion_cost_comparison: pd.DataFrame,
    *,
    stage_order: Sequence[str],
    color_map: Mapping[str, str],
    output_path: str | Path | None = None,
    title: str = "Heuristic Expansion Cost by Asset Type",
    show: bool = True,
):
    """Plot total cable and transformer expansion cost by stage."""
    if expansion_cost_comparison.empty:
        print("No post-electrification expansion cost summaries available yet.")
        return None

    cost_data = expansion_cost_comparison.copy()
    if "data_source" not in cost_data.columns:
        cost_data["data_source"] = "Synthetic"
    plotted_stages = [stage for stage in stage_order if stage in set(cost_data["stage"])]
    sources = [str(source) for source in cost_data["data_source"].dropna().drop_duplicates()]
    components = ["Cables", "Transformers"]
    fig, ax = plt.subplots(figsize=(max(7.2, 2.7 * len(plotted_stages) * max(len(sources), 1)), 4.4))
    x_positions = range(len(components))
    series = [(stage, source) for stage in plotted_stages for source in sources]
    bar_width = min(0.22, 0.72 / max(len(series), 1))
    offsets = {
        item: (index - (len(series) - 1) / 2) * bar_width
        for index, item in enumerate(series)
    }
    hatch_map = {"Synthetic": "", "Real SWF": "//"}

    for stage, source in series:
        values_million_eur = []
        for component in components:
            match = cost_data[
                (cost_data["stage"] == stage)
                & (cost_data["data_source"].astype(str) == source)
                & (cost_data["component"] == component)
            ]
            values_million_eur.append(float(match["cost_eur"].sum()) / 1_000_000.0)
        bars = ax.bar(
            [position + offsets[(stage, source)] for position in x_positions],
            values_million_eur,
            width=bar_width * 0.92,
            label=f"{stage} - {source}",
            color=color_map.get(stage),
            hatch=hatch_map.get(source, ".."),
            edgecolor="#444444" if hatch_map.get(source, "") else None,
            linewidth=0.6 if hatch_map.get(source, "") else 0.0,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(components)
    ax.set_ylabel("Estimated cost [M€]")
    ax.set_title(title)
    ax.legend(frameon=False, title="Case")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if output_path is not None:
        base_path = Path(output_path).with_suffix("")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    return fig
