"""Plots for the graph-normalized synthetic/real feeder audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_COLORS = {"Synthetic": "#326891", "Real SWF": "#D07A32"}


def _save(fig, save_path: str | Path | None, save_formats: tuple[str, ...]) -> None:
    if save_path is None:
        return
    base = Path(save_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    for image_format in save_formats:
        fig.savefig(
            base.with_suffix(f".{image_format.lstrip('.')}"), bbox_inches="tight"
        )


def plot_feeder_structure_comparison(
    grid_metrics: pd.DataFrame,
    *,
    title: str = "Feeder Structure: Synthetic vs Real SWF",
    figsize: tuple[float, float] = (14.5, 8.0),
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("svg", "pdf"),
):
    """Show grid-level feeder, demand concentration, capacity, and depth metrics."""
    metrics = (
        ("outgoing_feeders", "Outgoing feeders", 1.0),
        ("feeder_sections", "Feeder sections", 1.0),
        (
            "median_downstream_demand_per_capacity_kwh_per_a",
            "Median downstream demand\n[kWh/a per A]",
            1.0,
        ),
        (
            "downstream_demand_weighted_capacity_a",
            "Demand-weighted capacity [A]",
            1.0,
        ),
        (
            "demand_weighted_section_depth",
            "Demand-weighted path depth\n[sections]",
            1.0,
        ),
        ("demand_weighted_path_length_km", "Demand-weighted path length [km]", 1.0),
    )
    sources = [
        source
        for source in ("Synthetic", "Real SWF")
        if source in set(grid_metrics["data_source"])
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, (column, label, scale) in zip(axes.ravel(), metrics):
        values = [
            pd.to_numeric(
                grid_metrics.loc[grid_metrics["data_source"].eq(source), column],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
            * scale
            for source in sources
        ]
        boxes = ax.boxplot(
            values,
            tick_labels=sources,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 2.0},
        )
        for patch, source in zip(boxes["boxes"], sources):
            patch.set_facecolor(SOURCE_COLORS[source])
            patch.set_alpha(0.55)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.grid(False, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.995)
    fig.subplots_adjust(
        top=0.90, bottom=0.09, left=0.075, right=0.985, hspace=0.38, wspace=0.30
    )
    _save(fig, save_path, save_formats)
    return fig


def plot_feeder_section_loading_percentiles(
    section_loading: pd.DataFrame,
    *,
    stage_order: tuple[str, ...],
    color_map: dict[str, str],
    percentiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    title: str = "Physical Feeder-Section Maximum Loading",
    figsize: tuple[float, float] = (9.5, 5.5),
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("svg", "pdf"),
):
    """Compare section bottleneck-loading percentiles after excluding control lines."""
    sources = [
        source
        for source in ("Synthetic", "Real SWF")
        if source in set(section_loading["data_source"])
    ]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(percentiles, dtype=float)
    for stage in stage_order:
        for source in sources:
            values = pd.to_numeric(
                section_loading.loc[
                    section_loading["comparison_stage"].eq(stage)
                    & section_loading["data_source"].eq(source),
                    "section_max_loading_percent",
                ],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            ax.plot(
                x,
                [values.quantile(value) for value in x],
                color=color_map.get(stage, "#555555"),
                linestyle="-" if source == "Synthetic" else "--",
                marker="o" if source == "Synthetic" else "s",
                linewidth=2.4,
                markersize=5.5,
                label=f"{stage} - {source}",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{round(value * 100):02d}" for value in x])
    ax.set_xlabel("Feeder-section percentile")
    ax.set_ylabel("Annual maximum loading [%]")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path, save_formats)
    return fig
