"""Asset-level power-flow comparison plots and cutoff visualizations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cutoff_data import (
    CRITICAL_PERCENTILE,
    FALLBACK_PALETTE,
    Y_TITLES,
    color_defaults,
    normalize_filter_scope,
    normalize_quantiles,
    retained_curve,
    retained_frame,
    select_metrics,
    select_worst_asset_per_grid,
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(51, 92, 129, {float(alpha):.3f})"
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {float(alpha):.3f})"


def _powerflow_y_axis_ranges(
    y_axis_limits: tuple[float | None, float | None, float | None] | None,
) -> dict[str, list[float]]:
    if y_axis_limits is None:
        return {}
    if len(y_axis_limits) != 3:
        raise ValueError(
            "y_axis_limits must be (transformer_upper_percent, cable_upper_percent, voltage_lower_pu)."
        )

    transformer_upper, cable_upper, voltage_lower = y_axis_limits
    ranges: dict[str, list[float]] = {}
    if transformer_upper is not None:
        ranges["Transformer"] = [0.0, float(transformer_upper)]
    if cable_upper is not None:
        ranges["Cables"] = [0.0, float(cable_upper)]
    if voltage_lower is not None:
        ranges["Voltage"] = [float(voltage_lower), 1.0]
    return ranges


def plot_cable_max_loading_ecdf(
    profile: pd.DataFrame,
    *,
    group_col: str = "comparison_group",
    color_map: dict[str, str] | None = None,
    title: str = "Cable Maximum Loading ECDF",
    thresholds: tuple[float, ...] = (80.0, 100.0),
    show: bool = True,
) -> go.Figure:
    """Plot the ECDF of each cable's maximum loading for every comparison group."""
    required = {group_col, "metric", "percentile", "value"}
    missing = required.difference(profile.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(
            "plot_cable_max_loading_ecdf expects the asset-level percentile "
            f"profile dataframe; missing column(s): {missing_columns}."
        )

    cable_maxima = profile.loc[
        profile["metric"].eq("Cables") & profile["percentile"].eq("max")
    ].copy()
    cable_maxima["value"] = pd.to_numeric(cable_maxima["value"], errors="coerce")
    cable_maxima = cable_maxima.dropna(subset=[group_col, "value"])
    if cable_maxima.empty:
        raise ValueError("No cable maximum-loading values are available for the ECDF.")

    colors = color_map or {}
    figure = go.Figure()
    for group, group_frame in cable_maxima.groupby(group_col, sort=False, observed=True):
        values = np.sort(group_frame["value"].to_numpy(dtype=float))
        cumulative_share = np.arange(1, values.size + 1, dtype=float) / values.size * 100.0
        figure.add_trace(
            go.Scatter(
                x=values,
                y=cumulative_share,
                mode="lines",
                name=str(group),
                line={"width": 2.6, "color": colors.get(str(group))},
                hovertemplate=(
                    "Grid model: %{fullData.name}<br>"
                    "Cable maximum loading: %{x:.2f}%<br>"
                    "Share of cables at or below loading: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    for threshold in thresholds:
        figure.add_vline(
            x=float(threshold),
            line_dash="dash",
            line_color="#777777",
            annotation_text=f"{float(threshold):g}%",
            annotation_position="top right",
        )
    figure.update_layout(
        title=title,
        xaxis_title="Cable maximum loading [%]",
        yaxis_title="Share of cables at or below loading [%]",
        xaxis={"rangemode": "tozero"},
        yaxis={"range": [0, 100]},
        legend_title="Grid model",
        margin={"l": 80, "r": 35, "t": 70, "b": 65},
        height=460,
    )
    if show:
        figure.show()
    return figure


def plot_powerflow_asset_cutoff_overview(
    profile: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
    color_map: dict[str, str] | None = None,
    asset_percentiles: tuple[float, ...] | None = None,
    asset_cutoff_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str = "Power-Flow Stress by Retained-Asset Cutoff",
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
    center_stat: str = "median",
    show_band: bool = True,
    worst_asset_per_grid: bool = False,
    filter_scope: str = "asset",
):
    """Plot retained cutoff curves and matching asset distributions.

    Row 1 shows, for every retained cutoff, the selected center statistic and
    min-max range of the retained assets. Row 2 shows the distribution of the
    retained assets at the selected cutoff. ``filter_scope="asset"`` filters
    individual assets directly. ``filter_scope="grid"`` ranks whole grids by
    their most critical asset and then keeps all assets belonging to the
    retained grids. Set ``worst_asset_per_grid=True`` to draw only each grid's
    most critical retained transformer/cable/bus value in the violin row.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_asset_cutoff_overview expects the asset-level "
            f"percentile profile dataframe; missing column(s): {missing}."
        )

    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"median", "mean"}:
        raise ValueError("center_stat must be either 'median' or 'mean'.")
    center_label = center_stat.capitalize()

    filter_scope = normalize_filter_scope(filter_scope)
    cutoff_unit = "asset" if filter_scope == "asset" else "grid"
    cutoff_units = "assets" if filter_scope == "asset" else "grids"

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0, 0.99, 0.95, 0.90, 0.50)
    asset_cutoff_percentiles = normalize_quantiles(asset_cutoff_percentiles, "asset_cutoff_percentiles")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = normalize_quantiles(asset_percentiles, "asset_percentiles")
    asset_percentiles = tuple(sorted(set(asset_percentiles).union(asset_cutoff_percentiles)))
    asset_cutoff_percentiles = tuple(sorted(set(asset_cutoff_percentiles), reverse=True))

    selected_metrics = select_metrics(metrics)

    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)
    default_colors = color_defaults(color_map)

    def _cutoff_label(cutoff: float) -> str:
        if np.isclose(cutoff, 1.0):
            return f"Show {cutoff_unit} cutoffs through P100"
        return f"Show {cutoff_unit} cutoffs through P{int(round(cutoff * 100)):02d}"

    def _asset_percentile_label(q: float) -> str:
        return f"P{int(round(q * 100)):02d}" if q < 1 else "P100"

    def _visible_asset_percentiles(cutoff: float) -> tuple[float, ...]:
        visible = tuple(q for q in asset_percentiles if q <= cutoff or np.isclose(q, cutoff))
        return visible or (cutoff,)

    def _x_range(cutoff: float) -> list[float]:
        x_values = _visible_asset_percentiles(cutoff)
        lower = float(min(x_values))
        upper = float(max(x_values))
        if np.isclose(lower, upper):
            pad = 0.01 if upper >= 0.99 else min(0.01, upper / 2)
            return [max(0.0, lower - pad), min(1.0, upper + pad)]
        return [lower, upper]

    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)
    subplot_titles = selected_metrics + ["" for _ in selected_metrics]
    fig = make_subplots(
        rows=2,
        cols=len(selected_metrics),
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        row_heights=[0.56, 0.44],
    )

    def _axis_layout(cutoff: float) -> dict[str, object]:
        layout: dict[str, object] = {
            "autosize": False,
            "width": 1500,
            "height": 860,
            "margin": {"l": 60, "r": 25, "t": 100, "b": 110},
            "legend": {"title": {"text": f"{center_label}, range, distribution"}},
        }
        tickvals = _visible_asset_percentiles(cutoff)
        n_cols = len(selected_metrics)
        for col_idx, metric in enumerate(selected_metrics, start=1):
            top_xaxis = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            top_yaxis = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
            bottom_yaxis_index = n_cols + col_idx
            bottom_yaxis = "yaxis" if bottom_yaxis_index == 1 else f"yaxis{bottom_yaxis_index}"
            layout[f"{top_xaxis}.range"] = _x_range(cutoff)
            layout[f"{top_xaxis}.tickvals"] = [float(value) for value in tickvals]
            layout[f"{top_xaxis}.ticktext"] = [_asset_percentile_label(float(value)) for value in tickvals]
            for yaxis in (top_yaxis, bottom_yaxis):
                if metric in y_axis_ranges:
                    layout[f"{yaxis}.range"] = y_axis_ranges[metric]
                    layout[f"{yaxis}.autorange"] = False
                else:
                    layout[f"{yaxis}.autorange"] = True
        return layout

    traces_by_cutoff: list[list[int]] = []
    for cutoff_index, cutoff in enumerate(asset_cutoff_percentiles):
        is_visible = cutoff_index == 0
        cutoff_trace_indices: list[int] = []
        cutoff_label = _cutoff_label(cutoff)
        x_values = _visible_asset_percentiles(cutoff)

        for col_idx, metric in enumerate(selected_metrics, start=1):
            metric_df = df[
                (df["metric"] == metric) & (df["percentile_norm"] == CRITICAL_PERCENTILE[metric])
            ].dropna(subset=["value"]).copy()
            if metric_df.empty:
                continue

            for color_idx, (group, group_df) in enumerate(metric_df.groupby(group_col, sort=False)):
                values = group_df["value"].astype(float).dropna()
                if values.empty:
                    continue
                curve = retained_curve(
                    group_df, metric, x_values, filter_scope=filter_scope, center_stat=center_stat, counts=True
                )
                if curve.empty:
                    continue
                group_label = str(group)
                color = default_colors.get(group_label, FALLBACK_PALETTE[color_idx % len(FALLBACK_PALETTE)])
                customdata = np.column_stack(
                    [
                        curve["band_lower"].to_numpy(dtype=float),
                        curve["band_upper"].to_numpy(dtype=float),
                        curve["retained_assets"].to_numpy(dtype=int),
                        curve["total_assets"].to_numpy(dtype=int),
                    ]
                )

                if show_band:
                    for trace in (
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_upper"],
                            mode="lines",
                            line={"width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                            visible=is_visible,
                        ),
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_lower"],
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=_hex_to_rgba(color, 0.16),
                            name=f"{group_label}: range",
                            legendgroup=f"{group_label} retained asset range",
                            showlegend=col_idx == 1,
                            customdata=customdata,
                            hovertemplate=(
                                f"{cutoff_unit} cutoff %{{x:.0%}}<br>"
                                "range: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                                f"retained {cutoff_units}: %{{customdata[2]}} / %{{customdata[3]}}<br>"
                                f"{cutoff_label}<extra></extra>"
                            ),
                            visible=is_visible,
                        ),
                    ):
                        fig.add_trace(trace, row=1, col=col_idx)
                        cutoff_trace_indices.append(len(fig.data) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=curve["retained_asset_cutoff"],
                        y=curve["center"],
                        mode="lines+markers",
                        line={"color": color, "width": 2.7},
                        marker={"size": 6, "color": color},
                        name=f"{group_label}: {center_stat}",
                        legendgroup=f"{group_label} retained asset {center_stat}",
                        showlegend=col_idx == 1,
                        customdata=customdata,
                        hovertemplate=(
                            f"{cutoff_unit} cutoff %{{x:.0%}}<br>"
                            f"{center_stat}: %{{y:.4g}}<br>"
                            "range: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                            f"retained {cutoff_units}: %{{customdata[2]}} / %{{customdata[3]}}<br>"
                            f"{cutoff_label}<extra></extra>"
                        ),
                        visible=is_visible,
                    ),
                    row=1,
                    col=col_idx,
                )
                cutoff_trace_indices.append(len(fig.data) - 1)

                violin_df = retained_frame(group_df, metric, cutoff, filter_scope)
                if worst_asset_per_grid:
                    violin_df = select_worst_asset_per_grid(violin_df, metric, group_col)
                if violin_df.empty:
                    continue
                hover_parts = []
                for col, label in {
                    "grid": "grid",
                    "asset_label": "asset",
                    "asset_id": "asset_id",
                    "n_failed_timesteps": "failed_hours",
                }.items():
                    if col in violin_df.columns:
                        hover_parts.append(label + ": " + violin_df[col].astype(str))
                if hover_parts:
                    violin_df["hover_text"] = hover_parts[0]
                    for part in hover_parts[1:]:
                        violin_df["hover_text"] = violin_df["hover_text"] + "<br>" + part
                    violin_df["hover_text"] = violin_df["hover_text"] + "<br>" + cutoff_label
                else:
                    violin_df["hover_text"] = f"{group_label}<br>{cutoff_label}"
                fig.add_trace(
                    go.Violin(
                        x=violin_df[group_col].astype(str),
                        y=violin_df["value"].astype(float),
                        text=violin_df["hover_text"],
                        hovertemplate="%{text}<br>%{y:.4g}<extra></extra>",
                        box_visible=False,
                        meanline_visible=True,
                        points="all",
                        jitter=0.12,
                        width=0.5,
                        scalemode="width",
                        marker={"color": color, "opacity": 0.45, "size": 3.5},
                        line={"color": color, "width": 1.8},
                        fillcolor=_hex_to_rgba(color, 0.36),
                        opacity=0.82,
                        spanmode="hard",
                        name=f"{group_label}: distribution",
                        legendgroup=f"{group_label} retained asset distribution",
                        showlegend=col_idx == 1,
                        visible=is_visible,
                    ),
                    row=2,
                    col=col_idx,
                )
                cutoff_trace_indices.append(len(fig.data) - 1)

            fig.update_yaxes(
                title_text=Y_TITLES[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=Y_TITLES[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=2,
                col=col_idx,
            )
            if metric in y_axis_ranges:
                fig.update_yaxes(range=y_axis_ranges[metric], row=1, col=col_idx)
                fig.update_yaxes(range=y_axis_ranges[metric], row=2, col=col_idx)
            fig.update_xaxes(title_text=f"{cutoff_unit.capitalize()} cutoff", tickangle=-45, row=1, col=col_idx)
            fig.update_xaxes(title_text="", row=2, col=col_idx)
        traces_by_cutoff.append(cutoff_trace_indices)

    slider_steps = []
    n_traces = len(fig.data)
    for cutoff, cutoff_trace_indices in zip(asset_cutoff_percentiles, traces_by_cutoff):
        visible = [False] * n_traces
        for trace_index in cutoff_trace_indices:
            visible[trace_index] = True
        slider_steps.append(
            {
                "label": _cutoff_label(cutoff),
                "method": "update",
                "args": [{"visible": visible}, _axis_layout(cutoff)],
            }
        )

    if asset_cutoff_percentiles:
        fig.update_layout(_axis_layout(asset_cutoff_percentiles[0]))

    fig.update_layout(
        title={
            "text": (
                f"{title}<br>"
                f"<sup>Top: retained-{cutoff_unit} {center_stat} and min-max range. Bottom: distribution of assets retained by {cutoff_unit} cutoff.</sup>"
            )
        },
        autosize=False,
        legend={"title": {"text": f"{center_label}, range, distribution"}},
        height=1000,
        width=1800,
        margin={"l": 60, "r": 25, "t": 100, "b": 110},
        violingap=0.12,
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": f"Visible {cutoff_unit} cutoff range: "},
                "x": 0.08,
                "len": 0.84,
                "y": -0.08,
                "pad": {"t": 35},
                "steps": slider_steps,
            }
        ] if len(asset_cutoff_percentiles) > 1 else None,
    )
    if show:
        fig.show()
    return fig

# Line style and violin transparency of each network source, in the order the sources appear.
_SOURCE_STYLES = (
    {"linestyle": "-", "alpha": 0.40},
    {"linestyle": (0, (3.0, 1.6)), "alpha": 0.20},
    {"linestyle": "-.", "alpha": 0.30},
    {"linestyle": ":", "alpha": 0.25},
)


def _static_axis_limits(
    limits: Mapping[str, tuple[float | None, float | None]] | None, name: str
) -> dict[str, tuple[float | None, float | None]]:
    """``{metric: (low, high)}`` with canonical metric names; ``None`` bounds stay automatic."""
    result: dict[str, tuple[float | None, float | None]] = {}
    for metric, bounds in (limits or {}).items():
        (key,) = select_metrics((metric,))
        low, high = bounds
        if low is not None and high is not None and not float(low) < float(high):
            raise ValueError(f"{name}[{metric!r}] must be (low, high) with low < high, got {bounds!r}.")
        result[key] = (low, high)
    return result


def plot_powerflow_asset_cutoff_overview_static(
    profile: pd.DataFrame,
    group_col: str | None = None,
    color_map: dict[str, str] | None = None,
    asset_cutoff_percentile: float = 1.0,
    asset_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str | None = None,
    curve_y_axis_limits: Mapping[str, tuple[float | None, float | None]] | None = None,
    distribution_y_axis_limits: Mapping[str, tuple[float | None, float | None]] | None = None,
    center_stat: str = "mean",
    show_band: bool = False,
    worst_asset_per_grid: bool = True,
    filter_scope: str = "asset",
    source_col: str | None = None,
    source_style_map: dict[str, dict[str, object]] | None = None,
    group_labels: dict[str, str] | None = None,
    reference_group: str | None = None,
    width_mm: float = 180.0,
    height_mm: float | None = None,
    font_size: float = 8.0,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("pdf", "svg"),
):
    """Publication figure of the retained cutoff overview (Matplotlib, no slider).

    Top row: the ``center_stat`` of the retained critical values over the retained
    cutoffs up to ``asset_cutoff_percentile``. Bottom row: their distribution at that
    cutoff (violin, median, points). ``filter_scope`` ranks assets or whole grids.
    Colors follow ``group_col`` (the cases), line styles and violin offsets follow
    ``source_col`` (e.g. ``network`` for pooled Real vs Synthetic); the legend shows
    both. ``group_labels`` renames the groups for display, ``reference_group`` adds a
    dotted line at that group's median to every distribution panel.
    ``curve_y_axis_limits`` (top row) and ``distribution_y_axis_limits`` (bottom row)
    map a metric to ``(low, high)``; ``None`` or a missing metric keeps that bound
    automatic. Wide, fixed limits keep the scale the same across figures.

    The figure is sized for print (``width_mm``, ``font_size`` in pt, no title unless
    ``title`` is given; the caption carries it). PDF/SVG keep the text as text.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_asset_cutoff_overview_static expects the asset-level "
            f"percentile profile dataframe; missing column(s): {missing}."
        )
    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"median", "mean"}:
        raise ValueError("center_stat must be either 'median' or 'mean'.")
    filter_scope = normalize_filter_scope(filter_scope)
    cutoff_unit = "asset" if filter_scope == "asset" else "grid"
    cutoff = float(asset_cutoff_percentile)
    if cutoff > 1:
        cutoff = cutoff / 100
    if cutoff <= 0 or cutoff > 1:
        raise ValueError("asset_cutoff_percentile must satisfy 0 < value <= 1, or 0 < value <= 100.")
    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = normalize_quantiles(asset_percentiles, "asset_percentiles")
    x_values = tuple(q for q in sorted(set(asset_percentiles).union({cutoff})) if q <= cutoff or np.isclose(q, cutoff))

    df = profile.copy()
    group_col = group_col or "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"
    if source_col is not None and source_col not in df.columns:
        raise ValueError(f"source_col={source_col!r} is not present in the profile dataframe.")
    if source_col is None:
        source_col = "_plot_source"
        df[source_col] = ""
    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)
    selected_metrics = select_metrics(metrics)
    curve_limits = _static_axis_limits(curve_y_axis_limits, "curve_y_axis_limits")
    distribution_limits = _static_axis_limits(distribution_y_axis_limits, "distribution_y_axis_limits")

    def _order(column: str) -> list[str]:
        values = df[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            present = set(values.dropna().astype(str))
            return [str(value) for value in values.cat.categories if str(value) in present]
        return list(values.dropna().astype(str).drop_duplicates())

    groups, sources = _order(group_col), _order(source_col)
    base_colors = color_defaults(color_map)
    group_colors = {
        group: base_colors.get(group, FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)])
        for index, group in enumerate(groups)
    }
    violin_width = 0.8 / max(1, len(sources))
    source_styles = {}
    for index, source in enumerate(sources):
        style = {**_SOURCE_STYLES[index % len(_SOURCE_STYLES)],
                 "offset": (index - (len(sources) - 1) / 2) * violin_width}
        style.update((source_style_map or {}).get(source, {}))
        source_styles[source] = style
    group_labels = group_labels or {}

    def _group_label(group: str) -> str:
        text = str(group_labels.get(group, group)).replace("status-quo", "status quo")
        return "\n".join(textwrap.wrap(text, width=14, break_long_words=False)) or text

    def _cutoff_label(value: float) -> str:
        return f"P{int(round(value * 100)):02d}" if value < 1 else "P100"

    positions_by_cutoff = {q: index for index, q in enumerate(x_values)}

    def cutoff_position(value: float) -> int:
        return next(index for q, index in positions_by_cutoff.items() if np.isclose(q, value))

    width_in = width_mm / 25.4
    size = figsize or (width_in, (height_mm / 25.4) if height_mm else width_in * 0.56)
    rc = {
        "font.size": font_size, "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1, "legend.fontsize": font_size,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            2, len(selected_metrics), figsize=size, squeeze=False,
            gridspec_kw={"height_ratios": [1.0, 1.25]},
        )
        rng = np.random.default_rng(7)
        positions = np.arange(1, len(groups) + 1)
        for col_idx, metric in enumerate(selected_metrics):
            ax_curve, ax_dist = axes[0, col_idx], axes[1, col_idx]
            metric_df = df[
                (df["metric"] == metric) & (df["percentile_norm"] == CRITICAL_PERCENTILE[metric])
            ].dropna(subset=["value"])
            reference_values = []
            for group_index, group in enumerate(groups):
                for source in sources:
                    group_df = metric_df[
                        (metric_df[group_col].astype(str) == group) & (metric_df[source_col].astype(str) == source)
                    ]
                    if group_df.empty:
                        continue
                    color, style = group_colors[group], source_styles[source]
                    curve = retained_curve(group_df, metric, x_values, filter_scope=filter_scope, center_stat=center_stat)
                    # Evenly spaced cutoffs: P90/P95/P99 would overlap on a linear axis.
                    curve_x = curve["retained_asset_cutoff"].map(cutoff_position)
                    ax_curve.plot(
                        curve_x, curve["center"], color=color,
                        linestyle=style["linestyle"], marker="o", markersize=2.6,
                    )
                    if show_band:
                        ax_curve.fill_between(
                            curve_x.to_numpy(dtype=float),
                            curve["band_lower"].to_numpy(dtype=float), curve["band_upper"].to_numpy(dtype=float),
                            color=color, alpha=float(style["alpha"]) * 0.5, linewidth=0,
                        )
                    retained = retained_frame(group_df, metric, cutoff, filter_scope)
                    if worst_asset_per_grid:
                        retained = select_worst_asset_per_grid(retained, metric, group_col)
                    values = retained["value"].astype(float).dropna().to_numpy()
                    if values.size == 0:
                        continue
                    if group == reference_group:
                        reference_values.append(values)
                    position = positions[group_index] + float(style["offset"])
                    parts = ax_dist.violinplot(
                        [values], positions=[position], widths=violin_width * 0.92,
                        showmeans=False, showmedians=True, showextrema=False,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(float(style["alpha"]))
                        body.set_linewidth(0.8)
                        body.set_linestyle(style["linestyle"])
                    parts["cmedians"].set_color("#222222")
                    parts["cmedians"].set_linewidth(1.0)
                    jitter = rng.normal(0.0, violin_width * 0.08, size=values.size)
                    ax_dist.scatter(position + jitter, values, s=3, color=color,
                                    alpha=min(0.8, float(style["alpha"]) + 0.3), linewidths=0)
            if reference_values:
                ax_dist.axhline(float(np.median(np.concatenate(reference_values))), color=group_colors[reference_group],
                                linestyle=":", linewidth=0.8, zorder=0)
            ax_curve.set_title(metric, fontweight="bold")
            ax_curve.set_xticks(list(range(len(x_values))))
            ax_curve.set_xticklabels([_cutoff_label(q) for q in x_values], rotation=45, ha="right",
                                     rotation_mode="anchor")
            ax_curve.set_xlabel(f"Retained {cutoff_unit} cutoff")
            ax_curve.set_ylabel(f"{center_stat.capitalize()} {Y_TITLES[metric].lower()}")
            ax_dist.set_xticks(positions)
            ax_dist.set_xticklabels([_group_label(group) for group in groups])
            ax_dist.set_xlim(0.4, len(groups) + 0.6)
            ax_dist.tick_params(axis="x", length=0)
            ax_dist.set_ylabel(Y_TITLES[metric])
            ax_curve.set_ylim(*curve_limits.get(metric, (None, None)))
            ax_dist.set_ylim(*distribution_limits.get(metric, (None, None)))
            for ax in (ax_curve, ax_dist):
                if metric == "Voltage":
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
                ax.grid(False, axis="x")

        handles = [Patch(facecolor=group_colors[group], edgecolor="none", label=_group_label(group).replace("\n", " "))
                   for group in groups]
        if len(sources) > 1:
            handles += [Line2D([0], [0], color="#333333", linestyle=source_styles[source]["linestyle"], label=source)
                        for source in sources]
        # Panels fill the figure; legend and title sit on top of them (saved with bbox_inches="tight").
        fig.tight_layout(h_pad=1.2, w_pad=1.0)
        top = max(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
                  for ax in axes[0])
        if len(handles) > 1:
            legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                                bbox_to_anchor=(0.5, top), handlelength=2.4, columnspacing=1.4, borderaxespad=0.2)
            top = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
        if title:
            suffix = "" if np.isclose(cutoff, 1.0) else f" ({_cutoff_label(cutoff)} retained-{cutoff_unit} cutoff)"
            fig.suptitle(f"{title}{suffix}", y=top, va="bottom", fontweight="bold")

        if save_path is not None:
            save_path = Path(save_path)
            base_path = save_path.with_suffix("") if save_path.suffix else save_path
            base_path.parent.mkdir(parents=True, exist_ok=True)
            for image_format in save_formats:
                fig.savefig(base_path.with_suffix(f".{image_format.lstrip('.')}"), bbox_inches="tight")
    return fig


def _normalize_percentile_label(value) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"max", "min"}:
            return text
        if text.startswith("p"):
            suffix = text[1:]
            return f"p{int(suffix):02d}" if suffix.isdigit() else text
        return f"p{int(text):02d}" if text.isdigit() else f"p{text}"
    return f"p{int(value):02d}"


def plot_cable_capacity_current_loading_comparison(
    cable_data: pd.DataFrame,
    *,
    stage_order: tuple[str, ...],
    color_map: dict[str, str],
    asset_percentiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    title: str = "Cable Capacity, Peak Current, and Loading",
    figsize: tuple[float, float] = (16.0, 5.4),
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("svg", "pdf"),
):
    """Compare cable capacity, annual peak current, and annual peak loading."""
    required = {
        "data_source",
        "comparison_stage",
        "installed_capacity_a",
        "max_current_a",
        "max_loading_percent",
    }
    missing = required.difference(cable_data.columns)
    if missing:
        raise ValueError(f"Cable decomposition data is missing: {', '.join(sorted(missing))}.")
    if cable_data.empty:
        raise ValueError("No cable decomposition rows are available.")

    percentiles = tuple(
        sorted({float(value) / 100.0 if float(value) > 1 else float(value) for value in asset_percentiles})
    )
    if not percentiles or any(value <= 0 or value > 1 for value in percentiles):
        raise ValueError("asset_percentiles must satisfy 0 < value <= 1, or 0 < value <= 100.")

    metrics = (
        ("installed_capacity_a", "Installed capacity [A]"),
        ("max_current_a", "Annual peak current [A]"),
        ("max_loading_percent", "Annual max loading [%]"),
    )
    source_styles = {
        "Synthetic": {"linestyle": "-", "marker": "o"},
        "Real": {"linestyle": "--", "marker": "s"},
        "Real SWF": {"linestyle": "--", "marker": "s"},
        "Synthetic SWF": {"linestyle": "-", "marker": "o"},
        "Synthetic ÜZW": {"linestyle": "-.", "marker": "^"},
        "Real ÜZW": {"linestyle": ":", "marker": "D"},
    }
    stages = [
        stage for stage in stage_order if stage in set(cable_data["comparison_stage"].astype(str))
    ]
    sources = [
        source for source in source_styles
        if source in set(cable_data["data_source"].astype(str))
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, (column, y_label) in zip(axes, metrics):
        for stage in stages:
            for source in sources:
                values = pd.to_numeric(
                    cable_data.loc[
                        cable_data["comparison_stage"].astype(str).eq(stage)
                        & cable_data["data_source"].astype(str).eq(source),
                        column,
                    ],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                style = source_styles[source]
                curve = [float(values.quantile(percentile)) for percentile in percentiles]
                ax.plot(
                    percentiles,
                    curve,
                    color=color_map.get(stage, "#555555"),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.3,
                    markersize=5.5,
                    label=f"{stage} - {source}",
                )
        ax.set_xlabel("Asset percentile")
        ax.set_ylabel(y_label)
        ax.set_xticks(percentiles)
        ax.set_xticklabels(
            [f"P{int(round(value * 100)):02d}" for value in percentiles],
            rotation=35,
            ha="right",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
        ax.grid(False, axis="x")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.91),
            handlelength=2.8,
        )
    fig.suptitle(title, y=0.995, fontsize=17, fontweight="bold")
    fig.subplots_adjust(top=0.72, bottom=0.20, left=0.065, right=0.985, wspace=0.28)

    if save_path is not None:
        base_path = Path(save_path).with_suffix("")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        for image_format in save_formats:
            fig.savefig(base_path.with_suffix(f".{image_format.lstrip('.')}"), bbox_inches="tight")
    return fig
