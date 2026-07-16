"""Asset-level power-flow comparison plots and cutoff visualizations."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def _powerflow_y_axis_slider_layout(y_axis_ranges: dict[str, list[float]]) -> dict[str, object]:
    layout: dict[str, object] = {}
    for axis_index, metric in enumerate(("Transformer", "Cables", "Voltage"), start=1):
        axis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
        if metric in y_axis_ranges:
            layout[f"{axis_name}.range"] = y_axis_ranges[metric]
            layout[f"{axis_name}.autorange"] = False
        else:
            layout[f"{axis_name}.autorange"] = True
    return layout


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

    filter_scope = str(filter_scope).strip().lower()
    if filter_scope in {"asset", "assets"}:
        filter_scope = "asset"
    elif filter_scope in {"grid", "grids"}:
        filter_scope = "grid"
    else:
        raise ValueError("filter_scope must be either 'asset' or 'grid'.")
    cutoff_unit = "asset" if filter_scope == "asset" else "grid"
    cutoff_units = "assets" if filter_scope == "asset" else "grids"

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0, 0.99, 0.95, 0.90, 0.50)
    asset_cutoff_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_cutoff_percentiles)
    if any(q <= 0 or q > 1 for q in asset_cutoff_percentiles):
        raise ValueError("asset_cutoff_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_percentiles)
    if any(q <= 0 or q > 1 for q in asset_percentiles):
        raise ValueError("asset_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    asset_percentiles = tuple(sorted(set(asset_percentiles).union(asset_cutoff_percentiles)))
    asset_cutoff_percentiles = tuple(sorted(set(asset_cutoff_percentiles), reverse=True))

    metric_order = ["Transformer", "Cables", "Voltage"]
    metric_lookup = {metric.lower(): metric for metric in metric_order}
    selected_metrics = []
    for metric in metrics:
        metric_key = metric_lookup.get(str(metric).strip().lower())
        if metric_key is None:
            available = ", ".join(metric_order)
            raise ValueError(f"Unsupported metric {metric!r}. Available: {available}.")
        if metric_key not in selected_metrics:
            selected_metrics.append(metric_key)

    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)
    critical_percentile = {"Transformer": "max", "Cables": "max", "Voltage": "min"}
    critical_direction = {"Transformer": "high", "Cables": "high", "Voltage": "low"}
    y_titles = {
        "Transformer": "Max loading [%]",
        "Cables": "Max loading [%]",
        "Voltage": "Min voltage [p.u.]",
    }
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]

    def _cutoff_label(cutoff: float) -> str:
        if np.isclose(cutoff, 1.0):
            return f"Show {cutoff_unit} cutoffs through P100"
        return f"Show {cutoff_unit} cutoffs through P{int(round(cutoff * 100)):02d}"

    def _asset_percentile_label(q: float) -> str:
        return f"P{int(round(q * 100)):02d}" if q < 1 else "P100"

    def _visible_asset_percentiles(cutoff: float) -> tuple[float, ...]:
        visible = tuple(q for q in asset_percentiles if q <= cutoff or np.isclose(q, cutoff))
        return visible or (cutoff,)

    def _grid_keys(frame: pd.DataFrame) -> pd.Series:
        if "powerflow_run_id" in frame.columns:
            key_cols = [
                col
                for col in ("powerflow_source", "comparison_group", "run_name", "stage", "powerflow_run_id")
                if col in frame.columns
            ]
        elif "grid" in frame.columns:
            key_cols = [col for col in ("comparison_group", "run_name", "stage", "grid") if col in frame.columns]
        else:
            raise ValueError("filter_scope='grid' requires a 'powerflow_run_id' or 'grid' column.")
        return frame[key_cols].astype("string").fillna("<NA>").agg("|".join, axis=1)

    def _retained_mask(values: pd.Series, metric_name: str, retained_fraction: float) -> pd.Series:
        values = values.astype(float)
        if np.isclose(retained_fraction, 1.0):
            return pd.Series(True, index=values.index)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(retained_fraction)
            return values <= threshold
        threshold = values.quantile(1 - retained_fraction)
        return values >= threshold

    def _retained_frame(group_df: pd.DataFrame, metric_name: str, cutoff: float) -> pd.DataFrame:
        values = group_df["value"].astype(float)
        if filter_scope == "asset":
            return group_df.loc[_retained_mask(values, metric_name, cutoff)].copy()

        row_keys = _grid_keys(group_df)
        reducer = "max" if critical_direction[metric_name] == "high" else "min"
        grid_values = values.groupby(row_keys, sort=False).agg(reducer)
        retained_grid_keys = set(grid_values.loc[_retained_mask(grid_values, metric_name, cutoff)].index)
        return group_df.loc[row_keys.isin(retained_grid_keys)].copy()

    def _retained_curve(group_df: pd.DataFrame, metric_name: str, x_values: tuple[float, ...]) -> pd.DataFrame:
        rows = []
        group_df = group_df.dropna(subset=["value"]).copy()
        if group_df.empty:
            return pd.DataFrame(rows)
        total_count = int(group_df["value"].size) if filter_scope == "asset" else int(_grid_keys(group_df).nunique())
        for retained_fraction in x_values:
            retained_df = _retained_frame(group_df, metric_name, retained_fraction)
            retained = retained_df["value"].astype(float).dropna()
            if retained.empty:
                continue
            retained_count = int(retained.size) if filter_scope == "asset" else int(_grid_keys(retained_df).nunique())
            rows.append(
                {
                    "retained_asset_cutoff": retained_fraction,
                    "center": float(retained.median() if center_stat == "median" else retained.mean()),
                    "band_lower": float(retained.min()),
                    "band_upper": float(retained.max()),
                    "retained_assets": retained_count,
                    "total_assets": total_count,
                }
            )
        return pd.DataFrame(rows)

    def _select_worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        plot_df = plot_df.dropna(subset=["value"]).copy()
        if plot_df.empty:
            return plot_df
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        grouped = plot_df.groupby(group_keys, sort=False, observed=True)["value"]
        if critical_direction[metric_name] == "high":
            value_index = grouped.idxmax()
        else:
            value_index = grouped.idxmin()
        return plot_df.loc[value_index.dropna()].reset_index(drop=True)

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
                (df["metric"] == metric) & (df["percentile_norm"] == critical_percentile[metric])
            ].dropna(subset=["value"]).copy()
            if metric_df.empty:
                continue

            for color_idx, (group, group_df) in enumerate(metric_df.groupby(group_col, sort=False)):
                values = group_df["value"].astype(float).dropna()
                if values.empty:
                    continue
                curve = _retained_curve(group_df, metric, x_values)
                if curve.empty:
                    continue
                group_label = str(group)
                color = default_colors.get(group_label, fallback_palette[color_idx % len(fallback_palette)])
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

                violin_df = _retained_frame(group_df, metric, cutoff)
                violin_df = _select_worst_asset_per_grid(violin_df, metric)
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
                title_text=y_titles[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=y_titles[metric],
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
        height=860,
        width=1500,
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

def plot_powerflow_asset_cutoff_overview_static(
    profile: pd.DataFrame,
    group_col: str | None = None,
    color_map: dict[str, str] | None = None,
    asset_cutoff_percentile: float = 1.0,
    asset_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str = "Power-Flow Stress by Retained-Asset Cutoff",
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
    center_stat: str = "mean",
    show_band: bool = False,
    worst_asset_per_grid: bool = True,
    filter_scope: str = "asset",
    source_col: str | None = None,
    source_style_map: dict[str, dict[str, object]] | None = None,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("svg", "pdf"),
):
    """Draw a publication-oriented static retained cutoff overview.

    The figure mirrors :func:`plot_powerflow_asset_cutoff_overview` without a
    slider. ``asset_cutoff_percentile`` selects the retained cutoff shown in the
    bottom row and the maximum cutoff shown in the top-row curves.
    ``filter_scope`` switches between asset-level and grid-level filtering.
    Static Matplotlib output can be saved as SVG/PDF through ``save_path``.
    ``source_col`` adds a second comparison dimension, e.g. Synthetic vs Real SWF:
    colors still follow ``group_col`` while line style and violin/scatter offsets
    follow ``source_col``.
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

    filter_scope = str(filter_scope).strip().lower()
    if filter_scope in {"asset", "assets"}:
        filter_scope = "asset"
    elif filter_scope in {"grid", "grids"}:
        filter_scope = "grid"
    else:
        raise ValueError("filter_scope must be either 'asset' or 'grid'.")
    cutoff_unit = "asset" if filter_scope == "asset" else "grid"

    cutoff = float(asset_cutoff_percentile)
    if cutoff > 1:
        cutoff = cutoff / 100
    if cutoff <= 0 or cutoff > 1:
        raise ValueError("asset_cutoff_percentile must satisfy 0 < value <= 1, or 0 < value <= 100.")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_percentiles)
    if any(q <= 0 or q > 1 for q in asset_percentiles):
        raise ValueError("asset_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    x_values = tuple(q for q in sorted(set(asset_percentiles).union({cutoff})) if q <= cutoff or np.isclose(q, cutoff))
    if not x_values:
        x_values = (cutoff,)

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"
    if source_col is not None and source_col not in df.columns:
        raise ValueError(f"source_col={source_col!r} is not present in the profile dataframe.")
    if source_col is None:
        source_col = "_plot_source"
        df[source_col] = ""
    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)

    metric_order = ["Transformer", "Cables", "Voltage"]
    metric_lookup = {metric.lower(): metric for metric in metric_order}
    selected_metrics = []
    for metric in metrics:
        metric_key = metric_lookup.get(str(metric).strip().lower())
        if metric_key is None:
            available = ", ".join(metric_order)
            raise ValueError(f"Unsupported metric {metric!r}. Available: {available}.")
        if metric_key not in selected_metrics:
            selected_metrics.append(metric_key)

    critical_percentile = {"Transformer": "max", "Cables": "max", "Voltage": "min"}
    critical_direction = {"Transformer": "high", "Cables": "high", "Voltage": "low"}
    y_titles = {
        "Transformer": "Max loading [%]",
        "Cables": "Max loading [%]",
        "Voltage": "Min voltage [p.u.]",
    }
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]
    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)

    def _grid_keys(frame: pd.DataFrame) -> pd.Series:
        if "powerflow_run_id" in frame.columns:
            key_cols = [
                col
                for col in ("powerflow_source", "comparison_group", "run_name", "stage", "powerflow_run_id")
                if col in frame.columns
            ]
        elif "grid" in frame.columns:
            key_cols = [col for col in ("comparison_group", "run_name", "stage", "grid") if col in frame.columns]
        else:
            raise ValueError("filter_scope='grid' requires a 'powerflow_run_id' or 'grid' column.")
        return frame[key_cols].astype("string").fillna("<NA>").agg("|".join, axis=1)

    def _retained_mask(values: pd.Series, metric_name: str, retained_fraction: float) -> pd.Series:
        values = values.astype(float)
        if np.isclose(retained_fraction, 1.0):
            return pd.Series(True, index=values.index)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(retained_fraction)
            return values <= threshold
        threshold = values.quantile(1 - retained_fraction)
        return values >= threshold

    def _retained_frame_at_cutoff(group_df: pd.DataFrame, metric_name: str, retained_fraction: float) -> pd.DataFrame:
        values = group_df["value"].astype(float)
        if filter_scope == "asset":
            return group_df.loc[_retained_mask(values, metric_name, retained_fraction)].copy()

        row_keys = _grid_keys(group_df)
        reducer = "max" if critical_direction[metric_name] == "high" else "min"
        grid_values = values.groupby(row_keys, sort=False).agg(reducer)
        retained_grid_keys = set(grid_values.loc[_retained_mask(grid_values, metric_name, retained_fraction)].index)
        return group_df.loc[row_keys.isin(retained_grid_keys)].copy()

    def _retained_curve(group_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        group_df = group_df.dropna(subset=["value"]).copy()
        rows = []
        for retained_fraction in x_values:
            retained_df = _retained_frame_at_cutoff(group_df, metric_name, retained_fraction)
            retained = retained_df["value"].astype(float).dropna()
            if retained.empty:
                continue
            rows.append(
                {
                    "retained_asset_cutoff": retained_fraction,
                    "center": float(retained.median() if center_stat == "median" else retained.mean()),
                    "band_lower": float(retained.min()),
                    "band_upper": float(retained.max()),
                }
            )
        return pd.DataFrame(rows)

    def _retained_frame(group_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        return _retained_frame_at_cutoff(group_df, metric_name, cutoff)

    def _select_worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        plot_df = plot_df.dropna(subset=["value"]).copy()
        if plot_df.empty:
            return plot_df
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        grouped = plot_df.groupby(group_keys, sort=False, observed=True)["value"]
        if critical_direction[metric_name] == "high":
            value_index = grouped.idxmax()
        else:
            value_index = grouped.idxmin()
        return plot_df.loc[value_index.dropna()].reset_index(drop=True)

    def _cutoff_label(value: float) -> str:
        return f"P{int(round(value * 100)):02d}" if value < 1 else "P100"

    def _wrap_axis_label(value: object, width: int = 18) -> str:
        text = str(value).replace("100-electrification-", "100% electrification ")
        text = text.replace("status-quo", "status quo")
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=True))

    def _short_source_label(source: str) -> str:
        lowered = str(source).strip().lower()
        if lowered in {"synthetic", "syn"}:
            return "Syn."
        if lowered in {"real swf", "real_swf", "real"}:
            return "Real"
        return str(source)[:6]

    title_fontsize = 20
    panel_title_fontsize = 17
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 15

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2,
        len(selected_metrics),
        figsize=(4.6 * len(selected_metrics), 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.34, "wspace": 0.28},
        squeeze=False,
    )
    groups = list(df[group_col].astype(str).dropna().drop_duplicates())
    sources = list(df[source_col].astype(str).dropna().drop_duplicates())
    source_style_map = source_style_map or {}
    default_source_styles = {
        "": {"linestyle": "-", "offset": 0.0, "alpha": 0.30, "marker_alpha": 0.42},
        "Synthetic": {"linestyle": "-", "offset": -0.18, "alpha": 0.30, "marker_alpha": 0.42},
        "Real SWF": {"linestyle": "--", "dashes": (3.0, 2.0), "offset": 0.18, "alpha": 0.18, "marker_alpha": 0.46},
        "synthetic": {"linestyle": "-", "offset": -0.18, "alpha": 0.30, "marker_alpha": 0.42},
        "real_swf": {"linestyle": "--", "dashes": (3.0, 2.0), "offset": 0.18, "alpha": 0.18, "marker_alpha": 0.46},
    }
    for key, value in source_style_map.items():
        default_source_styles[str(key)] = {**default_source_styles.get(str(key), {}), **value}
    group_colors = {
        group: default_colors.get(group, fallback_palette[index % len(fallback_palette)])
        for index, group in enumerate(groups)
    }

    def _source_style(source: str) -> dict[str, object]:
        return default_source_styles.get(str(source), {"linestyle": "--", "dashes": (3.0, 2.0), "offset": 0.18, "alpha": 0.18, "marker_alpha": 0.46})

    def _legend_label(group: str, source: str) -> str:
        return group if source == "" else f"{group} - {source}"

    for col_idx, metric in enumerate(selected_metrics):
        metric_df = df[
            (df["metric"] == metric)
            & (df["percentile_norm"] == critical_percentile[metric])
        ].dropna(subset=["value"]).copy()
        if metric_df.empty:
            continue

        ax_curve = axes[0, col_idx]
        ax_dist = axes[1, col_idx]
        for group in groups:
            for source in sources:
                group_df = metric_df[
                    (metric_df[group_col].astype(str) == group)
                    & (metric_df[source_col].astype(str) == source)
                ]
                values = group_df["value"].astype(float).dropna()
                if values.empty:
                    continue
                curve = _retained_curve(group_df, metric)
                color = group_colors[group]
                style = _source_style(source)
                (line,) = ax_curve.plot(
                    curve["retained_asset_cutoff"],
                    curve["center"],
                    marker="o",
                    linewidth=2.8,
                    markersize=6.5,
                    color=color,
                    linestyle=str(style.get("linestyle", "-")),
                    label=_legend_label(group, source),
                )
                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])
                if show_band:
                    ax_curve.fill_between(
                        curve["retained_asset_cutoff"].to_numpy(dtype=float),
                        curve["band_lower"].to_numpy(dtype=float),
                        curve["band_upper"].to_numpy(dtype=float),
                        color=color,
                        alpha=float(style.get("alpha", 0.13)),
                        linewidth=0,
                    )

        violin_values = []
        violin_positions = []
        violin_colors = []
        violin_alphas = []
        violin_sources = []
        base_positions = np.arange(1, len(groups) + 1)
        for group_index, group in enumerate(groups):
            for source in sources:
                group_df = metric_df[
                    (metric_df[group_col].astype(str) == group)
                    & (metric_df[source_col].astype(str) == source)
                ]
                retained = _select_worst_asset_per_grid(_retained_frame(group_df, metric), metric)
                values = retained["value"].astype(float).dropna().to_numpy()
                if values.size == 0:
                    continue
                style = _source_style(source)
                violin_values.append(values)
                violin_positions.append(float(base_positions[group_index] + float(style.get("offset", 0.0))))
                violin_colors.append(group_colors[group])
                violin_alphas.append(float(style.get("alpha", 0.28)))
                violin_sources.append(str(source))

        if violin_values:
            violins = ax_dist.violinplot(
                violin_values,
                positions=violin_positions,
                widths=0.30 if len(sources) > 1 else 0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body, color, alpha in zip(violins["bodies"], violin_colors, violin_alphas):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(alpha)
                body.set_linewidth(1.2)
            if "cmedians" in violins:
                violins["cmedians"].set_color("#222222")
                violins["cmedians"].set_linewidth(2.0)
            rng = np.random.default_rng(7)
            for position, values, color, alpha in zip(violin_positions, violin_values, violin_colors, violin_alphas):
                jitter = rng.normal(0, 0.022 if len(sources) > 1 else 0.035, size=values.size)
                ax_dist.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=18,
                    color=color,
                    alpha=min(0.65, alpha + 0.18),
                    linewidths=0,
                )
            ax_dist.set_xticks(base_positions)
            ax_dist.set_xticklabels([])
            ax_dist.tick_params(axis="x", length=0)
            for position, group in zip(base_positions, groups):
                ax_dist.text(
                    position,
                    -0.25 if len(sources) > 1 else -0.15,
                    _wrap_axis_label(group, width=18),
                    transform=ax_dist.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=tick_fontsize - 2,
                    color="#2f2f2f",
                    linespacing=1.05,
                )
            if len(sources) > 1:
                for position, source in zip(violin_positions, violin_sources):
                    ax_dist.text(
                        position,
                        -0.07,
                        _short_source_label(source),
                        transform=ax_dist.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=max(tick_fontsize - 4, 8),
                        color="#666666",
                    )

        ax_curve.set_title(metric, fontsize=panel_title_fontsize, fontweight="bold")
        ax_curve.set_xlabel("")
        ax_curve.set_ylabel(f"{center_stat.capitalize()} {y_titles[metric].lower()}", fontsize=label_fontsize)
        ax_curve.set_xticks(list(x_values))
        ax_curve.set_xticklabels([_cutoff_label(q) for q in x_values], rotation=35, ha="right", rotation_mode="anchor", fontsize=tick_fontsize)
        ax_dist.set_xlabel("")
        ax_dist.set_ylabel(y_titles[metric], fontsize=label_fontsize)
        if metric in y_axis_ranges:
            ax_curve.set_ylim(y_axis_ranges[metric])
            ax_dist.set_ylim(y_axis_ranges[metric])
        for ax in (ax_curve, ax_dist):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8)
            ax.grid(False, axis="x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and len(labels) > 1:
        legend_ncols = min(3, len(labels))
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=legend_ncols,
            frameon=False,
            bbox_to_anchor=(0.5, 0.915),
            fontsize=legend_fontsize,
            handlelength=2.8,
            columnspacing=1.2,
            labelspacing=0.65,
        )
    title_text = title if np.isclose(cutoff, 1.0) else f"{title} ({_cutoff_label(cutoff)} retained-{cutoff_unit} cutoff)"
    fig.suptitle(
        title_text,
        y=0.99,
        fontsize=title_fontsize,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.72, bottom=0.27, left=0.08, right=0.985, hspace=0.42, wspace=0.35)

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


def _normalize_time_quantile(value) -> str:
    if isinstance(value, str):
        return _normalize_percentile_label(value)
    quantile = float(value)
    if quantile == 0:
        return "min"
    if quantile == 1:
        return "max"
    if 0 < quantile < 1:
        return f"p{int(round(quantile * 100)):02d}"
    if 1 < quantile <= 100:
        return f"p{int(round(quantile)):02d}"
    raise ValueError("time_quantile must be 0..1, 0..100, or one of 'min'/'max'.")

def plot_powerflow_headline_violins(
    profile: pd.DataFrame,
    group_col: str | None = None,
    metric_config: dict[str, dict[str, object]] | None = None,
    asset_cutoff_percentiles: tuple[float, ...] | None = None,
    worst_asset_per_grid: bool = False,
    points: str | bool | None = "all",
    show: bool = True,
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
):
    """Plot one critical annual value per transformer, cable, or bus.

    By default, transformer and cable values use annual maximum loading and
    voltage values use annual minimum voltage. ``asset_cutoff_percentiles`` adds
    the same value-based outlier slider used by the retained-asset cutoff overview:
    P99 removes loading assets above global P99 and voltage assets below global P01
    before drawing.
    ``y_axis_limits`` optionally fixes the panels as
    ``(transformer_upper_percent, cable_upper_percent, voltage_lower_pu)``.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_headline_violins expects the asset-level percentile "
            f"profile dataframe; missing column(s): {missing}. Pass the dataframe "
            "loaded by powerflow_percentile_profile_db / "
            "real_powerflow_percentile_profile_db instead of the compact summary."
        )

    df = profile.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All assets"

    config = {
        "Transformer": {"time_quantile": 1.0, "asset_quantile": 1.0},
        "Cables": {"time_quantile": 1.0, "asset_quantile": 1.0},
        "Voltage": {"time_quantile": 0.0, "asset_quantile": 0.0},
    }
    critical_direction = {
        "Transformer": "high",
        "Cables": "high",
        "Voltage": "low",
    }
    if metric_config:
        metric_names = {metric.lower(): metric for metric in config}
        allowed_keys = {"time_quantile", "asset_quantile"}
        for metric, user_config in metric_config.items():
            metric_key = metric_names.get(str(metric).strip().lower())
            if metric_key is None:
                available = ", ".join(config)
                raise ValueError(f"Unsupported headline metric {metric!r}. Available: {available}.")
            if not isinstance(user_config, dict):
                raise ValueError(
                    "Each metric_config entry must be a dict with keys "
                    "'time_quantile' and/or 'asset_quantile'."
                )
            unknown_keys = set(user_config).difference(allowed_keys)
            if unknown_keys:
                unknown = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Unsupported metric_config key(s) for {metric_key}: {unknown}.")
            config[metric_key].update(user_config)

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0,)
    asset_cutoff_percentiles = tuple(
        float(q) / 100 if float(q) > 1 else float(q) for q in asset_cutoff_percentiles
    )
    if any(q <= 0 or q > 1 for q in asset_cutoff_percentiles):
        raise ValueError("asset_cutoff_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)

    def _cutoff_label(cutoff: float) -> str:
        return "All assets" if np.isclose(cutoff, 1.0) else f"P{int(round(cutoff * 100)):02d} cutoff"

    def _normalize_asset_quantile(value) -> float | None:
        if value is None:
            return None
        quantile = float(value)
        if quantile > 1:
            quantile = quantile / 100
        if not 0 <= quantile <= 1:
            raise ValueError("asset_quantile must be between 0 and 1, or 0 and 100.")
        return quantile

    def _filter_asset_quantile(plot_df: pd.DataFrame, metric_name: str, asset_quantile) -> pd.DataFrame:
        quantile = _normalize_asset_quantile(asset_quantile)
        if quantile is None or plot_df.empty:
            return plot_df
        direction = critical_direction[metric_name]
        if direction == "high" and quantile == 1:
            return plot_df
        if direction == "low" and quantile == 0:
            return plot_df
        frames = []
        for _, group_df in plot_df.groupby(group_col, sort=False):
            threshold = group_df["value"].quantile(quantile)
            if direction == "high":
                frames.append(group_df[group_df["value"] <= threshold])
            else:
                frames.append(group_df[group_df["value"] >= threshold])
        return pd.concat(frames, ignore_index=True) if frames else plot_df.iloc[0:0].copy()

    def _filter_by_value_cutoff(plot_df: pd.DataFrame, metric_name: str, cutoff: float) -> pd.DataFrame:
        if np.isclose(cutoff, 1.0) or plot_df.empty:
            return plot_df
        values = plot_df["value"].astype(float)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(cutoff)
            return plot_df[values <= threshold]
        threshold = values.quantile(1 - cutoff)
        return plot_df[values >= threshold]

    def _worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        direction = critical_direction[metric_name]
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        value_index = (
            plot_df.groupby(group_keys, sort=False)["value"].idxmax()
            if direction == "high"
            else plot_df.groupby(group_keys, sort=False)["value"].idxmin()
        )
        return plot_df.loc[value_index].reset_index(drop=True)

    y_titles = {
        "Transformer": "Annual loading [%]",
        "Cables": "Annual loading [%]",
        "Voltage": "Annual voltage [p.u.]",
    }
    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)
    metrics = [
        ("Transformer", y_titles["Transformer"]),
        ("Cables", y_titles["Cables"]),
        ("Voltage", y_titles["Voltage"]),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in metrics])
    traces_by_cutoff: list[list[int]] = []

    for cutoff_index, cutoff in enumerate(asset_cutoff_percentiles):
        is_visible = cutoff_index == 0
        cutoff_trace_indices: list[int] = []
        cutoff_label = _cutoff_label(cutoff)

        for col_idx, (metric, y_title) in enumerate(metrics, start=1):
            time_quantile = _normalize_time_quantile(config[metric]["time_quantile"])
            asset_quantile = config[metric]["asset_quantile"]
            plot_df = df[(df["metric"] == metric) & (df["percentile_norm"] == time_quantile)].copy()
            if plot_df.empty:
                available = sorted(df.loc[df["metric"] == metric, "percentile_norm"].dropna().unique())
                raise ValueError(f"No {metric} rows found for time_quantile {time_quantile!r}. Available: {available}.")
            plot_df = plot_df.dropna(subset=["value"])
            if asset_cutoff_percentiles == (1.0,):
                plot_df = _filter_asset_quantile(plot_df, metric, asset_quantile)
            plot_df = _filter_by_value_cutoff(plot_df, metric, cutoff)
            plot_df = _worst_asset_per_grid(plot_df, metric)
            hover_parts = []
            hover_labels = {
                "grid": "grid",
                "asset_label": "asset",
                "asset_id": "asset_id",
                "n_failed_timesteps": "failed_hours",
                "n_converged_timesteps": "converged_hours",
            }
            for col, label in hover_labels.items():
                if col in plot_df.columns:
                    hover_parts.append(label + ": " + plot_df[col].astype(str))
            if hover_parts:
                plot_df["hover_text"] = hover_parts[0]
                for part in hover_parts[1:]:
                    plot_df["hover_text"] = plot_df["hover_text"] + "<br>" + part
                plot_df["hover_text"] = plot_df["hover_text"] + "<br>" + cutoff_label
            else:
                plot_df["hover_text"] = metric + "<br>" + cutoff_label
            fig.add_trace(
                go.Violin(
                    x=plot_df[group_col].astype(str),
                    y=plot_df["value"].astype(float),
                    text=plot_df["hover_text"],
                    hovertemplate="%{text}<br>%{y:.4g}<extra></extra>",
                    box_visible=True,
                    meanline_visible=True,
                    points=points,
                    jitter=0.18,
                    scalemode="width",
                    name=f"{metric} {time_quantile}",
                    showlegend=False,
                    visible=is_visible,
                ),
                row=1,
                col=col_idx,
            )
            cutoff_trace_indices.append(len(fig.data) - 1)
            fig.update_yaxes(
                title_text=f"{time_quantile.upper()} {y_title}",
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            if metric in y_axis_ranges:
                fig.update_yaxes(range=y_axis_ranges[metric], row=1, col=col_idx)
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
                "args": [
                    {"visible": visible},
                    _powerflow_y_axis_slider_layout(y_axis_ranges),
                ],
            }
        )

    fig.update_layout(
        title="Headline Power-Flow Quality Metrics by Grid" if worst_asset_per_grid else "Headline Power-Flow Quality Metrics by Asset",
        violingap=0.12,
        height=470,
        margin={"l": 55, "r": 25, "t": 75, "b": 150},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Outlier filter: "},
                "x": 0.08,
                "len": 0.84,
                "y": -0.26,
                "pad": {"t": 65},
                "steps": slider_steps,
            }
        ] if len(asset_cutoff_percentiles) > 1 else None,
    )
    if show:
        fig.show()
    return fig
