"""Voltage deviation DB summaries and plotting helpers."""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots






def _format_pu_limit(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def plot_voltage_deviation_histogram_comparison(
    summaries: dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]],
    lower_limit: float = 0.9,
    upper_limit: float = 1.1,
    bin_size: float = 0.01,
    title: str = "Voltage Magnitude Extremes Across LV Grids",
    show: bool = True,
    ncols: int | None = None,
):
    """Plot voltage-extreme histograms for stages, optionally separated by source rows."""
    if not summaries:
        raise ValueError("At least one summary dataframe is required.")

    nested = all(isinstance(value, dict) for value in summaries.values())
    if nested:
        source_items = [(str(source), stage_map) for source, stage_map in summaries.items()]
    else:
        source_items = [("", summaries)]

    cleaned: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    x_min_values = [lower_limit]
    x_max_values = [upper_limit]
    stage_order: list[str] = []
    source_order: list[str] = []
    for source, stage_map in source_items:
        source_order.append(source)
        for label, summary in stage_map.items():
            if summary.empty:
                continue
            lower_values = pd.to_numeric(summary["min_vm_pu"], errors="coerce").dropna() if "min_vm_pu" in summary.columns else pd.Series(dtype=float)
            upper_values = pd.to_numeric(summary["max_vm_pu"], errors="coerce").dropna() if "max_vm_pu" in summary.columns else pd.Series(dtype=float)
            if lower_values.empty and upper_values.empty:
                continue
            cleaned[(source, str(label))] = (lower_values, upper_values)
            if str(label) not in stage_order:
                stage_order.append(str(label))
            if not lower_values.empty:
                x_min_values.append(float(lower_values.min()))
            if not upper_values.empty:
                x_max_values.append(float(upper_values.max()))
    if not cleaned:
        raise ValueError("No finite voltage summary values found.")

    x_min = min(x_min_values) - 0.03
    x_max = max(x_max_values) + 0.03
    if nested:
        nrows = len(source_order)
        ncols = len(stage_order)
    else:
        n_panels = len(stage_order)
        if ncols is None:
            ncols = n_panels
        ncols = max(1, min(int(ncols), n_panels))
        nrows = int(np.ceil(n_panels / ncols))
    subplot_titles = []
    for row_source in source_order if nested else [""]:
        for stage in stage_order:
            subplot_titles.append(stage if not row_source else f"{row_source}<br>{stage}")

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    colors = {"highest": "#2f92c5", "lowest": "#66c2a4"}
    panel_idx = 0
    for row_idx in range(1, nrows + 1):
        source = source_order[row_idx - 1] if nested else ""
        for col_idx in range(1, ncols + 1):
            if nested:
                if col_idx > len(stage_order):
                    continue
                label = stage_order[col_idx - 1]
            else:
                flat_idx = (row_idx - 1) * ncols + col_idx - 1
                if flat_idx >= len(stage_order):
                    continue
                label = stage_order[flat_idx]
            panel_idx += 1
            lower_values, upper_values = cleaned.get((source, label), (pd.Series(dtype=float), pd.Series(dtype=float)))
            if not upper_values.empty:
                upper_share = (upper_values > upper_limit).mean() * 100.0
                fig.add_trace(
                    go.Histogram(
                        x=upper_values,
                        name="Highest voltage per grid",
                        marker={"color": colors["highest"], "line": {"color": "white", "width": 0.5}},
                        xbins={"start": x_min, "end": x_max, "size": bin_size},
                        opacity=0.88,
                        legendgroup="highest",
                        showlegend=panel_idx == 1,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_annotation(
                    x=upper_limit + 0.012,
                    y=0.70,
                    xref="x" if panel_idx == 1 else f"x{panel_idx}",
                    yref="paper",
                    text=f"> {_format_pu_limit(upper_limit)} p.u.: {upper_share:.1f}%",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#d0d0d0",
                    borderwidth=1,
                )
            if not lower_values.empty:
                lower_share = (lower_values < lower_limit).mean() * 100.0
                fig.add_trace(
                    go.Histogram(
                        x=lower_values,
                        name="Lowest voltage per grid",
                        marker={"color": colors["lowest"], "line": {"color": "white", "width": 0.5}},
                        xbins={"start": x_min, "end": x_max, "size": bin_size},
                        opacity=0.88,
                        legendgroup="lowest",
                        showlegend=panel_idx == 1,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_annotation(
                    x=lower_limit - 0.012,
                    y=0.82,
                    xref="x" if panel_idx == 1 else f"x{panel_idx}",
                    yref="paper",
                    text=f"< {_format_pu_limit(lower_limit)} p.u.: {lower_share:.1f}%",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#d0d0d0",
                    borderwidth=1,
                )
            fig.add_vline(x=lower_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=row_idx, col=col_idx)
            fig.add_vline(x=upper_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=row_idx, col=col_idx)
            fig.update_xaxes(title_text="Voltage extremum [p.u.]", range=[x_min, x_max], showgrid=True, gridcolor="#d8d8d8", row=row_idx, col=col_idx)
            fig.update_yaxes(
                type="log",
                rangemode="tozero",
                tickmode="array",
                tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
                minor={"ticks": "outside"},
                showgrid=True,
                gridcolor="#d8d8d8",
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        barmode="overlay",
        title=title,
        yaxis_title="LV Grid Count (log scale)",
        legend={"orientation": "h", "x": 0.02, "y": 1.10},
        margin={"l": 70, "r": 30, "t": 110, "b": 65},
        width=max(820, 455 * ncols),
        height=max(440, 330 * nrows),
    )
    if show:
        fig.show()
    return fig
