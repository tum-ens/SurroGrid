"""Voltage deviation DB summaries and plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase

try:
    from .powerflow_heatmaps import (
        _normalize_optional_ags,
        _resolve_db_grid,
        _resolve_powerflow_run,
    )
except ImportError:
    from powerflow_heatmaps import (
        _normalize_optional_ags,
        _resolve_db_grid,
        _resolve_powerflow_run,
    )



def _grid_label_from_row(row: pd.Series) -> str:
    ags = str(int(row["ags"])).zfill(8)
    return f"{ags}-{int(row['plz'])}_{int(row['kcid'])}_{int(row['bcid'])}"

def voltage_deviation_summary_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stages: tuple[str, ...] = ("post",),
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Summarize DB voltage extrema for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    include all matching results, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
    db = SurroGridDatabase()
    run_id = None
    if input_id is not None:
        grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
        run_id = int(run["powerflow_run_id"])

    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.pylovo_grid_result_id,
               pbv.stage,
               MIN(pbv.vm_pu) AS min_vm_pu,
               MAX(pbv.vm_pu) AS max_vm_pu,
               COUNT(DISTINCT pbv.t_index) AS n_timesteps,
               COUNT(DISTINCT pbv.bus) AS n_buses
        FROM surrogrid.powerflow_bus_voltage pbv
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pbv.stage = ANY(:stages)
          AND (:run_id IS NULL OR pbv.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                 gc.pylovo_grid_result_id, pbv.stage
        ORDER BY pr.powerflow_run_id, pbv.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stages": list(stages),
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if summary.empty:
        compact_query = text(
            """
            SELECT pr.powerflow_run_id,
                   pr.run_name,
                   pr.scenario_id,
                   sc.scenario_key,
                   gc.ags,
                   gc.plz,
                   gc.kcid,
                   gc.bcid,
                   gc.pylovo_grid_result_id,
                   pbvs.stage,
                   MIN(pbvs.voltage_min_time_pu) AS min_vm_pu,
                   MAX(pbvs.voltage_max_time_pu) AS max_vm_pu,
                   MAX(pfs.n_timesteps) AS n_timesteps,
                   COUNT(DISTINCT pbvs.bus) AS n_buses
            FROM surrogrid.powerflow_bus_voltage_summary pbvs
            JOIN surrogrid.powerflow_summary pfs
              ON pfs.powerflow_run_id = pbvs.powerflow_run_id
             AND pfs.stage = pbvs.stage
            JOIN surrogrid.powerflow_run pr
              ON pr.powerflow_run_id = pbvs.powerflow_run_id
            JOIN surrogrid.scenario sc
              ON sc.scenario_id = pr.scenario_id
            JOIN surrogrid.grid_case gc
              ON gc.grid_case_id = pr.grid_case_id
            WHERE pr.run_name = :run_name
              AND pbvs.stage = ANY(:stages)
              AND (:run_id IS NULL OR pbvs.powerflow_run_id = :run_id)
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
              AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
              AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
            GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                     gc.pylovo_grid_result_id, pbvs.stage
            ORDER BY pr.powerflow_run_id, pbvs.stage
            """
        )
        with db.engine.connect() as conn:
            summary = pd.read_sql_query(
                compact_query,
                conn,
                params={
                    "run_name": run_name,
                    "stages": list(stages),
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "ags": _normalize_optional_ags(ags),
                    "filter_plz": plz if input_id is None else None,
                    "filter_kcid": kcid if input_id is None else None,
                    "filter_bcid": bcid if input_id is None else None,
                },
            )

    if summary.empty:
        raise ValueError(f"No DB voltage results found for run name {run_name!r}.")

    summary["grid"] = summary.apply(_grid_label_from_row, axis=1)
    return summary[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "n_timesteps",
            "n_buses",
            "min_vm_pu",
            "max_vm_pu",
        ]
    ].reset_index(drop=True)

def _format_pu_limit(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")

def plot_voltage_deviation_histogram(
    summary: pd.DataFrame,
    lower_limit: float = 0.9,
    upper_limit: float = 1.1,
    bin_size: float = 0.01,
    show: bool = True,
):
    lower_values = summary["min_vm_pu"].astype(float)
    upper_values = summary["max_vm_pu"].astype(float)
    lower_share = (lower_values < lower_limit).mean() * 100.0
    upper_share = (upper_values > upper_limit).mean() * 100.0

    x_min = min(float(lower_values.min()), lower_limit) - 0.03
    x_max = max(float(upper_values.max()), upper_limit) + 0.03

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=upper_values,
            name="Highest voltage per grid",
            marker={"color": "#2f92c5", "line": {"color": "white", "width": 0.5}},
            xbins={"start": x_min, "end": x_max, "size": bin_size},
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=lower_values,
            name="Lowest voltage per grid",
            marker={"color": "#66c2a4", "line": {"color": "white", "width": 0.5}},
            xbins={"start": x_min, "end": x_max, "size": bin_size},
            opacity=0.95,
        )
    )
    fig.add_vline(x=lower_limit, line_color="#3a3a3a", line_dash="dash", line_width=2)
    fig.add_vline(x=upper_limit, line_color="#3a3a3a", line_dash="dash", line_width=2)
    fig.add_annotation(
        x=lower_limit - 0.012,
        y=0.72,
        xref="x",
        yref="paper",
        text=f"< {_format_pu_limit(lower_limit)} p.u.: {lower_share:.1f}%",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )
    fig.add_annotation(
        x=upper_limit + 0.012,
        y=0.72,
        xref="x",
        yref="paper",
        text=f"> {_format_pu_limit(upper_limit)} p.u.: {upper_share:.1f}%",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )
    fig.update_layout(
        barmode="overlay",
        title="Voltage Magnitude Extremes Across LV Grids",
        xaxis_title="Grid-Level Voltage Extremum [p.u.]",
        yaxis_title="LV Grid Count (log scale)",
        yaxis={
            "type": "log",
            "rangemode": "tozero",
            "tickmode": "array",
            "tickvals": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            "ticktext": ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
            "minor": {"ticks": "outside"},
        },
        legend={"orientation": "h", "x": 0.02, "y": 1.12},
        margin={"l": 70, "r": 30, "t": 80, "b": 65},
        width=820,
        height=420,
    )
    fig.update_xaxes(range=[x_min, x_max], showgrid=True, gridcolor="#d8d8d8")
    fig.update_yaxes(showgrid=True, gridcolor="#d8d8d8")
    if show:
        fig.show()
    return fig

def plot_voltage_deviation_histogram_comparison(
    summaries: dict[str, pd.DataFrame],
    lower_limit: float = 0.9,
    upper_limit: float = 1.1,
    bin_size: float = 0.01,
    title: str = "Voltage Magnitude Extremes Across LV Grids",
    show: bool = True,
):
    """Plot voltage-extreme histograms for multiple stages in horizontal subplots."""
    if not summaries:
        raise ValueError("At least one summary dataframe is required.")

    cleaned: dict[str, tuple[pd.Series, pd.Series]] = {}
    x_min_values = [lower_limit]
    x_max_values = [upper_limit]
    for label, summary in summaries.items():
        if summary.empty:
            continue
        lower_values = summary["min_vm_pu"].astype(float).dropna()
        upper_values = summary["max_vm_pu"].astype(float).dropna()
        if lower_values.empty or upper_values.empty:
            continue
        cleaned[str(label)] = (lower_values, upper_values)
        x_min_values.append(float(lower_values.min()))
        x_max_values.append(float(upper_values.max()))
    if not cleaned:
        raise ValueError("No finite voltage summary values found.")

    x_min = min(x_min_values) - 0.03
    x_max = max(x_max_values) + 0.03
    fig = make_subplots(
        rows=1,
        cols=len(cleaned),
        subplot_titles=list(cleaned.keys()),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    colors = {
        "highest": "#2f92c5",
        "lowest": "#66c2a4",
    }
    for col_idx, (label, (lower_values, upper_values)) in enumerate(cleaned.items(), start=1):
        lower_share = (lower_values < lower_limit).mean() * 100.0
        upper_share = (upper_values > upper_limit).mean() * 100.0
        fig.add_trace(
            go.Histogram(
                x=upper_values,
                name="Highest voltage per grid",
                marker={"color": colors["highest"], "line": {"color": "white", "width": 0.5}},
                xbins={"start": x_min, "end": x_max, "size": bin_size},
                opacity=0.88,
                legendgroup="highest",
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Histogram(
                x=lower_values,
                name="Lowest voltage per grid",
                marker={"color": colors["lowest"], "line": {"color": "white", "width": 0.5}},
                xbins={"start": x_min, "end": x_max, "size": bin_size},
                opacity=0.88,
                legendgroup="lowest",
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_vline(x=lower_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=1, col=col_idx)
        fig.add_vline(x=upper_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=1, col=col_idx)
        fig.add_annotation(
            x=lower_limit - 0.012,
            y=0.82,
            xref="x" if col_idx == 1 else f"x{col_idx}",
            yref="paper",
            text=f"< {_format_pu_limit(lower_limit)} p.u.: {lower_share:.1f}%",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        )
        fig.add_annotation(
            x=upper_limit + 0.012,
            y=0.70,
            xref="x" if col_idx == 1 else f"x{col_idx}",
            yref="paper",
            text=f"> {_format_pu_limit(upper_limit)} p.u.: {upper_share:.1f}%",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        )

    fig.update_layout(
        barmode="overlay",
        title=title,
        yaxis_title="LV Grid Count (log scale)",
        legend={"orientation": "h", "x": 0.02, "y": 1.16},
        margin={"l": 70, "r": 30, "t": 90, "b": 65},
        width=max(820, 455 * len(cleaned)),
        height=440,
    )
    for col_idx in range(1, len(cleaned) + 1):
        fig.update_xaxes(title_text="Grid-Level Voltage Extremum [p.u.]", range=[x_min, x_max], showgrid=True, gridcolor="#d8d8d8", row=1, col=col_idx)
        fig.update_yaxes(
            type="log",
            rangemode="tozero",
            tickmode="array",
            tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
            minor={"ticks": "outside"},
            showgrid=True,
            gridcolor="#d8d8d8",
            row=1,
            col=col_idx,
        )
    if show:
        fig.show()
    return fig
