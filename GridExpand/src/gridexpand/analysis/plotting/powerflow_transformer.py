"""Transformer import distributions and stage-comparison plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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

def transformer_import_distribution_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stage: str = "post",
    reactive_magnitude: bool = True,
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read transformer import time series for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    aggregate all matching DB runs, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
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
               pi.stage,
               pi.ts,
               pi.t_index,
               pi.p_mw,
               pi.q_mvar
        FROM surrogrid.powerflow_import pi
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pi.stage = :stage
          AND (:run_id IS NULL OR pi.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        ORDER BY pr.powerflow_run_id, pi.t_index
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stage": stage,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if df.empty:
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
                   ptd.stage,
                   ptd.diagnostic,
                   ptd.point_index,
                   ptd.x_value,
                   ptd.t_index,
                   ptd.ts,
                   ptd.p_mw,
                   ptd.q_mvar,
                   ptd.q_abs_mvar,
                   ptd.s_mva,
                   ptd.mean_s_mva,
                   ptd.max_s_mva
            FROM surrogrid.powerflow_transformer_diagnostic ptd
            JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
            JOIN surrogrid.scenario sc USING (scenario_id)
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND ptd.stage = :stage
              AND (:run_id IS NULL OR ptd.powerflow_run_id = :run_id)
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
              AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
              AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
            ORDER BY pr.powerflow_run_id, ptd.diagnostic, ptd.point_index
            """
        )
        with db.engine.connect() as conn:
            df = pd.read_sql_query(
                compact_query,
                conn,
                params={
                    "run_name": run_name,
                    "stage": stage,
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "ags": _normalize_optional_ags(ags),
                    "filter_plz": plz if input_id is None else None,
                    "filter_kcid": kcid if input_id is None else None,
                    "filter_bcid": bcid if input_id is None else None,
                },
            )
        if df.empty:
            raise ValueError(f"No DB transformer import results found for run name {run_name!r}.")
        df["grid"] = df.apply(_grid_label_from_row, axis=1)
        df["q_import_mvar"] = df["q_abs_mvar"] if reactive_magnitude else df["q_mvar"]
        df["s_import_mva"] = df["s_mva"]
        mean_s = df["mean_s_mva"].replace(0.0, np.nan)
        max_s_by_grid = df.groupby("powerflow_run_id")["max_s_mva"].first().replace(0.0, np.nan)
        ldc_scale = float(max_s_by_grid.mean())
        if not np.isfinite(ldc_scale) or ldc_scale == 0.0:
            ldc_scale = np.nan
        df["p_ts_norm"] = df["p_mw"] / mean_s
        df["q_ts_norm"] = df["q_import_mvar"] / mean_s
        df["s_ts_norm"] = df["s_import_mva"] / mean_s
        df["p_ldc_norm"] = df["p_mw"] / ldc_scale
        df["q_ldc_norm"] = df["q_import_mvar"] / ldc_scale
        df["s_ldc_norm"] = df["s_import_mva"] / ldc_scale
        df.attrs["ldc_scale_mva"] = ldc_scale
        return df.reset_index(drop=True)

    df["grid"] = df.apply(_grid_label_from_row, axis=1)
    df["q_import_mvar"] = df["q_mvar"].abs() if reactive_magnitude else df["q_mvar"]
    df["s_import_mva"] = np.hypot(df["p_mw"].astype(float), df["q_mvar"].astype(float))

    mean_s = df.groupby("powerflow_run_id")["s_import_mva"].transform("mean").replace(0.0, np.nan)
    max_s_by_grid = df.groupby("powerflow_run_id")["s_import_mva"].max().replace(0.0, np.nan)
    ldc_scale = float(max_s_by_grid.mean())
    if not np.isfinite(ldc_scale) or ldc_scale == 0.0:
        ldc_scale = np.nan
    df["p_ts_norm"] = df["p_mw"] / mean_s
    df["q_ts_norm"] = df["q_import_mvar"] / mean_s
    df["s_ts_norm"] = df["s_import_mva"] / mean_s
    df["p_ldc_norm"] = df["p_mw"] / ldc_scale
    df["q_ldc_norm"] = df["q_import_mvar"] / ldc_scale
    df["s_ldc_norm"] = df["s_import_mva"] / ldc_scale
    df.attrs["ldc_scale_mva"] = ldc_scale
    return df.reset_index(drop=True)

def _band_by_x(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df.groupby(x_col)[y_col]
        .agg(
            expected="mean",
            q02=lambda series: series.quantile(0.02275),
            q16=lambda series: series.quantile(0.15865),
            q84=lambda series: series.quantile(0.84135),
            q98=lambda series: series.quantile(0.97725),
        )
        .reset_index()
        .sort_values(x_col)
    )

def _daily_transformer_bands(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    daily = df[["powerflow_run_id", "t_index", y_col]].copy()
    daily["day_index"] = daily["t_index"] // 24
    daily = daily.groupby(["powerflow_run_id", "day_index"], as_index=False)[y_col].mean()
    return _band_by_x(daily, "day_index", y_col)

def _ldc_transformer_bands(df: pd.DataFrame, y_col: str, n_points: int = 101) -> pd.DataFrame:
    percent_grid = np.linspace(0.0, 100.0, n_points)
    rows = []
    for run_id, group in df.groupby("powerflow_run_id"):
        values = group[y_col].dropna().sort_values(ascending=False).to_numpy()
        if len(values) == 0:
            continue
        percent = np.linspace(0.0, 100.0, len(values))
        rows.append(
            pd.DataFrame(
                {
                    "powerflow_run_id": run_id,
                    "duration_percent": percent_grid,
                    y_col: np.interp(percent_grid, percent, values),
                }
            )
        )
    if not rows:
        raise ValueError(f"No values available to build LDC bands for {y_col}.")
    return _band_by_x(pd.concat(rows, ignore_index=True), "duration_percent", y_col)

def _add_distribution_panel(
    fig,
    row: int,
    col: int,
    band: pd.DataFrame,
    x_col: str,
    color: str,
    fill68: str,
    fill96: str,
    expected_name: str,
    showlegend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q02"],
            mode="lines",
            line={"width": 0, "color": fill96},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q98"],
            mode="lines",
            line={"width": 0, "color": fill96},
            fill="tonexty",
            fillcolor=fill96,
            name="96% Percentile Band",
            hoverinfo="skip",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q16"],
            mode="lines",
            line={"width": 0, "color": fill68},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q84"],
            mode="lines",
            line={"width": 0, "color": fill68},
            fill="tonexty",
            fillcolor=fill68,
            name="68% Percentile Band",
            hoverinfo="skip",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["expected"],
            mode="lines",
            line={"color": color, "width": 2},
            name=expected_name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )

def plot_transformer_import_distributions(
    df: pd.DataFrame,
    show: bool = True,
):
    series = [
        {
            "key": "p",
            "ts_col": "p_ts_norm",
            "ldc_col": "p_ldc_norm",
            "title": "Net Transformer Active Power P Import",
            "ts_y": "Norm. Active Power Import P_i(t) / <|S_i|>",
            "ldc_y": "Norm. Active Power Import P_i / max|S|",
            "color": "#ef3b2c",
            "fill68": "rgba(239, 59, 44, 0.25)",
            "fill96": "rgba(239, 59, 44, 0.14)",
        },
        {
            "key": "q",
            "ts_col": "q_ts_norm",
            "ldc_col": "q_ldc_norm",
            "title": "Net Transformer Reactive Power |Q| Import",
            "ts_y": "Norm. Reactive Power Import |Q_i(t)| / <|S_i|>",
            "ldc_y": "Norm. Reactive Power Import |Q_i| / max|S|",
            "color": "#5b54ff",
            "fill68": "rgba(91, 84, 255, 0.22)",
            "fill96": "rgba(91, 84, 255, 0.12)",
        },
        {
            "key": "s",
            "ts_col": "s_ts_norm",
            "ldc_col": "s_ldc_norm",
            "title": "Net Transformer Apparent Power |S| Load",
            "ts_y": "Norm. Apparent Power Load |S_i(t)| / <|S_i|>",
            "ldc_y": "Norm. Apparent Power Load |S_i| / max|S|",
            "color": "#174a7e",
            "fill68": "rgba(49, 130, 189, 0.35)",
            "fill96": "rgba(49, 130, 189, 0.18)",
        },
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[item for serie in series for item in (serie["title"], serie["title"])],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for row, serie in enumerate(series, start=1):
        ts_band = _daily_transformer_bands(df, serie["ts_col"])
        ldc_band = _ldc_transformer_bands(df, serie["ldc_col"])
        _add_distribution_panel(
            fig,
            row=row,
            col=1,
            band=ts_band,
            x_col="day_index",
            color=serie["color"],
            fill68=serie["fill68"],
            fill96=serie["fill96"],
            expected_name="Expected Timeseries (24 h Agg.)",
            showlegend=(row == 1),
        )
        _add_distribution_panel(
            fig,
            row=row,
            col=2,
            band=ldc_band,
            x_col="duration_percent",
            color=serie["color"],
            fill68=serie["fill68"],
            fill96=serie["fill96"],
            expected_name="Expected LDC (Hourly)",
            showlegend=False,
        )
        fig.update_yaxes(title_text=serie["ts_y"], row=row, col=1)
        fig.update_yaxes(title_text=serie["ldc_y"], row=row, col=2)

    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for row in range(1, 4):
        fig.update_xaxes(tickmode="array", tickvals=month_days, ticktext=month_labels, row=row, col=1)
        fig.update_xaxes(ticksuffix="%", range=[0, 100], row=row, col=2)

    fig.update_layout(
        height=980,
        width=1120,
        hovermode="x unified",
        legend={"orientation": "h", "x": 0.02, "y": 1.05},
        margin={"l": 80, "r": 35, "t": 80, "b": 55},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#d8d8d8")
    fig.update_yaxes(showgrid=True, gridcolor="#d8d8d8", zeroline=True, zerolinecolor="#b8b8b8")
    if show:
        fig.show()
    return fig

def _uses_relative_timeslice_axis(df: pd.DataFrame) -> bool:
    if "t_index" not in df.columns:
        return False
    max_horizon = df.groupby("powerflow_run_id")["t_index"].nunique().max()
    return bool(pd.notna(max_horizon) and int(max_horizon) <= 24 * 14)

def _wide_transformer_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    index_col = "t_index" if relative_axis else "ts"
    if index_col not in df.columns:
        raise ValueError(f"Transformer import plotting requires a {index_col!r} column.")
    wide = df.pivot_table(
        index=index_col,
        columns="powerflow_run_id",
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    if wide.empty:
        raise ValueError(f"No transformer import values available for {value_col}.")
    if relative_axis:
        wide.index = wide.index.astype(int)
        wide.index.name = "t_index"
    else:
        wide.index = pd.to_datetime(wide.index)
        if wide.index.isna().any():
            wide.index = pd.date_range("2009-01-01", periods=len(wide), freq="h")
    return wide

def _matplotlib_quantile_summary(wide: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected": wide.mean(axis=1),
            "q02": wide.quantile(0.02275, axis=1),
            "q16": wide.quantile(0.15865, axis=1),
            "q84": wide.quantile(0.84135, axis=1),
            "q98": wide.quantile(0.97725, axis=1),
        },
        index=wide.index,
    )

def _daily_matplotlib_transformer_bands(
    df: pd.DataFrame,
    value_col: str,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    if "diagnostic" in df.columns:
        daily_df = df[df["diagnostic"] == "daily_mean"].copy()
        if daily_df.empty:
            raise ValueError(f"No compact daily transformer diagnostic values available for {value_col}.")
        index_col = "t_index" if relative_axis else "ts"
        wide = daily_df.pivot_table(
            index=index_col,
            columns="powerflow_run_id",
            values=value_col,
            aggfunc="mean",
        ).sort_index()
        if relative_axis:
            wide.index = (wide.index.astype(int) // 24).astype(int)
            wide.index.name = "day_index"
        else:
            wide.index = pd.to_datetime(wide.index)
        return _matplotlib_quantile_summary(wide)

    wide = _wide_transformer_frame(df, value_col, relative_axis=relative_axis)
    if relative_axis:
        daily = wide.groupby(wide.index // 24).mean()
        daily.index.name = "day_index"
    else:
        daily = wide.resample("24h").mean()
    return _matplotlib_quantile_summary(daily)

def _ldc_matplotlib_transformer_bands(
    df: pd.DataFrame,
    value_col: str,
    n_points: int = 101,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    if "diagnostic" in df.columns:
        ldc_df = df[df["diagnostic"] == "ldc"].copy()
        if ldc_df.empty:
            raise ValueError(f"No compact LDC transformer diagnostic values available for {value_col}.")
        ldc = ldc_df.pivot_table(
            index="x_value",
            columns="powerflow_run_id",
            values=value_col,
            aggfunc="mean",
        ).sort_index()
        return _matplotlib_quantile_summary(ldc)

    wide = _wide_transformer_frame(df, value_col, relative_axis=relative_axis)
    duration_percent = np.linspace(0.0, 100.0, n_points)
    curves = []
    for column in wide.columns:
        values = wide[column].dropna().sort_values(ascending=False).to_numpy()
        if len(values) == 0:
            continue
        source_percent = np.linspace(0.0, 100.0, len(values))
        curves.append(pd.Series(np.interp(duration_percent, source_percent, values), name=column))
    if not curves:
        raise ValueError(f"No transformer import values available for {value_col}.")
    ldc = pd.concat(curves, axis=1)
    ldc.index = duration_percent
    return _matplotlib_quantile_summary(ldc)

def _style_transformer_axis(ax) -> None:
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.55, alpha=0.45)
    ax.grid(which="major", axis="x", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="both", which="both", direction="inout", length=5, width=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _format_month_axis(ax, index: pd.DatetimeIndex) -> None:
    start = index[0].normalize()
    end = index[-1].normalize() + pd.offsets.MonthBegin(1)
    boundaries = pd.date_range(start, end, freq="MS")
    if len(boundaries) < 2:
        return
    centers = boundaries[:-1] + (boundaries[1:] - boundaries[:-1]) / 2
    labels = [value.strftime("%b") for value in boundaries[:-1]]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xticks(boundaries, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", direction="out", length=5, width=0.8)
    ax.set_xlim(index[0], index[-1])

def _format_relative_day_axis(ax, day_index: pd.Index) -> None:
    if len(day_index) == 0:
        return
    first_day = int(day_index.min())
    last_day = int(day_index.max())
    n_days = last_day - first_day + 1
    if n_days <= 14:
        ticks = np.arange(first_day, last_day + 1)
        labels = [f"Day {int(day - first_day + 1)}" for day in ticks]
    else:
        month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        month_labels = np.array([
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ])
        mask = (month_days >= first_day) & (month_days <= last_day)
        ticks = month_days[mask]
        labels = month_labels[mask].tolist()
        if len(ticks) == 0:
            ticks = np.linspace(first_day, last_day, min(6, n_days))
            labels = [f"Day {int(round(day - first_day + 1))}" for day in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(first_day, last_day)

def _plot_matplotlib_band(
    ax,
    band: pd.DataFrame,
    x,
    color: str,
    band68_color: str,
    band96_color: str,
    expected_label: str,
    show_legend: bool,
    band68_label: str = "68% Percentile Band",
    band96_label: str = "96% Percentile Band",
) -> None:
    ax.fill_between(
        x,
        band["q02"].to_numpy(),
        band["q98"].to_numpy(),
        facecolor=band96_color,
        edgecolor="none",
        alpha=0.42,
        label=band96_label,
    )
    ax.fill_between(
        x,
        band["q16"].to_numpy(),
        band["q84"].to_numpy(),
        facecolor=band68_color,
        edgecolor="none",
        alpha=0.38,
        label=band68_label,
    )
    ax.plot(
        x,
        band["expected"].to_numpy(),
        color=color,
        linewidth=1.6,
        alpha=0.98,
        label=expected_label,
    )
    _style_transformer_axis(ax)
    if show_legend:
        legend = ax.legend(
            frameon=True,
            framealpha=0.72,
            fontsize=7,
            loc="upper right",
            handlelength=1.2,
            borderpad=0.25,
            labelspacing=0.25,
        )
        legend.get_frame().set_edgecolor("#cccccc")

def _select_tsam_week_indices(ts_band: pd.DataFrame, requested: tuple[int, int] | None) -> tuple[int, int]:
    n_weeks = int(len(ts_band) // 7)
    if n_weeks < 2:
        return (0, 0)
    if requested is not None:
        first, second = requested
        first = max(0, min(int(first), n_weeks - 1))
        second = max(0, min(int(second), n_weeks - 1))
        return first, second
    week_ids = pd.Series(np.arange(len(ts_band)) // 7, index=ts_band.index)
    weekly_mean = ts_band["expected"].groupby(week_ids).mean().dropna()
    if weekly_mean.empty:
        return (0, min(1, n_weeks - 1))
    high_week = int(weekly_mean.idxmax())
    low_week = int(weekly_mean.idxmin())
    if high_week == low_week:
        low_week = min(high_week + 1, n_weeks - 1) if high_week == 0 else 0
    return high_week, low_week

def _week_slice(band: pd.DataFrame, week_index: int) -> pd.DataFrame:
    start = int(week_index) * 7
    end = start + 7
    return band.iloc[start:end].copy()

def _x_values_for_tsam_week(band: pd.DataFrame):
    if isinstance(band.index, pd.DatetimeIndex):
        return mdates.date2num(band.index.to_pydatetime()), True
    return np.arange(len(band), dtype=float), False

def _format_tsam_week_axis(ax, band: pd.DataFrame) -> None:
    if isinstance(band.index, pd.DatetimeIndex):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
        ax.set_xlim(band.index[0], band.index[-1])
    else:
        ax.set_xticks(np.arange(len(band), dtype=float))
        ax.set_xticklabels([f"Day {idx + 1}" for idx in range(len(band))])
        ax.set_xlim(0, max(len(band) - 1, 1))

def _plot_transformer_import_tsam_week_panels_matplotlib(
    df: pd.DataFrame,
    spec: dict[str, str],
    grouped_frames: list[tuple[str, pd.DataFrame]],
    default_group_styles: dict[str, dict[str, str]],
    fallback_styles: list[dict[str, str]],
    *,
    tsam_week_indices: tuple[int, int] | None,
    tsam_week_labels: tuple[str, str],
    show: bool,
):
    use_calendar_axis = "ts" in df.columns and df["ts"].notna().any()
    relative_axis = not use_calendar_axis

    reference_band = _daily_matplotlib_transformer_bands(
        grouped_frames[-1][1] if grouped_frames else df,
        spec["ts_col"],
        relative_axis=relative_axis,
    )
    week_indices = _select_tsam_week_indices(reference_band, tsam_week_indices)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig = plt.figure(figsize=(11.2, 6.2), dpi=150)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.30)
        week_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        ax_ldc = fig.add_subplot(gs[:, 1])

        for row, (ax_ts, week_index, week_label) in enumerate(zip(week_axes, week_indices, tsam_week_labels)):
            for group_index, (group_label, group_df) in enumerate(grouped_frames):
                if group_df.empty:
                    continue
                style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
                color = style.get("color", spec["color"])
                band68 = style.get("band68", spec["band68"])
                band96 = style.get("band96", spec["band96"])
                ts_band = _daily_matplotlib_transformer_bands(
                    group_df,
                    spec["ts_col"],
                    relative_axis=relative_axis,
                )
                week_band = _week_slice(ts_band, week_index)
                if week_band.empty:
                    continue
                x_ts, _ = _x_values_for_tsam_week(week_band)
                prefix = f"{group_label} " if group_label else ""
                _plot_matplotlib_band(
                    ax_ts,
                    week_band,
                    x_ts,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=row == 0,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )
            _format_tsam_week_axis(ax_ts, _week_slice(reference_band, week_index))
            ax_ts.set_title(f"{week_label} (representative week {week_index + 1})")
            ax_ts.set_ylabel(spec["ts_y"])

        for group_index, (group_label, group_df) in enumerate(grouped_frames):
            if group_df.empty:
                continue
            style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
            color = style.get("color", spec["color"])
            band68 = style.get("band68", spec["band68"])
            band96 = style.get("band96", spec["band96"])
            ldc_band = _ldc_matplotlib_transformer_bands(
                group_df,
                spec["ldc_col"],
                relative_axis=relative_axis,
            )
            x_ldc = ldc_band.index.to_numpy(dtype=float)
            prefix = f"{group_label} " if group_label else ""
            _plot_matplotlib_band(
                ax_ldc,
                ldc_band,
                x_ldc,
                color,
                band68,
                band96,
                f"{prefix}expected",
                show_legend=True,
                band68_label=f"{prefix}68% band",
                band96_label=f"{prefix}96% band",
            )
        ax_ldc.set_xlim(0, 100)
        ax_ldc.set_xticks(np.arange(0, 101, 20))
        ax_ldc.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
        ax_ldc.set_title(f"{spec['title']} duration curve")
        ax_ldc.set_ylabel(spec["ldc_y"])
        fig.suptitle(spec["title"], y=0.99, fontsize=10, fontweight="bold")
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.98, hspace=0.42, wspace=0.30)
        if show:
            plt.show()
        return fig

def plot_transformer_import_distributions_matplotlib(
    df: pd.DataFrame,
    show: bool = True,
    metrics: tuple[str, ...] = ("p", "q", "s"),
    group_col: str | None = None,
    group_styles: dict[str, dict[str, str]] | None = None,
    tsam_week_panels: bool = False,
    tsam_week_indices: tuple[int, int] | None = None,
    tsam_week_labels: tuple[str, str] = ("Min-temperature week", "Max-solar week"),
):
    specs = [
        {
            "key": "p",
            "ts_col": "p_ts_norm",
            "ldc_col": "p_ldc_norm",
            "title": "Net Transformer Active Power P Import",
            "ts_y": r"Norm. Active Power Import $P_i(t) / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. Active Power Import $P_i / \langle \max |S| \rangle$",
            "color": "#ef3b2c",
            "band68": "#fcae91",
            "band96": "#fee0d2",
        },
        {
            "key": "q",
            "ts_col": "q_ts_norm",
            "ldc_col": "q_ldc_norm",
            "title": "Net Transformer Reactive Power |Q| Import",
            "ts_y": r"Norm. React. Power Import $|Q_i(t)| / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. React. Power Import $|Q_i| / \langle \max |S| \rangle$",
            "color": "#5b54ff",
            "band68": "#bcbddc",
            "band96": "#efedf5",
        },
        {
            "key": "s",
            "ts_col": "s_ts_norm",
            "ldc_col": "s_ldc_norm",
            "title": "Net Transformer Apparent Power |S| Load",
            "ts_y": r"Norm. Apparent Power Load $|S_i(t)| / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. Apparent Power Load $|S_i| / \langle \max |S| \rangle$",
            "color": "#08306b",
            "band68": "#3182bd",
            "band96": "#9ecae1",
        },
    ]
    metric_aliases = {
        "p": "p",
        "active": "p",
        "active_power": "p",
        "q": "q",
        "reactive": "q",
        "reactive_power": "q",
        "s": "s",
        "apparent": "s",
        "apparent_power": "s",
    }
    selected_keys = []
    for metric in metrics:
        key = metric_aliases.get(str(metric).strip().lower())
        if key is None:
            available = ", ".join(sorted(metric_aliases))
            raise ValueError(f"Unsupported transformer metric {metric!r}. Available aliases: {available}.")
        if key not in selected_keys:
            selected_keys.append(key)
    specs = [spec for spec in specs if spec["key"] in selected_keys]
    if not specs:
        raise ValueError("At least one transformer metric must be selected.")

    default_group_styles = {
        "Pre": {"color": "#335C81", "band68": "#9ecae1", "band96": "#deebf7"},
        "Post-all": {"color": "#D95D39", "band68": "#f4a582", "band96": "#fde0c5"},
        "Post": {"color": "#D95D39", "band68": "#f4a582", "band96": "#fde0c5"},
    }
    if group_styles:
        default_group_styles.update({str(key): value for key, value in group_styles.items()})
    fallback_styles = [
        {"color": "#08306b", "band68": "#3182bd", "band96": "#9ecae1"},
        {"color": "#d73027", "band68": "#fc8d59", "band96": "#fee0d2"},
        {"color": "#1a9850", "band68": "#91cf60", "band96": "#d9ef8b"},
        {"color": "#6a3d9a", "band68": "#b2abd2", "band96": "#e0d6f0"},
    ]

    if group_col is not None and group_col in df.columns:
        group_labels = [str(label) for label in df[group_col].dropna().drop_duplicates()]
        grouped_frames = [(label, df[df[group_col].astype(str) == label].copy()) for label in group_labels]
    else:
        grouped_frames = [("", df.copy())]

    if tsam_week_panels:
        if len(specs) != 1:
            raise ValueError("tsam_week_panels=True currently supports exactly one transformer metric.")
        return _plot_transformer_import_tsam_week_panels_matplotlib(
            df,
            specs[0],
            grouped_frames,
            default_group_styles,
            fallback_styles,
            tsam_week_indices=tsam_week_indices,
            tsam_week_labels=tsam_week_labels,
            show=show,
        )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig_height = 3.25 * len(specs)
        fig, axes = plt.subplots(len(specs), 2, figsize=(10, fig_height), dpi=150, squeeze=False)
        relative_axis = _uses_relative_timeslice_axis(df)
        for row, spec in enumerate(specs):
            ax_ts = axes[row, 0]
            ax_ldc = axes[row, 1]
            for group_index, (group_label, group_df) in enumerate(grouped_frames):
                if group_df.empty:
                    continue
                style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
                color = style.get("color", spec["color"])
                band68 = style.get("band68", spec["band68"])
                band96 = style.get("band96", spec["band96"])
                ts_band = _daily_matplotlib_transformer_bands(
                    group_df,
                    spec["ts_col"],
                    relative_axis=relative_axis,
                )
                ldc_band = _ldc_matplotlib_transformer_bands(
                    group_df,
                    spec["ldc_col"],
                    relative_axis=relative_axis,
                )
                if relative_axis:
                    x_ts = ts_band.index.to_numpy(dtype=float)
                else:
                    x_ts = mdates.date2num(ts_band.index.to_pydatetime())
                x_ldc = ldc_band.index.to_numpy(dtype=float)

                prefix = f"{group_label} " if group_label else ""
                _plot_matplotlib_band(
                    ax_ts,
                    ts_band,
                    x_ts,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=True,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )
                _plot_matplotlib_band(
                    ax_ldc,
                    ldc_band,
                    x_ldc,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=True,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )

            if relative_axis:
                sample_band = _daily_matplotlib_transformer_bands(grouped_frames[0][1], spec["ts_col"], relative_axis=relative_axis)
                _format_relative_day_axis(ax_ts, sample_band.index)
            else:
                ax_ts.xaxis_date()
                sample_band = _daily_matplotlib_transformer_bands(grouped_frames[0][1], spec["ts_col"], relative_axis=relative_axis)
                _format_month_axis(ax_ts, sample_band.index)
            ax_ldc.set_xlim(0, 100)
            ax_ldc.set_xticks(np.arange(0, 101, 20))
            ax_ldc.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
            ax_ts.set_title(spec["title"])
            ax_ldc.set_title(spec["title"])
            ax_ts.set_ylabel(spec["ts_y"])
            ax_ldc.set_ylabel(spec["ldc_y"])

        fig.tight_layout(h_pad=1.8, w_pad=2.0)
        if show:
            plt.show()
        return fig

def plot_transformer_apparent_power_stage_comparison_matplotlib(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    stage_titles: tuple[str, str] = ("Pre: electricity demand", "Post: electrification"),
    show: bool = True,
):
    datasets = [
        (pre_df, stage_titles[0]),
        (post_df, stage_titles[1]),
    ]
    color = "#08306b"
    band68 = "#3182bd"
    band96 = "#9ecae1"

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=150, sharey=True)
        for ax, (df, title) in zip(axes, datasets):
            relative_axis = _uses_relative_timeslice_axis(df)
            band = _daily_matplotlib_transformer_bands(
                df,
                "s_ts_norm",
                relative_axis=relative_axis,
            )
            if relative_axis:
                x_values = band.index.to_numpy(dtype=float)
            else:
                x_values = mdates.date2num(band.index.to_pydatetime())
            _plot_matplotlib_band(
                ax,
                band,
                x_values,
                color,
                band68,
                band96,
                "Expected Timeseries (24 h Agg.)",
                show_legend=True,
            )
            if relative_axis:
                _format_relative_day_axis(ax, band.index)
            else:
                ax.xaxis_date()
                _format_month_axis(ax, band.index)
            ax.set_title(title)
            ax.set_ylabel(r"Norm. Apparent Power Load $|S_i(t)| / \langle |S_i| \rangle$")

        fig.tight_layout(w_pad=2.0)
        if show:
            plt.show()
        return fig
