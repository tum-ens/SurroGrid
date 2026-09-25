"""Data preparation of the retained-asset cutoff plots (``powerflow_asset_plots``).

A cutoff keeps the least critical share of assets (``filter_scope="asset"``) or of
whole grids ranked by their most critical asset (``filter_scope="grid"``).
Critical means high loading for transformers and cables and low voltage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

METRIC_ORDER = ("Transformer", "Cables", "Voltage")
CRITICAL_PERCENTILE = {"Transformer": "max", "Cables": "max", "Voltage": "min"}
CRITICAL_DIRECTION = {"Transformer": "high", "Cables": "high", "Voltage": "low"}
Y_TITLES = {
    "Transformer": "Max loading [%]",
    "Cables": "Max loading [%]",
    "Voltage": "Min voltage [p.u.]",
}
DEFAULT_COLORS = {
    "Synthetic": "#335C81",
    "Real SWF": "#D95D39",
    "synthetic": "#335C81",
    "real_swf": "#D95D39",
}
FALLBACK_PALETTE = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]


def normalize_filter_scope(filter_scope: str) -> str:
    """``asset``/``assets`` -> ``asset``, ``grid``/``grids`` -> ``grid``."""
    value = str(filter_scope).strip().lower()
    if value in {"asset", "assets"}:
        return "asset"
    if value in {"grid", "grids"}:
        return "grid"
    raise ValueError("filter_scope must be either 'asset' or 'grid'.")


def normalize_quantiles(values: Iterable[float], name: str) -> tuple[float, ...]:
    """Fractions in (0, 1]; values above 1 are read as percent."""
    quantiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in values)
    if any(q <= 0 or q > 1 for q in quantiles):
        raise ValueError(f"{name} values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    return quantiles


def select_metrics(metrics: Iterable[str]) -> list[str]:
    """Canonical metric names (case-insensitive, unique, in the given order)."""
    lookup = {metric.lower(): metric for metric in METRIC_ORDER}
    selected: list[str] = []
    for metric in metrics:
        key = lookup.get(str(metric).strip().lower())
        if key is None:
            raise ValueError(f"Unsupported metric {metric!r}. Available: {', '.join(METRIC_ORDER)}.")
        if key not in selected:
            selected.append(key)
    return selected


def color_defaults(color_map: Mapping[str, str] | None) -> dict[str, str]:
    """Group colors: the defaults updated by ``color_map``."""
    colors = dict(DEFAULT_COLORS)
    if color_map:
        colors.update({str(key): value for key, value in color_map.items()})
    return colors


def grid_keys(frame: pd.DataFrame) -> pd.Series:
    """One text key per grid and run of each profile row (grid-scope filtering)."""
    if frame.empty:
        # DataFrame.agg on no rows returns a frame, which groupby rejects.
        return pd.Series(index=frame.index, dtype="string")
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


def retained_mask(values: pd.Series, metric: str, retained_fraction: float) -> pd.Series:
    """Rows kept by the cutoff (the least critical ``retained_fraction``)."""
    values = values.astype(float)
    if np.isclose(retained_fraction, 1.0):
        return pd.Series(True, index=values.index)
    if CRITICAL_DIRECTION[metric] == "high":
        threshold = values.quantile(retained_fraction)
        return values <= threshold
    threshold = values.quantile(1 - retained_fraction)
    return values >= threshold


def retained_frame(group_df: pd.DataFrame, metric: str, retained_fraction: float, filter_scope: str) -> pd.DataFrame:
    """Profile rows kept by the cutoff, per asset or per whole grid."""
    values = group_df["value"].astype(float)
    if filter_scope == "asset":
        return group_df.loc[retained_mask(values, metric, retained_fraction)].copy()
    row_keys = grid_keys(group_df)
    reducer = "max" if CRITICAL_DIRECTION[metric] == "high" else "min"
    grid_values = values.groupby(row_keys, sort=False).agg(reducer)
    retained_grid_keys = set(grid_values.loc[retained_mask(grid_values, metric, retained_fraction)].index)
    return group_df.loc[row_keys.isin(retained_grid_keys)].copy()


def retained_curve(
    group_df: pd.DataFrame,
    metric: str,
    x_values: Iterable[float],
    *,
    filter_scope: str,
    center_stat: str,
    counts: bool = False,
) -> pd.DataFrame:
    """Center (median/mean) and min-max band of the retained values for each cutoff.

    ``counts`` adds ``retained_assets``/``total_assets`` (assets or grids).
    """
    rows: list[dict[str, float]] = []
    group_df = group_df.dropna(subset=["value"]).copy()
    if counts:
        if group_df.empty:
            return pd.DataFrame(rows)
        total_count = int(group_df["value"].size) if filter_scope == "asset" else int(grid_keys(group_df).nunique())
    for retained_fraction in x_values:
        retained_df = retained_frame(group_df, metric, retained_fraction, filter_scope)
        retained = retained_df["value"].astype(float).dropna()
        if retained.empty:
            continue
        row = {
            "retained_asset_cutoff": retained_fraction,
            "center": float(retained.median() if center_stat == "median" else retained.mean()),
            "band_lower": float(retained.min()),
            "band_upper": float(retained.max()),
        }
        if counts:
            row["retained_assets"] = (
                int(retained.size) if filter_scope == "asset" else int(grid_keys(retained_df).nunique())
            )
            row["total_assets"] = total_count
        rows.append(row)
    return pd.DataFrame(rows)


def select_worst_asset_per_grid(plot_df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """The most critical row of each grid (per ``group_col`` if present)."""
    if plot_df.empty:
        return plot_df
    if "grid" not in plot_df.columns:
        raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
    plot_df = plot_df.dropna(subset=["value"]).copy()
    if plot_df.empty:
        return plot_df
    group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
    grouped = plot_df.groupby(group_keys, sort=False, observed=True)["value"]
    value_index = grouped.idxmax() if CRITICAL_DIRECTION[metric] == "high" else grouped.idxmin()
    return plot_df.loc[value_index.dropna()].reset_index(drop=True)
