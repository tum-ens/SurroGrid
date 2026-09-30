"""Time-series power flow: one solve per timestep, results as matrices.

``run_timeseries`` solves every row of a demand frame and keeps the external-grid
exchange, bus voltages and the line from-side flows as ``T x n`` arrays. Both
outputs of Step 4 are derived from the same matrices: ``raw_tables`` (the
``pwrflw/output`` tables) and ``summarize_powerflow_matrices`` (the compact
summary). Each timestep is solved from pandapower's flat/DC start (``init="auto"``,
no recycling), so it does not depend on the previous timestep (converged or
not) or on the chunking over workers; all timesteps of a chunk reuse one net.

Demand frames have timesteps as rows and ``(bus, component)`` columns with
``electricity`` (kW) and ``electricity-reactive`` (kvar) in the load convention
(positive P consumes, positive Q absorbs); they are converted to MW/Mvar.
"""

from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandapower as pp
import pandas as pd

LINE_QUANTITIES = ("p_from_mw", "q_from_mvar", "i_from_ka")


@dataclass
class PowerflowMatrices:
    """Per-timestep results of one power-flow time series (rows = timesteps)."""

    ext_grid: np.ndarray  # T x E x len(ext_grid_columns)
    ext_grid_index: pd.Index
    ext_grid_columns: pd.Index
    vm_pu: np.ndarray  # T x B
    bus_index: pd.Index
    line: np.ndarray  # T x L x 3 (LINE_QUANTITIES)
    line_index: pd.Index
    failed: list[int] = field(default_factory=list)

    @property
    def n_timesteps(self) -> int:
        return int(self.vm_pu.shape[0])


def demand_arrays(grid, demand: pd.DataFrame):
    """P and Q (MW, Mvar) per timestep and ``grid.load`` row.

    A bus without a component column gets 0 (as before: every load is reset to
    zero each timestep and only the given values are assigned).

    Raises:
        ValueError: missing values in the demand, a demand bus without exactly
            one load row, or no active-power columns.
    """
    if getattr(demand.columns, "nlevels", 1) < 2:
        raise ValueError("Power-flow demand must use (bus, component) MultiIndex columns.")
    values = demand.to_numpy(dtype=float)
    if np.isnan(values).any():
        columns = demand.columns[np.isnan(values).any(axis=0)].tolist()
        raise ValueError(
            "Power-flow demand contains missing values (they would silently become "
            f"zero load), e.g. in {columns[:5]}."
        )
    components = demand.columns.get_level_values(1)
    if not (components == "electricity").any():
        raise ValueError("Each power-flow timestep must provide active p_mw demand.")
    load_bus = grid.load["bus"].astype(int).to_numpy()
    row_of_bus = {}
    for row, bus in enumerate(load_bus):
        if bus in row_of_bus:
            raise ValueError(f"Bus {bus} has more than one load row.")
        row_of_bus[bus] = row
    p = np.zeros((len(demand), len(load_bus)), dtype=float)
    q = np.zeros((len(demand), len(load_bus)), dtype=float)
    seen = set()
    for position, (bus, component) in enumerate(demand.columns.to_flat_index()):
        if component not in ("electricity", "electricity-reactive"):
            continue
        if (bus, component) in seen:
            raise ValueError(f"Demand column {(bus, component)!r} appears twice.")
        seen.add((bus, component))
        row = row_of_bus.get(int(bus))
        if row is None:
            raise ValueError(f"Demand bus {bus} has no load row in the network.")
        target = p if component == "electricity" else q
        target[:, row] = values[:, position] / 1000
    return p, q


def _solve(grid, algorithm):
    algorithms = list(algorithm) if isinstance(algorithm, (list, tuple)) else [algorithm]
    last_error = None
    for solver in algorithms:
        try:
            stdout_context = contextlib.redirect_stdout(io.StringIO()) if solver == "iwamoto_nr" else contextlib.nullcontext()
            with stdout_context:
                pp.runpp(
                    grid,
                    algorithm=solver,
                    max_iteration=100 if solver == "iwamoto_nr" else 50,
                    tolerance_mva=1e-6,
                )
            return
        except pp.LoadflowNotConverged as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _run_chunk(grid, p, q, algorithm, on_nonconvergence, offset):
    """Solve the timesteps of one chunk on ``grid`` (a private copy)."""
    n = len(p)
    bus_index = pd.Index(grid.bus.index)
    line_index = pd.Index(grid.line.index)
    ext_index = pd.Index(grid.ext_grid.index)
    ext_columns = None
    ext = None
    vm = np.full((n, len(bus_index)), np.nan, dtype=float)
    line = np.full((n, len(line_index), len(LINE_QUANTITIES)), np.nan, dtype=float)
    failed = []
    for t in range(n):
        grid.load["p_mw"] = p[t]
        grid.load["q_mvar"] = q[t]
        try:
            _solve(grid, algorithm)
        except pp.LoadflowNotConverged:
            if on_nonconvergence == "raise":
                raise
            failed.append(offset + t)
            continue
        res_ext = grid.res_ext_grid
        if ext is None:
            ext_columns = pd.Index(res_ext.columns)
            ext = np.full((n, len(ext_index), len(ext_columns)), np.nan, dtype=float)
        ext[t] = res_ext.reindex(index=ext_index, columns=ext_columns).to_numpy(dtype=float)
        vm[t] = grid.res_bus["vm_pu"].reindex(bus_index).to_numpy(dtype=float)
        line[t] = grid.res_line[list(LINE_QUANTITIES)].reindex(line_index).to_numpy(dtype=float)
    if ext is None:
        ext_columns = pd.Index(["p_mw", "q_mvar"])
        ext = np.full((n, len(ext_index), len(ext_columns)), np.nan, dtype=float)
    return PowerflowMatrices(ext, ext_index, ext_columns, vm, bus_index, line, line_index, failed)


def _run_chunk_star(args):
    return _run_chunk(*args)


def run_timeseries(
    grid,
    demand: pd.DataFrame,
    *,
    algorithm="bfsw",
    on_nonconvergence="raise",
    n_workers=1,
) -> PowerflowMatrices:
    """Solve the power flow for every demand row.

    Args:
        grid: prepared pandapower net (not modified; each chunk works on a copy).
        demand: timesteps x (bus, component) in kW/kvar.
        algorithm: pandapower algorithm or a list tried in order per timestep.
        on_nonconvergence: ``"raise"`` or ``"nan"`` (record the timestep as failed).
        n_workers: time chunks solved in parallel processes (empty chunks dropped).
    """
    if on_nonconvergence not in {"raise", "nan"}:
        raise ValueError("on_nonconvergence must be either 'raise' or 'nan'.")
    p, q = demand_arrays(grid, demand)
    n = len(demand)
    n_workers = max(1, int(n_workers))
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    bounds = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)] or [(0, 0)]
    jobs = [
        (deepcopy(grid), p[start:stop], q[start:stop], algorithm, on_nonconvergence, start)
        for start, stop in bounds
    ]
    if len(jobs) == 1:
        parts = [_run_chunk(*jobs[0])]
    else:
        with Pool(processes=len(jobs)) as pool:
            parts = pool.map(_run_chunk_star, jobs)
    first = parts[0]
    return PowerflowMatrices(
        ext_grid=np.concatenate([part.ext_grid for part in parts], axis=0),
        ext_grid_index=first.ext_grid_index,
        ext_grid_columns=first.ext_grid_columns,
        vm_pu=np.concatenate([part.vm_pu for part in parts], axis=0),
        bus_index=first.bus_index,
        line=np.concatenate([part.line for part in parts], axis=0),
        line_index=first.line_index,
        failed=[t for part in parts for t in part.failed],
    )


def raw_tables(matrices: PowerflowMatrices):
    """The raw output tables ``(demand_import, vm, line_loads)``.

    Same layout as the former per-timestep concatenation: ``demand_import`` has
    one row per timestep and external grid, ``vm`` one column per bus and
    ``line_loads`` ``(line, quantity)`` columns in line-major order.
    """
    if matrices.failed:
        raise ValueError(
            f"Raw power-flow tables need converged timesteps; failed: {matrices.failed[:10]}."
        )
    n = matrices.n_timesteps
    ext = matrices.ext_grid.reshape(n * len(matrices.ext_grid_index), len(matrices.ext_grid_columns))
    ext_imports = pd.DataFrame(ext, columns=matrices.ext_grid_columns.copy())
    vm = pd.DataFrame(matrices.vm_pu, columns=matrices.bus_index.copy())
    vm.index = pd.RangeIndex(n)
    line_columns = pd.DataFrame(
        np.zeros((len(matrices.line_index), len(LINE_QUANTITIES))),
        index=matrices.line_index.copy(),
        columns=list(LINE_QUANTITIES),
    ).stack().index
    if np.isnan(matrices.line).any():
        # stack() drops missing values; rebuild row by row like the former code.
        rows = [
            pd.DataFrame(matrices.line[t], index=matrices.line_index, columns=list(LINE_QUANTITIES))
            .stack().to_frame().T.reset_index(drop=True)
            for t in range(n)
        ]
        line_loads = pd.concat(rows, axis=0).reset_index(drop=True)
    else:
        line_loads = pd.DataFrame(matrices.line.reshape(n, -1), columns=line_columns)
    return ext_imports, vm, line_loads


# Summary --------------------------------------------------------------------

def _safe_nanpercentile(values, percentile, axis=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        if axis is None:
            return np.nan
        axis_length = values.shape[1] if axis == 0 and values.ndim > 1 else 0
        return np.full(axis_length, np.nan, dtype=float)
    if axis == 0 and values.ndim > 1:
        result = np.full(values.shape[1], np.nan, dtype=float)
        valid_cols = ~np.isnan(values).all(axis=0)
        if valid_cols.any():
            result[valid_cols] = np.nanpercentile(values[:, valid_cols], percentile, axis=0)
        return result
    return np.nanpercentile(values, percentile, axis=axis)


def _safe_nanmax(values, axis=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        if axis is None:
            return np.nan
        axis_length = values.shape[1] if axis == 0 and values.ndim > 1 else 0
        return np.full(axis_length, np.nan, dtype=float)
    if axis == 0 and values.ndim > 1:
        result = np.full(values.shape[1], np.nan, dtype=float)
        valid_cols = ~np.isnan(values).all(axis=0)
        if valid_cols.any():
            result[valid_cols] = np.nanmax(values[:, valid_cols], axis=0)
        return result
    return np.nanmax(values, axis=axis)


def _tail_values_frame(values, asset_ids, metric, asset_type, tail, threshold_percentile):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.shape[1] == 0:
        return pd.DataFrame(
            columns=["metric", "asset_type", "asset_id", "tail", "threshold_value", "t_index", "value"]
        )

    thresholds = _safe_nanpercentile(values, threshold_percentile, axis=0)
    if tail == "upper":
        mask = values >= thresholds[np.newaxis, :]
    elif tail == "lower":
        mask = values <= thresholds[np.newaxis, :]
    else:
        raise ValueError(f"Unknown tail {tail!r}.")
    mask &= ~np.isnan(values)

    timestep_idx, asset_idx = np.nonzero(mask)
    return pd.DataFrame(
        {
            "metric": metric,
            "asset_type": asset_type,
            "asset_id": np.asarray(asset_ids, dtype=int)[asset_idx],
            "tail": f"p{int(threshold_percentile):02d}_{tail}",
            "threshold_value": thresholds[asset_idx],
            "t_index": timestep_idx.astype(int),
            "value": values[timestep_idx, asset_idx],
        }
    )


def annual_boundary_diagnostic(transformer_loadings, *, bands=(24, 168)):
    """Diagnose sensitivity of the annual peak to the horizon boundary.

    A cyclic annual boundary can concentrate flexible load in the first and last
    hours of the modeled year. These bands measure that directly: they report the
    peak inside the leading and trailing windows against the peak outside them, so
    a boundary artifact is visible instead of being averaged away. The bands
    diagnose sensitivity; they are not themselves a correction.
    """
    values = np.asarray(transformer_loadings, dtype=float)
    total = len(values)
    diagnostic = {"boundary_n_timesteps": int(total)}
    if total == 0:
        return diagnostic
    overall_max = _safe_nanmax(values)
    argmax = int(np.nanargmax(values)) if not np.isnan(values).all() else -1
    diagnostic["boundary_peak_t_index"] = argmax
    for band in bands:
        band = int(band)
        if total < 2 * band:
            continue
        leading = values[:band]
        trailing = values[-band:]
        interior = values[band:-band]
        leading_max = float(_safe_nanmax(leading))
        trailing_max = float(_safe_nanmax(trailing))
        interior_max = float(_safe_nanmax(interior))
        diagnostic.update(
            {
                f"boundary_first_{band}h_max_percent": leading_max,
                f"boundary_last_{band}h_max_percent": trailing_max,
                f"boundary_outside_{band}h_max_percent": interior_max,
                f"boundary_peak_in_first_{band}h": bool(0 <= argmax < band),
                f"boundary_peak_in_last_{band}h": bool(total - band <= argmax < total),
                f"boundary_{band}h_excess_percent": float(
                    max(leading_max, trailing_max) - interior_max
                ),
            }
        )
    diagnostic["boundary_overall_max_percent"] = float(overall_max)
    return diagnostic


def _interp_ldc(values: np.ndarray, duration_percent: np.ndarray) -> np.ndarray:
    values = pd.Series(values).dropna().sort_values(ascending=False).to_numpy(dtype=float)
    if len(values) == 0:
        return np.full(len(duration_percent), np.nan, dtype=float)
    source_percent = np.linspace(0.0, 100.0, len(values))
    return np.interp(duration_percent, source_percent, values)


def _transformer_import_diagnostic_frame(
    p_mw: np.ndarray,
    q_mvar: np.ndarray,
    s_mva: np.ndarray,
    ldc_points: int = 101,
) -> pd.DataFrame:
    hourly = pd.DataFrame(
        {
            "t_index": np.arange(len(s_mva), dtype=int),
            "p_mw": p_mw,
            "q_mvar": q_mvar,
            "q_abs_mvar": np.abs(q_mvar),
            "s_mva": s_mva,
        }
    )
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "diagnostic",
                "point_index",
                "x_value",
                "t_index",
                "p_mw",
                "q_mvar",
                "q_abs_mvar",
                "s_mva",
                "mean_s_mva",
                "max_s_mva",
            ]
        )

    mean_s_mva = float(np.nanmean(s_mva)) if not np.all(np.isnan(s_mva)) else np.nan
    max_s_mva = float(np.nanmax(s_mva)) if not np.all(np.isnan(s_mva)) else np.nan

    hourly["day_index"] = hourly["t_index"] // 24
    daily = (
        hourly.groupby("day_index", as_index=False)[["p_mw", "q_mvar", "q_abs_mvar", "s_mva"]]
        .mean()
        .rename(columns={"day_index": "point_index"})
    )
    daily["diagnostic"] = "daily_mean"
    daily["x_value"] = daily["point_index"].astype(float)
    daily["t_index"] = daily["point_index"].astype(int) * 24

    duration_percent = np.linspace(0.0, 100.0, ldc_points)
    ldc = pd.DataFrame(
        {
            "diagnostic": "ldc",
            "point_index": np.arange(ldc_points, dtype=int),
            "x_value": duration_percent,
            "t_index": pd.NA,
            "p_mw": _interp_ldc(hourly["p_mw"].to_numpy(dtype=float), duration_percent),
            "q_mvar": _interp_ldc(hourly["q_mvar"].to_numpy(dtype=float), duration_percent),
            "q_abs_mvar": _interp_ldc(hourly["q_abs_mvar"].to_numpy(dtype=float), duration_percent),
            "s_mva": _interp_ldc(hourly["s_mva"].to_numpy(dtype=float), duration_percent),
        }
    )

    out = pd.concat([daily, ldc], ignore_index=True, sort=False)
    out["mean_s_mva"] = mean_s_mva
    out["max_s_mva"] = max_s_mva
    return out[
        [
            "diagnostic",
            "point_index",
            "x_value",
            "t_index",
            "p_mw",
            "q_mvar",
            "q_abs_mvar",
            "s_mva",
            "mean_s_mva",
            "max_s_mva",
        ]
    ]


def transformer_exchange(matrices: PowerflowMatrices):
    """External-grid P, Q and S per timestep (sum over external grids; NaN if failed)."""
    n = matrices.n_timesteps
    failed = set(matrices.failed)
    columns = list(matrices.ext_grid_columns)
    p_mw = np.full(n, np.nan, dtype=float)
    q_mvar = np.full(n, np.nan, dtype=float)
    if {"p_mw", "q_mvar"}.issubset(columns):
        p_position, q_position = columns.index("p_mw"), columns.index("q_mvar")
        for t in range(n):
            if t in failed:
                continue
            p_mw[t] = float(pd.Series(matrices.ext_grid[t, :, p_position]).sum())
            q_mvar[t] = float(pd.Series(matrices.ext_grid[t, :, q_position]).sum())
    s_mva = np.array([float(np.hypot(p, q)) for p, q in zip(p_mw, q_mvar)], dtype=float)
    return p_mw, q_mvar, s_mva


def summarize_powerflow_matrices(
    matrices: PowerflowMatrices,
    *,
    transformer_s_rated_mva,
    cable_ids,
    cable_max_i_ka,
    cable_parallel,
    voltage_buses,
):
    """Compact violation-hour and percentile metrics of one power-flow time series.

    Args:
        matrices: output of ``run_timeseries``.
        transformer_s_rated_mva: station rating (MVA).
        cable_ids: evaluated lines (pd.Index, subset of the network lines).
        cable_max_i_ka: rated current per evaluated line (0 -> NaN).
        cable_parallel: parallel systems per evaluated line.
        voltage_buses: evaluated buses (pd.Index, subset of the network buses).
    """
    n = matrices.n_timesteps
    failed_timesteps = list(matrices.failed)
    cable_capacity = (cable_max_i_ka * cable_parallel).to_numpy(dtype=float)
    line_positions = matrices.line_index.get_indexer(cable_ids)
    bus_positions = matrices.bus_index.get_indexer(voltage_buses)
    i_position = LINE_QUANTITIES.index("i_from_ka")
    cable_loading_matrix = (np.abs(matrices.line[:, line_positions, i_position]) / cable_capacity) * 100.0
    voltage_matrix = matrices.vm_pu[:, bus_positions].astype(float)

    transformer_p_mw, transformer_q_mvar, transformer_s_mva = transformer_exchange(matrices)
    if transformer_s_rated_mva > 0:
        transformer_loadings = (transformer_s_mva / transformer_s_rated_mva) * 100.0
    else:
        transformer_loadings = np.full(n, np.nan, dtype=float)

    voltage_all = voltage_matrix[~np.isnan(voltage_matrix)]
    cable_max_loading = _safe_nanmax(cable_loading_matrix, axis=0) if len(cable_ids) else np.array([], dtype=float)
    cable_values = cable_max_loading[~np.isnan(cable_max_loading)]

    trafo_hours_above_100 = int(np.nansum(transformer_loadings > 100.0)) if transformer_loadings.size else 0
    cable_hours_above_100 = np.nansum(cable_loading_matrix > 100.0, axis=0).astype(int) if len(cable_ids) else np.array([], dtype=int)
    voltage_hours_below_0_90 = np.nansum(voltage_matrix < 0.90, axis=0).astype(int) if len(voltage_buses) else np.array([], dtype=int)
    voltage_hours_above_1_03 = np.nansum(voltage_matrix > 1.03, axis=0).astype(int) if len(voltage_buses) else np.array([], dtype=int)
    voltage_hours_above_1_10 = np.nansum(voltage_matrix > 1.10, axis=0).astype(int) if len(voltage_buses) else np.array([], dtype=int)
    cable_max_t_index = (
        np.nanargmax(cable_loading_matrix, axis=0).astype(int)
        if len(cable_ids) and not np.all(np.isnan(cable_loading_matrix), axis=0).any()
        else np.array([
            int(np.nanargmax(cable_loading_matrix[:, idx])) if not np.all(np.isnan(cable_loading_matrix[:, idx])) else -1
            for idx in range(len(cable_ids))
        ], dtype=int)
    )
    if transformer_s_mva.size and not np.all(np.isnan(transformer_s_mva)):
        trafo_critical_t_index = int(np.nanargmax(transformer_s_mva))
        trafo_max_s_mva = float(transformer_s_mva[trafo_critical_t_index])
        trafo_max_p_mw = float(transformer_p_mw[trafo_critical_t_index])
        trafo_max_q_mvar = float(transformer_q_mvar[trafo_critical_t_index])
    else:
        trafo_critical_t_index = None
        trafo_max_s_mva = np.nan
        trafo_max_p_mw = np.nan
        trafo_max_q_mvar = np.nan
    trafo_mean_s_mva = float(np.nanmean(transformer_s_mva)) if transformer_s_mva.size and not np.all(np.isnan(transformer_s_mva)) else np.nan

    cable_summary = pd.DataFrame(
        {
            "cable": cable_ids,
            "cable_max_i_ka": cable_max_i_ka.to_numpy(dtype=float),
            "cable_parallel": cable_parallel.to_numpy(dtype=float),
            "cable_installed_capacity_ka": cable_capacity,
            "cable_loading_p50_time_percent": _safe_nanpercentile(cable_loading_matrix, 50, axis=0),
            "cable_loading_p90_time_percent": _safe_nanpercentile(cable_loading_matrix, 90, axis=0),
            "cable_loading_p95_time_percent": _safe_nanpercentile(cable_loading_matrix, 95, axis=0),
            "cable_loading_p99_time_percent": _safe_nanpercentile(cable_loading_matrix, 99, axis=0),
            "cable_loading_max_time_percent": cable_max_loading,
            "cable_loading_max_t_index": cable_max_t_index,
            "cable_loading_hours_above_100": cable_hours_above_100,
        }
    ).dropna(subset=["cable_loading_max_time_percent"])
    bus_voltage_summary = pd.DataFrame(
        {
            "bus": voltage_buses,
            "voltage_p50_time_pu": _safe_nanpercentile(voltage_matrix, 50, axis=0),
            "voltage_p10_time_pu": _safe_nanpercentile(voltage_matrix, 10, axis=0),
            "voltage_p05_time_pu": _safe_nanpercentile(voltage_matrix, 5, axis=0),
            "voltage_p01_time_pu": _safe_nanpercentile(voltage_matrix, 1, axis=0),
            "voltage_min_time_pu": _safe_nanmax(-voltage_matrix, axis=0) * -1.0,
            "voltage_max_time_pu": _safe_nanmax(voltage_matrix, axis=0),
            "voltage_hours_below_0_90": voltage_hours_below_0_90,
            "voltage_hours_above_1_03": voltage_hours_above_1_03,
            "voltage_hours_above_1_10": voltage_hours_above_1_10,
        }
    ).dropna(subset=["voltage_p05_time_pu"])

    n_failed_timesteps = int(len(failed_timesteps))
    grid_summary = {
        **annual_boundary_diagnostic(transformer_loadings),
        "n_timesteps": int(n),
        "n_converged_timesteps": int(n - n_failed_timesteps),
        "n_failed_timesteps": n_failed_timesteps,
        "n_voltage_buses": int(len(voltage_buses)),
        "n_cables": int(len(cable_values)),
        "transformer_s_rated_mva": float(transformer_s_rated_mva),
        "trafo_mean_s_mva": trafo_mean_s_mva,
        "trafo_max_s_mva": trafo_max_s_mva,
        "trafo_max_p_mw": trafo_max_p_mw,
        "trafo_max_q_mvar": trafo_max_q_mvar,
        "trafo_critical_t_index": trafo_critical_t_index,
        "trafo_loading_p50_time_percent": float(_safe_nanpercentile(transformer_loadings, 50)),
        "trafo_loading_p90_time_percent": float(_safe_nanpercentile(transformer_loadings, 90)),
        "trafo_loading_p95_time_percent": float(_safe_nanpercentile(transformer_loadings, 95)),
        "trafo_loading_p99_time_percent": float(_safe_nanpercentile(transformer_loadings, 99)),
        "trafo_loading_max_time_percent": float(_safe_nanmax(transformer_loadings)),
        "trafo_loading_hours_above_100": trafo_hours_above_100,
        "cable_loading_p95_asset_percent": float(_safe_nanpercentile(cable_values, 95)),
        "cable_hours_above_100_p95_asset": float(_safe_nanpercentile(cable_hours_above_100, 95)) if cable_hours_above_100.size else np.nan,
        "voltage_p05_load_bus_hour_pu": float(_safe_nanpercentile(voltage_all, 5)),
        "voltage_hours_below_0_90_p95_asset": float(_safe_nanpercentile(voltage_hours_below_0_90, 95)) if voltage_hours_below_0_90.size else np.nan,
        "voltage_hours_above_1_03_p95_asset": float(_safe_nanpercentile(voltage_hours_above_1_03, 95)) if voltage_hours_above_1_03.size else np.nan,
        "voltage_hours_above_1_10_p95_asset": float(_safe_nanpercentile(voltage_hours_above_1_10, 95)) if voltage_hours_above_1_10.size else np.nan,
    }

    transformer_diagnostic = _transformer_import_diagnostic_frame(
        transformer_p_mw,
        transformer_q_mvar,
        transformer_s_mva,
    )

    transformer_matrix = transformer_loadings.reshape(-1, 1) if transformer_loadings.size else np.empty((0, 1))
    tail_frames = [
        _tail_values_frame(
            transformer_matrix,
            [0],
            metric="Transformer",
            asset_type="transformer",
            tail="upper",
            threshold_percentile=99,
        ),
        _tail_values_frame(
            cable_loading_matrix,
            cable_ids.to_numpy(dtype=int),
            metric="Cables",
            asset_type="cable",
            tail="upper",
            threshold_percentile=99,
        ),
        _tail_values_frame(
            voltage_matrix,
            voltage_buses.to_numpy(dtype=int),
            metric="Voltage",
            asset_type="bus",
            tail="lower",
            threshold_percentile=1,
        ),
        pd.DataFrame(
            {
                "metric": "Power-flow convergence",
                "asset_type": "grid",
                "asset_id": 0,
                "tail": "failure",
                "threshold_value": 0.0,
                "t_index": failed_timesteps,
                "value": 1.0,
            }
        ),
    ]
    tail_frames = [frame for frame in tail_frames if not frame.empty]
    if tail_frames:
        tail_summary = pd.concat(tail_frames, ignore_index=True)
    else:
        tail_summary = pd.DataFrame(
            columns=["metric", "asset_type", "asset_id", "tail", "threshold_value", "t_index", "value"]
        )

    return {
        "grid_summary": grid_summary,
        "cable_summary": cable_summary,
        "bus_voltage_summary": bus_voltage_summary,
        "tail_summary": tail_summary,
        "transformer_diagnostic": transformer_diagnostic,
        "failed_timesteps": failed_timesteps,
    }


def summary_scope(grid, cable_max_i_ka, voltage_buses, cable_ids=None):
    """Evaluated cables and buses of a network, their ratings and parallel systems."""
    if cable_ids is None:
        cable_ids = pd.Index([int(line) for line in grid.line.index], name="cable")
    else:
        cable_ids = pd.Index([int(line) for line in cable_ids if int(line) in grid.line.index], name="cable")
    cable_max_i_ka = cable_max_i_ka.reindex(cable_ids).astype(float).replace(0.0, np.nan)
    if "parallel" in grid.line.columns:
        cable_parallel = grid.line["parallel"].reindex(cable_ids).fillna(1).astype(float)
    else:
        cable_parallel = pd.Series(1.0, index=cable_ids)
    voltage_buses = pd.Index([int(bus) for bus in voltage_buses if int(bus) in grid.bus.index], name="bus")
    return cable_ids, cable_max_i_ka, cable_parallel, voltage_buses


def summarize(grid, matrices, *, transformer_s_rated_mva, cable_max_i_ka, voltage_buses, cable_ids=None):
    """``summarize_powerflow_matrices`` with the evaluation scope taken from ``grid``."""
    cable_ids, cable_max_i_ka, cable_parallel, voltage_buses = summary_scope(
        grid, cable_max_i_ka, voltage_buses, cable_ids
    )
    return summarize_powerflow_matrices(
        matrices,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_ids=cable_ids,
        cable_max_i_ka=cable_max_i_ka,
        cable_parallel=cable_parallel,
        voltage_buses=voltage_buses,
    )
