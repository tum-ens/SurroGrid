"""Pandapower execution helpers for time-series power flow.

This module provides:

- `prepare_grid(net)`: relaxes constraints and replaces the transformer with a
    closed bus-bus switch to simplify the boundary condition.
- `pf(net, df_demand, parallel, n_cpu)`: runs a pandapower load-flow for each
    timestep in `df_demand` and returns external-grid imports, bus voltages, and
    line loadings.

Expected demand format:

- `df_demand` rows are timesteps.
- Columns are a 2-level MultiIndex `(bus, component)` where `component` is
    `electricity` (P) or `electricity-reactive` (Q). Values are assumed to be in
    kW/kVAr and are converted to MW/MVAr internally.
"""

import contextlib
import io
from collections import deque

import pandapower as pp
import pandas as pd
import numpy as np

from multiprocessing import Pool
from copy import deepcopy

def prepare_grid(grid):
    """ Set max_i to very high value. Remove transformer and replace with small line. """
    # Increase max line capacity
    df_lines = grid.line
    df_lines["max_i_ka"] = 1000
    grid.line = df_lines

    # Remove load max restrictions
    df_loads = grid.load
    df_loads["max_p_mw"] = 1000
    for column, value in {
        "const_z_percent": 0.0,
        "const_i_percent": 0.0,
        "scaling": 1.0,
        "in_service": True,
    }.items():
        if column not in df_loads.columns:
            df_loads[column] = value
    grid.load = df_loads

    # Remove voltage restrictions
    df_buses = grid.bus
    df_buses[["min_vm_pu", "max_vm_pu"]] = (0, 10)
    grid.bus = df_buses

    # Remove trafo and replace with switch
    trafo_buses = grid.trafo[["hv_bus", "lv_bus"]].values[0]
    grid.trafo.drop(index=0, inplace=True)

    ext_grid_bus = int(grid.ext_grid.loc[0, "bus"]) # bus which is the external import bus
    lv_bus = [bus for bus in trafo_buses if bus!=ext_grid_bus][0]
    grid.bus.loc[ext_grid_bus, "vn_kv"] = grid.bus.loc[lv_bus, "vn_kv"]

    pp.create_switch(
        grid,
        bus     = ext_grid_bus,
        element = lv_bus,
        et      = "b",
        closed  = True,
        type    = "CB",
        name    = f"SW_replacing_T0"
    )

    return grid


def run_single_pf(grid, new_load, algorithm="bfsw"):
    # 1. Ensure that 'bus' is the index in both DataFrames
    df_load = grid.load.copy()
    df_load_indexed = df_load.set_index('bus')
    new_load_indexed = new_load.set_index('bus')
    # 2. Use DataFrame.update to overwrite only the overlapping entries
    df_load_indexed.update(new_load_indexed)
    # 3. If you need to restore 'bus' as a column rather than the index:
    df_load_updated = df_load_indexed.reset_index()

    grid.load = df_load_updated
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
            return grid
        except pp.LoadflowNotConverged as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return grid


def _demand_row_to_load(row):
    reshaped_load = row.unstack(level=1)
    reshaped_load.index.name = 'bus'
    reshaped_load = reshaped_load.reset_index()
    reshaped_load.rename(columns={"electricity": "p_mw", "electricity-reactive": "q_mvar"}, inplace=True)
    reshaped_load[["p_mw", "q_mvar"]] = reshaped_load[["p_mw", "q_mvar"]] / 1000
    return reshaped_load


def run_full_pf(grid, df_demand):
    ext_imports_list = []
    line_loads_list = []
    vm_list = []

    for i, row in df_demand.iterrows():
        ### Prepare
        reshaped_load = _demand_row_to_load(row)
        
        ### Run powerflow
        grid_res = run_single_pf(grid, reshaped_load)

        ### Save
        ext_imports_list.append(grid_res.res_ext_grid)                          # Transformer import of p,q
        vm_list.append(grid_res.res_bus[["vm_pu"]].T.reset_index(drop=True))    # Voltage magnitude at buses
        line_loads_list.append(grid_res.res_line[["p_from_mw", "q_from_mvar", "i_from_ka"]].stack().to_frame().T.reset_index(drop=True))   # Line loads

    # Concatenate results
    ext_imports = pd.concat(ext_imports_list, axis=0).reset_index(drop=True)
    vm = pd.concat(vm_list, axis=0).reset_index(drop=True)
    line_loads = pd.concat(line_loads_list, axis=0).reset_index(drop=True)

    return ext_imports, vm, line_loads


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


def _active_line_index(grid):
    if grid.line.empty:
        return pd.Index([], dtype=int)
    if "in_service" not in grid.line.columns:
        return pd.Index(grid.line.index)
    active = grid.line["in_service"].fillna(True).astype(bool)
    return pd.Index(grid.line.index[active])


def _grid_adjacency(grid):
    adjacency = {}
    for _, line in grid.line.loc[_active_line_index(grid)].iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        adjacency.setdefault(from_bus, set()).add(to_bus)
        adjacency.setdefault(to_bus, set()).add(from_bus)

    if hasattr(grid, "switch") and not grid.switch.empty:
        switches = grid.switch
        if "closed" in switches.columns:
            switches = switches[switches["closed"].fillna(True).astype(bool)]
        if "et" in switches.columns:
            switches = switches[switches["et"].astype(str).eq("b")]
        for _, switch in switches.iterrows():
            bus = int(switch["bus"])
            element = int(switch["element"])
            adjacency.setdefault(bus, set()).add(element)
            adjacency.setdefault(element, set()).add(bus)
    return adjacency


def _root_bus(grid):
    if hasattr(grid, "ext_grid") and not grid.ext_grid.empty and "bus" in grid.ext_grid.columns:
        return int(grid.ext_grid.iloc[0]["bus"])
    if hasattr(grid, "trafo") and not grid.trafo.empty and "lv_bus" in grid.trafo.columns:
        return int(grid.trafo.iloc[0]["lv_bus"])
    return int(grid.bus.index[0])


def _parent_tree_from_root(adjacency, root):
    parents = {int(root): None}
    queue = deque([int(root)])
    while queue:
        bus = queue.popleft()
        for neighbor in sorted(adjacency.get(bus, [])):
            if neighbor in parents:
                continue
            parents[neighbor] = bus
            queue.append(neighbor)
    return parents


def comparison_backbone_scope(grid, load_buses):
    """Return backbone cable ids and upstream backbone voltage buses.

    The comparison scope excludes service/house-connection lines that touch the
    selected load endpoints. Voltages are evaluated at the nearest upstream bus
    that remains part of the retained line backbone.
    """
    load_buses = {
        int(bus)
        for bus in load_buses
        if pd.notna(bus) and int(bus) in grid.bus.index
    }
    active_line_ids = _active_line_index(grid)
    if len(active_line_ids) == 0:
        return [], []

    lines = grid.line.loc[active_line_ids]
    endpoint_mask = (
        lines["from_bus"].astype(int).isin(load_buses)
        | lines["to_bus"].astype(int).isin(load_buses)
    )
    backbone_cable_ids = pd.Index(active_line_ids[~endpoint_mask])
    backbone_lines = grid.line.loc[backbone_cable_ids]
    backbone_buses = set(backbone_lines["from_bus"].astype(int)).union(
        set(backbone_lines["to_bus"].astype(int))
    )
    if not backbone_buses:
        backbone_buses = set(lines["from_bus"].astype(int)).union(set(lines["to_bus"].astype(int))) - load_buses

    adjacency = _grid_adjacency(grid)
    parents = _parent_tree_from_root(adjacency, _root_bus(grid))
    voltage_buses = []
    for load_bus in sorted(load_buses):
        bus = load_bus
        seen = set()
        mapped_bus = None
        while bus in parents and bus not in seen:
            seen.add(bus)
            bus = parents[bus]
            if bus is None:
                break
            if bus in backbone_buses:
                mapped_bus = bus
                break
        if mapped_bus is None:
            for neighbor in sorted(adjacency.get(load_bus, [])):
                if neighbor in backbone_buses:
                    mapped_bus = neighbor
                    break
        if mapped_bus is not None:
            voltage_buses.append(int(mapped_bus))

    voltage_buses = pd.Index(voltage_buses, dtype=int).drop_duplicates().tolist()
    return backbone_cable_ids.astype(int).tolist(), voltage_buses


def pf_summary(grid, df, transformer_s_rated_mva, cable_max_i_ka, voltage_buses, algorithm="bfsw", cable_ids=None):
    """Run power flow and return compact violation-hour and percentile metrics."""
    transformer_loadings = []
    if cable_ids is None:
        cable_ids = pd.Index([int(line) for line in grid.line.index], name="cable")
    else:
        cable_ids = pd.Index([int(line) for line in cable_ids if int(line) in grid.line.index], name="cable")
    cable_max_i_ka = cable_max_i_ka.reindex(cable_ids).astype(float).replace(0.0, np.nan)
    cable_capacity = cable_max_i_ka.to_numpy(dtype=float)
    cable_loading_matrix = np.full((len(df), len(cable_ids)), np.nan, dtype=float)
    voltage_buses = pd.Index([int(bus) for bus in voltage_buses if int(bus) in grid.bus.index], name="bus")
    voltage_matrix = np.full((len(df), len(voltage_buses)), np.nan, dtype=float)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        grid_res = run_single_pf(grid, _demand_row_to_load(row), algorithm=algorithm)

        ext_grid = grid_res.res_ext_grid
        if {"p_mw", "q_mvar"}.issubset(ext_grid.columns) and transformer_s_rated_mva > 0:
            s_mva = np.hypot(ext_grid["p_mw"].sum(), ext_grid["q_mvar"].sum())
            transformer_loadings.append((s_mva / transformer_s_rated_mva) * 100.0)

        voltage_matrix[row_idx, :] = grid_res.res_bus["vm_pu"].reindex(voltage_buses).to_numpy(dtype=float)

        i_from_ka = grid_res.res_line["i_from_ka"].abs().reindex(cable_ids).to_numpy(dtype=float)
        cable_loading_matrix[row_idx, :] = (i_from_ka / cable_capacity) * 100.0

    voltage_all = voltage_matrix[~np.isnan(voltage_matrix)]
    transformer_loadings = np.asarray(transformer_loadings, dtype=float)
    cable_max_loading = _safe_nanmax(cable_loading_matrix, axis=0) if len(cable_ids) else np.array([], dtype=float)
    cable_values = cable_max_loading[~np.isnan(cable_max_loading)]

    trafo_hours_above_100 = int(np.nansum(transformer_loadings > 100.0)) if transformer_loadings.size else 0
    cable_hours_above_100 = np.nansum(cable_loading_matrix > 100.0, axis=0).astype(int) if len(cable_ids) else np.array([], dtype=int)
    voltage_hours_below_0_90 = np.nansum(voltage_matrix < 0.90, axis=0).astype(int) if len(voltage_buses) else np.array([], dtype=int)

    cable_summary = pd.DataFrame(
        {
            "cable": cable_ids,
            "cable_max_i_ka": cable_capacity,
            "cable_loading_p50_time_percent": _safe_nanpercentile(cable_loading_matrix, 50, axis=0),
            "cable_loading_p90_time_percent": _safe_nanpercentile(cable_loading_matrix, 90, axis=0),
            "cable_loading_p95_time_percent": _safe_nanpercentile(cable_loading_matrix, 95, axis=0),
            "cable_loading_p99_time_percent": _safe_nanpercentile(cable_loading_matrix, 99, axis=0),
            "cable_loading_max_time_percent": cable_max_loading,
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
            "voltage_hours_below_0_90": voltage_hours_below_0_90,
        }
    ).dropna(subset=["voltage_p05_time_pu"])

    grid_summary = {
        "n_timesteps": int(len(df)),
        "n_voltage_buses": int(len(voltage_buses)),
        "n_cables": int(len(cable_values)),
        "transformer_s_rated_mva": float(transformer_s_rated_mva),
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
    }

    transformer_matrix = transformer_loadings.reshape(-1, 1) if transformer_loadings.size else np.empty((0, 1))
    tail_summary = pd.concat(
        [
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
        ],
        ignore_index=True,
    )

    return {
        "grid_summary": grid_summary,
        "cable_summary": cable_summary,
        "bus_voltage_summary": bus_voltage_summary,
        "tail_summary": tail_summary,
    }

def pf(grid, df, parallel, n_cpu):
    if parallel:
        # Partition df into n_cpu equal chunks
        chunk_size = (len(df) + n_cpu - 1) // n_cpu  # ceiling division
        chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n_cpu)]
        # Create job arguments
        job_args = [(deepcopy(grid), chunk) for chunk in chunks]
        # Run parallel
        with Pool() as pool: results = pool.starmap(run_full_pf, job_args)
        # Concatenate results
        ext_imports = pd.concat([results[i][0] for i in range(len(results))], axis=0).reset_index(drop=True)
        vm = pd.concat([results[i][1] for i in range(len(results))], axis=0).reset_index(drop=True)
        line_loads = pd.concat([results[i][2] for i in range(len(results))], axis=0).reset_index(drop=True)
    else:
        ext_imports, vm, line_loads = run_full_pf(grid, df)
    
    return ext_imports, vm, line_loads