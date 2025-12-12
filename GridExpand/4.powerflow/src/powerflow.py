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

import pandapower as pp
import pandas as pd

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


def run_single_pf(grid, new_load):
    # 1. Ensure that 'bus' is the index in both DataFrames
    df_load = grid.load.copy()
    df_load_indexed = df_load.set_index('bus')
    new_load_indexed = new_load.set_index('bus')
    # 2. Use DataFrame.update to overwrite only the overlapping entries
    df_load_indexed.update(new_load_indexed)
    # 3. If you need to restore 'bus' as a column rather than the index:
    df_load_updated = df_load_indexed.reset_index()

    grid.load = df_load_updated
    pp.runpp(grid, algorithm="bfsw", max_iteration=50, tolerance_mva=1e-6)
    return grid


def run_full_pf(grid, df_demand):
    ext_imports_list = []
    line_loads_list = []
    vm_list = []

    for i, row in df_demand.iterrows():
        ### Prepare
        # Unstack the row to pivot the power_type level into columns, resulting in a DataFrame with index as 'bus' and columns as 'p_mw' and 'q_mvar'
        reshaped_load = row.unstack(level=1)
        reshaped_load.index.name = 'bus'
        reshaped_load = reshaped_load.reset_index()
        reshaped_load.rename(columns={"electricity":"p_mw", "electricity-reactive":"q_mvar"}, inplace=True)
        reshaped_load[["p_mw", "q_mvar"]] = reshaped_load[["p_mw", "q_mvar"]]/1000
        
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