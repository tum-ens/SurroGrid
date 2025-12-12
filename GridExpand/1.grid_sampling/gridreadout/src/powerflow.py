"""Small pandapower power-flow helpers.

This module is mainly used for quick validation/experiments.

Note: `read_net()` currently expects the pandapower JSON at the HDF5 key
`grid_top/net`. The sampling notebooks in this step write to `/raw_data/net`.
If you want to run these helpers on freshly sampled files, either adjust the
key here or write an additional copy under `grid_top/net`.
"""

import pandapower as pp
import pandas as pd
import numpy as np
import h5py

def read_net(file_path):
    with h5py.File(file_path, 'r') as f:
        json_data = f['grid_top/net'][()]
    net = pp.from_json_string(json_data)
    return net


def remove_limits(net):
    """ Set max_i to very high value. Remove transformer and replace with small line. """
    # Increase max line capacity
    df_lines = net.line
    df_lines["max_i_ka"] = 1000
    net.line = df_lines

    # Remove load max restrictions
    df_loads = net.load
    df_loads["max_p_mw"] = 100
    net.load = df_loads

    # Remove voltage restrictions
    df_buses = net.bus
    df_buses[["min_vm_pu", "max_vm_pu"]] = (0, 10)
    # net.bus = df_buses

    # Replace trafo with switch
    hv_bus = int(net.trafo.loc[0, "hv_bus"])
    lv_bus = int(net.trafo.loc[0, "lv_bus"])
    pp.drop_trafos(net, [0])
    switch_idx = pp.create_switch(net, bus=hv_bus, element=lv_bus, et="b", closed=True, type="CB")

    df_buses.loc[hv_bus, "vn_kv"] = df_buses.loc[lv_bus, "vn_kv"]
    net.bus = df_buses

    return net


def generate_random_loads(consumers):
    # Define the bus values and time index
    bus_values = consumers["bus"].to_list()
    time_index = np.arange(0, 11)

    # Create MultiIndex for columns
    columns = pd.MultiIndex.from_product([bus_values, ['p_mw', 'q_mvar']], names=['bus', 'power_type'])

    # Create a DataFrame with the time index and MultiIndex columns
    # data = np.array([[0.001 + 0.001 * t] * len(bus_values) * 2 for t in time_index])
    data = np.array([[0.05] * len(bus_values) * 2 for t in time_index])
    df = pd.DataFrame(data, index=time_index, columns=columns)
    df.index.name = 't'
    return df


def run_single_pf(net, new_load):
    df_load = net.load
    df_load = df_load.drop(['p_mw', 'q_mvar'], axis=1).merge(
        new_load, on='bus', how='left')
    net.load = df_load
    # IMPLEMENT: Why does this not converge??
    pp.runpp(net, algorithm="nr", max_iteration=200, tolerance_mva=1e-3)

    return net


def run_full_pf(net, new_loads):
    ext_grid_bus = int(net.ext_grid.loc[0, "bus"])  # Which bus is the external import bus 
    line_loadings = pd.DataFrame()
    ext_imports = pd.DataFrame()

    for i, row in new_loads.iterrows():
        # Unstack the row to pivot the power_type level into columns, resulting in a DataFrame with index as 'bus' and columns as 'p_mw' and 'q_mvar'
        reshaped_load = row.unstack(level='power_type')
        reshaped_load.index.name = 'bus'
        reshaped_load = reshaped_load.reset_index()
        reshaped_load.columns.name = None

        # Run powerflow
        net_res = run_single_pf(net, reshaped_load)

        # Concat line loadings
        cur_line_load = pd.DataFrame(net_res.res_line["i_ka"]).T
        if line_loadings.empty:
            line_loadings = cur_line_load
            line_loadings.index.name = 't'
        else:
            line_loadings = pd.concat([line_loadings, cur_line_load], axis=0, ignore_index=True)

        # Concat ext import
        cur_ext_import = pd.DataFrame(net_res.res_bus.loc[ext_grid_bus, ["p_mw","q_mvar"]]).T
        if ext_imports.empty:
            ext_imports = cur_ext_import
            ext_imports.index.name = 't'
        else:
            ext_imports = pd.concat([ext_imports, cur_ext_import], axis=0, ignore_index=True) 

    return line_loadings, ext_imports