"""Demand reconstruction for time-series power flow.

This module converts the scenario demand data stored in the input `.h5` file
into per-bus active and reactive power time series that can be fed into
pandapower.

Inputs (via `SaveFile.get_input_demands()`):

- `urbs_in/demand`: pre-expansion household electricity demand (active power)
- `urbs_out/MILP/tau_pro`: post-expansion urbs results used to reconstruct:
    - net electricity import (import - feed_in)
    - heat pump electricity consumption
    - rooftop PV production

Outputs:

- Returns `(df_pre_demand, df_post_demand)` with MultiIndex columns identifying
    site/bus and power component (`electricity` and `electricity-reactive`).
- Writes `pwrflw/urbs_out/MILP/reactive` to the output `.h5` for traceability.

Important conventions:

- Reactive power is derived from fixed power factors in `config.py`.
- Inductive/lagging demand is represented as negative Q.
"""

from config import config
import pandas as pd
import numpy as np


def _process_pre_demands(df_pre_demand):
    ### Pre-urbs raw household (reactive) electrical demand
    df_raw_demand_elec = df_pre_demand.loc[:, df_pre_demand.columns.get_level_values(1) == 'electricity']
    df_raw_demand_react = df_raw_demand_elec.copy()*np.tan(np.arccos(config.PF_ELC))*(-1)   # -1 as inductive/lagging and thus a demand
    df_raw_demand_react.columns = df_raw_demand_react.columns.map(lambda x: (x[0], "electricity-reactive"))
    return df_raw_demand_elec, df_raw_demand_react

def _extract_relevant_demands(df_net_demand):
    ### Post-urbs net imported elec/react
    # 1. The list of 'pro' values to keep:
    pro_vals = ["import", "feed_in", "heatpump_air"]
    # 2. Make a boolean mask on the 'pro' level of the row‐index
    pro_level = df_net_demand.index.get_level_values("pro")
    mask = pro_level.isin(pro_vals) | pro_level.str.startswith("Rooftop")
    # 3. Filter to those rows only
    df_net_demand = df_net_demand[mask]
    # 4. Reset the index so that 'sit', 'pro', and 't' become ordinary rows
    df_net_demand = df_net_demand.reset_index().drop(columns=["stf"])
    # 5. Pivot row indices to column indices:
    df_net_demand = df_net_demand.pivot(
        index="t",
        columns=["sit", "pro"],
        values="tau_pro")
    df_net_demand.reset_index(drop=True, inplace=True)  # To start counting rows from 0 instead of 1

    # 6. Subtract feed-ins from imports to get net import, then drop feed-ins:
    sites = df_net_demand.columns.get_level_values(0).unique()
    # Adjust imports
    for site in sites:
        df_net_demand[(site, 'import')] -= df_net_demand[(site, 'feed_in')]
    # Remove feed-in columns
    to_drop = [col for col in df_net_demand.columns if col[1] == 'feed_in']
    df_net_demand = df_net_demand.drop(columns=to_drop)
    df_net_demand.rename(columns={"import":"electricity"}, inplace=True)

    # 7. Split by net elec and HP,PV (needed for their reactive power) 
    df_net_demand_elec = df_net_demand.loc[:, df_net_demand.columns.get_level_values("pro") == 'electricity']
    df_demand_HP_elec = df_net_demand.loc[:, df_net_demand.columns.get_level_values("pro") == 'heatpump_air']
    df_prod_PV_elec = df_net_demand.loc[:, df_net_demand.columns.get_level_values("pro").str.startswith("Rooftop")]
    # 8. Sum all PV productions for a single site
    df_prod_PV_elec = df_prod_PV_elec.T.groupby(level=0).sum().T
    df_prod_PV_elec.columns = pd.MultiIndex.from_product([df_prod_PV_elec.columns, ["solar"]])

    return df_net_demand_elec, df_demand_HP_elec, df_prod_PV_elec

def _obtain_post_reactive_power(df_pre_demand_react, df_demand_HP_elec, df_prod_PV_elec):
    df_pre_demand_react.index.name    = None
    df_demand_HP_elec.index.name      = None
    df_prod_PV_elec.index.name        = None
    df_demand_HP_elec.columns.names   = [None,None]
    df_pre_demand_react.columns.names = [None,None]
    df_prod_PV_elec.columns.names     = [None,None]
    
    ### Heat pump
    df_demand_HP_react = df_demand_HP_elec*np.tan(np.arccos(config.PF_HP))*(-1) # -1 as inductive/lagging and thus a demand
    df_demand_HP_react.columns = df_demand_HP_react.columns.map(lambda x: (x[0], "electricity-reactive"))

    ### Determine PV as optimal operation between -tan(phi) <= Q/P <= tan(phi) to obtain minimal reactive power demand from grid
    # Constraints as above:
    upper_constraint = df_prod_PV_elec*np.tan(np.arccos(config.PF_PV_MIN))
    upper_constraint.columns = upper_constraint.columns.map(lambda x: (x[0], "electricity-reactive"))
    lower_constraint = -upper_constraint
    # Ideal PV react production would be cancelling out other react demands

    df_prod_PV_react = -(df_pre_demand_react + df_demand_HP_react)
    # Now clip ideal PV production to constraints
    df_prod_PV_react = df_prod_PV_react.clip(lower=lower_constraint, upper=upper_constraint)

    ### Total reactive demand
    df_post_demand_react = df_prod_PV_react + df_pre_demand_react + df_demand_HP_react

    return df_post_demand_react, df_prod_PV_react, df_demand_HP_react

def _concat_react_demands(df_HH_reactive, df_HP_reactive, df_PV_reactive):
    ### Convert PV, HP, HH react demand to be saved as urbs-output
    new_tuples = [tup + ('household',) for tup in df_HH_reactive.columns.to_flat_index()]
    new_columns = pd.MultiIndex.from_tuples(new_tuples, names=[None, *df_HH_reactive.columns.names])
    df_HH_reactive.columns = new_columns

    new_tuples = [tup + ('heatpump_air',) for tup in df_HP_reactive.columns.to_flat_index()]
    new_columns = pd.MultiIndex.from_tuples(new_tuples, names=[None, *df_HP_reactive.columns.names])
    df_HP_reactive.columns = new_columns

    new_tuples = [tup + ('solar',) for tup in df_PV_reactive.columns.to_flat_index()]
    new_columns = pd.MultiIndex.from_tuples(new_tuples, names=[None, *df_PV_reactive.columns.names])
    df_PV_reactive.columns = new_columns

    df_react_save = pd.concat([df_HH_reactive, df_HP_reactive, df_PV_reactive], axis=1)
    return df_react_save

def _process_post_demands(df_urbs_demand, df_pre_demand_react):
    # Obtain demand after urbs simulation which are necessary for reactive power calculation
    df_post_demand_elec, df_demand_HP_elec, df_prod_PV_elec = _extract_relevant_demands(df_urbs_demand)
    # Obtain reactive demands post urbs
    df_post_demand_react, df_prod_PV_react, df_demand_HP_react = _obtain_post_reactive_power(df_pre_demand_react, df_demand_HP_elec, df_prod_PV_elec)
    # Get reactive demands of HP,HH,PV as concate output to be saved:
    df_react_save = _concat_react_demands(df_pre_demand_react.copy(), df_demand_HP_react, df_prod_PV_react)

    return df_post_demand_elec, df_post_demand_react, df_react_save

def obtain_demand(SF):
    # Read-out demands:
    df_raw_demand, df_urbs_demand = SF.get_input_demands()

    # Obtain pre-urbs raw household demands as imports
    df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
    # Extract post-urbs imports
    df_post_demand_elec, df_post_demand_react, df_react_save = _process_post_demands(df_urbs_demand, df_pre_demand_react)
    SF.save_df(df_react_save, "pwrflw/urbs_out/MILP/reactive")

    # Concat demands
    df_pre_demand = pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)
    df_post_demand = pd.concat([df_post_demand_elec, df_post_demand_react], axis=1)

    return df_pre_demand, df_post_demand