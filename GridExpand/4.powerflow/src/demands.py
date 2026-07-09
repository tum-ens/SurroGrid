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
- Optionally writes `pwrflw/urbs_out/MILP/reactive` to the output `.h5` for traceability.

Important conventions:

- Reactive power is derived from fixed power factors in `config.py`.
- Inductive/lagging demand is represented as negative Q.
"""

from config import config
import pandas as pd
import numpy as np


def _use_t_as_index(df):
    if not isinstance(df.index, pd.MultiIndex):
        return df.copy()
    if "t" not in df.index.names:
        return df.copy()
    result = df.copy()
    result.index = result.index.get_level_values("t")
    result.index.name = "t"
    return result


def _drop_tsam_initial_timestep(df_pre_demand):
    df_pre_demand = _use_t_as_index(df_pre_demand)
    if len(df_pre_demand) > 1 and df_pre_demand.index.min() == 0:
        return df_pre_demand.iloc[1:].copy()
    return df_pre_demand


def _align_pre_demand_to_urbs(df_pre_demand, df_urbs_demand):
    df_pre_demand = _use_t_as_index(df_pre_demand)
    urbs_timesteps = df_urbs_demand.index.get_level_values("t").nunique()
    if len(df_pre_demand) == urbs_timesteps + 1 and df_pre_demand.index.min() == 0:
        df_pre_demand = df_pre_demand.iloc[1:].copy()
    if len(df_pre_demand) != urbs_timesteps:
        raise ValueError(
            "Pre-demand and urbs output have incompatible timesteps: "
            f"pre={len(df_pre_demand)}, urbs={urbs_timesteps}."
        )
    df_pre_demand.index = range(len(df_pre_demand))
    return df_pre_demand


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

def _reference_timestep_count(reference):
    if reference is None:
        return None
    if isinstance(reference.index, pd.MultiIndex) and "t" in reference.index.names:
        return reference.index.get_level_values("t").nunique()
    return len(reference)


def _align_table_to_timesteps(df, timesteps, label, reference_label):
    df = _use_t_as_index(df)
    if len(df) == timesteps + 1 and df.index.min() == 0:
        df = df.iloc[1:].copy()
    if len(df) != timesteps:
        raise ValueError(
            f"{label} and {reference_label} have incompatible timesteps: "
            f"{label}={len(df)}, {reference_label}={timesteps}."
        )
    df.index = range(len(df))
    return df


def _empty_electricity_frame(index):
    return pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples([]), dtype=float)


def _columns_with_component(df, component):
    if df.empty or getattr(df.columns, "nlevels", 1) < 2:
        return []
    return [column for column in df.columns if str(column[1]) == component]


def _columns_starting_with(df, prefix):
    if df.empty or getattr(df.columns, "nlevels", 1) < 2:
        return []
    return [column for column in df.columns if str(column[1]).startswith(prefix)]


def _sum_columns_by_bus(df, component="electricity"):
    if df.empty:
        return _empty_electricity_frame(df.index)
    summed = df.T.groupby(level=0).sum().T
    summed.columns = pd.MultiIndex.from_tuples([(bus, component) for bus in summed.columns])
    return summed


def _heat_pump_electricity(df_raw_demand, df_eff_factor):
    heat_columns = [
        column for column in df_raw_demand.columns
        if getattr(df_raw_demand.columns, "nlevels", 1) >= 2 and str(column[1]) in {"space_heat", "water_heat"}
    ]
    if not heat_columns:
        return _empty_electricity_frame(df_raw_demand.index)

    cop_columns = _columns_with_component(df_eff_factor, "heatpump_air")
    if not cop_columns:
        raise ValueError("No-flex post demand requires heatpump_air COP columns in eff_factor when heat demand is present.")

    heat_by_bus = df_raw_demand.loc[:, heat_columns].T.groupby(level=0).sum().T
    cop_by_bus = df_eff_factor.loc[:, cop_columns].copy()
    cop_by_bus.columns = cop_by_bus.columns.get_level_values(0)
    cop_by_bus = cop_by_bus.T.groupby(level=0).mean().T
    cop_by_bus = cop_by_bus.reindex(columns=heat_by_bus.columns)
    heat_electricity = heat_by_bus.divide(cop_by_bus.replace(0, np.nan)).fillna(0.0)
    heat_electricity.columns = pd.MultiIndex.from_tuples([(bus, "electricity") for bus in heat_electricity.columns])
    return heat_electricity


def _redistribute_ev_energy(energy, availability, charger_kw):
    energy_values = np.asarray(energy, dtype=float)
    available = np.asarray(availability, dtype=float) > 0.5
    result = np.zeros_like(energy_values, dtype=float)
    assigned_energy = 0.0
    i = 0
    while i < len(energy_values):
        if not available[i]:
            i += 1
            continue
        j = i
        while j < len(energy_values) and available[j]:
            j += 1

        required = float(energy_values[i:j].sum())
        assigned_energy += required
        k = i
        while required > 1e-9 and k < j:
            charge = min(float(charger_kw), required)
            result[k] = charge
            required -= charge
            k += 1
        if required > 1e-9:
            result[j - 1] += required
        i = j

    total_energy = float(energy_values.sum())
    if abs(total_energy - assigned_energy) > 1e-6:
        raise ValueError(
            "No-flex EV redistribution found mobility energy outside home availability. "
            f"total={total_energy:.6f} kWh, assigned={assigned_energy:.6f} kWh."
        )
    return result


def _mobility_electricity(df_raw_demand, df_eff_factor, ev_charger_kw):
    mobility_columns = _columns_starting_with(df_raw_demand, "mobility")
    if not mobility_columns:
        return _empty_electricity_frame(df_raw_demand.index)

    direct_profiles = {}
    for column in mobility_columns:
        bus, label = column
        suffix = str(label).replace("mobility", "")
        availability_column = (bus, f"charging_station{suffix}")
        if availability_column not in df_eff_factor.columns:
            raise ValueError(f"Missing EV availability column {availability_column!r} for mobility column {column!r}.")
        direct_profiles[column] = _redistribute_ev_energy(
            df_raw_demand[column],
            df_eff_factor[availability_column],
            ev_charger_kw,
        )

    mobility = pd.DataFrame(direct_profiles, index=df_raw_demand.index)
    return _sum_columns_by_bus(mobility)


def _pv_generation(df_supim, df_process):
    if df_supim.empty or df_process.empty:
        return _empty_electricity_frame(df_supim.index)

    process = df_process.reset_index()
    required = {"Site", "Process", "cap-up"}
    if not required.issubset(process.columns):
        return _empty_electricity_frame(df_supim.index)

    parts = []
    labels = []
    for _, row in process.iterrows():
        process_name = str(row["Process"])
        if not process_name.startswith("Rooftop PV"):
            continue
        commodity = process_name.replace("Rooftop PV", "solar", 1)
        column = (row["Site"], commodity)
        if column not in df_supim.columns:
            continue
        parts.append(pd.to_numeric(df_supim[column], errors="coerce").fillna(0.0) * float(row["cap-up"]))
        labels.append((row["Site"], commodity))

    if not parts:
        return _empty_electricity_frame(df_supim.index)
    pv = pd.concat(parts, axis=1, keys=labels)
    return _sum_columns_by_bus(pv)


def _reactive_from_no_flex_components(df_pre_demand_react, df_heat_elec, df_pv_elec):
    heat_react = df_heat_elec * np.tan(np.arccos(config.PF_HP)) * (-1)
    heat_react.columns = heat_react.columns.map(lambda x: (x[0], "electricity-reactive"))

    react_without_pv = df_pre_demand_react.add(heat_react, fill_value=0.0)
    if df_pv_elec.empty:
        pv_react = pd.DataFrame(index=df_pre_demand_react.index, columns=pd.MultiIndex.from_tuples([]), dtype=float)
        return react_without_pv, pv_react, heat_react

    pv_limit = df_pv_elec * np.tan(np.arccos(config.PF_PV_MIN))
    pv_limit.columns = pv_limit.columns.map(lambda x: (x[0], "electricity-reactive"))
    ideal_pv_react = -react_without_pv.reindex(columns=pv_limit.columns, fill_value=0.0)
    pv_react = ideal_pv_react.clip(lower=-pv_limit, upper=pv_limit)
    post_react = react_without_pv.add(pv_react, fill_value=0.0)
    return post_react, pv_react, heat_react


def _process_no_flex_demands(no_flex_inputs, df_pre_demand_elec, df_pre_demand_react, ev_charger_kw):
    reference = no_flex_inputs.get("reference")
    timesteps = _reference_timestep_count(reference)
    reference_label = "urbs output" if reference is not None else "raw no-flex demand"
    if timesteps is None:
        timesteps = len(_use_t_as_index(no_flex_inputs["demand"]))

    df_raw_demand = _align_table_to_timesteps(no_flex_inputs["demand"], timesteps, "No-flex demand", reference_label)
    df_eff_factor = _align_table_to_timesteps(no_flex_inputs["eff_factor"], timesteps, "No-flex eff_factor", reference_label)
    df_supim = _align_table_to_timesteps(no_flex_inputs["supim"], timesteps, "No-flex supim", reference_label)
    df_process = no_flex_inputs["process"]

    df_heat_elec = _heat_pump_electricity(df_raw_demand, df_eff_factor)
    df_ev_elec = _mobility_electricity(df_raw_demand, df_eff_factor, ev_charger_kw)
    df_pv_elec = _pv_generation(df_supim, df_process)

    active_parts = [df_pre_demand_elec, df_heat_elec, df_ev_elec, -df_pv_elec]
    df_post_demand_elec = pd.concat(active_parts, axis=1).T.groupby(level=0).sum().T
    df_post_demand_elec.columns = pd.MultiIndex.from_tuples([(bus, "electricity") for bus in df_post_demand_elec.columns])

    df_post_demand_react, df_prod_PV_react, df_demand_HP_react = _reactive_from_no_flex_components(
        df_pre_demand_react,
        df_heat_elec,
        df_pv_elec,
    )
    df_react_save = _concat_react_demands(df_pre_demand_react.copy(), df_demand_HP_react, df_prod_PV_react)
    return df_post_demand_elec, df_post_demand_react, df_react_save


def obtain_pre_demand(SF):
    df_raw_demand = SF.get_pre_demand()
    if SF.uses_reduced_demand():
        df_raw_demand = _drop_tsam_initial_timestep(df_raw_demand)
        df_raw_demand.index = range(len(df_raw_demand))
    df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
    return pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)

def obtain_demand(SF, save_reactive=True, post_demand_mode="flexible", ev_charger_kw=None, no_flex_source="auto"):
    if post_demand_mode not in {"flexible", "no-flex"}:
        raise ValueError("post_demand_mode must be 'flexible' or 'no-flex'.")

    if post_demand_mode == "flexible":
        df_raw_demand, df_urbs_demand = SF.get_input_demands()
        df_raw_demand = _align_pre_demand_to_urbs(df_raw_demand, df_urbs_demand)
        df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
        df_post_demand_elec, df_post_demand_react, df_react_save = _process_post_demands(df_urbs_demand, df_pre_demand_react)
    else:
        charger_kw = config.EV_HOME_CHARGER_KW if ev_charger_kw is None else float(ev_charger_kw)
        if charger_kw <= 0:
            raise ValueError("ev_charger_kw must be greater than zero.")
        no_flex_inputs = SF.get_no_flex_inputs(source=no_flex_source)
        reference = no_flex_inputs.get("reference")
        timesteps = _reference_timestep_count(reference)
        if timesteps is None:
            timesteps = len(_use_t_as_index(no_flex_inputs["demand"]))
        reference_label = "urbs output" if reference is not None else "raw no-flex demand"
        df_raw_demand = _align_table_to_timesteps(no_flex_inputs["demand"], timesteps, "No-flex demand", reference_label)
        df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
        df_post_demand_elec, df_post_demand_react, df_react_save = _process_no_flex_demands(
            no_flex_inputs,
            df_pre_demand_elec,
            df_pre_demand_react,
            charger_kw,
        )
    if save_reactive:
        SF.save_df(df_react_save, "pwrflw/urbs_out/MILP/reactive")

    # Concat demands
    df_pre_demand = pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)
    df_post_demand = pd.concat([df_post_demand_elec, df_post_demand_react], axis=1)

    return df_pre_demand, df_post_demand
