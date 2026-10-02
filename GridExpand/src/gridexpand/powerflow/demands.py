"""Demand reconstruction for time-series power flow.

This module converts the scenario demand data stored in the input `.h5` file
into per-bus active and reactive power time series that can be fed into
pandapower.

Inputs (via `io.ScenarioResultReader`):

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
- Net consumption uses pandapower's load convention: positive P consumes active
  power; positive Q absorbs inductive reactive power. PV compensation contributes
  negative Q to this net-load representation.
"""

from gridexpand.powerflow.config import config
import pandas as pd
import numpy as np

from gridexpand.common.ev_sessions import (
    ENERGY_TOL_KWH as SESSION_ENERGY_TOL_KWH,
    POWER_TOL_KW as SESSION_POWER_TOL_KW,
    earliest_feasible_schedule,
    validate_sessions,
)


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
    df_raw_demand_elec = df_pre_demand.loc[
        :, df_pre_demand.columns.get_level_values(1) == "electricity"
    ]
    # Normalize level names before arithmetic with reconstructed heat, mobility,
    # PV, and battery frames; their semantic alignment is positional.
    df_raw_demand_elec = _set_electricity_component(
        df_raw_demand_elec, "electricity"
    )
    df_raw_demand_react = (
        df_raw_demand_elec.copy() * np.tan(np.arccos(config.PF_ELC))
    )
    df_raw_demand_react = _set_electricity_component(
        df_raw_demand_react, "electricity-reactive"
    )
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
    """Return net, PV and HP Q in the consumer convention for both dispatch modes.

    HP electricity and PV production are positive component magnitudes. PV Q is
    a signed contribution to net load, not a pandapower sgen setpoint.
    """
    df_pre_demand_react.index.name    = None
    df_demand_HP_elec.index.name      = None
    df_prod_PV_elec.index.name        = None
    df_demand_HP_elec.columns.names   = [None,None]
    df_pre_demand_react.columns.names = [None,None]
    df_prod_PV_elec.columns.names     = [None,None]
    
    ### Heat pump
    df_demand_HP_react = df_demand_HP_elec * np.tan(np.arccos(config.PF_HP))
    df_demand_HP_react = _set_electricity_component(
        df_demand_HP_react,
        "electricity-reactive",
    )

    react_without_pv = df_pre_demand_react.add(
        df_demand_HP_react,
        fill_value=0.0,
    )
    if df_prod_PV_elec.empty:
        df_prod_PV_react = _empty_electricity_frame(df_pre_demand_react.index)
        return react_without_pv, df_prod_PV_react, df_demand_HP_react

    # Local compensation: minimize |Q_load + Q_PV| within the assumed
    # generation-dependent inverter limit. At zero PV output the limit is zero.
    upper_constraint = df_prod_PV_elec*np.tan(np.arccos(config.PF_PV_MIN))
    upper_constraint = _set_electricity_component(
        upper_constraint,
        "electricity-reactive",
    )
    ideal_pv_react = -react_without_pv.reindex(
        columns=upper_constraint.columns,
        fill_value=0.0,
    )
    df_prod_PV_react = ideal_pv_react.clip(
        lower=-upper_constraint,
        upper=upper_constraint,
    )
    df_post_demand_react = react_without_pv.add(
        df_prod_PV_react,
        fill_value=0.0,
    )

    return df_post_demand_react, df_prod_PV_react, df_demand_HP_react

def _concat_react_demands(df_HH_reactive, df_HP_reactive, df_PV_reactive):
    ### Convert PV, HP, HH react demand to be saved as urbs-output
    df_HH_reactive = _append_component_level(df_HH_reactive, "household")
    df_HP_reactive = _append_component_level(df_HP_reactive, "heatpump_air")
    df_PV_reactive = _append_component_level(df_PV_reactive, "solar")

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

def _reference_timestep_count(reference, drop_initial_timestep=False):
    if reference is None:
        return None
    if isinstance(reference.index, pd.MultiIndex) and "t" in reference.index.names:
        timesteps = reference.index.get_level_values("t").nunique()
        first_timestep = reference.index.get_level_values("t").min()
    else:
        timesteps = len(reference)
        first_timestep = reference.index.min() if len(reference) else None
    if drop_initial_timestep and timesteps > 1 and first_timestep == 0:
        return timesteps - 1
    return timesteps


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
    columns = pd.MultiIndex.from_arrays([[], []], names=[None, None])
    return pd.DataFrame(index=index, columns=columns, dtype=float)


def _set_electricity_component(df, component):
    result = df.copy()
    if result.shape[1] == 0:
        return _empty_electricity_frame(result.index)
    result.columns = pd.MultiIndex.from_tuples(
        [(column[0], component) for column in result.columns],
        names=[None, None],
    )
    return result


def _append_component_level(df, component):
    result = df.copy()
    if result.shape[1] == 0:
        result.columns = pd.MultiIndex.from_arrays(
            [[], [], []],
            names=[None, None, None],
        )
        return result
    result.columns = pd.MultiIndex.from_tuples(
        [tuple(column) + (component,) for column in result.columns.to_flat_index()],
        names=[None, *result.columns.names],
    )
    return result


def _columns_with_component(df, component):
    if df.empty or getattr(df.columns, "nlevels", 1) < 2:
        return []
    return [column for column in df.columns if str(column[1]) == component]


def _sum_columns_by_bus(df, component="electricity"):
    if df.empty:
        return _empty_electricity_frame(df.index)
    summed = df.T.groupby(level=0).sum().T
    summed.columns = pd.MultiIndex.from_tuples([(bus, component) for bus in summed.columns])
    return summed


def project_scenario_units_to_buses(df, allocation):
    """Project canonical scenario-unit columns onto one target network."""
    if df is None or df.empty:
        return df
    required = {"scenario_unit_id", "allocation_bus"}
    missing = required.difference(allocation.columns)
    if missing:
        raise ValueError(
            "Scenario-unit projection requires allocation columns "
            f"{sorted(missing)}."
        )
    mapping = allocation[["scenario_unit_id", "allocation_bus"]].copy()
    mapping["scenario_unit_id"] = pd.to_numeric(
        mapping["scenario_unit_id"], errors="raise"
    ).astype(int)
    mapping["allocation_bus"] = pd.to_numeric(
        mapping["allocation_bus"], errors="raise"
    ).astype(int)
    ambiguous = mapping.groupby("scenario_unit_id", observed=True)[
        "allocation_bus"
    ].nunique()
    if ambiguous.gt(1).any():
        units = ambiguous[ambiguous.gt(1)].index.tolist()[:10]
        raise ValueError(
            "Scenario units map to multiple buses within one target plan: "
            f"{units}."
        )
    bus_by_unit = (
        mapping.drop_duplicates("scenario_unit_id")
        .set_index("scenario_unit_id")["allocation_bus"]
        .to_dict()
    )
    if getattr(df.columns, "nlevels", 1) < 2:
        raise ValueError("Projected power-flow demand requires MultiIndex columns.")
    projected_columns = []
    missing_units = set()
    for column in df.columns.to_flat_index():
        unit = int(column[0])
        if unit not in bus_by_unit:
            missing_units.add(unit)
            continue
        projected_columns.append((bus_by_unit[unit], *column[1:]))
    if missing_units:
        raise ValueError(
            "Demand contains scenario units absent from the target plan: "
            f"{sorted(missing_units)[:10]}."
        )
    projected = df.copy()
    projected.columns = pd.MultiIndex.from_tuples(projected_columns)
    levels = list(range(projected.columns.nlevels))
    projected = projected.T.groupby(level=levels, observed=True, sort=False).sum().T
    return projected.sort_index(axis=1)


def _heat_and_cop_by_bus(df_raw_demand, df_eff_factor):
    heat_columns = [
        column for column in df_raw_demand.columns
        if getattr(df_raw_demand.columns, "nlevels", 1) >= 2 and str(column[1]) in {"space_heat", "water_heat"}
    ]
    if not heat_columns:
        empty = _empty_electricity_frame(df_raw_demand.index)
        return empty, empty

    cop_columns = _columns_with_component(df_eff_factor, "heatpump_air")
    if not cop_columns:
        raise ValueError("INFLEX post demand requires heatpump_air COP columns in eff_factor when heat demand is present.")

    heat_by_bus = df_raw_demand.loc[:, heat_columns].T.groupby(level=0).sum().T
    cop_by_bus = df_eff_factor.loc[:, cop_columns].copy()
    cop_by_bus.columns = cop_by_bus.columns.get_level_values(0)
    cop_by_bus = cop_by_bus.T.groupby(level=0).mean().T
    cop_by_bus = cop_by_bus.reindex(columns=heat_by_bus.columns)
    return heat_by_bus, cop_by_bus


def _capacity_by_bus(cap_pro, process, buses):
    if cap_pro is None:
        raise ValueError("INFLEX heat split requires optimized post-flex cap_pro results.")
    if not isinstance(cap_pro.index, pd.MultiIndex):
        raise ValueError("INFLEX heat split expects cap_pro with MultiIndex levels stf, sit, pro.")
    if "sit" not in cap_pro.index.names or "pro" not in cap_pro.index.names:
        raise ValueError("INFLEX heat split expects cap_pro index levels named 'sit' and 'pro'.")

    process_mask = cap_pro.index.get_level_values("pro") == process
    process_caps = pd.to_numeric(cap_pro.loc[process_mask], errors="coerce").fillna(0.0)
    if process_caps.empty:
        return pd.Series(0.0, index=buses, dtype=float)

    by_site = process_caps.groupby(level="sit").sum()
    by_site_lookup = {str(site): float(value) for site, value in by_site.items()}
    return pd.Series([by_site_lookup.get(str(bus), 0.0) for bus in buses], index=buses, dtype=float)


def _input_capacity_by_bus(process, process_name, buses):
    """Read fixed installed process capacities from an urbs input table."""
    if process is None or process.empty:
        raise ValueError("INFLEX heat dispatch requires urbs process inputs.")
    frame = process.reset_index() if isinstance(process.index, pd.MultiIndex) else process.copy()
    site_column = "Site" if "Site" in frame else "sit"
    process_column = "Process" if "Process" in frame else "pro"
    capacity_column = "inst-cap" if "inst-cap" in frame else "inst_cap"
    required = {site_column, process_column, capacity_column}
    if not required.issubset(frame.columns):
        raise ValueError("INFLEX heat dispatch cannot identify Site, Process, and inst-cap columns.")
    selected = frame[frame[process_column].astype(str).eq(process_name)].copy()
    selected[capacity_column] = pd.to_numeric(selected[capacity_column], errors="coerce").fillna(0.0)
    by_site = selected.groupby(site_column)[capacity_column].sum()
    lookup = {str(site): float(value) for site, value in by_site.items()}
    return pd.Series([lookup.get(str(bus), 0.0) for bus in buses], index=buses, dtype=float)


def _inflex_heat_electricity(df_raw_demand, df_eff_factor, process):
    heat_by_bus, cop_by_bus = _heat_and_cop_by_bus(df_raw_demand, df_eff_factor)
    if heat_by_bus.empty:
        empty = _empty_electricity_frame(df_raw_demand.index)
        return empty, empty, empty

    buses = list(heat_by_bus.columns)
    hp_capacity_el = _input_capacity_by_bus(process, "heatpump_air", buses)
    booster_capacity_el = _input_capacity_by_bus(process, "heatpump_booster", buses)
    cop_safe = cop_by_bus.replace(0, np.nan)
    hp_thermal_limit = cop_safe.multiply(hp_capacity_el, axis=1).fillna(0.0)
    hp_heat = heat_by_bus.where(heat_by_bus.le(hp_thermal_limit), hp_thermal_limit)
    auxiliary_heat = (heat_by_bus - hp_heat).clip(lower=0.0)
    excess = auxiliary_heat.subtract(booster_capacity_el, axis=1).clip(lower=0.0)
    maximum_excess = float(excess.max().max()) if not excess.empty else 0.0
    if maximum_excess > 1e-6:
        raise ValueError(
            "Fixed inflex HP and auxiliary capacities cannot cover heat demand; "
            f"maximum residual is {maximum_excess:.6f} kW."
        )
    hp_electricity = hp_heat.divide(cop_safe).fillna(0.0)
    auxiliary_electricity = auxiliary_heat
    total_electricity = hp_electricity.add(auxiliary_electricity, fill_value=0.0)
    for frame in (total_electricity, hp_electricity, auxiliary_electricity):
        frame.columns = pd.MultiIndex.from_tuples([(bus, "electricity") for bus in frame.columns])
    print(
        "INFLEX heat split from fixed scenario inputs: "
        f"heat-pump electricity={float(hp_electricity.sum().sum()):.1f} kWh, "
        f"auxiliary electricity={float(auxiliary_electricity.sum().sum()):.1f} kWh, "
        f"auxiliary peak={float(auxiliary_electricity.sum(axis=1).max()):.3f} kW.",
        flush=True,
    )
    return total_electricity, hp_electricity, auxiliary_electricity


def _mobility_electricity(sessions, session_hours, horizon_hours, index):
    """INFLEX EV charging: fill every session from its own arrival, earliest first.

    The session tables are the shared EV service contract; both controllers serve
    exactly the same per-session energy inside exactly the same window at exactly
    the same charger rating. Charging power per vehicle comes from its own
    ``charger_kw``, never from a global default.
    """
    if sessions is None or sessions.empty:
        return _empty_electricity_frame(index)

    schedule = earliest_feasible_schedule(
        sessions, session_hours, horizon_hours=horizon_hours
    )
    if schedule.empty:
        return _empty_electricity_frame(index)

    delivered = float(schedule.to_numpy().sum())
    required = float(sessions["energy_kwh"].sum())
    if abs(delivered - required) > max(SESSION_ENERGY_TOL_KWH, 1e-9 * required):
        raise ValueError(
            "INFLEX EV schedule does not deliver the required session energy: "
            f"required={required:.9f} kWh, delivered={delivered:.9f} kWh."
        )

    charger_kw = sessions.set_index(["site", "process"])["charger_kw"]
    charger_kw = charger_kw[~charger_kw.index.duplicated()]
    residual = 0.0
    for column in schedule.columns:
        limit = float(charger_kw.loc[column])
        peak = float(schedule[column].max())
        residual = max(residual, peak - limit)
    if residual > SESSION_POWER_TOL_KW:
        raise ValueError(
            f"INFLEX EV schedule exceeds a charger rating by {residual:.9f} kW."
        )

    schedule = schedule.copy()
    schedule.index = index
    print(
        f"INFLEX EV sessions: vehicles={schedule.shape[1]}, "
        f"sessions={len(sessions)}, delivered={delivered:.1f} kWh, "
        f"max_power_residual={residual:.3e} kW.",
        flush=True,
    )
    return _sum_columns_by_bus(schedule)


def _pv_generation(df_supim, df_process, cap_pro):
    if df_supim.empty or df_process.empty:
        return _empty_electricity_frame(df_supim.index)

    process = df_process.reset_index()
    required = {"Site", "Process", "inst-cap", "cap-up"}
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
        installed_capacity_kw = float(row["inst-cap"])
        capacity_upper_kw = float(row["cap-up"])
        if np.isclose(installed_capacity_kw, capacity_upper_kw):
            # Heuristic asset plans are fixed upstream and independent of HEMS.
            pv_capacity_kw = installed_capacity_kw
        else:
            # Endogenous HEMS sizing still uses the solved process capacity.
            pv_capacity_kw = float(
                _capacity_by_bus(cap_pro, process_name, [row["Site"]]).iloc[0]
            )
        parts.append(
            pd.to_numeric(df_supim[column], errors="coerce").fillna(0.0)
            * pv_capacity_kw
        )
        labels.append((row["Site"], commodity))

    if not parts:
        return _empty_electricity_frame(df_supim.index)
    pv = pd.concat(parts, axis=1, keys=labels)
    return _sum_columns_by_bus(pv)


def _fixed_stationary_batteries(df_storage):
    """Return fixed SWF batteries; ignore generic zero-installed potentials."""
    if df_storage is None or df_storage.empty:
        return pd.DataFrame()

    storage = df_storage.reset_index()
    required = {
        "Site",
        "Storage",
        "inst-cap-c",
        "cap-up-c",
        "inst-cap-p",
        "cap-up-p",
        "eff-in",
        "eff-out",
    }
    if not required.issubset(storage.columns):
        raise ValueError(
            "INFLEX battery control requires storage columns "
            f"{sorted(required)}."
        )
    storage = storage[storage["Storage"].astype(str).eq("battery_private")].copy()
    storage["Site"] = pd.to_numeric(storage["Site"], errors="raise").astype(int)
    for column in required.difference({"Site", "Storage"}):
        storage[column] = pd.to_numeric(storage[column], errors="coerce").fillna(0.0)
    storage = storage[storage["inst-cap-c"].gt(0.0)]
    if storage.empty:
        return storage

    fixed_energy = np.isclose(storage["inst-cap-c"], storage["cap-up-c"])
    fixed_power = np.isclose(storage["inst-cap-p"], storage["cap-up-p"])
    if not bool((fixed_energy & fixed_power).all()):
        raise ValueError(
            "INFLEX battery control only accepts fixed installed capacities; "
            "found an endogenous battery investment row."
        )
    if storage["Site"].duplicated().any():
        raise ValueError("Expected at most one stationary battery per scenario unit.")
    return storage.set_index("Site").sort_index()


def _simulate_self_consumption_period(
    net_demand,
    *,
    initial_soc_kwh,
    energy_kwh,
    power_kw,
    charge_efficiency,
    discharge_efficiency,
    self_discharge_per_timestep=0.0,
    delta_t=1.0,
):
    """Greedy self-consumption dispatch with explicit losses and timestep length.

    ``net_demand`` is in kW; the returned trajectory is in kW. Energy quantities
    are in kWh. Self-discharge is applied to the stored energy at the start of
    each timestep, exactly as URBS does in ``def_storage_state_rule``.
    """
    soc = float(initial_soc_kwh)
    adjusted = np.asarray(net_demand, dtype=float).copy()
    soc_trajectory = np.empty(len(adjusted) + 1, dtype=float)
    soc_trajectory[0] = soc
    charged_kwh = 0.0
    discharged_kwh = 0.0
    self_loss_kwh = 0.0
    retention = (1.0 - float(self_discharge_per_timestep)) ** float(delta_t)
    for index, net_kw in enumerate(adjusted):
        before = soc
        soc *= retention
        self_loss_kwh += before - soc
        if net_kw < 0.0:
            charge_kw = min(
                -float(net_kw),
                power_kw,
                max(energy_kwh - soc, 0.0) / (charge_efficiency * delta_t),
            )
            soc += charge_kw * charge_efficiency * delta_t
            adjusted[index] += charge_kw
            charged_kwh += charge_kw * delta_t
        elif net_kw > 0.0:
            discharge_kw = min(
                float(net_kw),
                power_kw,
                soc * discharge_efficiency / delta_t,
            )
            soc -= discharge_kw * delta_t / discharge_efficiency
            adjusted[index] -= discharge_kw
            discharged_kwh += discharge_kw * delta_t
        # Bounds must hold after every transition, never by resetting the state.
        if soc < -1e-9 or soc > energy_kwh + 1e-9:
            raise ValueError(
                f"Stationary-battery state left its bounds at step {index}: "
                f"soc={soc:.9f} kWh, capacity={energy_kwh:.9f} kWh."
            )
        soc_trajectory[index + 1] = soc
    return adjusted, soc, charged_kwh, discharged_kwh, self_loss_kwh, soc_trajectory


def _cyclic_self_consumption_period(net_demand, battery, *, delta_t=1.0):
    energy_kwh = float(battery["inst-cap-c"])
    power_kw = float(battery["inst-cap-p"])
    charge_efficiency = float(battery["eff-in"])
    discharge_efficiency = float(battery["eff-out"])
    self_discharge = float(battery.get("discharge", 0.0) or 0.0)
    if energy_kwh <= 0.0 or power_kw <= 0.0:
        return np.asarray(net_demand, dtype=float), _empty_battery_diagnostics()
    if not 0.0 < charge_efficiency <= 1.0:
        raise ValueError("Stationary-battery charging efficiency must be in (0, 1].")
    if not 0.0 < discharge_efficiency <= 1.0:
        raise ValueError("Stationary-battery discharging efficiency must be in (0, 1].")
    if not 0.0 <= self_discharge < 1.0:
        raise ValueError(
            f"Unsupported stationary-battery self-discharge {self_discharge!r}; "
            "it must lie in [0, 1)."
        )

    initial_soc = energy_kwh / 2.0
    for _ in range(1000):
        _, final_soc, _, _, _, _ = _simulate_self_consumption_period(
            net_demand,
            initial_soc_kwh=initial_soc,
            energy_kwh=energy_kwh,
            power_kw=power_kw,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            self_discharge_per_timestep=self_discharge,
            delta_t=delta_t,
        )
        if abs(final_soc - initial_soc) <= 1e-7:
            break
        initial_soc = final_soc
    else:
        raise RuntimeError("Stationary-battery cyclic state did not converge.")

    adjusted, final_soc, charged, discharged, self_loss, trajectory = (
        _simulate_self_consumption_period(
            net_demand,
            initial_soc_kwh=initial_soc,
            energy_kwh=energy_kwh,
            power_kw=power_kw,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            self_discharge_per_timestep=self_discharge,
            delta_t=delta_t,
        )
    )

    # Validate the trajectory that is actually returned, not only the preceding
    # fixed-point iterate.
    cyclic_residual = final_soc - trajectory[0]
    tolerance = max(1e-6, 1e-9 * energy_kwh)
    if abs(cyclic_residual) > tolerance:
        raise ValueError(
            "Stationary-battery annual closure is not satisfied by the returned "
            f"trajectory: residual={cyclic_residual:.9e} kWh, tolerance={tolerance:.3e}."
        )
    balance_residual = (
        charge_efficiency * charged - discharged / discharge_efficiency - self_loss
    ) - cyclic_residual
    if abs(balance_residual) > tolerance:
        raise ValueError(
            "Stationary-battery energy balance does not close: "
            f"residual={balance_residual:.9e} kWh."
        )
    diagnostics = {
        "initial_soc_kwh": float(trajectory[0]),
        "final_soc_kwh": float(final_soc),
        "min_soc_kwh": float(trajectory.min()),
        "max_soc_kwh": float(trajectory.max()),
        "charged_kwh": float(charged),
        "discharged_kwh": float(discharged),
        "self_loss_kwh": float(self_loss),
        "cyclic_residual_kwh": float(cyclic_residual),
        "balance_residual_kwh": float(balance_residual),
    }
    return adjusted, diagnostics


def _empty_battery_diagnostics():
    return {
        "initial_soc_kwh": 0.0,
        "final_soc_kwh": 0.0,
        "min_soc_kwh": 0.0,
        "max_soc_kwh": 0.0,
        "charged_kwh": 0.0,
        "discharged_kwh": 0.0,
        "self_loss_kwh": 0.0,
        "cyclic_residual_kwh": 0.0,
        "balance_residual_kwh": 0.0,
    }


def _apply_inflex_battery_control(
    net_demand,
    df_storage,
    *,
    hours_per_period=None,
    delta_t=1.0,
):
    batteries = _fixed_stationary_batteries(df_storage)
    if batteries.empty:
        return net_demand, pd.DataFrame()

    adjusted = net_demand.copy()
    adjusted.columns = adjusted.columns.get_level_values(0)
    period_hours = int(hours_per_period or len(adjusted))
    if period_hours <= 0:
        raise ValueError("Battery-control period length must be positive.")

    total_charged = 0.0
    total_discharged = 0.0
    total_self_loss = 0.0
    max_cyclic_residual = 0.0
    max_balance_residual = 0.0
    diagnostics_rows = []
    for site, battery in batteries.iterrows():
        if site not in adjusted.columns:
            adjusted[site] = 0.0
        values = adjusted[site].to_numpy(dtype=float)
        controlled = values.copy()
        for start in range(0, len(values), period_hours):
            stop = min(start + period_hours, len(values))
            segment, diagnostics = _cyclic_self_consumption_period(
                values[start:stop], battery, delta_t=delta_t
            )
            controlled[start:stop] = segment
            total_charged += diagnostics["charged_kwh"]
            total_discharged += diagnostics["discharged_kwh"]
            total_self_loss += diagnostics["self_loss_kwh"]
            max_cyclic_residual = max(
                max_cyclic_residual, abs(diagnostics["cyclic_residual_kwh"])
            )
            max_balance_residual = max(
                max_balance_residual, abs(diagnostics["balance_residual_kwh"])
            )
            diagnostics_rows.append(
                {"site": site, "period_start": start, **diagnostics}
            )
        adjusted[site] = controlled

    adjusted = adjusted.sort_index(axis=1)
    adjusted.columns = pd.MultiIndex.from_tuples(
        [(site, "electricity") for site in adjusted.columns]
    )
    print(
        "INFLEX stationary batteries: "
        f"sites={len(batteries)}, charged={total_charged:.1f} kWh, "
        f"discharged={total_discharged:.1f} kWh, "
        f"self_loss={total_self_loss:.3f} kWh, "
        f"max_cyclic_residual={max_cyclic_residual:.3e} kWh, "
        f"max_balance_residual={max_balance_residual:.3e} kWh, "
        f"control_period_hours={period_hours}.",
        flush=True,
    )
    return adjusted, pd.DataFrame(diagnostics_rows)


def _reactive_from_inflex_components(df_pre_demand_react, df_heat_elec, df_pv_elec):
    """Use the same physical Q assumptions as HEMS; heat input is HP-only."""
    return _obtain_post_reactive_power(
        df_pre_demand_react, df_heat_elec, df_pv_elec
    )


def _inflex_timesteps(inflex_inputs):
    """Timestep count of the inflex reconstruction and the label of its reference."""
    reference = inflex_inputs.get("reference")
    timesteps = _reference_timestep_count(reference, inflex_inputs.get("drop_initial_timestep", False))
    reference_label = "urbs output" if reference is not None else "raw inflex demand"
    if timesteps is None:
        timesteps = len(_use_t_as_index(inflex_inputs["demand"]))
    return timesteps, reference_label


def _process_inflex_demands(inflex_inputs, df_raw_demand, df_pre_demand_elec, df_pre_demand_react, *, timesteps, reference_label):
    delta_t = inflex_inputs.get("delta_t_hours")
    if delta_t is not None and abs(float(delta_t) - 1.0) > 1e-9:
        raise ValueError(
            f"INFLEX reconstruction supports hourly timesteps only, but this "
            f"result records delta_t_hours={delta_t}. Pass the duration through "
            "EV scheduling and battery control before advertising other "
            "resolutions."
        )
    df_eff_factor = _align_table_to_timesteps(inflex_inputs["eff_factor"], timesteps, "INFLEX eff_factor", reference_label)
    df_supim = _align_table_to_timesteps(inflex_inputs["supim"], timesteps, "INFLEX supim", reference_label)
    df_process = inflex_inputs["process"]

    if inflex_inputs.get("thermal_parameters") is not None:
        df_heat_elec, df_heat_hp_elec, _df_heat_auxiliary_elec = _inflex_internal_heat(inflex_inputs,df_raw_demand.index)
    else:
        df_heat_elec, df_heat_hp_elec, _df_heat_auxiliary_elec = _inflex_heat_electricity(
            df_raw_demand, df_eff_factor, inflex_inputs["process"],
        )
    df_ev_elec = _mobility_electricity(
        inflex_inputs["ev_sessions"],
        inflex_inputs["ev_session_hours"],
        timesteps,
        df_raw_demand.index,
    )
    df_pv_elec = _pv_generation(
        df_supim, df_process, inflex_inputs["cap_pro"]
    )

    active_parts = [df_pre_demand_elec, df_heat_elec, df_ev_elec, -df_pv_elec]
    net_before_battery = pd.concat(active_parts, axis=1).T.groupby(
        level=0, observed=True
    ).sum().T
    net_before_battery.columns = pd.MultiIndex.from_tuples(
        [(site, "electricity") for site in net_before_battery.columns]
    )
    df_post_demand_elec, battery_diagnostics = _apply_inflex_battery_control(
        net_before_battery,
        inflex_inputs["storage"],
        hours_per_period=inflex_inputs.get("tsam_hours_per_period"),
    )

    df_post_demand_react, df_prod_PV_react, df_demand_HP_react = _reactive_from_inflex_components(
        df_pre_demand_react,
        df_heat_hp_elec,
        df_pv_elec,
    )
    df_react_save = _concat_react_demands(df_pre_demand_react.copy(), df_demand_HP_react, df_prod_PV_react)
    return df_post_demand_elec, df_post_demand_react, df_react_save, battery_diagnostics


def obtain_pre_demand(SF):
    df_raw_demand = SF.get_pre_demand()
    if SF.uses_reduced_demand():
        df_raw_demand = _drop_tsam_initial_timestep(df_raw_demand)
        df_raw_demand.index = range(len(df_raw_demand))
    df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
    return pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)

def reconstruct_demands(SF, post_demand_mode="flexible", ev_charger_kw=None):
    """Pre and post demand of one scenario result (no side effects).

    Args:
        SF: reader with ``get_input_demands`` / ``get_inflex_inputs``
            (``io.ScenarioResultReader``).
        post_demand_mode: ``flexible`` (optimized URBS net import) or ``inflex``.
        ev_charger_kw: optional inflex cross-check of every charger rating.

    Returns:
        ``(df_pre_demand, df_post_demand, df_reactive_components, battery_diagnostics)``.
    """
    if post_demand_mode not in {"flexible", "inflex"}:
        raise ValueError("post_demand_mode must be 'flexible' or 'inflex'.")

    battery_diagnostics = pd.DataFrame()
    if post_demand_mode == "flexible":
        df_raw_demand, df_urbs_demand = SF.get_input_demands()
        df_raw_demand = _align_pre_demand_to_urbs(df_raw_demand, df_urbs_demand)
        df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
        df_post_demand_elec, df_post_demand_react, df_react_save = _process_post_demands(df_urbs_demand, df_pre_demand_react)
    else:
        inflex_inputs = SF.get_inflex_inputs()
        timesteps, reference_label = _inflex_timesteps(inflex_inputs)
        df_raw_demand = _align_table_to_timesteps(inflex_inputs["demand"], timesteps, "INFLEX demand", reference_label)
        # Charger power is per vehicle and comes from the session table, which is
        # validated against the scenario process rows. A run-level override is
        # accepted only if it agrees with every vehicle's actual rating.
        sessions = inflex_inputs["ev_sessions"]
        validate_sessions(
            sessions,
            inflex_inputs["ev_session_hours"],
            horizon_hours=timesteps,
            process_table=inflex_inputs["process"],
        )
        if ev_charger_kw is not None and not sessions.empty:
            mismatched = sessions.loc[
                ~np.isclose(
                    sessions["charger_kw"].astype(float),
                    float(ev_charger_kw),
                    rtol=0.0,
                    atol=SESSION_POWER_TOL_KW,
                )
            ]
            if not mismatched.empty:
                raise ValueError(
                    f"--inflex-ev-charger-kw={ev_charger_kw} disagrees with "
                    f"{len(mismatched)} session charger rating(s), for example "
                    f"{mismatched['charger_kw'].iloc[0]} kW at site "
                    f"{mismatched['site'].iloc[0]}."
                )
        df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
        df_post_demand_elec, df_post_demand_react, df_react_save, battery_diagnostics = (
            _process_inflex_demands(
                inflex_inputs,
                df_raw_demand,
                df_pre_demand_elec,
                df_pre_demand_react,
                timesteps=timesteps,
                reference_label=reference_label,
            )
        )

    df_pre_demand = pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)
    df_post_demand = pd.concat([df_post_demand_elec, df_post_demand_react], axis=1)
    return df_pre_demand, df_post_demand, df_react_save, battery_diagnostics


def save_battery_audit(SF, battery_diagnostics):
    """Keep the inflex stationary-battery diagnostics in the run's audit sidecar.

    Component audits are retained regardless of whether the reactive time-series
    tables are written: the compact-summary paths disable those, and the
    database has no matching table.
    """
    if battery_diagnostics is None or battery_diagnostics.empty:
        return None
    saver = getattr(SF, "save_component_audit", None)
    if saver is None:
        print(
            "No component-audit sink on this adapter; stationary-battery "
            "diagnostics were not retained.",
            flush=True,
        )
        return None
    location = saver(battery_diagnostics, "inflex_battery_state")
    print(f"INFLEX stationary-battery audit retained at {location}.", flush=True)
    return location


def obtain_demand(SF, save_reactive=True, post_demand_mode="flexible", ev_charger_kw=None):
    """Pre and post demand; saves the battery audit and (optionally) the reactive table via ``SF``."""
    df_pre_demand, df_post_demand, df_react_save, battery_diagnostics = reconstruct_demands(
        SF, post_demand_mode=post_demand_mode, ev_charger_kw=ev_charger_kw
    )
    save_battery_audit(SF, battery_diagnostics)
    if save_reactive:
        SF.save_df(df_react_save, "pwrflw/urbs_out/MILP/reactive")
    return df_pre_demand, df_post_demand


def _inflex_internal_heat(inputs,index):
    """Independent forward thermostat, service COPs and fixed shared HP/rod assets."""
    from gridexpand.common.thermal import dispatch_heat_services
    params = inputs["thermal_parameters"]
    reference = inputs.get("internal_heat_reference")
    if reference is None or len(reference) != len(index):
        raise ValueError("Internal InFlex requires the complete independent thermostat reference.")
    buses = sorted(params.Site.unique())
    hp_caps = _input_capacity_by_bus(inputs["process"],"heatpump_air",buses)
    rod_caps = _input_capacity_by_bus(inputs["process"],"heatpump_booster",buses)
    hp_elec,rod_elec = {},{}
    for site,group in params.groupby("Site"):
        heat,cops = [],[]
        for bid in group.building_objectid.astype(str):
            for service in ("space","water"):
                heat.append(reference[bid,service+"_heat_kw"].to_numpy())
                cops.append(reference[bid,service+"_cop"].to_numpy())
        hp,rod=dispatch_heat_services(np.column_stack(heat),np.column_stack(cops),hp_caps[site])
        if (rod.sum(axis=1)>rod_caps[site]+1e-7).any():
            raise ValueError(f"Fixed internal InFlex heat capacity cannot maintain comfort at site {site}.")
        hp_elec[site,"electricity"]=hp.sum(axis=1)
        rod_elec[site,"electricity"]=rod.sum(axis=1)
    hp_frame=pd.DataFrame(hp_elec,index=index)
    rod_frame=pd.DataFrame(rod_elec,index=index)
    return hp_frame+rod_frame,hp_frame,rod_frame
