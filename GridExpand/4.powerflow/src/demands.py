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

    ### Determine PV as optimal operation between -tan(phi) <= Q/P <= tan(phi) to obtain minimal reactive power demand from grid
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
        raise ValueError("No-flex post demand requires heatpump_air COP columns in eff_factor when heat demand is present.")

    heat_by_bus = df_raw_demand.loc[:, heat_columns].T.groupby(level=0).sum().T
    cop_by_bus = df_eff_factor.loc[:, cop_columns].copy()
    cop_by_bus.columns = cop_by_bus.columns.get_level_values(0)
    cop_by_bus = cop_by_bus.T.groupby(level=0).mean().T
    cop_by_bus = cop_by_bus.reindex(columns=heat_by_bus.columns)
    return heat_by_bus, cop_by_bus


def _capacity_by_bus(cap_pro, process, buses):
    if cap_pro is None:
        raise ValueError("No-flex heat split requires optimized post-flex cap_pro results.")
    if not isinstance(cap_pro.index, pd.MultiIndex):
        raise ValueError("No-flex heat split expects cap_pro with MultiIndex levels stf, sit, pro.")
    if "sit" not in cap_pro.index.names or "pro" not in cap_pro.index.names:
        raise ValueError("No-flex heat split expects cap_pro index levels named 'sit' and 'pro'.")

    process_mask = cap_pro.index.get_level_values("pro") == process
    process_caps = pd.to_numeric(cap_pro.loc[process_mask], errors="coerce").fillna(0.0)
    if process_caps.empty:
        return pd.Series(0.0, index=buses, dtype=float)

    by_site = process_caps.groupby(level="sit").sum()
    by_site_lookup = {str(site): float(value) for site, value in by_site.items()}
    return pd.Series([by_site_lookup.get(str(bus), 0.0) for bus in buses], index=buses, dtype=float)


def _no_flex_heat_electricity(df_raw_demand, df_eff_factor, cap_pro):
    heat_by_bus, cop_by_bus = _heat_and_cop_by_bus(df_raw_demand, df_eff_factor)
    if heat_by_bus.empty:
        empty = _empty_electricity_frame(df_raw_demand.index)
        return empty, empty, empty

    buses = list(heat_by_bus.columns)
    hp_capacity_el = _capacity_by_bus(cap_pro, "heatpump_air", buses)
    booster_capacity_el = _capacity_by_bus(cap_pro, "heatpump_booster", buses)

    cop_safe = cop_by_bus.replace(0, np.nan)
    hp_thermal_limit = cop_safe.multiply(hp_capacity_el, axis=1).fillna(0.0)

    heat_values = heat_by_bus.astype(float).to_numpy()
    hp_limit_values = hp_thermal_limit.to_numpy(dtype=float)
    uses_auxiliary = (booster_capacity_el.to_numpy(dtype=float) > 1e-9)[None, :]

    # If the optimized flex case installed auxiliary capacity at a bus, the
    # no-flex reconstruction lets the heat pump serve demand up to its optimized
    # electric capacity and assigns the high-demand residual to direct auxiliary
    # electric heating. Buses without optimized auxiliary capacity remain
    # heat-pump-only to preserve the post-flex technology choice.
    hp_heat_values = np.where(uses_auxiliary, np.minimum(heat_values, hp_limit_values), heat_values)
    hp_heat = pd.DataFrame(hp_heat_values, index=heat_by_bus.index, columns=heat_by_bus.columns)
    auxiliary_heat = (heat_by_bus - hp_heat).clip(lower=0.0)

    hp_electricity = hp_heat.divide(cop_safe).fillna(0.0)
    auxiliary_electricity = auxiliary_heat
    total_electricity = hp_electricity.add(auxiliary_electricity, fill_value=0.0)

    for frame in (total_electricity, hp_electricity, auxiliary_electricity):
        frame.columns = pd.MultiIndex.from_tuples([(bus, "electricity") for bus in frame.columns])

    auxiliary_peak = float(auxiliary_electricity.sum(axis=1).max()) if not auxiliary_electricity.empty else 0.0
    auxiliary_energy = float(auxiliary_electricity.sum().sum()) if not auxiliary_electricity.empty else 0.0
    hp_energy = float(hp_electricity.sum().sum()) if not hp_electricity.empty else 0.0
    print(
        "No-flex heat split from post-flex capacities: "
        f"heat-pump electricity={hp_energy:.1f} kWh, "
        f"auxiliary electricity={auxiliary_energy:.1f} kWh, "
        f"auxiliary peak={auxiliary_peak:.3f} kW.",
        flush=True,
    )
    return total_electricity, hp_electricity, auxiliary_electricity


def _redistribute_ev_energy_linear(energy_values, available, charger_kw):
    """Assign EV energy to the earliest available hours in non-cyclic windows."""
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

    return result, assigned_energy


def _cyclic_availability_start(available):
    starts = np.where((~available[:-1]) & (available[1:]))[0] + 1
    if len(starts) == 0:
        return None
    return int(starts[0])


def _redistribute_ev_energy(energy, availability, charger_kw):
    energy_values = np.asarray(energy, dtype=float)
    available = np.asarray(availability, dtype=float) > 0.5
    if len(energy_values) == 0:
        return np.zeros_like(energy_values, dtype=float)

    if available[0] and available[-1]:
        start = _cyclic_availability_start(available)
        if start is not None:
            rotated_result, assigned_energy = _redistribute_ev_energy_linear(
                np.roll(energy_values, -start),
                np.roll(available, -start),
                charger_kw,
            )
            result = np.roll(rotated_result, start)
        else:
            result, assigned_energy = _redistribute_ev_energy_linear(
                energy_values,
                available,
                charger_kw,
            )
    else:
        result, assigned_energy = _redistribute_ev_energy_linear(
            energy_values,
            available,
            charger_kw,
        )

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


def _pv_generation(df_supim, df_process, cap_pro):
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
        optimized_capacity_kw = float(
            _capacity_by_bus(cap_pro, process_name, [row["Site"]]).iloc[0]
        )
        parts.append(
            pd.to_numeric(df_supim[column], errors="coerce").fillna(0.0)
            * optimized_capacity_kw
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
            "No-flex battery control requires storage columns "
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
            "No-flex battery control only accepts fixed installed capacities; "
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
):
    soc = float(initial_soc_kwh)
    adjusted = np.asarray(net_demand, dtype=float).copy()
    charged_kwh = 0.0
    discharged_kwh = 0.0
    for index, net_kw in enumerate(adjusted):
        if net_kw < 0.0:
            charge_kw = min(
                -float(net_kw),
                power_kw,
                max(energy_kwh - soc, 0.0) / charge_efficiency,
            )
            soc += charge_kw * charge_efficiency
            adjusted[index] += charge_kw
            charged_kwh += charge_kw
        elif net_kw > 0.0:
            discharge_kw = min(
                float(net_kw),
                power_kw,
                soc * discharge_efficiency,
            )
            soc -= discharge_kw / discharge_efficiency
            adjusted[index] -= discharge_kw
            discharged_kwh += discharge_kw
    return adjusted, soc, charged_kwh, discharged_kwh


def _cyclic_self_consumption_period(net_demand, battery):
    energy_kwh = float(battery["inst-cap-c"])
    power_kw = float(battery["inst-cap-p"])
    charge_efficiency = float(battery["eff-in"])
    discharge_efficiency = float(battery["eff-out"])
    if energy_kwh <= 0.0 or power_kw <= 0.0:
        return np.asarray(net_demand, dtype=float), 0.0, 0.0
    if not 0.0 < charge_efficiency <= 1.0:
        raise ValueError("Stationary-battery charging efficiency must be in (0, 1].")
    if not 0.0 < discharge_efficiency <= 1.0:
        raise ValueError("Stationary-battery discharging efficiency must be in (0, 1].")

    initial_soc = energy_kwh / 2.0
    for _ in range(1000):
        _, final_soc, _, _ = _simulate_self_consumption_period(
            net_demand,
            initial_soc_kwh=initial_soc,
            energy_kwh=energy_kwh,
            power_kw=power_kw,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
        )
        if abs(final_soc - initial_soc) <= 1e-7:
            break
        initial_soc = final_soc
    else:
        raise RuntimeError("Stationary-battery cyclic state did not converge.")

    adjusted, _, charged, discharged = _simulate_self_consumption_period(
        net_demand,
        initial_soc_kwh=initial_soc,
        energy_kwh=energy_kwh,
        power_kw=power_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )
    return adjusted, charged, discharged


def _apply_no_flex_battery_control(
    net_demand,
    df_storage,
    *,
    hours_per_period=None,
):
    batteries = _fixed_stationary_batteries(df_storage)
    if batteries.empty:
        return net_demand

    adjusted = net_demand.copy()
    adjusted.columns = adjusted.columns.get_level_values(0)
    period_hours = int(hours_per_period or len(adjusted))
    if period_hours <= 0:
        raise ValueError("Battery-control period length must be positive.")

    total_charged = 0.0
    total_discharged = 0.0
    for site, battery in batteries.iterrows():
        if site not in adjusted.columns:
            adjusted[site] = 0.0
        values = adjusted[site].to_numpy(dtype=float)
        controlled = values.copy()
        for start in range(0, len(values), period_hours):
            stop = min(start + period_hours, len(values))
            segment, charged, discharged = _cyclic_self_consumption_period(
                values[start:stop], battery
            )
            controlled[start:stop] = segment
            total_charged += charged
            total_discharged += discharged
        adjusted[site] = controlled

    adjusted = adjusted.sort_index(axis=1)
    adjusted.columns = pd.MultiIndex.from_tuples(
        [(site, "electricity") for site in adjusted.columns]
    )
    print(
        "No-flex stationary batteries: "
        f"sites={len(batteries)}, charged={total_charged:.1f} kWh, "
        f"discharged={total_discharged:.1f} kWh, "
        f"control_period_hours={period_hours}.",
        flush=True,
    )
    return adjusted


def _reactive_from_no_flex_components(df_pre_demand_react, df_heat_elec, df_pv_elec):
    heat_react = df_heat_elec * np.tan(np.arccos(config.PF_HP)) * (-1)
    heat_react = _set_electricity_component(heat_react, "electricity-reactive")

    react_without_pv = df_pre_demand_react.add(heat_react, fill_value=0.0)
    if df_pv_elec.empty:
        pv_react = _empty_electricity_frame(df_pre_demand_react.index)
        return react_without_pv, pv_react, heat_react

    pv_limit = df_pv_elec * np.tan(np.arccos(config.PF_PV_MIN))
    pv_limit = _set_electricity_component(pv_limit, "electricity-reactive")
    ideal_pv_react = -react_without_pv.reindex(columns=pv_limit.columns, fill_value=0.0)
    pv_react = ideal_pv_react.clip(lower=-pv_limit, upper=pv_limit)
    post_react = react_without_pv.add(pv_react, fill_value=0.0)
    return post_react, pv_react, heat_react


def _process_no_flex_demands(no_flex_inputs, df_pre_demand_elec, df_pre_demand_react, ev_charger_kw):
    reference = no_flex_inputs.get("reference")
    timesteps = _reference_timestep_count(reference, no_flex_inputs.get("drop_initial_timestep", False))
    reference_label = "urbs output" if reference is not None else "raw no-flex demand"
    if timesteps is None:
        timesteps = len(_use_t_as_index(no_flex_inputs["demand"]))

    df_raw_demand = _align_table_to_timesteps(no_flex_inputs["demand"], timesteps, "No-flex demand", reference_label)
    df_eff_factor = _align_table_to_timesteps(no_flex_inputs["eff_factor"], timesteps, "No-flex eff_factor", reference_label)
    df_supim = _align_table_to_timesteps(no_flex_inputs["supim"], timesteps, "No-flex supim", reference_label)
    df_process = no_flex_inputs["process"]

    df_heat_elec, df_heat_hp_elec, _df_heat_auxiliary_elec = _no_flex_heat_electricity(
        df_raw_demand,
        df_eff_factor,
        no_flex_inputs["cap_pro"],
    )
    df_ev_elec = _mobility_electricity(df_raw_demand, df_eff_factor, ev_charger_kw)
    df_pv_elec = _pv_generation(
        df_supim, df_process, no_flex_inputs["cap_pro"]
    )

    active_parts = [df_pre_demand_elec, df_heat_elec, df_ev_elec, -df_pv_elec]
    net_before_battery = pd.concat(active_parts, axis=1).T.groupby(
        level=0, observed=True
    ).sum().T
    net_before_battery.columns = pd.MultiIndex.from_tuples(
        [(site, "electricity") for site in net_before_battery.columns]
    )
    df_post_demand_elec = _apply_no_flex_battery_control(
        net_before_battery,
        no_flex_inputs["storage"],
        hours_per_period=no_flex_inputs.get("tsam_hours_per_period"),
    )

    df_post_demand_react, df_prod_PV_react, df_demand_HP_react = _reactive_from_no_flex_components(
        df_pre_demand_react,
        df_heat_hp_elec,
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

def obtain_demand(SF, save_reactive=True, post_demand_mode="flexible", ev_charger_kw=None):
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
        no_flex_inputs = SF.get_no_flex_inputs()
        reference = no_flex_inputs.get("reference")
        timesteps = _reference_timestep_count(reference, no_flex_inputs.get("drop_initial_timestep", False))
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
