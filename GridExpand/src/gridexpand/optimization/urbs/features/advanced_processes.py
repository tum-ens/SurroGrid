"""Time-variable process efficiency (``TimeVarEff`` / ``eff_factor`` input).

Only the time-variable output ratio of urbs' advanced processes is used by
GridExpand (heat-pump COP and charging-station availability profiles). For a
charging station the factor is the connected share of the hour, so it also limits
the station's input: without that limit a station could draw electricity with
zero output while the car is away.
"""

import pyomo.core as pyomo

# Processes whose eff_factor is an availability (connected share of the hour)
CHARGING_STATION_PREFIX = "charging_station"


def add_advanced_processes(m):
    """Add the time-variable efficiency outputs to ``m`` (requires ``m.mode['tve']``)."""
    # all support timeframes for which time variable efficiency is enabled
    tve_stflist = set()
    for key in m.eff_factor_dict[tuple(m.eff_factor_dict.keys())[0]]:
        tve_stflist.add(tuple(key)[0])
    m.pro_timevar_output_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro * m.com,
        initialize=[(stf, site, process, commodity)
                    for stf in tve_stflist
                    for (site, process) in tuple(m.eff_factor_dict.keys())
                    for (st, pro, commodity) in tuple(m.r_out_dict.keys())
                    if process == pro and st == stf],
        doc='Outputs of processes with time dependent efficiency')

    m.def_process_timevar_output = pyomo.Constraint(
        m.tm, m.pro_timevar_output_tuples,
        rule=def_pro_timevar_output_rule,
        doc='e_pro_out = tau_pro * r_out * eff_factor')

    # Chargers of dedicated EV sessions are already limited to their sessions.
    sessions = set(getattr(m, 'ev_charger_tuples', ()))
    m.pro_availability_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro,
        initialize=[(stf, site, process)
                    for stf in sorted(tve_stflist)
                    for (site, process) in tuple(m.eff_factor_dict.keys())
                    if str(process).startswith(CHARGING_STATION_PREFIX)
                    and (stf, site, process) in m.pro_tuples
                    and (stf, site, process) not in sessions],
        doc='Charging stations whose eff_factor is the connected share of the hour')
    m.res_process_availability = pyomo.Constraint(
        m.tm, m.pro_availability_tuples,
        rule=res_process_availability_rule,
        doc='tau_pro <= dt * cap_pro * eff_factor (no charging while the car is away)')
    return m


# charging-station throughput <= capacity * connected share of the hour
def res_process_availability_rule(m, tm, stf, sit, pro):
    return (m.tau_pro[tm, stf, sit, pro] <=
            m.dt * m.cap_pro[stf, sit, pro] * m.eff_factor_dict[(sit, pro)][stf, tm])


# process output == process throughput * output ratio * efficiency factor
def def_pro_timevar_output_rule(m, tm, stf, sit, pro, com):
    return (m.e_pro_out[tm, stf, sit, pro, com] ==
            m.tau_pro[tm, stf, sit, pro] * m.r_out_dict[(stf, pro, com)] *
            m.eff_factor_dict[(sit, pro)][stf, tm])
