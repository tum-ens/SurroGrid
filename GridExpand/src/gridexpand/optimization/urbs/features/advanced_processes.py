"""Time-variable process efficiency (``TimeVarEff`` / ``eff_factor`` input).

Only the time-variable output ratio of urbs' advanced processes is used by
GridExpand (heat-pump COP and charging-station availability profiles).
"""

import pyomo.core as pyomo


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
    return m


# process output == process throughput * output ratio * efficiency factor
def def_pro_timevar_output_rule(m, tm, stf, sit, pro, com):
    return (m.e_pro_out[tm, stf, sit, pro, com] ==
            m.tau_pro[tm, stf, sit, pro] * m.r_out_dict[(stf, pro, com)] *
            m.eff_factor_dict[(sit, pro)][stf, tm])
