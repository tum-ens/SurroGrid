"""Storage feature: stationary batteries, heat storages and legacy mobility buffers."""

import pyomo.core as pyomo


def add_storage(m):
    """Add storage sets, variables, capacity expressions and constraints to ``m``."""
    # storage (e.g. hydrogen, pump storage)
    indexlist = set()
    for key in m.storage_dict["eff-in"]:
        indexlist.add(tuple(key)[2])
    m.sto = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of storage technologies')

    # storage tuples
    m.sto_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.storage_dict["eff-in"].keys()),
        doc='Combinations of possible storage by site,'
            'e.g. (2020,Mid,Bat,Elec)')

    # storage tuples for storages with given energy to power ratio
    m.sto_ep_ratio_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.sto_ep_ratio_dict.keys()),
        doc='storages with given energy to power ratio')

    m.sto_linked_capacity_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.sto_linked_capacity_dict.keys()),
        doc='Storages whose energy capacity is linked to a process capacity')

    # Variables
    m.cap_sto_c_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New storage size (MWh)')
    m.cap_sto_p_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New  storage power (MW)')

    # storage capacities as expression objects
    m.cap_sto_c = pyomo.Expression(
        m.sto_tuples,
        rule=def_storage_capacity_rule,
        doc='Total storage size (MWh)')
    m.cap_sto_p = pyomo.Expression(
        m.sto_tuples,
        rule=def_storage_power_rule,
        doc='Total storage power (MW)')

    m.e_sto_in = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow into storage (MW) per timestep')
    m.e_sto_out = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow out of storage (MW) per timestep')
    m.e_sto_con = pyomo.Var(
        m.t, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Energy content of storage (MWh) in timestep')

    # storage rules
    m.def_storage_state = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=def_storage_state_rule,
        doc='storage[t] = (1 - sd) * storage[t-1] + in * eff_i - out / eff_o')
    m.res_storage_input_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_input_by_power_rule,
        doc='storage input <= storage power')
    m.res_storage_output_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_output_by_power_rule,
        doc='storage output <= storage power')
    m.res_storage_state_by_capacity = pyomo.Constraint(
        m.t, m.sto_tuples,
        rule=res_storage_state_by_capacity_rule,
        doc='storage content <= storage capacity')
    m.res_storage_power = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_power_rule,
        doc='storage.cap-lo-p <= storage power <= storage.cap-up-p')
    m.res_storage_capacity = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_capacity_rule,
        doc='storage.cap-lo-c <= storage capacity <= storage.cap-up-c')
    m.res_storage_state_cyclicity = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_state_cyclicity_rule,
        doc='storage content initial <= final, both variable')
    m.def_storage_energy_power_ratio = pyomo.Constraint(
        m.sto_ep_ratio_tuples,
        rule=def_storage_energy_power_ratio_rule,
        doc='storage capacity = storage power * storage E/P ratio')
    m.res_storage_linked_process_capacity = pyomo.Constraint(
        m.sto_linked_capacity_tuples,
        rule=res_storage_linked_process_capacity_rule,
        doc='storage capacity <= linked process capacity * configured ratio')
    return m


# constraints

# storage content in timestep [t] == storage content[t-1] * (1-discharge)
# + newly stored energy * input efficiency
# - retrieved energy / output efficiency
def def_storage_state_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[t, stf, sit, sto, com] ==
            m.e_sto_con[t - 1, stf, sit, sto, com] *
            (1 - m.storage_dict['discharge']
            [(stf, sit, sto, com)]) ** m.dt.value +
            m.e_sto_in[t, stf, sit, sto, com] *
            m.storage_dict['eff-in'][(stf, sit, sto, com)] -
            m.e_sto_out[t, stf, sit, sto, com] /
            m.storage_dict['eff-out'][(stf, sit, sto, com)])


# storage capacity (for m.cap_sto_c expression)
def def_storage_capacity_rule(m, stf, sit, sto, com):
    return (m.cap_sto_c_new[stf, sit, sto, com] +
            m.storage_dict['inst-cap-c'][(stf, sit, sto, com)])


# storage power (for m.cap_sto_p expression)
def def_storage_power_rule(m, stf, sit, sto, com):
    return (m.cap_sto_p_new[stf, sit, sto, com] +
            m.storage_dict['inst-cap-p'][(stf, sit, sto, com)])


# storage input <= storage power
def res_storage_input_by_power_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_in[t, stf, sit, sto, com] <= m.dt *
            m.cap_sto_p[stf, sit, sto, com])


# storage output <= storage power
def res_storage_output_by_power_rule(m, t, stf, sit, sto, co):
    return (m.e_sto_out[t, stf, sit, sto, co] <= m.dt *
            m.cap_sto_p[stf, sit, sto, co])


# storage content <= storage capacity
def res_storage_state_by_capacity_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[t, stf, sit, sto, com] <=
            m.cap_sto_c[stf, sit, sto, com])


# storage power <= upper bound (no lower bound)
def res_storage_power_rule(m, stf, sit, sto, com):
    return (
            None,
            m.cap_sto_p[stf, sit, sto, com],
            m.storage_dict['cap-up-p'][(stf, sit, sto, com)])


# storage capacity <= upper bound (no lower bound)
def res_storage_capacity_rule(m, stf, sit, sto, com):
    return (
            None,
            m.cap_sto_c[stf, sit, sto, com],
            m.storage_dict['cap-up-c'][(stf, sit, sto, com)])


def res_storage_state_cyclicity_rule(m, stf, sit, sto, com):
    # Full-year chronological reference: one annual closure, as an equality, so
    # the horizon cannot be used as a free energy source or sink. With type
    # periods active the closure is handled by
    # res_storage_state_cyclicity_typeperiod_rule and this constraint keeps its
    # historical inequality form so existing TSAM runs are unchanged.
    if m.mode['tdy']:
        return (m.e_sto_con[m.t.at(1), stf, sit, sto, com] <=    # Indexing in pyomo starts at 1 not 0!
                m.e_sto_con[m.t.at(len(m.t)), stf, sit, sto, com])
    return (m.e_sto_con[m.t.at(1), stf, sit, sto, com] ==
            m.e_sto_con[m.t.at(len(m.t)), stf, sit, sto, com])


def def_storage_energy_power_ratio_rule(m, stf, sit, sto, com):
    return (m.cap_sto_c[stf, sit, sto, com] == m.cap_sto_p[stf, sit, sto, com] *
            m.storage_dict['ep-ratio'][(stf, sit, sto, com)])


def res_storage_linked_process_capacity_rule(m, stf, sit, sto, com):
    linked_process, max_energy_per_capacity = m.sto_linked_capacity_dict[
        (stf, sit, sto, com)
    ]
    return (
        m.cap_sto_c[stf, sit, sto, com]
        <= max_energy_per_capacity * m.cap_pro[stf, sit, linked_process]
    )


# storage balance
def storage_balance(m, tm, stf, sit, com):
    """Storage input minus output of commodity ``com`` at one site and timestep.

    Called in the commodity balance.
    """
    return sum(m.e_sto_in[(tm, stframe, site, storage, com)] -
               m.e_sto_out[(tm, stframe, site, storage, com)]
               # usage as input for storage increases consumption
               # output from storage decreases consumption
               for stframe, site, storage, commodity in m.sto_tuples
               if site == sit and stframe == stf and commodity == com)


# storage costs
def storage_cost(m, cost_type):
    """Storage part of the cost function for one cost type."""
    if cost_type == 'Invest':
        cost = sum(m.cap_sto_p_new[s] *
                   m.storage_dict['inv-cost-p'][s] *
                   m.storage_dict['invcost-factor'][s] +
                   m.cap_sto_c_new[s] *
                   m.storage_dict['inv-cost-c'][s] *
                   m.storage_dict['invcost-factor'][s]
                   for s in m.sto_tuples)
        return cost
    elif cost_type == 'Fixed':
        return sum((m.cap_sto_p[s] * m.storage_dict['fix-cost-p'][s] +
                    m.cap_sto_c[s] * m.storage_dict['fix-cost-c'][s]) *
                   m.storage_dict['cost_factor'][s]
                   for s in m.sto_tuples)
    elif cost_type == 'Variable':
        return sum(
                   (m.e_sto_in[(tm,) + s] + m.e_sto_out[(tm,) + s]) *
                   m.weight * m.typeperiod['weight_typeperiod'][(m.stf_list[0], tm)] * m.storage_dict['var-cost-p'][s] *
                   m.storage_dict['cost_factor'][s]
                   for tm in m.tm
                   for s in m.sto_tuples)
