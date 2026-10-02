"""Pyomo model of one building cluster (the urbs core used by GridExpand Step 3).

Declaration order of sets, variables, expressions and constraints defines the order
of rows and columns in the LP file handed to the solver. The heuristic model case is
a degenerate LP, so this order selects the optimal vertex the solver returns: do not
reorder declarations or the terms inside the rules.
"""

import pyomo.environ as pyomo

from .features.advanced_processes import add_advanced_processes
from .features.buy_sell_price import (
    add_buy_sell_price,
    bsp_surplus,
    purchase_costs,
    revenue_costs,
)
from .features.ev_sessions import add_ev_sessions
from .features.modelhelper import commodity_balance, commodity_subset
from .features.storage import add_storage, storage_cost
from .features.typeperiod import add_typeperiod
from .input import pyomo_model_prep
from .features.thermal import add_thermal


def create_model(data, global_settings):
    """Create the pyomo ConcreteModel of one building cluster.

    Args:
        data: urbs input dict of one cluster (see ``input.read_input_h5``).
        global_settings: run settings; uses ``timesteps``, ``dt`` and
            ``hoursPerPeriod`` (type periods only).

    Returns:
        The pyomo ConcreteModel.
    """
    m = pyomo_model_prep(data, global_settings["timesteps"])
    m = pyomo_assign_params(m, global_settings)
    m = pyomo_assign_basic_sets(m, global_settings)
    m = pyomo_assign_basic_variables(m, global_settings)
    m.cap_pro = pyomo.Expression(
        m.pro_tuples,
        rule=def_process_capacity_rule,
        doc='total process capacity')
    m = pyomo_assign_advanced_features(m, global_settings, data)
    if m.mode.get("thermal"):
        m = add_thermal(m, data)
    m = pyomo_assign_basic_constraints(m, global_settings)
    m = pyomo_assign_objective(m, global_settings)
    return m


def pyomo_assign_params(m, global_settings):
    # weight = length of year (hours) / length of simulation (hours)
    # weight scales costs and emissions from length of simulation to a full
    # year, making comparisons among cost types (invest is annualized, fixed
    # costs are annual by default, variable costs are scaled by weight) and
    # among different simulation durations meaningful.
    m.weight = pyomo.Param(
        initialize=float(8760) / (len(m.timesteps) - 1),
        doc='Pre-factor for variable costs and emissions for an annual result')

    # dt = spacing between timesteps. Required for storage equation that
    # converts between energy (storage content, e_sto_con) and power (all other
    # quantities that start with "e_")
    m.dt = pyomo.Param(
        initialize=global_settings["dt"],
        doc='Time step duration (in hours), default: 1')
    return m


def pyomo_assign_basic_sets(m, global_settings):
    # generate ordered time step sets
    m.t = pyomo.Set(
        initialize=m.timesteps,
        ordered=True,
        doc='Set of timesteps')

    # modelled (i.e. excluding init time step for storage) time steps
    m.tm = pyomo.Set(
        within=m.t,
        initialize=m.timesteps[1:],
        ordered=True,
        doc='Set of modelled timesteps')

    # support timeframes (e.g. 2020, 2030...)
    indexlist = set()
    for key in m.commodity_dict["price"]:
        indexlist.add(tuple(key)[0])
    m.stf = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of modeled support timeframes (e.g. years)')

    # site (e.g. north, middle, south...)
    indexlist = set()
    for key in m.commodity_dict["price"]:
        indexlist.add(tuple(key)[1])
    m.sit = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of sites')

    # commodity (e.g. solar, wind, coal...)
    indexlist = set()
    for key in m.commodity_dict["price"]:
        indexlist.add(tuple(key)[2])
    m.com = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of commodities')

    # commodity type (i.e. SupIm, Demand, Stock, Env)
    indexlist = set()
    for key in m.commodity_dict["price"]:
        indexlist.add(tuple(key)[3])
    m.com_type = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of commodity types')

    # process (e.g. Wind turbine, Gas plant, Photovoltaics...)
    indexlist = set()
    for key in m.process_dict["inv-cost"]:
        indexlist.add(tuple(key)[2])
    m.pro = pyomo.Set(
        initialize=indexlist,
        ordered=False,
        doc='Set of conversion processes')

    # cost_type
    m.cost_type = pyomo.Set(
        initialize=m.cost_type_list,
        doc='Set of cost types (hard-coded)')

    # tuple sets
    m.com_tuples = pyomo.Set(
        within=m.stf * m.sit * m.com * m.com_type,
        initialize=tuple(m.commodity_dict["price"].keys()),
        doc='Combinations of defined commodities, e.g. (2018,Mid,Elec,Demand)')
    m.pro_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro,
        initialize=tuple(m.process_dict["inv-cost"].keys()),
        doc='Combinations of possible processes, e.g. (2018,North,Coal plant)')

    # commodity type subsets
    m.com_supim = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'SupIm'),
        ordered=False,
        doc='Commodities that have intermittent (timeseries) input')
    m.com_demand = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Demand'),
        ordered=False,
        doc='Commodities that have a demand (implies timeseries)')

    m.pro_inv_cost_fix_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro,
        initialize=[(stf, site, process)
                    for (stf, site, process) in m.pro_tuples
                    for (s, si, pro) in tuple(m.pro_inv_cost_fix_dict.keys())
                    if process == pro and si == site and s == stf],
        doc='Processes with fixed investment cost portions')
    # process input/output
    m.pro_input_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro * m.com,
        initialize=[(stf, site, process, commodity)
                    for (stf, site, process) in m.pro_tuples
                    for (s, pro, commodity) in tuple(m.r_in_dict.keys())
                    if process == pro and s == stf],
        doc='Commodities consumed by process by site,'
            'e.g. (2020,Mid,PV,Solar)')

    m.pro_output_tuples = pyomo.Set(
        within=m.stf * m.sit * m.pro * m.com,
        initialize=[(stf, site, process, commodity)
                    for (stf, site, process) in m.pro_tuples
                    for (s, pro, commodity) in tuple(m.r_out_dict.keys())
                    if process == pro and s == stf],
        doc='Commodities produced by process by site, e.g. (2020,Mid,PV,Elec)')
    return m


def pyomo_assign_basic_variables(m, global_settings):
    # costs (cost_types act as index)
    m.costs = pyomo.Var(
        m.cost_type,
        within=pyomo.Reals,
        doc='Costs by type (EUR/a)')

    # process
    m.cap_pro_new = pyomo.Var(
        m.pro_tuples,
        within=pyomo.NonNegativeReals,
        doc='New process capacity (MW)')
    m.tau_pro = pyomo.Var(
        m.tm, m.pro_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow (MW) through process')
    m.e_pro_in = pyomo.Var(
        m.tm, m.pro_input_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow of commodity into process (MW) per timestep')
    m.e_pro_out = pyomo.Var(
        m.tm, m.pro_output_tuples,
        within=pyomo.Reals,
        doc='Power flow out of process (MW) per timestep')

    # process new capacity expansion boolean
    m.pro_cap_expands = pyomo.Var(
        m.pro_inv_cost_fix_tuples,
        within=pyomo.Boolean,
        doc='Boolean variable whether a process is expanded')
    return m


def pyomo_assign_advanced_features(m, global_settings, data=None):
    if m.mode['sto']:
        m = add_storage(m)
    if m.mode['bsp']:
        m = add_buy_sell_price(m)
    if m.mode['tdy']:
        m = add_typeperiod(m, global_settings["hoursPerPeriod"])
    if m.mode['evs']:
        m = add_ev_sessions(m, data)
    if m.mode['tve']:
        m = add_advanced_processes(m)
    else:
        m.pro_timevar_output_tuples = pyomo.Set(
            within=m.stf * m.sit * m.pro * m.com,
            doc='empty set needed for (partial) process output')
    return m


def pyomo_assign_basic_constraints(m, global_settings):
    # commodities
    m.res_vertex = pyomo.Constraint(
        m.tm, m.com_tuples,
        rule=res_vertex_rule,
        doc='storage + transmission + process + source + buy - sell == demand')

    # processes
    m.def_process_input = pyomo.Constraint(
        m.tm, m.pro_input_tuples,
        rule=def_process_input_rule,
        doc='process input = process throughput * input ratio')
    m.def_process_output = pyomo.Constraint(
        m.tm, m.pro_output_tuples - m.pro_timevar_output_tuples,
        rule=def_process_output_rule,
        doc='process output = process throughput * output ratio')
    m.def_intermittent_supply = pyomo.Constraint(
        m.tm, m.pro_input_tuples,
        rule=def_intermittent_supply_rule,
        doc='process output = process capacity * supim timeseries')
    m.res_process_throughput_by_capacity = pyomo.Constraint(
        m.tm, m.pro_tuples,
        rule=res_process_throughput_by_capacity_rule,
        doc='process throughput <= total process capacity')
    m.res_process_capacity = pyomo.Constraint(
        m.pro_tuples,
        rule=res_process_capacity_rule,
        doc='process.cap-lo <= total process capacity <= process.cap-up')
    # The expansion bound applies to the newly installed capacity, while the
    # input's cap-up is meant for the total capacity.
    m.res_process_capacity_fixed_inv_cost_upper = pyomo.Constraint(
        m.pro_inv_cost_fix_tuples,
        rule=res_process_capacity_fixed_inv_cost_upper_rule,
        doc='new process capacity <= pro_cap_expands * process.cap-up')

    # costs
    m.def_costs = pyomo.Constraint(
        m.cost_type,
        rule=def_costs_rule,
        doc='main cost function by cost type')
    return m


def pyomo_assign_objective(m, global_settings):
    m.objective_function = pyomo.Objective(
        rule=cost_rule,
        sense=pyomo.minimize,
        doc='minimize(cost = sum of all cost types)')
    return m


# Constraints

# vertex equation: calculate balance for given commodity and site
def res_vertex_rule(m, tm, stf, sit, com, com_type):
    # if power_surplus > 0: production/storage/imports create net positive
    #                       amount of commodity com
    # if power_surplus < 0: production/storage/exports consume a net
    #                       amount of the commodity com

    # SupIm commodities are converted by process to pro_out commodity (which is
    # accounted for here). "electricity_hp" is a legacy commodity name that no
    # current input carries; such a commodity would be left unbalanced.
    if (com in m.com_supim) or com == "electricity_hp":
        return pyomo.Constraint.Skip

    # storage, process production/consumption
    # com_bal = (pro_in+sto_in)-(pro_out+sto_out)
    power_surplus = - commodity_balance(m, tm, stf, sit, com)

    # surplus + buy - sell
    if m.mode['bsp']:
        power_surplus += bsp_surplus(m, tm, stf, sit, com, com_type)

    thermal_building = getattr(m, '_thermal_commodities', {}).get((stf, sit, com))
    if thermal_building is not None:
        power_surplus -= m.building_heat[tm, stf, sit, thermal_building]
    # surplus - demand
    elif com in m.com_demand:
        try:
            power_surplus -= m.demand_dict[(sit, com)][(stf, tm)]
        except KeyError:
            pass
    return power_surplus == 0


# process capacity (for m.cap_pro Expression)
def def_process_capacity_rule(m, stf, sit, pro):
    return (m.cap_pro_new[stf, sit, pro] +
            m.process_dict['inst-cap'][(stf, sit, pro)])


# process input power == process throughput * input ratio
def def_process_input_rule(m, tm, stf, sit, pro, com):
    return (m.e_pro_in[tm, stf, sit, pro, com] ==
            m.tau_pro[tm, stf, sit, pro] * m.r_in_dict[(stf, pro, com)])


# process output power = process throughput * output ratio
def def_process_output_rule(m, tm, stf, sit, pro, com):
    return (m.e_pro_out[tm, stf, sit, pro, com] ==
            m.tau_pro[tm, stf, sit, pro] * m.r_out_dict[(stf, pro, com)])


# process input (for supim commodity) = process capacity * timeseries
def def_intermittent_supply_rule(m, tm, stf, sit, pro, coin):
    if coin in m.com_supim:
        return (m.e_pro_in[tm, stf, sit, pro, coin] ==
                m.cap_pro[stf, sit, pro] * m.supim_dict[(sit, coin)]
                [(stf, tm)] * m.dt)
    else:
        return pyomo.Constraint.Skip


# process throughput <= process capacity
def res_process_throughput_by_capacity_rule(m, tm, stf, sit, pro):
    return (m.tau_pro[tm, stf, sit, pro] <= m.dt * m.cap_pro[stf, sit, pro])


# process capacity <= upper bound (no lower bound)
def res_process_capacity_rule(m, stf, sit, pro):
    return (
            None,
            m.cap_pro[stf, sit, pro],
            m.process_dict['cap-up'][stf, sit, pro])


def res_process_capacity_fixed_inv_cost_upper_rule(m, stf, sit, pro):
    return m.cap_pro_new[stf, sit, pro] <= m.pro_cap_expands[stf, sit, pro] * m.process_dict['cap-up'][stf, sit, pro]


# Costs
def def_costs_rule(m, cost_type):
    """Cost of one cost type (EUR/a).

    - Invest: new process power, storage power and storage capacity times the
      annuity factors (plus fixed investment portions of expanded processes).
    - Fixed: annual fixed costs of total process and storage capacities.
    - Variable: process throughput and storage in/out flows, scaled by the
      annual weight and the type-period weights.
    - Revenue and Purchase: see ``buy_sell_price.py``.
    """
    if cost_type == 'Invest':
        cost = \
            (sum(m.cap_pro_new[p] *
                 m.process_dict['inv-cost'][p] *
                 m.process_dict['invcost-factor'][p]
                 for p in m.pro_tuples)
             + sum(m.pro_cap_expands[p] *
                   m.process_dict['inv-cost-fix'][p] *
                   m.process_dict['invcost-factor'][p]
                   for p in m.pro_inv_cost_fix_tuples))
        if m.mode['sto']:
            cost += storage_cost(m, cost_type)
        return m.costs[cost_type] == cost

    elif cost_type == 'Fixed':
        cost = \
            sum(m.cap_pro[p] * m.process_dict['fix-cost'][p] *
                m.process_dict['cost_factor'][p]
                for p in m.pro_tuples)
        if m.mode['sto']:
            cost += storage_cost(m, cost_type)
        return m.costs[cost_type] == cost

    elif cost_type == 'Variable':
        cost = \
            sum(m.tau_pro[(tm,) + p] * m.weight * m.typeperiod['weight_typeperiod'][(m.stf_list[0], tm)] *
                m.process_dict['var-cost'][p] *
                m.process_dict['cost_factor'][p]
                for tm in m.tm
                for p in m.pro_tuples)
        if m.mode['sto']:
            cost += storage_cost(m, cost_type)
        return m.costs[cost_type] == cost

    elif cost_type == 'Revenue':
        return m.costs[cost_type] == revenue_costs(m)
    elif cost_type == 'Purchase':
        return m.costs[cost_type] == purchase_costs(m)
    else:
        raise NotImplementedError("Unknown cost type.")


def cost_rule(m):
    return pyomo.summation(m.costs)
