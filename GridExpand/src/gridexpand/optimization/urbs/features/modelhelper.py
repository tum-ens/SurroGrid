"""Helper functions shared by the urbs model rules."""

from .storage import storage_balance


def invcost_factor(dep_prd, interest):
    """Annuity factor that turns an investment into annual cost.

    Args:
        dep_prd: depreciation period (years).
        interest: interest rate (e.g. 0.06 means 6 %).

    Returns:
        ``1 / dep_prd`` for zero interest, else the capital recovery factor
        ``(1 + i)^n * i / ((1 + i)^n - 1)``.
    """
    if interest == 0:
        return 1 / dep_prd
    else:
        return ((1 + interest) ** dep_prd * interest /
                ((1 + interest) ** dep_prd - 1))


def _process_flow_index(m):
    """Process inputs/outputs of each (stf, site, commodity), built once per model.

    ``commodity_balance`` used to scan all ``m.pro_tuples`` for every balance row,
    which is quadratic in the number of buildings per cluster. The lists keep the
    ``m.pro_tuples`` order, so every balance expression (and the LP file) is
    unchanged.
    """
    index = getattr(m, "_process_flow_index", None)
    if index is None:
        in_coms, out_coms = {}, {}
        for stframe, process, com in m.r_in_dict:
            in_coms.setdefault((stframe, process), []).append(com)
        for stframe, process, com in m.r_out_dict:
            out_coms.setdefault((stframe, process), []).append(com)
        inputs, outputs = {}, {}
        for stframe, site, process in m.pro_tuples:
            for com in in_coms.get((stframe, process), ()):
                inputs.setdefault((stframe, site, com), []).append((stframe, site, process))
            for com in out_coms.get((stframe, process), ()):
                outputs.setdefault((stframe, site, com), []).append((stframe, site, process))
        index = m._process_flow_index = (inputs, outputs)
    return index


def commodity_balance(m, tm, stf, sit, com):
    """Calculate commodity balance at given timestep.

    For a given commodity co and timestep tm, calculate the balance of
    consumed (to process/storage, counts positive) and provided
    (from process/storage, counts negative) commodity flow. Used
    as helper function in create_model for constraints on demand and stock
    commodities.

    Args:
        m: the model object
        tm: the timestep
        stf: the support timeframe
        sit: the site
        com: the commodity

    Returns:
        balance: net value of consumed (positive) or provided (negative) power
    """
    inputs, outputs = _process_flow_index(m)
    balance = (sum(m.e_pro_in[(tm, stframe, site, process, com)]
                   # usage as input for process increases balance
                   for stframe, site, process in inputs.get((stf, sit, com), ())) -
               sum(m.e_pro_out[(tm, stframe, site, process, com)]
                   # output from processes decreases balance
                   for stframe, site, process in outputs.get((stf, sit, com), ())))
    if m.mode['sto']:
        balance += storage_balance(m, tm, stf, sit, com)

    return balance


def commodity_subset(com_tuples, type_name):
    """Unique list of commodity names for given type.

    Args:
        com_tuples: a list of (stf, site, commodity, commodity type) tuples
        type_name: a commodity type or a set of commodity names

    Returns:
        The set of commodity names of the desired type, or, for a set of names,
        the set of commodity tuples whose commodity is in it.
    """
    if type(type_name) is str:
        # type_name: ('Stock', 'SupIm', 'Env' or 'Demand')
        return set(com for stf, sit, com, com_type in com_tuples
                   if com_type == type_name)
    else:
        # type(type_name) is a pyomo Set of commodity names, e.g. m.com_buy
        return set((stf, sit, com, com_type) for stf, sit, com, com_type
                   in com_tuples if com in type_name)
