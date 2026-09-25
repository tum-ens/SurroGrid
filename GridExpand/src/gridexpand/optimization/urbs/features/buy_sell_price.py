"""Buy and sell commodities (grid import and feed-in) with time series prices."""

import pyomo.core as pyomo

from .modelhelper import commodity_subset


def add_buy_sell_price(m):
    """Add the buy/sell commodity sets and source variables to ``m``."""
    # Sets
    m.com_sell = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Sell'),
        ordered=False,
        doc='Commodities that can be sold')
    m.com_buy = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Buy'),
        ordered=False,
        doc='Commodities that can be purchased')

    m.com_buy_tuples = pyomo.Set(
        within = m.stf * m.sit * m.com_buy * m.com_type,
        initialize= tuple(key for key in m.commodity_dict["price"].keys() if key[3] == "Buy"),
        ordered= False,
        doc="Subset of commodity_type 'Buy' of commodity tuples")

    m.com_sell_tuples = pyomo.Set(
        within = m.stf * m.sit * m.com_sell * m.com_type,
        initialize = tuple(key for key in m.commodity_dict["price"].keys() if key[3] == "Sell"),
        ordered= False,
        doc="Subset of commodity_type 'Sell' of commodity tuples")

    # Variables
    m.e_co_sell = pyomo.Var(
        m.tm, m.com_sell_tuples,
        within=pyomo.NonNegativeReals,
        doc='Use of sell commodity source (MW) per timestep')
    m.e_co_buy = pyomo.Var(
        m.tm, m.com_buy_tuples,
        within=pyomo.NonNegativeReals,
        doc='Use of buy commodity source (MW) per timestep')
    return m


def bsp_surplus(m, tm, stf, sit, com, com_type):
    """Buy (+) and sell (-) source terms of one commodity balance."""
    power_surplus = 0

    # if com is a sell commodity, the commodity source term e_co_sell
    # can supply a possibly positive power_surplus
    if com in m.com_sell:
        power_surplus -= m.e_co_sell[tm, stf, sit, com, com_type]

    # if com is a buy commodity, the commodity source term e_co_buy
    # can supply a possibly negative power_surplus
    if com in m.com_buy:
        power_surplus += m.e_co_buy[tm, stf, sit, com, com_type]

    return power_surplus


def _price_series(m, c):
    """Price time series of commodity tuple ``c`` in ``m.buy_sell_price_dict``.

    The dict is keyed by commodity name, by a one-level tuple or by
    (site, commodity), depending on the column layout of the input table.
    """
    prices = m.buy_sell_price_dict
    for key in (c[2], (c[2],), (c[1], c[2])):
        if key in prices:
            return prices[key]
    raise KeyError(f"No buy/sell price time series for commodity {c!r}.")


def _weighted_price_sum(m, flow, com_tuples):
    return sum(
        flow[(tm,) + c] *
        _price_series(m, c)[(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
        m.commodity_dict['price'][c] *
        m.commodity_dict['cost_factor'][c]
        for tm in m.tm
        for c in com_tuples)


def revenue_costs(m):
    """Feed-in revenue (negative cost) over all sell commodities and timesteps."""
    return -_weighted_price_sum(m, m.e_co_sell, commodity_subset(m.com_tuples, m.com_sell))


def purchase_costs(m):
    """Grid purchase cost over all buy commodities and timesteps."""
    return _weighted_price_sum(m, m.e_co_buy, commodity_subset(m.com_tuples, m.com_buy))
