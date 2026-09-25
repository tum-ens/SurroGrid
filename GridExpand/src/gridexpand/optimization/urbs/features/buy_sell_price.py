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


def revenue_costs(m):
    sell_tuples = commodity_subset(m.com_tuples, m.com_sell)
    try:
        return -sum(
            m.e_co_sell[(tm,) + c] *
            m.buy_sell_price_dict[c[2]][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
            m.commodity_dict['price'][c] *
            m.commodity_dict['cost_factor'][c]
            for tm in m.tm
            for c in sell_tuples)
    except KeyError:
        try:
            return -sum(
                m.e_co_sell[(tm,) + c] *
                m.buy_sell_price_dict[c[2], ][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
                m.commodity_dict['price'][c] *
                m.commodity_dict['cost_factor'][c]
                for tm in m.tm
                for c in sell_tuples)
        except KeyError:
            return -sum(
                m.e_co_sell[(tm,) + c] *
                m.buy_sell_price_dict[c[1], c[2]][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
                m.commodity_dict['price'][c] *
                m.commodity_dict['cost_factor'][c]
                for tm in m.tm
                for c in sell_tuples)

def purchase_costs(m):
    buy_tuples = commodity_subset(m.com_tuples, m.com_buy)
    try:
        return sum(
            m.e_co_buy[(tm,) + c] *
            m.buy_sell_price_dict[c[2]][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
            m.commodity_dict['price'][c] *
            m.commodity_dict['cost_factor'][c]
            for tm in m.tm
            for c in buy_tuples)
    except KeyError:
        try:
            return sum(
                m.e_co_buy[(tm,) + c] *
                m.buy_sell_price_dict[c[2], ][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
                m.commodity_dict['price'][c] *
                m.commodity_dict['cost_factor'][c]
                for tm in m.tm
                for c in buy_tuples)
        except KeyError:
            return sum(
                m.e_co_buy[(tm,) + c] *
                m.buy_sell_price_dict[c[1],c[2]][(c[0], tm)] * m.weight *  m.typeperiod['weight_typeperiod'][(m.stf_list[0],tm)] *
                m.commodity_dict['price'][c] *
                m.commodity_dict['cost_factor'][c]
                for tm in m.tm
                for c in buy_tuples)
