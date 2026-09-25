"""Pure helpers of the urbs model: balances, prices, cluster data and partition."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyomo
import pytest

from gridexpand.optimization.urbs.features.buy_sell_price import _price_series
from gridexpand.optimization.urbs.features.modelhelper import (
    commodity_balance,
    invcost_factor,
)
from gridexpand.optimization.urbs.identify import get_parallel_building_clusters
from gridexpand.optimization.urbs.input import get_cluster_data

STF = 2026


def _reference_commodity_balance(m, tm, stf, sit, com):
    """The former full scan over all process and storage tuples."""
    balance = (sum(m.e_pro_in[(tm, stframe, site, process, com)]
                   for stframe, site, process in m.pro_tuples
                   if site == sit and stframe == stf and
                   (stframe, process, com) in m.r_in_dict) -
               sum(m.e_pro_out[(tm, stframe, site, process, com)]
                   for stframe, site, process in m.pro_tuples
                   if site == sit and stframe == stf and
                   (stframe, process, com) in m.r_out_dict))
    if m.mode['sto']:
        balance += sum(m.e_sto_in[(tm, stframe, site, storage, com)] -
                       m.e_sto_out[(tm, stframe, site, storage, com)]
                       for stframe, site, storage, commodity in m.sto_tuples
                       if site == sit and stframe == stf and commodity == com)
    return balance


def _small_model():
    m = pyomo.ConcreteModel()
    m.mode = {'sto': True}
    m.tm = pyomo.Set(initialize=[1, 2], ordered=True)
    m.pro_tuples = pyomo.Set(initialize=[
        (STF, 'a', 'import'), (STF, 'a', 'pv'), (STF, 'a', 'hp'),
        (STF, 'b', 'hp'), (STF, 'b', 'import'), (STF, 'b', 'feed_in'),
    ])
    m.r_in_dict = {
        (STF, 'import', 'grid'): 1, (STF, 'pv', 'solar'): 1,
        (STF, 'hp', 'electricity'): 1, (STF, 'feed_in', 'electricity'): 1,
    }
    m.r_out_dict = {
        (STF, 'import', 'electricity'): 1, (STF, 'pv', 'electricity'): 1,
        (STF, 'hp', 'heat'): 1, (STF, 'feed_in', 'grid_out'): 1,
    }
    m.pro_input_tuples = pyomo.Set(initialize=[
        (s, site, p, c) for (s, site, p) in m.pro_tuples
        for (s2, p2, c) in m.r_in_dict if p == p2 and s == s2])
    m.pro_output_tuples = pyomo.Set(initialize=[
        (s, site, p, c) for (s, site, p) in m.pro_tuples
        for (s2, p2, c) in m.r_out_dict if p == p2 and s == s2])
    m.sto_tuples = pyomo.Set(initialize=[
        (STF, 'a', 'battery', 'electricity'), (STF, 'b', 'tank', 'heat'),
        (STF, 'a', 'tank', 'heat'), (STF, 'a', 'battery2', 'electricity'),
    ])
    m.e_pro_in = pyomo.Var(m.tm, m.pro_input_tuples)
    m.e_pro_out = pyomo.Var(m.tm, m.pro_output_tuples)
    m.e_sto_in = pyomo.Var(m.tm, m.sto_tuples)
    m.e_sto_out = pyomo.Var(m.tm, m.sto_tuples)
    return m


@pytest.mark.parametrize("sto", [True, False])
def test_commodity_balance_equals_full_scan(sto):
    m = _small_model()
    m.mode['sto'] = sto
    for tm in m.tm:
        for sit in ('a', 'b', 'c'):
            for com in ('electricity', 'heat', 'grid', 'solar', 'grid_out', 'none'):
                expected = _reference_commodity_balance(m, tm, STF, sit, com)
                actual = commodity_balance(m, tm, STF, sit, com)
                assert str(actual) == str(expected), (tm, sit, com)


def test_price_series_resolves_all_column_layouts():
    series = {(STF, 1): 0.3}
    c = (STF, 'site', 'electricity_import', 'Buy')

    class M:
        pass

    for key in ('electricity_import', ('electricity_import',), ('site', 'electricity_import')):
        m = M()
        m.buy_sell_price_dict = {key: series}
        assert _price_series(m, c) is series
    m = M()
    m.buy_sell_price_dict = {'other': series}
    with pytest.raises(KeyError):
        _price_series(m, c)


def test_invcost_factor():
    assert invcost_factor(20, 0) == pytest.approx(1 / 20)
    assert invcost_factor(20, 0.05) == pytest.approx(0.08024258719069129)
    with pytest.raises(ZeroDivisionError):
        invcost_factor(0, 0)


def _data():
    index = pd.MultiIndex.from_tuples(
        [(STF, 1, 'import'), (STF, 2, 'import'), (STF, 3, 'import')],
        names=['support_timeframe', 'Site', 'Process'])
    columns = pd.MultiIndex.from_tuples([(1, 'electricity'), (2, 'electricity'), (3, 'heat')])
    return {
        'process': pd.DataFrame({'inst-cap': [1.0, 2.0, 3.0]}, index=index),
        'demand': pd.DataFrame([[1.0, 2.0, 3.0]], columns=columns),
        'global_prop': pd.DataFrame({'value': [1]}),
        'ev_sessions': pd.DataFrame({'session_id': ['s1', 's2'], 'site': [1, 3]}),
        'ev_session_hours': pd.DataFrame({'session_id': ['s1', 's2', 's2'], 't': [1, 1, 2]}),
    }


def test_get_cluster_data_filters_sites_and_copies():
    data = _data()
    cluster = get_cluster_data(data, [1, 3])
    assert list(cluster) == list(data)
    assert cluster['process'].index.get_level_values('Site').tolist() == [1, 3]
    assert cluster['demand'].columns.tolist() == [(1, 'electricity'), (3, 'heat')]
    assert cluster['ev_sessions']['session_id'].tolist() == ['s1', 's2']
    assert cluster['ev_session_hours']['session_id'].tolist() == ['s1', 's2', 's2']
    only_two = get_cluster_data(data, [2])
    assert only_two['ev_sessions'].empty and only_two['ev_session_hours'].empty
    cluster['global_prop'].loc[0, 'value'] = 99
    assert data['global_prop'].loc[0, 'value'] == 1


def test_building_partition_rule_is_unchanged():
    site = pd.DataFrame(index=pd.MultiIndex.from_product(
        [[STF], list(range(10))], names=['support_timeframe', 'Name']))
    clusters = get_parallel_building_clusters({'site': site}, 4)
    assert clusters == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert get_parallel_building_clusters({'site': site}, 12) == [[i] for i in range(10)]
