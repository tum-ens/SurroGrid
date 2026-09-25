"""Step 4 network preparation and evaluation scopes."""

from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pytest

from gridexpand.powerflow import network


def _feeder():
    """Root -- 1 -- 2 -- 3 (terminal) with a branch 2 -- 4 (terminal)."""
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=0.4) for _ in range(5)]
    pp.create_ext_grid(net, buses[0])
    for a, b in ((0, 1), (1, 2), (2, 3), (2, 4)):
        pp.create_line(net, buses[a], buses[b], length_km=0.1, std_type="NAYY 4x150 SE")
    return net


def test_transformer_rating_multiplies_parallel_units():
    net = pn.create_kerber_landnetz_kabel_1()
    net.trafo["parallel"] = 2
    assert network.transformer_rating_mva(net) == pytest.approx(0.2)
    net.trafo = net.trafo.drop(columns=["parallel"])
    assert network.transformer_rating_mva(net) == pytest.approx(0.1)


def test_prepare_synthetic_grid_requires_exactly_one_transformer():
    net = pn.create_kerber_landnetz_kabel_1()
    extra = net.trafo.iloc[[0]].copy()
    extra.index = [7]
    net.trafo = pd.concat([net.trafo, extra])
    with pytest.raises(ValueError, match="exactly one transformer"):
        network.prepare_synthetic_grid(net)


def test_prepare_synthetic_grid_replaces_transformer_with_switch():
    net = pn.create_kerber_landnetz_kabel_1()
    net.trafo.index = [5]
    grid = network.prepare_synthetic_grid(network.set_scenario_load_buses(net, [2, 3]))
    assert grid.trafo.empty
    assert grid.switch["name"].tolist() == ["SW_replacing_T0"]
    assert (grid.line["max_i_ka"] == 1000).all()


def test_scenario_loads_are_one_zeroed_row_per_bus():
    net = pn.create_kerber_landnetz_kabel_1()
    grid = network.set_scenario_load_buses(net, [5, 3, 3])
    assert grid.load["bus"].tolist() == [3, 5]
    assert grid.load["name"].tolist() == ["Scenario_Profile_3", "Scenario_Profile_5"]
    assert (grid.load[["p_mw", "q_mvar"]] == 0).all().all()
    with pytest.raises(ValueError, match="missing from the pandapower grid"):
        network.set_scenario_load_buses(net, [999])


def test_backbone_scope_drops_terminal_service_lines():
    net = _feeder()
    cables, voltage_buses = network.comparison_backbone_scope(net, [3, 4])
    assert cables == [0, 1]
    assert voltage_buses == [2]
    full_cables, full_buses = network.comparison_evaluation_scope(net, [3, 4], scope="full")
    assert full_cables == [0, 1, 2, 3] and full_buses == [3, 4]
