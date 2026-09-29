"""Radial topology of the staged expansion rules (gridexpand.analysis.expansion.topology)."""

from __future__ import annotations

import math

import pandapower as pp
import pytest

from gridexpand.analysis.expansion import topology


def _real_like():
    """External grid on the LV busbar 0; zero-length station line 0-1; outlet 1-2; branches 2-3, 2-4-5.

    Loads at 3 (terminal: service line 2-3) and at 4 (not terminal: 4 has two line neighbours).
    """
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=0.4, geodata=(700000.0 + 10 * i, 5400000.0)) for i in range(6)]
    pp.create_ext_grid(net, buses[0])
    pp.create_line_from_parameters(net, buses[0], buses[1], length_km=0.0, r_ohm_per_km=0.0, x_ohm_per_km=0.0,
                                   c_nf_per_km=0.0, max_i_ka=1.0)
    for a, b, length in ((1, 2, 0.1), (2, 3, 0.05), (2, 4, 0.08), (4, 5, 0.04)):
        pp.create_line(net, buses[a], buses[b], length_km=length, std_type="NAYY 4x150 SE")
    pp.create_load(net, buses[3], p_mw=0.0)
    pp.create_load(net, buses[4], p_mw=0.0)
    return net


def test_station_outlets_services_and_distances():
    topo = topology.build_topology(_real_like())
    assert topo.root == 0 and topo.station_buses == frozenset({0, 1})
    assert topo.is_outlet(1, 2) and not topo.is_outlet(2, 4)
    assert topo.is_service(2, 3) and not topo.is_service(2, 4)
    assert topo.child(2, 3) == 3 and topo.child(3, 2) == 3
    assert topo.distance_km[3] == pytest.approx(0.15)
    assert topo.distance_km[5] == pytest.approx(0.22)
    assert topo.coordinates[1] == (700010.0, 5400000.0)  # already metres


def test_root_is_the_lv_side_of_the_fed_transformer():
    net = pp.create_empty_network()
    mv = pp.create_bus(net, vn_kv=20.0)
    lv = pp.create_bus(net, vn_kv=0.4)
    end = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, mv)
    pp.create_transformer(net, mv, lv, std_type="0.4 MVA 20/0.4 kV")
    pp.create_line(net, lv, end, length_km=0.1, std_type="NAYY 4x150 SE")
    topo = topology.build_topology(net)
    assert topo.root == lv and topo.is_outlet(lv, end)


def test_wgs84_coordinates_are_projected_to_metres():
    net = pp.create_empty_network()
    for lon in (12.1641, 12.1651):
        pp.create_bus(net, vn_kv=0.4, geodata=(lon, 48.571))
    points = list(topology.bus_coordinates(net).values())
    distance = math.dist(points[0], points[1])
    # 0.001 degrees of longitude at 48.571 N: about 73.7 m
    assert distance == pytest.approx(111320 * 0.001 * math.cos(math.radians(48.571)), rel=0.01)
    assert points[0][0] > 1000  # metres, not degrees
