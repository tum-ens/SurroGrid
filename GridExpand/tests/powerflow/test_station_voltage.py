"""Step 4 station voltage: LV busbar at the reference, off-load tap as in pylovo's validation power flow."""

from __future__ import annotations

import pandapower as pp
import pandas as pd
import pytest

from gridexpand.powerflow import engine, station_voltage


def _feeder(length_km):
    """External grid on the LV busbar -- one NAYY 4x150 line -- one load bus (a prepared Step 4 grid)."""
    net = pp.create_empty_network()
    busbar = pp.create_bus(net, vn_kv=0.4)
    end = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, busbar, vm_pu=1.0)
    pp.create_line(net, busbar, end, length_km=length_km, std_type="NAYY 4x150 SE")
    pp.create_load(net, end, p_mw=0.0)
    return net, end


def _solve(length_km, kw):
    net, end = _feeder(length_km)
    demand = pd.DataFrame({(end, "electricity"): [float(value) for value in kw]})
    demand.columns = pd.MultiIndex.from_tuples(demand.columns)
    matrices, station = station_voltage.solve(net, [end], lambda grid: engine.run_timeseries(grid, demand))
    low, high = station_voltage.voltage_extremes(matrices, [end])
    return net, station, low, high


def test_busbar_voltage_of_the_hv_side_tap():
    assert station_voltage.busbar_voltage_pu(0) == pytest.approx(0.96)
    assert station_voltage.busbar_voltage_pu(1) == pytest.approx(0.96 / 0.975)
    assert station_voltage.busbar_voltage_pu(2) == pytest.approx(0.96 / 0.95)


@pytest.mark.parametrize(
    ("length_km", "kw", "steps", "within_band"),
    [
        (0.2, [60, 20], 0, True),     # the reference is enough
        (0.35, [120, 20], 1, True),   # 0.899 p.u. at the reference, 0.925 after one step
        (0.5, [150, 20], 2, True),    # 0.843 / 0.872 / 0.902 p.u.
        (0.6, [150, 20], 2, False),   # the tap range is exhausted (0.876 p.u.)
    ],
)
def test_tap_lifts_the_busbar_while_a_bus_is_below_the_band(length_km, kw, steps, within_band):
    _, station, low, _ = _solve(length_km, kw)
    assert station.tap_steps == steps
    assert station.lv_busbar_vm_pu == pytest.approx(station_voltage.busbar_voltage_pu(steps))
    assert (low >= 0.90) is within_band


@pytest.mark.parametrize(
    ("length_km", "kw", "steps"),
    [
        (0.35, [120, -320], 0),  # one step would lift the feed-in hour to 1.114 p.u.
        (0.5, [150, -180], 1),   # the second step would reach 1.115 p.u.
    ],
)
def test_a_step_above_the_upper_limit_is_not_used(length_km, kw, steps):
    _, station, low, high = _solve(length_km, kw)
    assert station.tap_steps == steps
    assert low < 0.90 and high <= 1.10


def test_the_callers_grid_keeps_its_voltage():
    net, _, _, _ = _solve(0.35, [120, 20])
    assert net.ext_grid["vm_pu"].tolist() == [1.0]


def test_external_grid_must_sit_on_the_lv_busbar():
    net, end = _feeder(0.2)
    net.bus.loc[net.ext_grid["bus"], "vn_kv"] = 20.0
    with pytest.raises(ValueError, match="LV busbar"):
        station_voltage.solve(net, [end], lambda grid: None)


def test_summary_records_the_station_voltage():
    from gridexpand.powerflow.powerflow import pf_summary

    net, end = _feeder(0.35)
    demand = pd.DataFrame({(end, "electricity"): [120.0, 20.0]})
    demand.columns = pd.MultiIndex.from_tuples(demand.columns)
    summary = pf_summary(net, demand, transformer_s_rated_mva=0.4,
                         cable_max_i_ka=pd.Series({0: 0.27}), voltage_buses=[end])
    grid = summary["grid_summary"]
    assert grid["tap_steps"] == 1
    assert grid["lv_busbar_vm_pu"] == pytest.approx(0.96 / 0.975)
    assert summary["bus_voltage_summary"]["voltage_min_time_pu"].min() >= 0.90
