"""Step 4 engine: one solve per timestep, raw tables and summary from the same matrices."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pytest

from gridexpand.powerflow import engine, network, station_voltage
from gridexpand.powerflow.powerflow import pf, pf_summary


def _prepared_grid():
    net = pn.create_kerber_landnetz_kabel_1()
    buses = sorted(int(bus) for bus in net.load["bus"])
    grid = network.set_scenario_load_buses(net, buses)
    rating = network.transformer_rating_mva(grid)
    cable_max_i_ka = network.rated_cable_currents(grid)
    grid = network.prepare_synthetic_grid(grid)
    return grid, buses, rating, cable_max_i_ka


def _demand(buses, n=12, seed=3):
    rng = np.random.default_rng(seed)
    columns = pd.MultiIndex.from_tuples(
        [(bus, component) for bus in buses for component in ("electricity", "electricity-reactive")]
    )
    values = rng.uniform(-4.0, 12.0, size=(n, len(columns)))
    return pd.DataFrame(values, columns=columns)


def _reference_raw(grid, demand):
    """The former per-timestep loop (reset loads, assign kW/1000, runpp, collect)."""
    grid = deepcopy(grid)
    ext, vm, lines = [], [], []
    for _, row in demand.iterrows():
        load = row.unstack(level=1)
        grid.load["p_mw"] = 0.0
        grid.load["q_mvar"] = 0.0
        for bus, values in load.iterrows():
            mask = grid.load["bus"] == bus
            grid.load.loc[mask, "p_mw"] = values["electricity"] / 1000
            grid.load.loc[mask, "q_mvar"] = values["electricity-reactive"] / 1000
        pp.runpp(grid, algorithm="bfsw", max_iteration=50, tolerance_mva=1e-6)
        ext.append(grid.res_ext_grid)
        vm.append(grid.res_bus[["vm_pu"]].T.reset_index(drop=True))
        lines.append(grid.res_line[["p_from_mw", "q_from_mvar", "i_from_ka"]].stack().to_frame().T.reset_index(drop=True))
    return (
        pd.concat(ext, axis=0).reset_index(drop=True),
        pd.concat(vm, axis=0).reset_index(drop=True),
        pd.concat(lines, axis=0).reset_index(drop=True),
    )


def test_raw_tables_match_the_per_timestep_loop_for_any_chunking():
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses)
    expected = _reference_raw(grid, demand)
    for n_workers in (1, 5):
        for actual, reference in zip(pf(grid, demand, True, n_workers), expected):
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            assert actual.columns.equals(reference.columns)


def test_summary_is_independent_of_chunking():
    grid, buses, rating, cable_max_i_ka = _prepared_grid()
    demand = _demand(buses, n=30)
    cables, voltage_buses = network.comparison_evaluation_scope(grid, buses, scope="full")
    kwargs = dict(transformer_s_rated_mva=rating, cable_max_i_ka=cable_max_i_ka,
                  voltage_buses=voltage_buses, cable_ids=cables, on_nonconvergence="nan")
    serial = pf_summary(grid, demand, **kwargs)
    parallel = pf_summary(grid, demand, n_workers=4, **kwargs)
    matrices, station = station_voltage.solve(
        grid, voltage_buses, lambda net: engine.run_timeseries(net, demand, n_workers=3)
    )
    one_pass = engine.summarize(grid, matrices, **{k: v for k, v in kwargs.items() if k != "on_nonconvergence"})
    one_pass["grid_summary"].update(station.as_summary())
    for other in (parallel, one_pass):
        for key in ("cable_summary", "bus_voltage_summary", "tail_summary", "transformer_diagnostic"):
            pd.testing.assert_frame_equal(serial[key], other[key], check_exact=True)
        assert serial["grid_summary"].keys() == other["grid_summary"].keys()
        for key, value in serial["grid_summary"].items():
            assert value == other["grid_summary"][key] or (value != value and other["grid_summary"][key] != other["grid_summary"][key]), key
    assert serial["grid_summary"]["n_timesteps"] == 30


def test_nonconverged_timestep_does_not_change_the_next_one():
    """A chunk reuses one net, so a failed solve must leave nothing behind (real grids: NR, Iwamoto)."""
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses, n=3)
    demand.iloc[1] = demand.iloc[1].abs() * 400.0  # far beyond the grid's capacity: no convergence
    kwargs = dict(algorithm=["nr", "iwamoto_nr"], on_nonconvergence="nan")
    together = engine.run_timeseries(grid, demand, **kwargs)
    assert together.failed == [1]
    for t in (0, 2):
        alone = engine.run_timeseries(grid, demand.iloc[[t]], **kwargs)
        assert alone.failed == []
        np.testing.assert_array_equal(together.vm_pu[t], alone.vm_pu[0])
        np.testing.assert_array_equal(together.line[t], alone.line[0])
        np.testing.assert_array_equal(together.ext_grid[t], alone.ext_grid[0])


def test_more_workers_than_timesteps_does_not_create_empty_chunks():
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses, n=24)
    reference = pf(grid, demand, True, 1)
    for n_workers in (13, 23, 40):
        for actual, expected in zip(pf(grid, demand, True, n_workers), reference):
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_caller_grid_is_not_modified():
    grid, buses, _, _ = _prepared_grid()
    before = grid.load.copy()
    engine.run_timeseries(grid, _demand(buses, n=3))
    pd.testing.assert_frame_equal(grid.load, before)


def test_demand_with_missing_values_is_rejected():
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses, n=3)
    demand.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        engine.run_timeseries(grid, demand)


def test_demand_bus_without_load_row_is_rejected():
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses, n=2)
    demand[(9999, "electricity")] = 1.0
    with pytest.raises(ValueError, match="no load row"):
        engine.run_timeseries(grid, demand)


def test_missing_component_counts_as_zero():
    grid, buses, _, _ = _prepared_grid()
    demand = _demand(buses, n=2).drop(columns=[(buses[0], "electricity-reactive")])
    p, q = engine.demand_arrays(grid, demand)
    row = list(grid.load["bus"]).index(buses[0])
    assert (q[:, row] == 0).all() and (p[:, row] != 0).all()
