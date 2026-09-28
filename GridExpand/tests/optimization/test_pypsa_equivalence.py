"""The PyPSA optimizer solves the urbs model: same objective, costs, capacities and hours.

The scenario of ``deterministic_fixture.py`` has a unique optimum (certified below
on the PyPSA model), so every result of a correct model, down to the hourly
dispatch and the storage contents, must equal the urbs result. Both models are
solved with HiGHS to optimality (MIP gap 0).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from deterministic_fixture import VARIANTS, read_prepared, write_input

EXACT = {"mip_rel_gap": 0.0}
COMPARED = ("tau_pro", "cap_pro", "cap_sto_c", "cap_sto_p", "e_sto_in", "e_sto_out", "e_sto_con", "costs")
CONTINUOUS = ("Generator-p", "Link-p", "StorageUnit-p_store", "StorageUnit-p_dispatch",
              "StorageUnit-state_of_charge", "Generator-p_nom", "Link-p_nom", "StorageUnit-p_nom")


def solve_urbs(data, settings):
    from pyomo.environ import SolverFactory
    from pyomo.opt import check_optimal_termination

    from gridexpand.optimization.urbs.model import create_model
    from gridexpand.optimization.urbs.saveload import create_result_cache

    model = create_model(copy.deepcopy(data), settings)
    solver = SolverFactory("appsi_highs")
    solver.options = dict(EXACT)
    assert check_optimal_termination(solver.solve(model))
    return float(model.objective_function()), create_result_cache(model)


@pytest.fixture(scope="module", params=VARIANTS)
def solved(request, tmp_path_factory):
    from gridexpand.optimization.pypsa_model.solve import solve_group, solve_network

    path = write_input(tmp_path_factory.mktemp(request.param) / "input.h5", request.param)
    data, mode, settings = read_prepared(path)
    objective, urbs = solve_urbs(data, settings)
    pypsa = solve_group(copy.deepcopy(data), mode, solver_name="appsi_highs", options=EXACT)
    network, parts, _ = solve_network(copy.deepcopy(data), mode, solver_name="appsi_highs", options=EXACT)
    return request.param, mode, objective, urbs, pypsa, network


def test_the_fixture_covers_the_variant(solved):
    variant, mode, _, urbs, _, network = solved
    assert mode["sto"] and mode["bsp"] and mode["tve"] and not mode["tdy"]
    assert mode["evs"] == (variant == "ev_sessions")
    assert network.model.type == ("MILP" if variant == "optimized" else "LP")
    if variant == "ev_sessions":
        assert "EV-session-energy" in network.model.constraints
    # every flow family is active, so nothing is compared trivially at zero
    tau = urbs["tau_pro"].groupby(level="pro").sum()
    assert (tau.filter(like="Rooftop PV") > 0).all() and tau["feed_in"] > 0 and tau["heatpump_air"] > 0
    assert (urbs["e_sto_in"].groupby(level="sto").sum() > 0).all()
    if variant == "optimized":  # some fixed-cost decisions are "build", some "do not build"
        built = pd.concat([network.model[name].solution.to_pandas()
                           for name in ("Generator-build", "Link-build") if name in network.model.variables])
        assert (built > 0.5).any() and (built < 0.5).any()
    else:  # the heating rod covers the heat pump's shortfall
        assert tau["heatpump_booster"] > 0


def test_the_optimum_is_unique(solved):
    """Minimise and maximise a random direction over the optimal face: one point.

    The face is widened by 1e-11 of the objective (the order of the solver's
    feasibility tolerance). The spread grows linearly with that width, which marks
    a unique optimum with some almost-free directions rather than a tie; within it
    every solution lies within 1e-4 of the solver's point, so an hourly comparison
    with urbs is meaningful.
    """
    variant, _, _, _, _, network = solved
    m = network.model
    optimum = float(m.objective.value)
    for name in ("Generator-build", "Link-build"):  # fix the MILP's decisions
        if name in m.variables:
            m.variables[name].lower = m.variables[name].solution
            m.variables[name].upper = m.variables[name].solution
    m.add_constraints(m.objective.expression <= optimum + 1e-11 * max(1.0, abs(optimum)), name="optimal-face")
    rng = np.random.default_rng(0)
    names = [name for name in CONTINUOUS if name in m.variables]
    direction = sum(
        (m[name] * xr.DataArray(rng.uniform(0.5, 1.5, m[name].shape), coords=m[name].labels.coords,
                                dims=m[name].dims)).sum()
        for name in names
    )
    points = []
    for sign in (1.0, -1.0):
        m.objective = sign * direction
        status, condition = m.solve(solver_name="highs", io_api="direct", set_names=False, output_flag=False, **EXACT)
        assert (status, condition) == ("ok", "optimal")
        points.append({name: m[name].solution.values.copy() for name in names})
    for name in names:
        np.testing.assert_allclose(points[0][name], points[1][name], atol=1e-4, err_msg=f"{variant}: {name}")


def test_objective_equals_urbs(solved):
    _, _, objective, _, pypsa, _ = solved
    assert pypsa["audit"]["objective"] == pytest.approx(objective, rel=1e-9)
    assert float(pypsa["results"]["costs"].sum()) == pytest.approx(objective, rel=1e-9)


@pytest.mark.parametrize("key", COMPARED)
def test_results_equal_urbs(solved, key):
    variant, _, _, urbs, pypsa, _ = solved
    expected = urbs[key].sort_index()
    actual = pypsa["results"][key].sort_index()
    assert list(actual.index.names) == list(expected.index.names)
    pd.testing.assert_index_equal(actual.index, expected.index, check_exact=True)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-5, rtol=1e-7,
                               err_msg=f"{variant}: {key}")


def test_no_optimum_charges_while_the_car_is_away(tmp_path):
    """Availability limits the charging-station input in both optimizers.

    At a feed-in tariff of 0 surplus PV is free, so without that limit an optimal
    solution could burn it in a station whose car is away. Maximise exactly that
    over the optimal solutions: it must be 0.
    """
    import pyomo.environ as pyo
    from pyomo.environ import SolverFactory

    from gridexpand.optimization.pypsa_model.solve import solve_network
    from gridexpand.optimization.urbs.model import create_model

    data, mode, settings = read_prepared(write_input(tmp_path / "input.h5", "heuristic"))
    data["buy_sell_price"] = data["buy_sell_price"].assign(electricity_feed_in=0.0)
    eff = data["eff_factor"]
    away = {(int(t), site, pro): 1.0 - float(v) for (site, pro) in eff.columns if pro.startswith("charging_station")
            for (_, t), v in eff[(site, pro)].items() if t > 0 and v < 1.0}
    assert away

    # urbs
    model = create_model(copy.deepcopy(data), settings)
    solver = SolverFactory("appsi_highs")
    solver.options = dict(EXACT)
    solver.solve(model)
    optimum = pyo.value(model.objective_function)
    model.face = pyo.Constraint(expr=model.objective_function.expr <= optimum + 1e-9 * abs(optimum))
    model.objective_function.deactivate()
    stf = next(iter(model.stf))
    model.lost = pyo.Objective(expr=sum(share * model.tau_pro[t, stf, site, pro] for (t, site, pro), share in away.items()),
                               sense=pyo.maximize)
    solver.solve(model)
    assert pyo.value(model.lost) == pytest.approx(0.0, abs=1e-6)

    # PyPSA
    network, parts, _ = solve_network(copy.deepcopy(data), mode, solver_name="appsi_highs", options=EXACT)
    m = network.model
    optimum = float(m.objective.value)
    m.add_constraints(m.objective.expression <= optimum + 1e-9 * abs(optimum), name="face")
    names = [f"{site}|{pro}" for site, pro in eff.columns if pro.startswith("charging_station")]
    share = xr.DataArray(1.0 - eff.loc[eff.index.get_level_values("t") > 0, [c for c in eff.columns if c[1].startswith("charging_station")]].to_numpy(),
                         coords={"snapshot": parts.snapshots, "name": names}, dims=("snapshot", "name"))
    m.objective = -1 * (m["Link-p"].sel(name=names) * share).sum()
    m.solve(solver_name="highs", io_api="direct", set_names=False, output_flag=False, **EXACT)
    assert -float(m.objective.value) == pytest.approx(0.0, abs=1e-6)
