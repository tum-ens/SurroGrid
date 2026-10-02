"""HEMS smart charging: each session spread over its stay, except in hours of local PV surplus."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from deterministic_fixture import read_prepared, write_input

from gridexpand.common.ev_sessions import HEMS_FRACTION_COLUMN, hems_session_fractions
from gridexpand.optimization.urbs.runfunctions import apply_hems_session_cap, pv_surplus_hours


def _session(energy=11.0, hours=10, charger=11.0):
    sessions = pd.DataFrame([{"session_id": "s", "site": 1, "charger_kw": charger, "energy_kwh": energy}])
    rows = pd.DataFrame({"session_id": "s", "t": np.arange(1, hours + 1), "order": np.arange(hours),
                         "available_fraction": 1.0})
    return sessions, rows


def _uncapped(*hours):
    index = pd.MultiIndex.from_product([[1], range(1, 11)], names=["site", "t"])
    return pd.Series([t in hours for t in range(1, 11)], index=index)


def test_cap_is_factor_times_average_power_and_keeps_pv_surplus_hours():
    sessions, hours = _session()
    capped = hems_session_fractions(sessions, hours, factor=2.0, uncapped=_uncapped(5))
    # average 1.1 kW over 10 connected hours; 2 x 1.1 kW of an 11 kW charger = 0.2
    expected = np.full(10, 0.2)
    expected[4] = 1.0
    assert np.allclose(capped[HEMS_FRACTION_COLUMN], expected)
    assert capped["available_fraction"].tolist() == hours["available_fraction"].tolist()


def test_cap_never_exceeds_the_connected_fraction_and_stays_feasible():
    sessions, hours = _session(energy=90.0)
    hours.loc[[0, 9], "available_fraction"] = 0.5
    capped = hems_session_fractions(sessions, hours, factor=1.0, uncapped=_uncapped())
    assert (capped[HEMS_FRACTION_COLUMN] <= capped["available_fraction"] + 1e-12).all()
    assert (capped[HEMS_FRACTION_COLUMN] * 11.0).sum() >= 90.0 - 1e-9


def test_factor_below_one_is_rejected():
    sessions, hours = _session()
    with pytest.raises(ValueError, match="at least 1"):
        hems_session_fractions(sessions, hours, factor=0.9, uncapped=_uncapped())


def test_pv_surplus_hours_compare_pv_potential_with_fixed_demand(tmp_path):
    data, _, _ = read_prepared(write_input(tmp_path / "input.h5", "ev_sessions"))
    surplus = pv_surplus_hours(data)
    assert surplus.dtype == bool and surplus.index.names == ["site", "t"]
    supim = data["supim"].droplevel("support_timeframe")
    demand = data["demand"].droplevel("support_timeframe")
    process = data["process"].reset_index()
    for site in surplus.index.get_level_values("site").unique():
        pv = process[process["Site"].eq(site) & process["Process"].str.startswith("Rooftop PV")]
        columns = [c for c in supim.columns if c[0] == site]
        potential = sum(supim[c] * float(pv["cap-up"].iloc[i]) for i, c in enumerate(columns))
        expected = (potential > demand[(site, "electricity")]).to_numpy()
        assert np.array_equal(surplus.xs(site, level="site").to_numpy(), expected)
    assert surplus.any() and not surplus.all()


def test_no_factor_leaves_the_sessions_unchanged(tmp_path):
    data, _, _ = read_prepared(write_input(tmp_path / "input.h5", "ev_sessions"))
    before = data["ev_session_hours"].copy()
    assert apply_hems_session_cap(data, None)["ev_session_hours"].equals(before)


def test_backends_agree_with_the_cap_and_respect_it(tmp_path):
    from pyomo.environ import SolverFactory
    from pyomo.opt import check_optimal_termination

    from gridexpand.optimization.pypsa_model.solve import solve_group
    from gridexpand.optimization.urbs.model import create_model

    data, mode, settings = read_prepared(write_input(tmp_path / "input.h5", "ev_sessions"))
    data = apply_hems_session_cap(data, 2.0)
    hours = data["ev_session_hours"]
    assert (hours[HEMS_FRACTION_COLUMN] < hours["available_fraction"] - 1e-9).any()

    model = create_model(copy.deepcopy(data), settings)
    solver = SolverFactory("appsi_highs")
    solver.options = {"mip_rel_gap": 0.0}
    assert check_optimal_termination(solver.solve(model))
    result = solve_group(copy.deepcopy(data), mode, solver_name="appsi_highs", options={"mip_rel_gap": 0.0})
    assert result["audit"]["objective"] == pytest.approx(float(model.objective_function()), rel=1e-8)

    tau = result["results"]["tau_pro"].reset_index()
    sessions = data["ev_sessions"].set_index("session_id")
    merged = hours.merge(sessions[["site", "process", "charger_kw"]], left_on="session_id", right_index=True)
    flows = tau.set_index(["t", "sit", "pro"])["tau_pro"]
    charged = flows.reindex(pd.MultiIndex.from_arrays([merged["t"], merged["site"], merged["process"]])).to_numpy()
    assert (charged <= merged["charger_kw"].to_numpy() * merged[HEMS_FRACTION_COLUMN].to_numpy() + 1e-6).all()
