"""A Gurobi model that presolve declares infeasible or unbounded is solved once more without presolve."""

from __future__ import annotations

import linopy
import pytest
from deterministic_fixture import read_prepared, write_input

from gridexpand.optimization.pypsa_model.runfunctions import building_groups
from gridexpand.optimization.pypsa_model.solve import solve_network
from gridexpand.optimization.urbs.input import get_cluster_data


def _group(tmp_path):
    data, mode, _ = read_prepared(write_input(tmp_path / "input.h5"))
    return get_cluster_data(data, building_groups(data, 1)[0]), mode


def test_gurobi_solves_again_without_presolve(tmp_path, monkeypatch):
    group, mode = _group(tmp_path)
    calls, solve = [], linopy.Model.solve

    def presolve_fails_once(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return "warning", "infeasible_or_unbounded"
        return solve(self, solver_name="highs", io_api="direct")  # stands in for Gurobi without presolve

    monkeypatch.setattr(linopy.Model, "solve", presolve_fails_once)
    _, _, info = solve_network(group, mode, solver_name="gurobi")
    assert [call["Presolve"] for call in calls] == [2, 0]
    assert info["condition"] == "optimal" and info["options"]["Presolve"] == 0


def test_no_second_solve_for_highs_and_a_second_failure_raises(tmp_path, monkeypatch):
    group, mode = _group(tmp_path)
    calls = []

    def infeasible(self, **kwargs):
        calls.append(kwargs)
        return "warning", "infeasible"

    monkeypatch.setattr(linopy.Model, "solve", infeasible)
    with pytest.raises(RuntimeError, match="highs ended"):
        solve_network(group, mode, solver_name="appsi_highs")
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(RuntimeError, match="gurobi ended"):
        solve_network(group, mode, solver_name="gurobi")
    assert [call["Presolve"] for call in calls] == [2, 0]
