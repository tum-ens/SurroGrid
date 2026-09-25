"""Step 4 CLI options, run assumptions and the scenario result reader (no database)."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.powerflow import io
from gridexpand.powerflow.run_pwrflw import build_assumptions, demand_buses, parse_args


def test_outputs_default_and_summary_only_alias():
    assert parse_args(["x.h5"]).outputs == ("raw",)
    assert parse_args(["x.h5", "--storage", "db", "--summary-only"]).outputs == ("summary",)
    args = parse_args(["x.h5", "--storage", "db", "--outputs", "raw,summary", "--summary-run-name", "s"])
    assert args.outputs == ("raw", "summary") and args.summary_run_name == "s"
    assert args.summary_nonconvergence == "nan"


@pytest.mark.parametrize(
    "argv",
    [
        ["x.h5", "--summary-only"],  # summary needs the database
        ["x.h5", "--storage", "db", "--summary-only", "--outputs", "raw"],
        ["x.h5", "--storage", "db", "--summary-run-name", "s"],
        ["x.h5", "--summary-nonconvergence", "raise"],
        ["x.h5", "--outputs", "raw,other"],
        ["x.h5", "--pre-only", "--post-demand-mode", "inflex"],
    ],
)
def test_invalid_option_combinations(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


class _Reader:
    path = "unused.h5"

    def __init__(self, temporal=None, audit=None):
        self._temporal, self._audit = temporal, audit

    def temporal_method(self):
        return self._temporal

    def solver_audit(self):
        return self._audit


def test_assumptions_carry_temporal_and_solver_provenance():
    audit = pd.DataFrame({
        "solver": ["gurobi"], "solver_version": ["13.0.2"], "solver_options": ['{"MIPGap": 0.05}'],
        "termination_condition": ["optimal"], "mip_gap": [0.03],
    })
    temporal = {"temporal_method": "full_year_no_tsam", "operating_hours": 168,
                "storage_boundary_policy": "annual_equality", "ev_boundary_policy": "legacy_mobility_buffer"}
    assumptions = build_assumptions(parse_args(["x.h5"]), _Reader(temporal, audit))
    assert assumptions["post_demand_mode"] == "flexible"
    assert assumptions["temporal_method"] == "full_year_no_tsam"
    assert assumptions["optimization_solver"] == "gurobi"
    assert assumptions["optimization_max_mip_gap"] == pytest.approx(0.03)
    pre_only = build_assumptions(parse_args(["x.h5", "--pre-only"]), _Reader(None, audit))
    assert "optimization_solver" not in pre_only and "temporal_method" not in pre_only
    inflex = build_assumptions(parse_args(["x.h5", "--post-demand-mode", "inflex"]), _Reader())
    assert "inst-cap" in inflex["inflex_capacity_source"]


def test_demand_buses():
    frame = pd.DataFrame([[1.0, 2.0]], columns=pd.MultiIndex.from_tuples([(3, "electricity"), (1, "electricity-reactive")]))
    assert demand_buses(frame, None) == {1, 3}
    with pytest.raises(ValueError):
        demand_buses(pd.DataFrame())


def test_reader_prefers_reduced_tables_and_reads_provenance(tmp_path):
    path = tmp_path / "result.h5"
    raw = pd.DataFrame({"a": [1.0, 2.0]})
    reduced = pd.DataFrame({"a": [3.0]})
    with pd.HDFStore(path, "w") as store:
        store["urbs_in/demand"] = raw
        store["urbs_out/temporal_method"] = pd.Series({"temporal_method": "full_year_no_tsam"}, dtype=object)
    reader = io.ScenarioResultReader(path)
    assert not reader.uses_reduced_demand()
    pd.testing.assert_frame_equal(reader.get_pre_demand(), raw)
    assert reader.temporal_method() == {"temporal_method": "full_year_no_tsam"}
    assert reader.solver_audit() is None
    with pytest.raises(ValueError, match="shared_weather_tsam"):
        io.require_temporal_method(path, "shared_weather_tsam")
    with pd.HDFStore(path, "a") as store:
        store["urbs_out/reduced_data/demand"] = reduced
    reader = io.ScenarioResultReader(path)
    assert reader.uses_reduced_demand()
    pd.testing.assert_frame_equal(reader.get_pre_demand(), reduced)
    assert io.key_exists(path, "/urbs_in/demand") and not io.key_exists(path, "raw_data/net")


def test_component_audit_sidecar(tmp_path):
    path = io.component_audit_path(tmp_path / "grid.h5", "run_a")
    assert path.name == "grid.run_a.component_audit.h5"
    assert io.write_component_audit(path, pd.DataFrame(), "x") is None
    assert io.write_component_audit(path, pd.DataFrame({"v": [1]}), "x") == str(path)
    assert pd.read_hdf(path, "component_audit/x")["v"].tolist() == [1]


def test_run_plan():
    from gridexpand.powerflow.run_pwrflw import run_plan

    assert run_plan(("raw",), "r") == {"raw": "r"}
    assert run_plan(("summary",), "s") == {"summary": "s"}
    assert run_plan(("raw", "summary"), "r") == {"raw": "r", "summary": "r", "shared": True}
    assert run_plan(("raw", "summary"), "r", "s") == {"raw": "r", "summary": "s"}
    assert run_plan(("raw", "summary"), None) == {"raw": None, "summary": None, "shared": True}
