"""Step 3 with the PyPSA optimizer: result file, audit, partition and selection."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from deterministic_fixture import read_prepared, write_input

from gridexpand.optimization.pypsa_model import RESULT_KEYS, run_pypsa_opt
from gridexpand.optimization.pypsa_model.building_model import build_building_model
from gridexpand.optimization.pypsa_model.runfunctions import building_groups, default_concurrency
from gridexpand.optimization.solver import (
    DEFAULT_OPTIMIZER,
    OPTIMIZER_ENV,
    resolve_optimizer_name,
    step3_concurrency,
    summarize_audit,
)
from gridexpand.optimization.urbs import run_lvds_opt

URBS_AUDIT_COLUMNS = {
    "cluster", "n_sites", "first_site", "solver", "solver_interface", "solver_version", "solver_options",
    "solver_status", "termination_condition", "objective", "best_bound", "mip_gap", "build_seconds",
    "solve_seconds",
}


def _settings(input_file: str) -> dict:
    return {
        "input_file": input_file, "tsam": False, "noTypicalPeriods": 12, "hoursPerPeriod": 168,
        "tsamExtremePeriodMethod": "replace_cluster_center", "tsamMethodSettings": {},
        "scenario_id": "fixture", "scenario_hash": "0" * 64, "scenario_key": "fixture_key",
        "electrification_assignment_hash": None, "reduce_only": False, "n_cpu": 1,
    }


@pytest.fixture(scope="module")
def result_files(tmp_path_factory):
    """Step 3 of the deterministic heuristic input with both optimizers (HiGHS)."""
    root = tmp_path_factory.mktemp("step3")
    path = write_input(root / "fixture_input.h5", "heuristic")
    files = {}
    for optimizer, run in (("pypsa", run_pypsa_opt), ("urbs", run_lvds_opt)):
        result_dir = root / optimizer
        result_dir.mkdir()
        kwargs = {"concurrency": 2} if optimizer == "pypsa" else {}
        files[optimizer] = run(os.fspath(path), result_dir, _settings(path.name), log_dir=root / "logs",
                               solver="appsi_highs", **kwargs)
    return files


def test_result_file_has_the_urbs_layout(result_files):
    path = result_files["pypsa"]
    assert path.name == "fixture_input_fixture_key.h5" and path.exists()
    with pd.HDFStore(path, "r") as store:
        keys = set(store.keys())
    assert {f"/urbs_out/MILP/{key}" for key in RESULT_KEYS} <= keys
    assert {"/urbs_out/temporal_method", "/urbs_out/solver_audit", "/urbs_out/tsam/kept_timesteps",
            "/urbs_out/reduced_data/demand", "/urbs_out/reduced_data/process", "/urbs_in/demand"} <= keys
    temporal = pd.read_hdf(path, "urbs_out/temporal_method")
    assert temporal["temporal_method"] == "full_year_no_tsam"
    assert temporal["storage_boundary_policy"] == "annual_equality"


def test_solver_audit_has_one_row_per_building(result_files):
    audit = pd.read_hdf(result_files["pypsa"], "urbs_out/solver_audit")
    assert URBS_AUDIT_COLUMNS <= set(audit.columns)
    assert list(audit["cluster"]) == [0, 1, 2] and (audit["n_sites"] == 1).all()
    assert set(audit["optimizer"]) == {"pypsa"} and set(audit["termination_condition"]) == {"optimal"}
    summary = summarize_audit(audit)
    assert summary["optimization_optimizer"] == "pypsa" and summary["optimization_partitions"] == 3
    urbs_audit = pd.read_hdf(result_files["urbs"], "urbs_out/solver_audit")
    assert summarize_audit(urbs_audit)["optimization_optimizer"] == "urbs"
    assert audit["objective"].sum() == pytest.approx(urbs_audit["objective"].sum(), rel=1e-9)


@pytest.mark.parametrize("key", RESULT_KEYS)
def test_results_equal_the_urbs_result_file(result_files, key):
    """Per-building models give the urbs result of the one-cluster model (unique optimum)."""
    actual = pd.read_hdf(result_files["pypsa"], f"urbs_out/MILP/{key}").sort_index()
    expected = pd.read_hdf(result_files["urbs"], f"urbs_out/MILP/{key}").sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-5, rtol=1e-7)


def test_step4_reconstructs_the_same_demand(result_files):
    from gridexpand.powerflow.demands import _extract_relevant_demands

    frames = {name: _extract_relevant_demands(pd.read_hdf(path, "urbs_out/MILP/tau_pro"))
              for name, path in result_files.items()}
    for actual, expected in zip(frames["pypsa"], frames["urbs"]):
        pd.testing.assert_frame_equal(actual.sort_index(axis=1), expected.sort_index(axis=1), atol=1e-5, rtol=1e-7)


def test_step4_installed_assets_are_the_same(result_files):
    from gridexpand.powerflow.assets import installed_assets

    assets = {name: installed_assets(*(pd.read_hdf(path, f"urbs_out/MILP/{key}")
                                        for key in ("cap_pro", "cap_sto_c", "cap_sto_p")))
              for name, path in result_files.items()}
    pd.testing.assert_frame_equal(assets["pypsa"].reset_index(drop=True), assets["urbs"].reset_index(drop=True))


def test_one_model_per_building(tmp_path):
    data, _, _ = read_prepared(write_input(tmp_path / "input.h5", "heuristic"))
    assert building_groups(data) == [[1], [2], [3]]
    assert building_groups(data, 2) == [[1, 2], [3]]
    with pytest.raises(ValueError):
        building_groups(data, 0)
    assert 1 <= default_concurrency("gurobi", 3) <= 3
    assert default_concurrency("appsi_highs", 1) == 1


def test_time_series_aggregation_is_rejected(tmp_path):
    data, mode, _ = read_prepared(write_input(tmp_path / "input.h5", "heuristic"))
    with pytest.raises(NotImplementedError, match="full-year"):
        build_building_model(data, {**mode, "tdy": True})


def test_optimizer_resolution(monkeypatch):
    monkeypatch.delenv(OPTIMIZER_ENV, raising=False)
    assert resolve_optimizer_name(None) == DEFAULT_OPTIMIZER == "urbs"
    monkeypatch.setenv(OPTIMIZER_ENV, "pypsa")
    assert resolve_optimizer_name(None) == "pypsa"
    assert resolve_optimizer_name("urbs") == "urbs"
    with pytest.raises(ValueError):
        resolve_optimizer_name("oemof")


def test_runner_concurrency_default(monkeypatch):
    monkeypatch.delenv(OPTIMIZER_ENV, raising=False)
    assert step3_concurrency(None) == 1  # urbs: one cluster model (several GB) at a time
    assert step3_concurrency(None, "pypsa") is None  # gridexpand optimize chooses the workers
    assert step3_concurrency(3, "pypsa") == 3 and step3_concurrency(2, "urbs") == 2
    monkeypatch.setenv(OPTIMIZER_ENV, "pypsa")
    assert step3_concurrency(None) is None


def test_cli_rejects_tsam_with_pypsa(monkeypatch):
    from gridexpand.optimization.run_urbs_cluster import parse_args
    from gridexpand.paths import SCENARIO_CONFIG_DIR

    monkeypatch.delenv(OPTIMIZER_ENV, raising=False)
    full_year = str(SCENARIO_CONFIG_DIR / "forchheim_2045_full_year.yaml")
    with_tsam = str(SCENARIO_CONFIG_DIR / "forchheim_2045_synthetic.yaml")  # time_aggregation.enabled
    assert parse_args(["x.h5", "--scenario-config", full_year]).optimizer == "urbs"
    assert parse_args(["x.h5", "--optimizer", "pypsa", "--scenario-config", full_year]).optimizer == "pypsa"
    assert parse_args(["x.h5", "--scenario-config", with_tsam]).optimizer == "urbs"
    for argv in (["--scenario-config", with_tsam], ["--tsam", "--scenario-config", full_year]):
        with pytest.raises(SystemExit):
            parse_args(["x.h5", "--optimizer", "pypsa", *argv])
