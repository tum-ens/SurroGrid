"""Step 3 input identity, input resolution, result merging and provenance."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from gridexpand.optimization.identity import resolve_input_file, validate_step2_input
from gridexpand.optimization.solver import (
    DEFAULT_SOLVER,
    SOLVER_ENV,
    relative_gap,
    resolve_solver_name,
    summarize_audit,
)
from gridexpand.optimization.urbs.runfunctions import final_result_path, temporal_audit
from gridexpand.optimization.urbs.saveload import merge_cluster_results
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key


@pytest.fixture(scope="module")
def scenario():
    return load_scenario_config(SCENARIO_CONFIG_DIR / "forchheim_2045_synthetic.yaml")


def _pre_metadata(scenario_config, config_hash, overrides=None):
    key = scenario_identity_key(scenario_config.scenario_id, config_hash)
    metadata = {
        "scenario_hash": config_hash,
        "model_case": "pre",
        "timeframe_mode": "max_base_electricity_demand_week",
        "scenario_key": f"{key}_max_base_electricity_demand_week",
    }
    metadata.update(overrides or {})
    return metadata


def test_validate_step2_input_accepts_canonical_pre_input(scenario):
    config, scenario_hash = scenario
    identity = validate_step2_input(_pre_metadata(config, scenario_hash), None, config, scenario_hash)
    assert identity.model_case == "pre"
    assert identity.assignment_hash is None
    assert identity.scenario_key.endswith("_max_base_electricity_demand_week")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"scenario_hash": "other"}, "scenario_hash"),
        ({"scenario_key": None}, "missing scenario_key"),
        ({"scenario_key": "scenario_x_123"}, "not canonical"),
        ({"model_case": "post-hems-heuristic"}, "electrification_assignment_hash"),
        ({"model_case": "post-hems-heuristic", "electrification_assignment_hash": "abc"},
         "raw_data/electrification_assignment"),
    ],
)
def test_validate_step2_input_rejects(scenario, overrides, message):
    config, scenario_hash = scenario
    with pytest.raises(ValueError, match=message):
        validate_step2_input(_pre_metadata(config, scenario_hash, overrides), None, config, scenario_hash)


def test_resolve_input_file(tmp_path):
    for name in ("9184137-00_a_pre.h5", "9184137-00_a_post.h5", "9184137-01_b_pre.h5", "notes.txt"):
        (tmp_path / name).write_text("x")
    assert resolve_input_file(tmp_path, "9184137-00_a_pre.h5").name == "9184137-00_a_pre.h5"
    assert resolve_input_file(tmp_path, "9184137-01").name == "9184137-01_b_pre.h5"
    assert resolve_input_file(tmp_path / "elsewhere", tmp_path / "9184137-01_b_pre.h5").name == "9184137-01_b_pre.h5"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_input_file(tmp_path, "9184137-00")
    with pytest.raises(FileNotFoundError):
        resolve_input_file(tmp_path, "9184137-02")
    with pytest.raises(FileNotFoundError):
        resolve_input_file(tmp_path, "missing.h5")


def _reference_merge(caches):
    """The former pairwise merge loop of saveload.save."""
    results_all = {}
    for model_res in caches:
        for name, result in model_res.items():
            if name not in results_all or results_all[name].empty:
                results_all[name] = result
            elif name == 'costs':
                results_all[name] += result
            else:
                results_all[name] = pd.concat([results_all[name], result])
    for name in results_all:
        results_all[name] = results_all[name][~results_all[name].index.duplicated(keep='first')]
    return results_all


def _cache(sites, offset):
    index = pd.MultiIndex.from_tuples(
        [(t, 2026, s, 'import') for t in (1, 2) for s in sites], names=['t', 'stf', 'sit', 'pro'])
    return {
        'tau_pro': pd.Series([offset + i * 0.5 for i in range(len(index))], index=index, name='tau_pro'),
        'costs': pd.Series([1.0 + offset, 2.0], index=pd.Index(['Invest', 'Fixed'], name='cost_type'), name='costs'),
        'weight': pd.Series([52.1], index=pd.Index([None], name='None'), name='weight'),
        'e_co_sell': pd.Series(name='e_co_sell', dtype=float) if offset == 0 else pd.Series(
            [0.1], index=pd.MultiIndex.from_tuples([(1, 2026, sites[0], 'feed', 'Sell')],
                                                   names=['t', 'stf', 'sit', 'com', 'com_type']), name='e_co_sell'),
    }


def test_merge_cluster_results_matches_former_loop():
    caches = [_cache([1, 2], 0), _cache([3], 10), _cache([4, 5], 20)]
    expected = _reference_merge([{k: v.copy() for k, v in c.items()} for c in caches])
    actual = merge_cluster_results([{k: v.copy() for k, v in c.items()} for c in caches])
    assert list(actual) == list(expected)
    for name in expected:
        pd.testing.assert_series_equal(actual[name], expected[name], check_exact=True)


def _audit_data(weights):
    index = pd.MultiIndex.from_product([[2026], range(3)], names=['support_timeframe', 't'])
    return {
        'storage': pd.DataFrame(), 'buy_sell_price': pd.DataFrame(), 'eff_factor': pd.DataFrame(),
        'type_period': pd.DataFrame({'weight_typeperiod': weights}, index=index),
        'global_prop': pd.DataFrame(
            {'value': [False]}, index=pd.MultiIndex.from_tuples([(2026, 'tsam')])),
    }


def test_temporal_audit_describes_the_solved_model():
    settings = {"timesteps": range(0, 169), "dt": 1, "scenario_key": "k", "scenario_hash": "h"}
    chronological = temporal_audit({"tsam": False, "tdy": False}, _audit_data([float('nan')] * 3), settings)
    assert chronological["annual_weight"] == pytest.approx(8760 / 168)
    assert chronological["storage_boundary_policy"] == "annual_equality"
    tsam = temporal_audit({"tsam": True, "tdy": False}, _audit_data([0.0, 8.0, 8.0]), settings)
    assert tsam["temporal_method"] == "shared_weather_tsam"
    assert tsam["annual_weight"] == 1.0
    assert tsam["storage_boundary_policy"] == "typeperiod_common_initial_state"


def test_final_result_path(tmp_path):
    assert final_result_path(tmp_path, "a_b.h5", "scn").name == "a_b_scn.h5"


def test_solver_resolution(monkeypatch):
    monkeypatch.delenv(SOLVER_ENV, raising=False)
    assert resolve_solver_name(None) == DEFAULT_SOLVER == "gurobi"
    monkeypatch.setenv(SOLVER_ENV, "appsi_highs")
    assert resolve_solver_name(None) == "appsi_highs"
    assert resolve_solver_name("gurobi") == "gurobi"
    with pytest.raises(ValueError):
        resolve_solver_name("cplex")


def test_gap_and_audit_summary():
    assert relative_gap(100.0, 95.0) == pytest.approx(0.05)
    assert relative_gap(None, 1.0) is None
    audit = pd.DataFrame({
        "solver": ["gurobi", "gurobi"], "solver_version": ["13.0.2", "13.0.2"],
        "solver_options": ['{"MIPGap": 0.05}'] * 2, "termination_condition": ["optimal"] * 2,
        "mip_gap": [0.01, 0.04],
    })
    summary = summarize_audit(audit)
    assert summary["optimization_partitions"] == 2
    assert math.isclose(summary["optimization_max_mip_gap"], 0.04)
    assert summary["optimization_solver_options"] == {"MIPGap": 0.05}
    assert summarize_audit(pd.DataFrame()) == {}
