"""Run YAML loading of the synthetic pipeline (no database); paired_aligned YAMLs point to the validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from gridexpand.paths import RUN_CONFIG_DIR, SCENARIO_CONFIG_DIR
from gridexpand.scenario.run_config import (
    SyntheticRun,
    load_run_config,
    run_config_from_dict,
    with_cases,
)

SYNTHETIC = {
    "run": {"id": "demo", "scenario": str(SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml"), "pipeline": "synthetic"},
    "resources": {"pylovo_version_id": 1, "ags": 9184137, "plz": None, "kcid": None, "bcid": None,
                  "storage": "db", "output_directory": None},
    "execution": {"model_cases": ["pre", "post-hems-heuristic"], "n_cpu": 2, "mobility_source": "pool",
                  "demand_scope": "all", "timeframe_mode": "full_year", "powerflow_grid_scope": "full",
                  "profile_seed": 481527},
}


def synthetic(**changes):
    raw = copy.deepcopy(SYNTHETIC)
    for dotted, value in changes.items():
        block, key = dotted.split("__")
        raw[block][key] = value
    return run_config_from_dict(raw, base_dir=Path("/"))


@pytest.mark.parametrize("path", sorted(RUN_CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_repository_run_yamls(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "CHANGE_ME" in path.read_text(encoding="utf-8"):
        pytest.skip("template")
    cases = raw["execution"]["model_cases"]
    if raw["run"]["pipeline"] == "synthetic" and "post-inflex-heuristic" in cases:
        with pytest.raises(ValueError, match="post-inflex-heuristic is not available"):
            load_run_config(path)
        return
    run, run_hash = load_run_config(path)
    assert run.pipeline == raw["run"]["pipeline"] and len(run_hash) == 64
    assert run.scenario_path.is_file()


def test_synthetic_defaults_and_aliases():
    run = synthetic()
    assert isinstance(run, SyntheticRun)
    assert (run.ags, run.pylovo_version_id, run.step2_cpus, run.powerflow_output) == (9184137, "1", 2, "summary")
    assert (run.min_buildings, run.workers, run.step3_cpus, run.pilot_gate) == (5, 1, 16, True)
    one_grid = synthetic(resources__plz=85653, resources__kcid=1, resources__bcid=-1)
    assert (one_grid.plz, one_grid.kcid, one_grid.bcid) == (85653, 1, -1)
    assert synthetic(resources__plz="-").plz is None
    assert "region" in run.identity() and "model_cases" not in run.identity()


@pytest.mark.parametrize("changes, message", [
    ({"run__pipeline": "scenario"}, "renamed to 'synthetic'"),
    ({"run__pipeline": "nope"}, "run.pipeline"),
    ({"run__id": "../x"}, "directory-safe"),
    ({"resources__storage": "h5"}, "storage must be db"),
    ({"resources__output_directory": "/tmp/x"}, "output_directory"),
    ({"resources__ags": None}, "resources.ags is required"),
    ({"resources__kcid": 1}, "kcid and resources.bcid"),
    ({"resources__pylovo_version_id": ""}, "pylovo_version_id"),
    ({"execution__model_cases": ["pre", "post-inflex-heuristic"]}, "not available for the synthetic"),
    ({"execution__step2_cpus": 3}, "not both"),
    ({"execution__mobility_source": "emobpy"}, "mobility_source must be pool"),
    ({"execution__resume": "false"}, "true or false"),
    ({"execution__workers": 0}, "positive"),
    ({"execution__timeframe_mode": "winter"}, "timeframe_mode"),
    ({"execution__unknown": 1}, "Unknown synthetic execution option"),
])
def test_synthetic_rejects(changes, message):
    with pytest.raises(ValueError, match=message):
        synthetic(**changes)


def test_with_cases():
    run = synthetic()
    assert with_cases(run, ("pre",)).model_cases == ("pre",)
    with pytest.raises(ValueError):
        with_cases(run, ("post-inflex-heuristic",))


def test_paired_aligned_points_to_the_validation(tmp_path):
    raw = {**SYNTHETIC, "run": {**SYNTHETIC["run"], "pipeline": "paired_aligned"}}
    with pytest.raises(ValueError, match="python -m validation run"):
        run_config_from_dict(raw, base_dir=tmp_path)


def test_optimizer():
    assert synthetic().optimizer is None  # $GRIDEXPAND_OPTIMIZER, else urbs
    assert synthetic().step3_cluster_concurrency is None  # the optimizer's default
    assert synthetic(execution__step3_cluster_concurrency=3).step3_cluster_concurrency == 3
    assert synthetic(execution__optimizer="pypsa").optimizer == "pypsa"
    with pytest.raises(ValueError, match="execution.optimizer"):
        synthetic(execution__optimizer="oemof")
