"""Run YAML loading of the synthetic pipeline (no database); paired_aligned YAMLs point to the validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.fixture_paths import FIXTURE_SCENARIOS
from gridexpand.paths import RUN_CONFIG_DIR
from gridexpand.scenario.run_config import (
    SyntheticRun,
    load_run_config,
    run_config_from_dict,
    with_cases,
)

SYNTHETIC = {
    "run": "demo",
    "scenario": str(FIXTURE_SCENARIOS / "teaser_2045.yaml"),
    "grids": {"pylovo": {"version": 1, "ags": 9184137, "plz": None, "kcid": None, "bcid": None}},
    "model_cases": ["pre", "post-hems-heuristic"],
    "execution": {"step2_cpus": 2, "demand_scope": "all", "timeframe_mode": "full_year",
                  "powerflow_grid_scope": "full", "profile_seed": 481527},
}


def synthetic(**changes):
    """The demo run with changes: ``pylovo__ags`` (grids.pylovo), ``execution__x`` or a top-level key."""
    raw = copy.deepcopy(SYNTHETIC)
    for dotted, value in changes.items():
        if "__" not in dotted:
            raw[dotted] = value
            continue
        block, key = dotted.split("__")
        target = raw["grids"]["pylovo"] if block == "pylovo" else raw.setdefault(block, {})
        target[key] = value
    return run_config_from_dict(raw, base_dir=Path("/"))


@pytest.mark.parametrize("path", sorted(RUN_CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_repository_run_yamls(path):
    if "CHANGE_ME" in path.read_text(encoding="utf-8"):
        pytest.skip("template")
    run, run_hash = load_run_config(path)
    assert len(run_hash) == 64 and run.scenario_path.is_file()


def test_the_template_only_needs_its_change_me_values(tmp_path):
    text = (RUN_CONFIG_DIR / "00_run_template.yaml").read_text(encoding="utf-8")
    filled = text.replace("CHANGE_ME_run_id", "town").replace('"CHANGE_ME"', '"3"')
    assert "CHANGE_ME" not in "\n".join(line.split("#")[0] for line in filled.splitlines())
    (tmp_path / "run.yaml").write_text(filled, encoding="utf-8")
    run, _ = load_run_config(tmp_path / "run.yaml")
    assert (run.run_id, run.pylovo_version_id, run.scenario_path.name) == ("town", "3", "default_2045.yaml")


def test_synthetic_defaults():
    run = synthetic()
    assert isinstance(run, SyntheticRun)
    assert (run.ags, run.pylovo_version_id, run.step2_cpus, run.powerflow_output) == (9184137, "1", 2, "summary")
    assert (run.min_buildings, run.workers, run.step3_cpus, run.pilot_gate) == (5, 1, 16, True)
    one_grid = synthetic(pylovo__plz=85653, pylovo__kcid=1, pylovo__bcid=-1)
    assert (one_grid.plz, one_grid.kcid, one_grid.bcid) == (85653, 1, -1)
    assert synthetic(pylovo__plz="-").plz is None
    assert "region" in run.identity() and "model_cases" not in run.identity()
    raw = copy.deepcopy(SYNTHETIC)
    del raw["model_cases"], raw["execution"]
    assert run_config_from_dict(raw, base_dir=Path("/")).model_cases == ("pre", "post-hems-heuristic",
                                                                           "post-hems-optimized")


@pytest.mark.parametrize("changes, message", [
    ({"run": "../x"}, "directory-safe"),
    ({"pylovo__ags": None}, "grids.pylovo.ags"),
    ({"pylovo__kcid": 1}, "kcid and grids.pylovo.bcid"),
    ({"pylovo__version": ""}, "grids.pylovo.version"),
    ({"pylovo__storage": "db"}, "Unknown"),
    ({"model_cases": ["pre", "post-inflex-heuristic"]}, "not available for the synthetic"),
    ({"execution__n_cpu": 3}, "Unknown"),
    ({"execution__resume": "false"}, "true or false"),
    ({"execution__workers": 0}, "positive"),
    ({"execution__timeframe_mode": "winter"}, "timeframe_mode"),
    ({"execution__unknown": 1}, "Unknown"),
    ({"scenario_overrides": [1]}, "scenario_overrides"),
])
def test_synthetic_rejects(changes, message):
    with pytest.raises(ValueError, match=message):
        synthetic(**changes)


def test_with_cases():
    run = synthetic()
    assert with_cases(run, ("pre",)).model_cases == ("pre",)
    with pytest.raises(ValueError):
        with_cases(run, ("post-inflex-heuristic",))


def test_former_and_validation_formats_point_to_the_right_tool(tmp_path):
    former = {"run": {"id": "x", "scenario": "s.yaml", "pipeline": "paired_aligned"}, "resources": {},
              "execution": {}}
    with pytest.raises(ValueError, match="python -m validation run"):
        run_config_from_dict(former, base_dir=tmp_path)


def test_optimizer():
    assert synthetic().optimizer is None  # $GRIDEXPAND_OPTIMIZER, else urbs
    assert synthetic().step3_cluster_concurrency is None  # the optimizer's default
    assert synthetic(execution__step3_cluster_concurrency=3).step3_cluster_concurrency == 3
    assert synthetic(execution__optimizer="pypsa").optimizer == "pypsa"
    with pytest.raises(ValueError, match="execution.optimizer"):
        synthetic(execution__optimizer="oemof")
