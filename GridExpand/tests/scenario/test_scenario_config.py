"""Scenario YAMLs load; sizing methods follow the model-case asset plan; reference year is fixed."""

from __future__ import annotations

import copy

import pytest
import yaml

from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config
from gridexpand.scenario.scenario_config import ScenarioConfig

SCENARIOS = sorted(SCENARIO_CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.name)
def test_every_scenario_yaml_loads(path):
    scenario, scenario_hash = load_scenario_config(path)
    assert scenario.mobility.reference_year == 2009 and len(scenario_hash) == 64


def test_sizing_methods_by_asset_plan():
    scenario, _ = load_scenario_config(SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml")
    assert scenario.pv_sizing_method("pre") == "none"
    assert scenario.pv_sizing_method("post-hems-optimized") == scenario.pv.optimized_method
    assert scenario.pv_sizing_method("post-inflex-heuristic") == scenario.pv.heuristic_method
    assert scenario.heat_sizing_method("post-hems-heuristic") == "full_load_hours_rule"
    assert scenario.battery_capacity_coefficients("pre") == (0.0, 0.0)
    with pytest.raises(ValueError, match="Unknown model case"):
        scenario.battery_sizing_method("post")


def test_reference_year_must_be_2009():
    raw = yaml.safe_load((SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml").read_text(encoding="utf-8"))
    ScenarioConfig.from_dict(copy.deepcopy(raw))
    raw["mobility"]["reference_year"] = 2010
    with pytest.raises(ValueError, match="reference_year must be 2009"):
        ScenarioConfig.from_dict(raw)
