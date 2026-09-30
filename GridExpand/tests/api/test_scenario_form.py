"""Scenario editor without HTTP: field catalogue, comment-preserving edits, validation issues."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import configuration_hash, load_scenario_config
from gridexpand.scenario.scenario_config import ADOPTION_MODES
from gridexpand.api import scenario_form as sf

SHIPPED = sorted(SCENARIO_CONFIG_DIR.glob("*.yaml"))
SCHWEINFURT = SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml"
INVENTORY = SCENARIO_CONFIG_DIR / "forchheim_2045_full_year.yaml"  # source_inventory, TSAM off


def text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def comments(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("#") or " # " in line]


def test_shipped_scenarios_exist():
    assert {p.name for p in SHIPPED} >= {"schweinfurt_2045.yaml", "forchheim_2045_synthetic.yaml",
                                         "forchheim_2045_full_year.yaml", "joint_2045_full_year.yaml"}


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_catalogue_paths_exist_in_every_shipped_file(path):
    data = yaml.safe_load(text_of(path))
    for key, field in sf.FIELDS.items():
        value = sf.get_value(data, key, None)
        if field.get("optional"):
            assert sf.get_value(data, key.rsplit(".", 1)[0], None) is not None, key
        elif field.get("requires"):
            active = sf.get_value(data, field["requires"]["key"]) == field["requires"]["value"]
            assert (value is not None) == active, key
        else:
            assert value is not None, f"{key} missing in {path.name}"
        if value is not None and field["type"] == "enum":
            assert value in [o["value"] for o in field["options"]], key


def test_catalogue_is_well_formed():
    assert len(sf.FIELDS) == sum(len(s["fields"]) for s in sf.SECTIONS)  # unique keys
    for key, field in sf.FIELDS.items():
        assert field["type"] in sf.FIELD_TYPES and field["label"] and field["hint"], key
        if field["type"] == "percent":
            assert field.get("min", 0) >= 0 and field.get("max", 1) <= 1, key
        for rule in (field.get("requires"),):
            if rule:
                assert rule["key"] in sf.FIELDS and rule["value"] in [o["value"] for o in sf.FIELDS[rule["key"]]["options"]]
    adoption = sf.FIELDS["electrification.heat.adoption_mode"]
    assert tuple(o["value"] for o in adoption["options"]) == ADOPTION_MODES


@pytest.mark.parametrize("key", [k for k, f in sf.FIELDS.items() if f["type"] == "enum"])
def test_every_enum_option_is_accepted_by_the_loader(key):
    base = text_of(SCHWEINFURT)
    for option in sf.FIELDS[key]["options"]:
        changes = {key: option["value"]}
        if key.endswith("adoption_mode") and option["value"] == "deterministic_share":
            changes[key.replace("adoption_mode", "building_share")] = 0.5
        result = sf.preview(base, base_name="schweinfurt_2045.yaml", changes=changes)
        assert result["ok"], (key, option, result["issues"])


def _other_value(key: str, field: dict, current):
    if field["type"] == "bool":
        return not current
    if field["type"] == "int":
        return current + 1
    if key.endswith("indoor_design_temperature_c"):
        return current + 1.0
    if key.endswith("degree_day_base_temperature_c"):
        return 15.0  # must lie between the heating limit and the indoor temperature
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        current = field.get("default")  # absent optional key: start from the loader default
    return round(current * 0.9, 6) if current else 0.1


def test_every_non_enum_field_can_be_changed_within_the_loader_rules():
    base = text_of(SCHWEINFURT)
    data = yaml.safe_load(base)
    for key, field in sf.FIELDS.items():
        if field["type"] == "enum" or field.get("requires"):
            continue
        new = _other_value(key, field, sf.get_value(data, key))
        result = sf.preview(base, base_name="s", changes={key: new})
        assert result["ok"], (key, new, result["issues"])
        assert sf.get_value(yaml.safe_load(result["text"]), key) == new
        assert [c["key"] for c in result["changes"]] == [key, *field.get("mirror", [])]


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_no_op_changes_keep_the_text_and_the_hash(path):
    text = text_of(path)
    data = yaml.safe_load(text)
    current = {k: v for k, v in sf.field_values(data).items() if v is not None}
    assert sf.apply_changes(text, current) == text
    result = sf.preview(text, base_name=path.name, changes=current)
    assert result["text"] == text and result["changes"] == []
    if result["ok"]:
        assert result["configuration_hash"] == load_scenario_config(path)[1]


def test_an_equal_number_is_no_change():
    text = text_of(SCHWEINFURT)  # demand_multiplier: 2.0, hours_per_period: 168
    assert sf.apply_changes(text, {"asset_sizing.pv.demand_multiplier": 2, "time_aggregation.hours_per_period": 168.0}) == text


def test_a_change_changes_hash_and_key_and_keeps_comments_and_layout():
    text = text_of(SCHWEINFURT)
    before = sf.preview(text, base_name="s")
    result = sf.preview(text, base_name="s", changes={"asset_sizing.pv.demand_multiplier": 2.5})
    assert result["ok"] and result["configuration_hash"] != before["configuration_hash"]
    assert result["scenario_key"] != before["scenario_key"] and result["id"] == "schweinfurt_2045"
    old_lines, new_lines = text.splitlines(), result["text"].splitlines()
    assert len(old_lines) == len(new_lines)
    assert [(a, b) for a, b in zip(old_lines, new_lines) if a != b] == [
        ("    demand_multiplier: 2.0", "    demand_multiplier: 2.5")]
    assert result["changes"] == [{"key": "asset_sizing.pv.demand_multiplier", "label": "Demand multiplier",
                                  "kind": "changed", "old": 2.0, "new": 2.5}]
    assert result["diff"].startswith("--- s (base)\n+++ new scenario\n")
    renamed = sf.preview(text, base_name="s", scenario_id="schweinfurt_2045_custom")
    assert renamed["id"] == "schweinfurt_2045_custom"
    assert renamed["scenario_key"].startswith("scenario_schweinfurt_2045_custom_")
    assert "  id: schweinfurt_2045_custom\n" in renamed["text"]


def test_dependent_and_optional_keys_keep_the_comments_around_them():
    text = text_of(INVENTORY)
    changes = {f"electrification.{tech}.adoption_mode": "deterministic_share" for tech in ("heat", "mobility", "pv_battery")}
    changes |= {f"electrification.{tech}.building_share": 0.6 for tech in ("heat", "mobility", "pv_battery")}
    changes["asset_sizing.heat.teaser_retrofit_level"] = 1
    result = sf.preview(text, base_name="f", changes=changes)
    assert result["ok"], result["issues"]
    new = result["text"]
    assert comments(new) == comments(text)
    assert "    adoption_mode: deterministic_share\n    building_share: 0.6\n\n# Mobility behavior" in new
    assert "    space_heat_source: infdb_ro_heat\n    teaser_retrofit_level: 1\n    # Degree-day" in new
    # and back: the share keys are removed, the comment below them stays
    back = {f"electrification.{tech}.adoption_mode": "source_inventory" for tech in ("heat", "mobility", "pv_battery")}
    back["asset_sizing.heat.teaser_retrofit_level"] = 0
    reverted = sf.apply_changes(new, back)
    assert reverted == text.replace("    space_heat_source: infdb_ro_heat\n",
                                    "    space_heat_source: infdb_ro_heat\n    teaser_retrofit_level: 0\n")
    # the optional key at its default is not inserted
    assert sf.apply_changes(text, {"asset_sizing.heat.teaser_retrofit_level": 0}) == text


def test_small_and_large_floats_stay_floats_for_the_loader():
    text = text_of(SCHWEINFURT)
    new = sf.apply_changes(text, {"economics.electricity.pv_feed_in_tariff_eur_per_kwh": 1e-05,
                                  "asset_sizing.pv.fallback_capacity_kwp": 2e16})
    assert "pv_feed_in_tariff_eur_per_kwh: 1.0e-05\n" in new and "fallback_capacity_kwp: 2.0e+16\n" in new
    pv = yaml.safe_load(new)
    assert pv["economics"]["electricity"]["pv_feed_in_tariff_eur_per_kwh"] == 1e-05


def test_dependent_keys_are_only_removed_when_their_rule_field_changes():
    broken = text_of(INVENTORY).replace("    adoption_mode: source_inventory\n  mobility:",
                                        "    adoption_mode: source_inventory\n    building_share: 0.5\n  mobility:")
    new = sf.apply_changes(broken, {"asset_sizing.pv.demand_multiplier": 2.5})
    assert "building_share: 0.5" in new  # the loader reports it; the editor does not repair unasked
    fixed = sf.apply_changes(broken, {"electrification.heat.adoption_mode": "source_inventory"})
    assert "building_share" not in fixed.split("mobility:")[0]


def test_the_home_charger_power_writes_both_capacities():
    text = text_of(SCHWEINFURT)
    new = yaml.safe_load(sf.apply_changes(text, {"technologies.processes.home_charger.installed_capacity_kw": 7.4}))
    charger = new["technologies"]["processes"]["home_charger"]
    assert charger["installed_capacity_kw"] == charger["capacity_upper_kw"] == 7.4
    labels = [c["label"] for c in sf.preview(text, base_name="s", changes={
        "technologies.processes.home_charger.installed_capacity_kw": 7.4})["changes"]]
    assert labels == ["Home charger power", "Home charger power (capacity_upper_kw)"]


@pytest.mark.parametrize(("changes", "key", "needle", "has_line"), [
    ({"asset_sizing.pv.demand_multiplier": -1}, "asset_sizing.pv.demand_multiplier", "must be positive", True),
    ({"electrification.heat.building_share": 1.5}, "electrification.heat.building_share", "[0, 1]", True),
    ({"asset_sizing.battery.heuristic_usable_kwh_per_pv_kwp": 2.0}, None, "HTW 2025", False),
    ({"asset_sizing.heat.heating_limit_temperature_c": 25.0}, "asset_sizing.heat.heating_limit_temperature_c",
     "must be below", True),
    ({"asset_sizing.pv.demand_multiplier": "a lot"}, "asset_sizing.pv.demand_multiplier", "must be a number", False),
    ({"time_aggregation.hours_per_period": 2.5}, "time_aggregation.hours_per_period", "whole number", False),
    ({"time_aggregation.enabled": "yes"}, "time_aggregation.enabled", "true or false", False),
    ({"asset_sizing.heat.space_heat_source": "oil"}, "asset_sizing.heat.space_heat_source", "must be one of", False),
    ({"asset_sizing.pv.tilt_bin_degrees": 3.0}, "asset_sizing.pv.tilt_bin_degrees", "not an editable field", False),
])
def test_invalid_values_give_readable_issues(changes, key, needle, has_line):
    result = sf.preview(text_of(SCHWEINFURT), base_name="s", changes=changes)
    assert not result["ok"]
    issue = result["issues"][0]
    assert issue["level"] == "error" and needle in issue["message"]
    assert issue.get("key") == key
    assert ("line" in issue) == has_line
    if has_line:
        line = text_of(SCHWEINFURT).splitlines()[issue["line"] - 1]
        assert line.strip().startswith(issue["key"].rsplit(".", 1)[1] + ":")


def test_text_issues_carry_line_numbers():
    text = text_of(SCHWEINFURT)
    broken = sf.preview(text, base_name="s", text="scenario:\n  id: [1, 2\n")
    assert not broken["ok"] and broken["issues"][0]["message"].startswith("YAML syntax error") and broken["issues"][0]["line"]
    missing = sf.preview(text, base_name="s", text=text.replace("    fallback_capacity_kwp: 14.5\n", ""))
    assert missing["issues"][0]["message"] == "Missing required key 'fallback_capacity_kwp'."
    assert missing["issues"][0]["key"] == "asset_sizing.pv.fallback_capacity_kwp"
    unknown = sf.preview(text, base_name="s", text=text.replace("  milestone_year: 2045\n", "  milestone_year: 2045\n  typo: 1\n"))
    assert "Unknown scenario option(s): ['typo']" in unknown["issues"][0]["message"]
    assert unknown["issues"][0]["line"] == unknown["text"].splitlines().index("  typo: 1") + 1
    # form changes are applied on top of an edited text
    edited = text.replace("demand_multiplier: 2.0", "demand_multiplier: 2.2")
    both = sf.preview(text, base_name="s", text=edited, changes={"asset_sizing.pv.fallback_capacity_kwp": 10.0})
    assert both["ok"] and both["values"]["asset_sizing.pv.demand_multiplier"] == 2.2
    assert both["values"]["asset_sizing.pv.fallback_capacity_kwp"] == 10.0
    assert {c["key"] for c in both["changes"]} == {"asset_sizing.pv.demand_multiplier",
                                                   "asset_sizing.pv.fallback_capacity_kwp"}


def test_source_inventory_and_template_ids_are_warnings():
    result = sf.preview(text_of(INVENTORY), base_name="f", scenario_id="CHANGE_ME_x")
    assert result["ok"]
    messages = [i["message"] for i in result["issues"] if i["level"] == "warning"]
    assert sum("source_inventory" in m for m in messages) == 3 and any("CHANGE_ME" in m for m in messages)
    bad_id = sf.preview(text_of(SCHWEINFURT), base_name="s", scenario_id="../x")
    assert not bad_id["ok"] and bad_id["issues"][0]["key"] == "scenario.id"


def test_help_texts_are_the_yaml_comments_above_the_key():
    helps = sf.help_texts(text_of(SCHWEINFURT))
    assert helps["asset_sizing.pv.demand_multiplier"].startswith("Central compromise across public recommendations")
    assert "P_PV = min(" in helps["asset_sizing.pv.demand_multiplier"]
    assert helps["asset_sizing.pv.maximum_fallback_share"] == ("Publication-quality runs fail if more than this share "
                                                              "uses the fallback.")
    assert "asset_sizing.pv.flat_roof_utilization" not in helps  # no comment directly above
    assert "electrification.heat.adoption_mode" not in helps  # the inline comment belongs to ``electrification``
    sections, _ = sf.form_sections(text_of(SCHWEINFURT))
    fields = {f["key"]: f for s in sections for f in s["fields"]}
    assert set(fields) == set(sf.FIELDS)
    assert fields["asset_sizing.heat.teaser_retrofit_level"]["present"] is False
    assert fields["asset_sizing.heat.teaser_retrofit_level"]["value"] == 0
    assert fields["electrification.heat.building_share"]["value"] == 0.75


def test_fields_a_file_does_not_have_are_skipped():
    text = text_of(INVENTORY)
    sections, _ = sf.form_sections(text)
    fields = {f["key"]: f for s in sections for f in s["fields"]}
    assert fields["electrification.heat.building_share"]["present"] is False  # dependent: shown, disabled in the UI
    stripped = text.replace("  commuting_probability: 0.62\n", "")
    keys = {f["key"] for s in sf.form_sections(stripped)[0] for f in s["fields"]}
    assert "mobility.commuting_probability" not in keys


def test_other_indentation_is_kept():
    data = yaml.safe_load(text_of(SCHWEINFURT))
    four = yaml.safe_dump(data, indent=4, sort_keys=False)
    four = "# four spaces\n" + four
    new = sf.apply_changes(four, {"asset_sizing.pv.demand_multiplier": 3.0})
    assert [line for line in new.splitlines() if line not in four.splitlines()] == ["        demand_multiplier: 3.0"]
    assert configuration_hash(yaml.safe_load(new)) != configuration_hash(data)


def test_changed_values_lists_added_and_removed_leaves():
    old = {"a": {"b": 1, "c": 2.0}, "d": [1, 2]}
    new = {"a": {"b": 1.0, "e": True}, "d": [1, 2]}
    assert sf.changed_values(old, new) == [
        {"key": "a.b", "label": None, "kind": "changed", "old": 1, "new": 1.0},
        {"key": "a.c", "label": None, "kind": "removed", "old": 2.0, "new": None},
        {"key": "a.e", "label": None, "kind": "added", "old": None, "new": True},
    ]
