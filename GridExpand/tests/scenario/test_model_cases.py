"""Model-case table and execution groups (the rules the runners used to repeat)."""

from __future__ import annotations

import pytest

from gridexpand.scenario.model_cases import (
    MODEL_CASES,
    POST_MODEL_CASES,
    compatible_result_cases,
    execution_groups,
    get_model_case,
    validate_cases,
)


def test_table():
    assert POST_MODEL_CASES == ("post-inflex-heuristic", "post-hems-optimized", "post-hems-heuristic")
    pre = MODEL_CASES["pre"]
    assert (pre.asset_plan, pre.powerflow_mode, pre.profiles, pre.stage) == ("none", None, "status_quo", "pre")
    assert pre.real_powerflow_mode == "pre-only"
    # analysis suffixes of the paired expansion keys (unchanged strings)
    assert {name: case.analysis_suffix for name, case in MODEL_CASES.items()} == {
        "pre": "pre",
        "post-inflex-heuristic": "post_inflex",
        "post-hems-heuristic": "post",
        "post-hems-optimized": "post_hems_optimized",
    }
    # real-grid adapter modes and labels (were a dict in sources/swf.py)
    assert {name: (MODEL_CASES[name].powerflow_mode, MODEL_CASES[name].label) for name in POST_MODEL_CASES} == {
        "post-hems-optimized": ("flexible", "optimized HEMS"),
        "post-hems-heuristic": ("flexible", "heuristic-assets HEMS"),
        "post-inflex-heuristic": ("inflex", "heuristic-assets INFLEX"),
    }
    assert compatible_result_cases("post-inflex-heuristic") == ("post-inflex-heuristic", "post-hems-heuristic")
    assert compatible_result_cases("post-hems-optimized") == ("post-hems-optimized",)
    with pytest.raises(ValueError):
        get_model_case("post")


def test_execution_groups_paired_rule():
    groups = execution_groups(["post-hems-optimized", "post-hems-heuristic", "post-inflex-heuristic"])
    assert [(g.name, g.materialization_case, g.result_cases, g.emits_pre) for g in groups] == [
        ("heuristic-assets", "post-hems-heuristic", ("post-inflex-heuristic", "post-hems-heuristic"), True),
        ("post-hems-optimized", "post-hems-optimized", ("post-hems-optimized",), False),
    ]
    (only,) = execution_groups(["post-hems-optimized"])
    assert only.emits_pre and only.name == "post-hems-optimized"


def test_execution_groups_with_pre():
    groups = execution_groups(["pre", "post-hems-heuristic", "post-hems-optimized"])
    assert [g.materialization_case for g in groups] == ["pre", "post-hems-heuristic", "post-hems-optimized"]
    assert [g.emits_pre for g in groups] == [True, False, False]


@pytest.mark.parametrize("cases", [[], ["pre", "pre"], ["nope"]])
def test_validate_cases_rejects(cases):
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_allowed():
    with pytest.raises(ValueError, match="not allowed"):
        validate_cases(["pre"], allowed=POST_MODEL_CASES)
