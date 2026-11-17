"""Expansion analyses of aligned runs and the notebook keys of their groups (no database)."""

from __future__ import annotations

import pytest

from gridexpand.analysis.expansion import grid_expansion as ge
from validation import aligned_expansion
from validation import notebook_workflow as nw


def test_aligned_groups_cover_every_model_case():
    groups = aligned_expansion.aligned_groups(
        "run1", providers=("dso_b",), cases=tuple(aligned_expansion.CASE_STAGES),
        excluded_real_grids={"dso_b": ("113",)},
    )
    parsed = [ge._build_parser().parse_args(argv) for argv in groups]
    keys = {(a.data_source, a.analysis_key, a.run_name, a.stage) for a in parsed}
    assert ("synthetic", "run1_dso_b_synthetic_post_hems_optimized", "run1_dso_b_synthetic_post-hems-optimized", "post") in keys
    assert ("real_dso_b", "run1_dso_b_real_post_inflex", "run1_dso_b_real_dso_b_post-inflex-heuristic", "post") in keys
    real = [a for a in parsed if a.data_source == "real_dso_b"]
    assert all(a.exclude_real_lv_id == ["113"] for a in real)
    assert all(a.plz == [] and a.replace for a in parsed)
    assert len({a.analysis_key for a in parsed}) == len(parsed) == 8


def test_parse_exclusions():
    assert aligned_expansion._parse_exclusions(["dso_a:113", "dso_b:area-5", "dso_a:7"]) == {
        "dso_a": ("113", "7"), "dso_b": ("area-5",)
    }
    with pytest.raises(SystemExit):
        aligned_expansion._parse_exclusions(["xyz:1"])


def test_case_table_matches_notebook_keys():
    from gridexpand.analysis.expansion.cases import CASE_STAGES

    specs = nw.scenario_powerflow_specs("p", nw.ALL_MODEL_CASE_STAGE_LABELS)
    assert specs["Synthetic"]["HEMS optimized"] == {"run_name": "p_synthetic_post-hems-optimized", "stage": "post"}
    assert list(specs["Real DSO A"]) == ["status-quo", "INFLEX", "HEMS heuristic", "HEMS optimized"]
    keys = nw.scenario_analysis_keys("p", nw.ALL_MODEL_CASE_STAGE_LABELS, data_source="Real DSO B", provider="dso_b")
    assert keys == {
        "status-quo": "p_dso_b_real_pre",
        "INFLEX": "p_dso_b_real_post_inflex",
        "HEMS heuristic": "p_dso_b_real_post",
        "HEMS optimized": "p_dso_b_real_post_hems_optimized",
    }
    assert set(CASE_STAGES) == set(aligned_expansion.CASE_STAGES)


def test_exclusions_are_canonical():
    assert nw._excluded_ids({"Real DSO A": (113, "LV_007")}, "Real DSO A") == {"113", "7"}
    assert nw._excluded_ids({"Real DSO B": ("area-12",)}, "Real DSO B") == {"12"}
    assert nw._excluded_ids({"Synthetic DSO A": ("99999_1_3",)}, "Synthetic DSO A") == {"99999_1_3"}
    assert nw._excluded_ids({"Real DSO B": ("area-12",)}, "Real DSO A") == set()
