"""Pure helpers of the expansion CLIs (no database)."""

from __future__ import annotations

import pytest

from gridexpand.analysis.expansion import aligned_expansion, grid_expansion as ge


def _args(*argv: str):
    return ge._build_parser().parse_args(list(argv))


def test_sql_fragments_are_inlined():
    line_sql = ge.sql_text("line_insert.sql")
    assert "/*CABLE_SELECTION*/" not in line_sql and "generate_series" in line_sql
    transformer_sql = ge.sql_text("transformer_insert.sql")
    assert "/*TRANSFORMER_COST*/" not in transformer_sql and "all_in_replacement_to_100kva" in transformer_sql
    # Deterministic tie-break of the critical component (B8).
    assert "loading_percent DESC NULLS LAST, component_line" in line_sql


def test_identity_and_replacement_guard():
    args = _args("--run-name", "r", "--stage", "pre", "--ags", "09184137", "--replace")
    identity = ge.analysis_identity(args)
    assert identity == {"run_name": "r", "stage": "pre", "data_source": "Synthetic", "ags": 9184137}
    assert ge.check_replacement("k", None, identity, replace=False) is False
    assert ge.check_replacement("k", dict(identity), identity, replace=True) is True
    with pytest.raises(RuntimeError, match="Use --replace"):
        ge.check_replacement("k", dict(identity), identity, replace=False)
    with pytest.raises(RuntimeError, match="run_name: stored 'other'"):
        ge.check_replacement("k", {**identity, "run_name": "other"}, identity, replace=True)
    with pytest.raises(RuntimeError, match="ags"):
        ge.check_replacement("k", {**identity, "ags": 9162000}, identity, replace=True)


def test_scenario_resolution(capsys):
    assert ge.resolve_scenario_id(None, {4}) == 4
    assert ge.resolve_scenario_id(7, {4, 5}) == 7
    assert ge.resolve_scenario_id(None, set()) is None
    assert ge.resolve_scenario_id(None, {4, 5}) is None
    assert "2 scenarios" in capsys.readouterr().out


def _audit(**overrides):
    row = {
        "selected_runs": 4, "active_components": 900, "unmapped_components": 0,
        "overloaded_unmapped_components": 0, "root_connector_like_components": 0,
        "overloaded_root_connector_like_components": 0, "max_unmapped_loading_percent": 0.0,
    }
    return {**row, **overrides}


def test_component_audit():
    assert ge.check_component_audit(_audit()) is None
    assert "ignored 3 active unmapped" in ge.check_component_audit(
        _audit(unmapped_components=3, root_connector_like_components=3)
    )
    with pytest.raises(RuntimeError, match="No power-flow runs"):
        ge.check_component_audit(_audit(selected_runs=0))
    with pytest.raises(RuntimeError, match="cable summaries"):
        ge.check_component_audit(_audit(active_components=0))
    with pytest.raises(RuntimeError, match="Refusing to hide"):
        ge.check_component_audit(_audit(unmapped_components=2, overloaded_unmapped_components=1))


def test_refresh_flags():
    assert _args().no_refresh is False
    assert _args("--no-refresh").no_refresh is True
    assert _args("--refresh-only").refresh_only is True


def test_aligned_groups_cover_every_model_case():
    groups = aligned_expansion.aligned_groups(
        "run1", providers=("uzw",), cases=tuple(aligned_expansion.CASE_STAGES),
        excluded_real_grids={"uzw": ("113",)},
    )
    parsed = [ge._build_parser().parse_args(argv) for argv in groups]
    keys = {(a.data_source, a.analysis_key, a.run_name, a.stage) for a in parsed}
    assert ("synthetic", "run1_uzw_synthetic_post_hems_optimized", "run1_uzw_synthetic_post-hems-optimized", "post") in keys
    assert ("real_uzw", "run1_uzw_real_post_inflex", "run1_uzw_real_uzw_post-inflex-heuristic", "post") in keys
    real = [a for a in parsed if a.data_source == "real_uzw"]
    assert all(a.exclude_real_lv_id == ["113"] for a in real)
    assert all(a.plz == [] and a.replace for a in parsed)
    assert len({a.analysis_key for a in parsed}) == len(parsed) == 8


def test_parse_exclusions():
    assert aligned_expansion._parse_exclusions(["swf:113", "uzw:area-5", "swf:7"]) == {
        "swf": ("113", "7"), "uzw": ("area-5",)
    }
    with pytest.raises(SystemExit):
        aligned_expansion._parse_exclusions(["xyz:1"])
