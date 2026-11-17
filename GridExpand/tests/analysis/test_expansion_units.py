"""Pure helpers of the expansion CLIs (no database)."""

from __future__ import annotations

import pytest

from gridexpand.analysis.expansion import grid_expansion as ge


def _args(*argv: str):
    return ge._build_parser().parse_args(list(argv))


def test_sql_files_join_the_python_decisions():
    line_sql = ge.sql_text("line_insert.sql")
    assert "JOIN expansion_component_selection sel" in line_sql
    transformer_sql = ge.sql_text("transformer_insert.sql")
    assert "JOIN expansion_station_selection sel USING (powerflow_run_id)" in transformer_sql
    peak_sql = ge.sql_text("transformer_peak.sql")
    # Station rating, not the unit rating s_max_kva of transformer_positions_with_grid.
    assert "gr.transformer_rated_power::DOUBLE PRECISION AS rated_kva" in peak_sql
    assert "s_max_kva" not in peak_sql
    # Deterministic tie-break of the critical component (B8).
    assert "loading_percent DESC NULLS LAST, component_line" in line_sql


def test_default_assumption_is_the_staged_row():
    assert _args().assumption_key == "de_lv_staged_2026"


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


def test_run_name_required_before_database_access(capsys):
    with pytest.raises(SystemExit) as exc:
        ge.main(["--stage", "post"])
    assert exc.value.code == 2 and "--run-name is required" in capsys.readouterr().err
