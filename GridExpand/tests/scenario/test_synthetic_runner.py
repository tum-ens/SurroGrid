"""Synthetic runner: settings checks, Step 4 passes, names, resume and expansion notes (no database)."""

from __future__ import annotations

import pytest

from gridexpand.common.orchestration import StatusLog
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario import synthetic_ags_runner as runner
from gridexpand.scenario.config_loader import load_scenario_config

SCENARIO_PATH = SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml"
SCENARIO, SCENARIO_HASH = load_scenario_config(SCENARIO_PATH)
SID = f"scenario_schweinfurt_2045_{SCENARIO_HASH[:12]}"
KEY = f"{SID}_max_base_electricity_demand_week"


def settings(tmp_path=None, **changes):
    base = runner.BatchSettings(
        ags="09184137", pylovo_version_id="1", scenario_config=SCENARIO_PATH, scenario=SCENARIO,
        scenario_hash=SCENARIO_HASH, run_dir=tmp_path or runner.Path("/runs/x"), model_case="post-hems-heuristic",
        profiles="all", timeframe_mode="max_base_electricity_demand_week", powerflow_output="both",
        case_qualified_output=True, step4_cpus=2,
    )
    return base.replace(**changes)


def candidate(index=3, plz=85653, kcid=1, bcid=4):
    return {"candidate_index": index, "ags": 9184137, "plz": plz, "kcid": kcid, "bcid": bcid,
            "bridge_filename": f"9184137-{index:02d}_{plz}_{kcid}_{bcid}.h5", "n_buildings": 63}


def test_parse_and_settings_from_args(tmp_path):
    args = runner.parse_args(["--ags", "9184137", "--pylovo-version-id", "1", "--scenario-config", str(SCENARIO_PATH),
                              "--run-dir", str(tmp_path), "--model-case", "pre", "--profiles", "status_quo",
                              "--no-pilot-gate", "--cleanup-completed-only", "--plz", "85653"])
    result = runner.settings_from_args(args)
    assert result.scenario_hash == SCENARIO_HASH and result.run_dir == tmp_path.resolve()
    assert not result.pilot_gate and result.resume and result.plz == 85653
    assert result.assignment_path == tmp_path.resolve() / "electrification_assignment.csv"
    with pytest.raises(SystemExit):  # the scenario YAML is required
        runner.parse_args(["--ags", "1", "--pylovo-version-id", "1", "--run-dir", str(tmp_path)])


@pytest.mark.parametrize("changes, message", [
    ({"model_case": "pre"}, "requires --profiles status_quo"),
    ({"profiles": "status_quo"}, "Post model cases"),
    ({"include_inflex_powerflow": True}, "INFLEX"),
    ({"inflex_only": True}, "INFLEX"),
    ({"inflex_ev_charger_kw": 11.0}, "--inflex-ev-charger-kw"),
    ({"kcid": 1}, "--kcid and --bcid"),
    ({"kcid": 1, "bcid": 4}, "--plz is required"),
    ({"workers": 0}, "--workers"),
])
def test_check_settings_rejects(changes, message):
    with pytest.raises(ValueError, match=message):
        runner.check_settings(settings(**changes))


def test_powerflow_passes_and_commands():
    pre = settings(model_case="pre", profiles="status_quo")
    (only,) = runner.powerflow_passes(pre)
    assert (only.mode, only.outputs, only.stage) == ("pre_only", ("raw", "summary"), "step4_powerflow_raw_summary_pre_only")
    argv = runner.powerflow_pass_command(pre, "in.h5", only)
    assert argv[argv.index("--run-name") + 1] == f"{KEY}_status_quo_pre_raw_powerflow"
    assert argv[argv.index("--summary-run-name") + 1] == f"{KEY}_status_quo_pre_summary_powerflow"
    assert "--pre-only" in argv and "--summary-grid-scope" not in argv
    post = settings()
    (flexible,) = runner.powerflow_passes(post)
    argv = runner.powerflow_pass_command(post, "in.h5", flexible)
    assert argv[argv.index("--run-name") + 1] == f"{KEY}_post_electrification_post-hems-heuristic_raw_powerflow"
    assert "--post-demand-mode" not in argv and flexible.expected_summary_stages == ("pre", "post")
    (summary,) = runner.powerflow_passes(settings(powerflow_output="summary", powerflow_grid_scope="backbone"))
    argv = runner.powerflow_pass_command(settings(powerflow_output="summary", powerflow_grid_scope="backbone"),
                                         "in.h5", summary)
    assert summary.stage == "step4_powerflow_summary" and "--summary-run-name" not in argv
    assert argv[argv.index("--run-name") + 1].endswith("_summary_powerflow")
    assert argv[argv.index("--summary-grid-scope") + 1] == "backbone"
    # INFLEX passes stay in the table for the methodological fix (refused by check_settings today).
    passes = runner.powerflow_passes(settings(include_inflex_powerflow=True))
    assert [(p.mode, p.run_token("summary")) for p in passes] == [("flexible", "summary"), ("inflex", "summary_inflex")]


def test_names():
    s = settings()
    assert runner.step2_filename(candidate(), s) == "9184137-03_85653_1_4_max_base_electricity_demand_week_post-hems-heuristic.h5"
    assert runner.job_key(candidate()) == "9184137-03_85653_1_4"
    assert SCENARIO.time_aggregation.enabled  # schweinfurt_2045 uses TSAM
    assert runner.expansion_analysis_prefix(s) == (
        f"09184137_{SID}_max_base_electricity_demand_week_post_electrification_tsam_post-hems-heuristic")
    assert runner.expansion_analysis_prefix(settings(case_qualified_output=False, demand_scope="residential")) == (
        f"09184137_{SID}_max_base_electricity_demand_week_post_electrification_hh_only_tsam")
    assert runner.expansion_analysis_prefix(settings(expansion_analysis_prefix="mine")) == "mine"


def test_select_region():
    cands = [candidate(0, 85653, 1, 1), candidate(1, 85653, 1, 4), candidate(2, 85654, 2, 1)]
    assert [c["candidate_index"] for c in runner.select_region(cands, plz=85653)] == [0, 1]
    assert [c["candidate_index"] for c in runner.select_region(cands, plz=85653, kcid=1, bcid=4)] == [1]
    assert runner.select_region(cands) == cands


def test_resume_needs_the_same_grid(tmp_path):
    s = settings(tmp_path, resume=True)
    cands = [candidate(0), candidate(1), candidate(2)]
    status = StatusLog(tmp_path, resume=False, echo=False)
    status.update(0, status="done", bridge_filename=runner.step2_filename(cands[0], s))
    status.update(1, status="done", bridge_filename="9184137-01_99999_9_9_other.h5")  # renumbered grid
    status.update(2, status="failed", bridge_filename=runner.step2_filename(cands[2], s))
    reloaded = StatusLog(tmp_path, resume=True, echo=False)
    assert runner.previous_status(cands, s, reloaded) == {0: "done", 2: "failed"}
    assert [c["candidate_index"] for c in runner.filter_candidates(cands, s, reloaded)] == [1]
    assert [c["candidate_index"] for c in runner.filter_candidates(cands, s.replace(rerun_failed=True), reloaded)] == [1, 2]
    # an event with the job name counts; one naming another grid does not
    reloaded.event(event="candidate_done", candidate_index=1, job="9184137-01_85653_1_4", status="done")
    assert runner.previous_status(cands, s, reloaded)[1] == "done"
    ranged = s.replace(start_index=1, limit=2, rerun_failed=True)
    assert [c["candidate_index"] for c in runner.filter_candidates(cands, ranged, reloaded)] == [2]


def test_batch_identity_guard(tmp_path):
    runner.check_batch_identity(settings(tmp_path))
    runner.check_batch_identity(settings(tmp_path, resume=True, workers=8))  # resources may change
    with pytest.raises(ValueError, match="another batch"):
        runner.check_batch_identity(settings(tmp_path, resume=True, min_buildings=60))


def test_expansion_notes_name_failed_grids(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_batch_command", lambda **kwargs: calls.append(kwargs["cmd"]))
    s = settings(tmp_path)
    status = StatusLog(tmp_path, echo=False)
    materialized = runner.materialize_expansion_analyses(settings=s, status=status, candidate_count=4)
    assert [m["analysis_key"] for m in materialized] == [
        f"09184137_{SID}_max_base_electricity_demand_week_post_electrification_tsam_post-hems-heuristic_pre",
        f"09184137_{SID}_max_base_electricity_demand_week_post_electrification_tsam_post-hems-heuristic_post",
    ]
    assert calls[0][calls[0].index("--note") + 1].endswith("summary stage=pre.")
    calls.clear()
    runner.materialize_expansion_analyses(settings=s, status=status, candidate_count=4,
                                          failures=[{"job": "9184137-01_85653_1_1"}])
    note = calls[0][calls[0].index("--note") + 1]
    assert "INCOMPLETE: 1 of 4 grids failed and are missing (9184137-01_85653_1_1)." in note
    assert runner.materialize_expansion_analyses(settings=s.replace(powerflow_output="raw"), status=status) == []


def test_choose_step3_settings(tmp_path):
    import pandas as pd

    path = tmp_path / "in.h5"
    pd.DataFrame([[0.0] * 70], columns=[f"c{i}" for i in range(70)]).to_hdf(path, key="urbs_in/demand", format="fixed")
    s = settings(step3_cpus=4, step3_max_cpus=8, step3_target_columns=35)
    assert runner.choose_step3_settings(path, s) == (4, 1, {"demand_columns": 70, "eff_factor_columns": 0})
    assert runner.choose_step3_settings(path, s.replace(step3_target_columns=10))[0] == 8
    assert runner.choose_step3_settings(path, s.replace(dynamic_step3=False, step3_cpus=3)) == (3, 1, {})


def test_status_quo_candidate_skips_step3(tmp_path, monkeypatch):
    """run_candidate: the pre case runs Step 2 and one Step 4 pass (raw+summary), then validates both runs."""
    s = settings(tmp_path, model_case="pre", profiles="status_quo")
    (tmp_path / "logs").mkdir()
    status = StatusLog(tmp_path, echo=False)
    stages, validated = [], []
    monkeypatch.setattr(runner, "run_command", lambda **kwargs: stages.append((kwargs["stage"], kwargs["cmd"])))
    step2_dir = tmp_path / "step2"
    step2_dir.mkdir()
    step2_file = step2_dir / runner.step2_filename(candidate(), s)
    step2_file.write_bytes(b"h5")
    monkeypatch.setattr(runner, "scenario_output_directory", lambda base, key: step2_dir)
    monkeypatch.setattr(runner, "read_hdf_metadata", lambda path: {"scenario_key": KEY, "horizon_hours": 168})
    monkeypatch.setattr(runner, "POWERFLOW_INPUT_DIR", tmp_path / "pf_in")
    monkeypatch.setattr(runner, "validate_powerflow_db", lambda name, **kw: validated.append((name, kw)) or {})
    result = runner.run_candidate(candidate=candidate(), settings=s, status=status)
    assert result["status"] == "done" and result["job"] == "9184137-03_85653_1_4"
    assert [stage for stage, _ in stages] == ["step2_demand_allocation", "step4_powerflow_raw_summary_pre_only"]
    assert [(kw["summary_only"], kw["pre_only"]) for _, kw in validated] == [(False, True), (True, True)]
    row = status.rows[3]
    assert (row["status"], row["step3_cpus"], row["message"]) == ("done", "skipped", "ok")
