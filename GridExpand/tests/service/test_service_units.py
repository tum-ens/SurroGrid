"""Pure helpers of the service: run names, runner events, command construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridexpand.service.commands import (
    PipelineSpec,
    contiguous_range,
    pipeline_steps,
    run_yaml,
)
from gridexpand.service.queries import parse_run_name, split_scenario_key
from gridexpand.service.runlog import (
    RunProgress,
    classify_line,
    format_event,
    log_label,
)

KEY = "scenario_sandbox_2045_7669656db28a_max_base_electricity_demand_week"


def test_parse_run_name():
    assert parse_run_name(f"{KEY}_status_quo_pre_summary_powerflow") == {
        "profile": "status_quo", "model_case": "pre", "mode": "summary"}
    assert parse_run_name(f"{KEY}_post_electrification_post-hems-heuristic_raw_powerflow") == {
        "profile": "post_electrification", "model_case": "post-hems-heuristic", "mode": "raw"}
    assert parse_run_name(f"{KEY}_post_electrification_post-inflex-heuristic_summary_inflex_powerflow")["mode"] == "summary_inflex"
    assert parse_run_name(f"{KEY}_status_quo_summary_powerflow")["model_case"] == "pre"  # not case-qualified
    assert parse_run_name(f"{KEY}_post_electrification_summary_powerflow")["model_case"] is None
    assert parse_run_name("baseline_static_pre_powerflow") == {"profile": None, "model_case": None, "mode": None}


def test_split_scenario_key():
    assert split_scenario_key(KEY) == {"scenario_key": KEY, "scenario": "scenario_sandbox_2045_7669656db28a",
                                       "timeframe_mode": "max_base_electricity_demand_week"}
    assert split_scenario_key("baseline_static")["timeframe_mode"] is None


def test_run_progress_from_events():
    progress = RunProgress()
    for event in [
        {"event": "candidates_selected", "count": 3},
        {"event": "pilot_start", "candidate_index": 4},
        {"event": "start", "candidate_index": 4, "stage": "step2_demand_allocation"},
        {"event": "pilot_finish", "candidate_index": 4, "status": "done"},
        {"event": "start", "candidate_index": 5, "stage": "step4_powerflow_summary"},
        {"event": "candidate_failed", "candidate_index": 5, "stage": "step4", "message": "boom"},
        {"event": "candidate_failed_recorded", "candidate_index": 5, "status": "failed"},
    ]:
        progress.apply(event)
    assert (progress.grids_total, progress.grids_done, progress.grids_failed) == (3, 1, 1)
    assert progress.fraction() == pytest.approx(2 / 3)
    progress.apply({"event": "start", "stage": "expansion_materialize_post"})
    assert progress.stage == "expansion_materialize_post"
    progress.apply({"event": "batch_finish", "status": "completed_with_failures"})
    assert progress.finished and progress.fraction() == 1.0


def test_format_and_classify_events():
    finish = format_event({"event": "finish", "candidate_index": 2, "stage": "step2", "returncode": 0, "seconds": 1.5}, "pre")
    assert finish.startswith("✓ [pre] grid #2") and classify_line(finish) == "success"
    failed = format_event({"event": "candidate_failed", "candidate_index": 2, "stage": "step3", "message": "x"})
    assert classify_line(failed) == "error"
    done = format_event({"event": "batch_finish", "status": "done", "completed_count": 1, "candidate_count": 1})
    assert classify_line(done) == "success"
    partial = format_event({"event": "batch_finish", "status": "completed_with_failures"})
    assert classify_line(partial) == "warning"
    assert classify_line("Traceback (most recent call last):") == "error"
    assert classify_line("ValueError: bad") == "error"
    assert classify_line("UserWarning: careful") == "warning"
    assert log_label(Path("candidate_012_x.h5.log")) == "#12"
    assert log_label(Path("expansion_materialization.log")) == "expansion"


def test_commands(tmp_path):
    from gridexpand.scenario.run_config import load_run_config

    scenario = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "schweinfurt_2045.yaml"
    spec = PipelineSpec(ags=9184137, pylovo_version_id="1", scenario_config=scenario,
                        model_cases=("pre", "post-hems-optimized"), timeframe_mode="full_year",
                        min_buildings=60, start_index=3, limit=1)
    job_dir = tmp_path / "runs" / "abc123"
    steps = pipeline_steps(spec, job_dir, python="py")
    assert [s.name for s in steps] == ["pre", "post-hems-optimized"]
    assert steps[0].argv == ["py", "-m", "gridexpand", "run", str(job_dir / "run.yaml"), "--run-dir", str(job_dir),
                             "--model-case", "pre"]
    assert steps[1].argv[-1] == "post-hems-optimized" and steps[1].run_dir == str(job_dir / "post-hems-optimized")
    # the generated run YAML is a valid synthetic run with the request's selection
    run, _ = load_run_config(job_dir / "run.yaml")
    assert (run.pipeline, run.run_id, run.ags, run.pylovo_version_id) == ("synthetic", "service_abc123", 9184137, "1")
    assert (run.min_buildings, run.start_index, run.limit, run.pilot_index) == (60, 3, 1, 3)
    assert run.model_cases == ("pre", "post-hems-optimized") and run.powerflow_output == "summary"
    assert run.scenario_path == scenario and run.timeframe_mode == "full_year"
    assert run_yaml(spec, "x")["execution"]["workers"] == 1
    assert contiguous_range([5, 3, 4]) == (3, 3)
    with pytest.raises(ValueError):
        contiguous_range([1, 3])
    with pytest.raises(ValueError):
        contiguous_range([])


def test_single_grid_run_yaml_and_terminal_commands(tmp_path):
    from gridexpand.scenario.run_config import load_run_config
    from gridexpand.service.commands import run_yaml_text, terminal_commands

    scenario = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "schweinfurt_2045.yaml"
    spec = PipelineSpec(ags=9184137, pylovo_version_id="1", scenario_config=scenario, model_cases=("pre",),
                        timeframe_mode="max_base_electricity_demand_week", min_buildings=1, plz=85653, kcid=1, bcid=-1)
    path = tmp_path / "grid.yaml"
    path.write_text(run_yaml_text(spec, "ui_grid", "Grid 1/-1\n\nsecond line"), encoding="utf-8")
    assert path.read_text().startswith("# Grid 1/-1\n#\n# second line\n")
    run, _ = load_run_config(path)
    assert (run.plz, run.kcid, run.bcid, run.min_buildings, run.start_index) == (85653, 1, -1, 1, None)
    assert run_yaml(spec, "x", scenario="../scenarios/s.yaml")["run"]["scenario"] == "../scenarios/s.yaml"
    local = terminal_commands(path, "ui_grid", in_container=False, project_dir=tmp_path)
    assert local[0]["command"] == f"tmux new-session -d -s ui_grid 'uv run gridexpand run {path}; exec bash'"
    assert local[-1]["command"].endswith("--resume")
    docker = terminal_commands(path, "ui grid!", in_container=True, project_dir=tmp_path)
    assert "docker compose exec gridexpand gridexpand run" in docker[0]["command"] and "-s ui-grid-" in docker[0]["command"]


def test_status_rows_and_stage_from_state(tmp_path):
    import json

    from gridexpand.service.runlog import RunTracker, read_state, read_status_rows

    root = tmp_path / "job"
    (root / "pre").mkdir(parents=True)
    state = {"status": "running", "stage": "prepare", "job_list": [
        {"job": "pre/9184137-03_85653_1_4", "group": "pre", "candidate_index": 3, "plz": 85653, "kcid": 1,
         "bcid": 4, "n_buildings": 63, "status": "queued", "step": None},
        {"job": "post-hems-optimized/9184137-03_85653_1_4", "group": "post-hems-optimized", "candidate_index": 3},
    ]}
    (root / "state.json").write_text(json.dumps(state))
    assert read_state(root / "pre")["stage"] == "prepare"
    rows = read_status_rows(root / "pre")
    assert [(r["candidate_index"], r["kcid"], r["bcid"], r["status"], r["log"]) for r in rows] == [
        ("3", "1", "4", "queued", "")]
    tracker = RunTracker(root / "pre")
    tracker.poll_events()
    assert tracker.progress.stage == "run · prepare"
    (root / "prepare" / "logs").mkdir(parents=True)
    (root / "prepare" / "logs" / "electrification_preparation.log").write_text("building inventory\n")
    assert ("prep", "building inventory") in tracker.poll_logs()
    assert read_state(tmp_path) is None and read_status_rows(tmp_path / "nothing") == []
