"""Pure helpers of the service: run names, runner events, command construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridexpand.service.commands import (
    PipelineSpec,
    contiguous_range,
    pipeline_steps,
    synthetic_command,
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


def test_commands():
    spec = PipelineSpec(ags=9184137, pylovo_version_id="1", scenario_config=Path("/s/x.yaml"),
                        model_cases=("pre", "post-hems-optimized"), timeframe_mode="full_year",
                        min_buildings=60, start_index=3, limit=1)
    argv = synthetic_command(spec, "pre", Path("/runs/j/pre"), python="py")
    assert argv[:4] == ["py", "-m", "gridexpand", "synthetic"]
    joined = " ".join(argv)
    for part in ("--ags 9184137", "--pylovo-version-id 1", "--min-buildings 60", "--scenario-config /s/x.yaml",
                 "--model-case pre", "--profiles status_quo", "--case-qualified-output", "--timeframe-mode full_year",
                 "--powerflow-output summary", "--run-dir /runs/j/pre", "--start-index 3", "--pilot-index 3",
                 "--limit 1"):
        assert part in joined
    steps = pipeline_steps(spec, Path("/runs/j"), python="py")
    assert [s.name for s in steps] == ["pre", "post-hems-optimized"]
    assert "--profiles all" in " ".join(steps[1].argv) and steps[1].run_dir == "/runs/j/post-hems-optimized"
    assert contiguous_range([5, 3, 4]) == (3, 3)
    with pytest.raises(ValueError):
        contiguous_range([1, 3])
    with pytest.raises(ValueError):
        contiguous_range([])
