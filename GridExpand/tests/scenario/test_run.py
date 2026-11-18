"""gridexpand run: plans and commands of the synthetic pipeline, run-directory state, CLI (no database)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixture_paths import FIXTURE_SCENARIOS, FIXTURE_RUNS
from gridexpand.paths import RUN_CONFIG_DIR
from gridexpand.scenario import run as run_module
from gridexpand.scenario.rundir import IdentityMismatch, RunState, check_identity, load_state
from gridexpand.scenario.tools import config_main, status_lines

SANDBOX = FIXTURE_RUNS / "sandbox_example.yaml"


def context(path: Path, *extra: str, run_dir: Path | None = None):
    argv = [str(path), *extra]
    if run_dir is not None:
        argv += ["--run-dir", str(run_dir)]
    return run_module.load_context(run_module.build_parser().parse_args(argv))


def test_synthetic_groups_match_the_harness_commands(tmp_path):
    ctx = context(SANDBOX, run_dir=tmp_path)
    groups = run_module.execution_groups(ctx.run.model_cases)
    commands = [run_module.synthetic_equivalent_command(run_module.synthetic_settings(ctx, g)) for g in groups]
    harness = ("--ags 9184137 --pylovo-version-id 1 --min-buildings 60 --workers 2 --step2-cpus 2 --step3-cpus 4 "
               "--step3-max-cpus 8").split()
    for argv, case, profiles in zip(commands, ("pre", "post-hems-heuristic", "post-hems-optimized"),
                                    ("status_quo", "all", "all")):
        joined = " ".join(argv)
        assert " ".join(harness) in joined
        for part in ("--step4-cpus 2", "--timeframe-mode max_base_electricity_demand_week", "--powerflow-output both",
                     "--case-qualified-output", "--no-pilot-gate", "--profile-seed 481527",
                     f"--model-case {case}", f"--profiles {profiles}", f"--run-dir {tmp_path / case}",
                     f"--electrification-assignment {tmp_path / 'prepare' / 'electrification_assignment.csv'}"):
            assert part in joined, part
    lines = run_module.plan_lines(ctx)
    assert lines[2].startswith("[prepare] candidate grids of AGS 9184137") and "electrification" in lines[2]
    assert not any(tmp_path.iterdir())  # planning writes nothing


def test_until_execute_skips_the_expansion(tmp_path):
    ctx = context(SANDBOX, "--until", "execute", run_dir=tmp_path)
    (group,) = run_module.execution_groups(("pre",))
    assert not run_module.synthetic_settings(ctx, group).materialize_expansion


@pytest.mark.parametrize("argv", [
    [],
])
def test_invalid_configuration_exit_code(argv, capsys):
    assert run_module.main(argv) == run_module.EXIT_INVALID
    assert "invalid configuration" in capsys.readouterr().err


def test_synthetic_run_refuses_inflex_before_database_access(tmp_path, capsys):
    config = tmp_path / "synthetic_inflex.yaml"
    config.write_text(f"run: inflex\nscenario: {FIXTURE_SCENARIOS / 'teaser_2045.yaml'}\n"
                      "grids: {pylovo: {version: '1', ags: 9184137}}\nmodel_cases: [pre, post-inflex-heuristic]\n")
    assert run_module.main([str(config)]) == run_module.EXIT_INVALID
    assert "post-inflex-heuristic is not available for the synthetic pipeline" in capsys.readouterr().err


def test_scenario_overrides_are_merged_and_written_into_the_run_directory(tmp_path):
    import yaml

    from gridexpand.scenario.config_loader import load_scenario_config

    config = tmp_path / "run.yaml"
    config.write_text("run: tweak\nscenario: default_2045\ngrids: {pylovo: {version: '1', ags: 9184137}}\n"
                      "scenario_overrides: {electrification: {heat: {building_share: 0.9}}}\n")
    _, default_hash = load_scenario_config(RUN_CONFIG_DIR.parent / "scenarios" / "default_2045.yaml")
    ctx = context(config, run_dir=tmp_path / "run")
    assert ctx.scenario.electrification.for_technology("heat").building_share == 0.9
    assert ctx.scenario_hash != default_hash and ctx.run.scenario_path == tmp_path / "run" / "inputs" / "scenario.yaml"
    assert not (tmp_path / "run").exists()  # planning writes nothing
    run_module.write_merged_scenario(ctx.run.scenario_path, ctx.scenario_raw, ctx.scenario_hash)
    written, written_hash = load_scenario_config(ctx.run.scenario_path)
    assert written_hash == ctx.scenario_hash and yaml.safe_load(ctx.run.scenario_path.read_text())["scenario"]["id"] \
        == "default_2045"


def test_the_default_run_yaml(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("run: plain\nscenario: default_2045\ngrids: {pylovo: {version: '1', ags: 9184137}}\n")
    ctx = context(config, run_dir=tmp_path / "run")
    assert ctx.run.model_cases == ("pre", "post-hems-heuristic", "post-hems-optimized")
    assert ctx.run.scenario_path == RUN_CONFIG_DIR.parent / "scenarios" / "default_2045.yaml"
    assert ctx.scenario_raw is None


@pytest.mark.parametrize("text, message", [
    ("run: {id: x, scenario: s.yaml, pipeline: synthetic}\nresources: {}\nexecution: {}\n", "former format"),
    ("run: x\nscenario: nope\ngrids: {pylovo: {version: '1', ags: 1}}\n", "not found"),
    ("run: x\nscenario: default_2045\ngrids: {files: grids}\n", "in preparation"),
    ("run: x\nscenario: default_2045\ngrids: {pylovo: {ags: 1}}\n", "grids.pylovo.version"),
    ("run: x\nscenario: default_2045\ngrids: {pylovo: {version: '1', ags: 1}}\nresources: {}\n", "former format"),
])
def test_run_yaml_errors(tmp_path, text, message):
    from gridexpand.scenario.run_config import load_run_config

    config = tmp_path / "run.yaml"
    config.write_text(text)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_run_config(config)


def test_dry_run_exits_zero(capsys):
    assert run_module.main([str(SANDBOX), "--dry-run"]) == 0
    assert "[execute:post-hems-optimized] gridexpand synthetic" in capsys.readouterr().out


def test_run_state_resume_and_stages(tmp_path):
    state = RunState(tmp_path, "r", "synthetic")
    todo = state.plan([{"job": "pre/a", "group": "pre"}, {"job": "pre/b", "group": "pre"}], resume=False)
    assert todo == ["pre/a", "pre/b"]
    with state.stage_context("execute"):
        state.job("pre/a", status="running", step="step2")
        assert load_state(tmp_path)["running"][0]["job"] == "pre/a"
        state.job("pre/a", status="done")
        state.job("pre/b", status="failed", message="boom")
    snapshot = load_state(tmp_path)
    assert snapshot["jobs"]["done"] == 1 and snapshot["jobs"]["failed"] == 1 and snapshot["stages"][0]["status"] == "done"
    again = RunState(tmp_path, "r", "synthetic")  # a later invocation keeps earlier jobs
    assert again.plan([{"job": "pre/a"}, {"job": "pre/b"}, {"job": "opt/a"}], resume=True) == ["pre/b", "opt/a"]
    with pytest.raises(RuntimeError):
        with again.stage_context("postprocess"):
            raise RuntimeError("x")
    assert load_state(tmp_path)["stages"][-1]["status"] == "failed"
    again.finish("cancelled", 143)
    final = load_state(tmp_path)
    assert final["status"] == "cancelled" and final["jobs"]["cancelled"] == 2 and final["exit_code"] == 143
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {e["event"] for e in events} >= {"stage_start", "stage_finish", "run_finish"}
    assert all(e["schema"] == 1 and e["run_id"] == "r" for e in events)
    assert "cancelled" in status_lines(final, alive=False)[0]


def test_identity_guard(tmp_path):
    check_identity(tmp_path, {"pipeline": "synthetic", "region": {"ags": 1}})
    check_identity(tmp_path, {"pipeline": "synthetic", "region": {"ags": 1}})
    with pytest.raises(IdentityMismatch, match="belongs to another run"):
        check_identity(tmp_path, {"pipeline": "synthetic", "region": {"ags": 2}})


def test_config_check(tmp_path, capsys):
    assert config_main(["check", str(SANDBOX), str(FIXTURE_SCENARIOS / "teaser_2045.yaml")]) == 0
    out = capsys.readouterr().out
    assert "run sandbox_example (synthetic)" in out and "scenario_sandbox_2045_7669656db28a" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: x\nscenario: nope.yaml\ngrids: {pylovo: {version: '1', ags: 1}}\n")
    assert config_main(["check", str(bad), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)[0]["valid"] is False
