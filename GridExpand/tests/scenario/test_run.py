"""gridexpand run: plans and commands of every pipeline, run-directory state, CLI (no database)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridexpand.paths import RUN_CONFIG_DIR
from gridexpand.scenario import run as run_module
from gridexpand.scenario.rundir import IdentityMismatch, RunState, check_identity, load_state
from gridexpand.scenario.tools import config_main, status_lines

SANDBOX = RUN_CONFIG_DIR / "sandbox_example.yaml"


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


def test_paired_plan_and_expansion_keys(tmp_path):
    ctx = context(RUN_CONFIG_DIR / "forchheim_2045_paired_full_year.yaml", run_dir=tmp_path)
    jobs = run_module.paired_jobs(ctx)
    assert [(key, run_dir.name) for key, _, run_dir in jobs] == [
        ("heuristic-assets", "heuristic-assets"), ("post-hems-optimized", "post-hems-optimized")]
    heuristic, optimized = (argv for _, argv, _ in jobs)
    assert heuristic[heuristic.index("--model-case") + 1] == "post-hems-heuristic" and "--skip-pre" not in heuristic
    assert "--skip-pre" in optimized and optimized[optimized.index("--scenario-label") + 1] == ctx.run.run_id
    expansion = run_module.paired_expansion_commands(ctx.run)
    keys = [(a[a.index("--analysis-key") + 1], a[a.index("--stage") + 1], a[a.index("--run-name") + 1])
            for a in expansion]
    rid = ctx.run.run_id
    assert keys == [
        (f"{rid}_pre", "pre", f"{rid}_synthetic_pre"),
        (f"{rid}_post_inflex", "post", f"{rid}_synthetic_post-inflex-heuristic"),
        (f"{rid}_post", "post", f"{rid}_synthetic_post-hems-heuristic"),
        (f"{rid}_post_hems_optimized", "post", f"{rid}_synthetic_post-hems-optimized"),
        (f"{rid}_real_pre", "pre", f"{rid}_real_swf_pre"),
        (f"{rid}_real_post_inflex", "post", f"{rid}_real_swf_post-inflex-heuristic"),
        (f"{rid}_real_post", "post", f"{rid}_real_swf_post-hems-heuristic"),
        (f"{rid}_real_post_hems_optimized", "post", f"{rid}_real_swf_post-hems-optimized"),
    ]
    assert expansion[0][expansion[0].index("--ags") + 1] == "9474126"
    assert expansion[4][expansion[4].index("--exclude-real-lv-id") + 1] == "113"
    # a diagnostic single grid gets its own runner directory and no regional expansion
    diagnostic = context(RUN_CONFIG_DIR / "forchheim_2045_paired_full_year.yaml", "--target-grid-id", "80",
                         run_dir=tmp_path)
    assert run_module.paired_jobs(diagnostic)[0][2].name == "heuristic-assets-grid80"
    assert run_module.paired_expansion_commands(diagnostic.run, diagnostic.target_grid_id) == []
    lv080 = context(RUN_CONFIG_DIR / "forchheim_2045_paired_full_year_lv080.yaml")
    assert run_module.paired_expansion_commands(lv080.run) == []
    assert any(line.startswith("[prepare:prepare_allocation]") for line in run_module.plan_lines(ctx))


def test_aligned_plan(tmp_path):
    ctx = context(RUN_CONFIG_DIR / "joint_2045_v1_islands.yaml", "--provider", "uzw", run_dir=tmp_path)
    (provider,) = run_module.aligned_providers(ctx)
    jobs = run_module.aligned_jobs(ctx, provider, tmp_path / "uzw" / "grid_subset.json")
    assert [key for key, _, _ in jobs] == ["uzw/heuristic-assets"]
    argv = jobs[0][1]
    assert argv[argv.index("--workers") + 1] == "2" and argv[argv.index("--provider") + 1] == "uzw"
    assert argv[argv.index("--job-subset") + 1] == str(tmp_path / "uzw" / "grid_subset.json")
    lines = run_module.plan_lines(ctx)
    assert any("grid_subset.json (computed after preparation)" in line for line in lines)
    assert any(line.startswith("[prepare:uzw:pv_library]") and "--reference-year 2009" in line for line in lines)
    assert not (tmp_path / "uzw").exists()  # dry-run planning writes no subset (review-orch B3)
    pre_only = context(RUN_CONFIG_DIR / "joint_2045_v1_islands.yaml", "--pre-only", run_dir=tmp_path)
    keys = [key for p in run_module.aligned_providers(pre_only) for key, _, _ in run_module.aligned_jobs(pre_only, p, None)]
    assert keys == ["swf/pre-only", "uzw/pre-only"]


@pytest.mark.parametrize("argv", [
    [str(SANDBOX), "--provider", "swf"],
    [str(SANDBOX), "--target-grid-id", "3"],
    [str(RUN_CONFIG_DIR / "forchheim_2045_paired_full_year.yaml"), "--pre-only"],
    [str(RUN_CONFIG_DIR / "schweinfurt_2045_synthetic.yaml")],
    [],
])
def test_invalid_configuration_exit_code(argv, capsys):
    assert run_module.main(argv) == run_module.EXIT_INVALID
    assert "invalid configuration" in capsys.readouterr().err


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
    assert config_main(["check", str(SANDBOX), str(RUN_CONFIG_DIR.parent / "scenarios" / "schweinfurt_2045.yaml")]) == 0
    out = capsys.readouterr().out
    assert "run sandbox_example (synthetic)" in out and "scenario_sandbox_2045_7669656db28a" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {id: x, scenario: nope.yaml, pipeline: synthetic}\nresources: {}\nexecution: {}\n")
    assert config_main(["check", str(bad), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)[0]["valid"] is False
