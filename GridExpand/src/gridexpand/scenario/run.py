#!/usr/bin/env python3
"""``gridexpand run <run.yaml>``: prepare, execute and postprocess one run YAML.

``pipeline: synthetic``: prepare = candidate grids + one regional
electrification assignment; execute = one ``gridexpand synthetic`` batch per
execution group (:func:`gridexpand.scenario.synthetic_ags_runner.run_batch`,
including its expansion analyses); postprocess = one QGIS view refresh. The
real-vs-synthetic validation (``GridExpand/validation``) reuses the run
context, :func:`execute` and :func:`plan_header` for ``pipeline:
paired_aligned``.

The run directory (default ``WORK_DIR/runs/<run.id>``) holds frozen inputs,
``identity.json``, ``state.json`` and ``events.jsonl`` (see
:mod:`gridexpand.scenario.rundir`). ``--resume`` skips jobs recorded as done;
SIGTERM/SIGINT cancel the running steps. Exit codes: 0 done, 1 failed,
2 finished with failed jobs, 3 invalid configuration, 143 cancelled.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gridexpand.common.orchestration import CANCEL, Cancelled, install_cancel_handlers, utc_now
from gridexpand.optimization.solver import OPTIMIZER_ENV
from gridexpand.paths import RUNS_DIR
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash, scenario_identity_key
from gridexpand.scenario.model_cases import MODEL_CASES, ExecutionGroup, execution_groups
from gridexpand.scenario.run_config import (
    SyntheticRun,
    load_run_config,
    with_cases,
)
from gridexpand.scenario.rundir import (
    IdentityMismatch,
    RunState,
    check_identity,
    freeze_inputs,
    git_revision,
    write_json_atomic,
)
from gridexpand.scenario.scenario_config import ScenarioConfig

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_JOB_FAILURES = 2
EXIT_INVALID = 3
EXIT_CANCELLED = 143
STAGES = ("prepare", "execute", "postprocess")


@dataclass
class RunContext:
    """One ``gridexpand run`` invocation (the validation adds its own fields in a subclass)."""

    run: Any
    run_hash: str
    run_yaml: Path
    scenario: ScenarioConfig
    scenario_hash: str
    run_dir: Path
    resume: bool = False
    until: str = "postprocess"
    state: RunState | None = None
    readiness: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)
    # The merged scenario of a run with scenario_overrides (written to run.scenario_path by execute).
    scenario_raw: dict[str, Any] | None = None

    def wants(self, stage: str) -> bool:
        return STAGES.index(stage) <= STAGES.index(self.until)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand run",
        description="Run one run YAML (config/runs/*.yaml): prepare, execute and postprocess.",
        epilog="Exit codes: 0 done, 1 failed, 2 finished with failed jobs, 3 invalid configuration, 143 cancelled.",
    )
    parser.add_argument("run_yaml", nargs="?", type=Path, help="Run YAML (config/runs/<run>.yaml).")
    parser.add_argument("--run-config", type=Path, help="Same as the positional run YAML (older form).")
    parser.add_argument("--run-dir", type=Path, help="Run directory (default: WORK_DIR/runs/<run.id>).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip jobs recorded as done (same as execution.resume: true).")
    parser.add_argument("--until", choices=STAGES, default="postprocess", help="Last stage to run.")
    parser.add_argument("--prepare-only", action="store_true", help="Same as --until prepare.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the commands; no database, no files.")
    parser.add_argument("--model-case", action="append", default=None,
                        help="Run only this model case of the YAML (repeatable).")
    return parser


# Synthetic ------------------------------------------------------------------------------


def synthetic_settings(ctx: RunContext, group: ExecutionGroup):
    """The :class:`BatchSettings` of one synthetic execution group."""
    from gridexpand.scenario.synthetic_ags_runner import BatchSettings

    run: SyntheticRun = ctx.run  # type: ignore[assignment]
    case = MODEL_CASES[group.materialization_case]
    return BatchSettings(
        ags=str(run.ags),
        pylovo_version_id=run.pylovo_version_id,
        scenario_config=run.scenario_path,
        scenario=ctx.scenario,
        scenario_hash=ctx.scenario_hash,
        run_dir=ctx.run_dir / group.materialization_case,
        model_case=group.materialization_case,
        profiles=case.profiles,
        min_buildings=run.min_buildings,
        demand_scope=run.demand_scope,
        timeframe_mode=run.timeframe_mode,
        plz=run.plz,
        kcid=run.kcid,
        bcid=run.bcid,
        start_index=run.start_index,
        limit=run.limit,
        workers=run.workers,
        step2_cpus=run.step2_cpus,
        step2_timeseries_storage=run.step2_timeseries_storage,
        step3_cpus=run.step3_cpus,
        step3_max_cpus=run.step3_max_cpus,
        step3_target_columns=run.step3_target_columns,
        step3_cluster_concurrency=run.step3_cluster_concurrency,
        dynamic_step3=run.dynamic_step3,
        step4_cpus=run.step4_cpus,
        solver=run.solver,
        optimizer=run.optimizer,
        powerflow_output=run.powerflow_output,
        powerflow_grid_scope=run.powerflow_grid_scope,
        case_qualified_output=True,
        profile_seed=run.profile_seed,
        electrification_assignment=ctx.run_dir / "prepare" / "electrification_assignment.csv",
        pilot_index=run.pilot_index,
        pilot_gate=run.pilot_gate,
        resume=ctx.resume,
        rerun_failed=run.rerun_failed,
        cleanup_intermediates=run.cleanup_intermediates,
        materialize_expansion=run.materialize_expansion and ctx.wants("postprocess"),
    )


def synthetic_equivalent_command(settings) -> list[str]:
    """The ``gridexpand synthetic`` command line that runs the same batch."""
    argv = [
        "gridexpand", "synthetic",
        "--ags", settings.ags, "--pylovo-version-id", settings.pylovo_version_id,
        "--min-buildings", settings.min_buildings,
        "--workers", settings.workers, "--step2-cpus", settings.step2_cpus,
        "--step3-cpus", settings.step3_cpus, "--step3-max-cpus", settings.step3_max_cpus,
        "--step3-target-columns", settings.step3_target_columns,
        "--step4-cpus", settings.step4_cpus,
        "--scenario-config", settings.scenario_config,
        "--timeframe-mode", settings.timeframe_mode,
        "--demand-scope", settings.demand_scope,
        "--step2-timeseries-storage", settings.step2_timeseries_storage,
        "--powerflow-output", settings.powerflow_output,
        "--powerflow-grid-scope", settings.powerflow_grid_scope,
        "--case-qualified-output",
        "--profile-seed", settings.profile_seed,
        "--model-case", settings.model_case, "--profiles", settings.profiles,
        "--cleanup-intermediates", settings.cleanup_intermediates,
        "--pilot-index", settings.pilot_index,
        "--electrification-assignment", settings.assignment_path,
        "--run-dir", settings.run_dir,
    ]
    for name in ("plz", "kcid", "bcid", "start_index", "limit", "step3_cluster_concurrency", "solver", "optimizer"):
        value = getattr(settings, name)
        if value is not None:
            argv += [f"--{name.replace('_', '-')}", value]
    for enabled, name in ((not settings.pilot_gate, "--no-pilot-gate"), (not settings.dynamic_step3, "--no-dynamic-step3"),
                          (not settings.materialize_expansion, "--no-materialize-expansion"),
                          (settings.resume, "--resume"), (settings.rerun_failed, "--rerun-failed")):
        if enabled:
            argv.append(name)
    return [str(value) for value in argv]


def _mirrored(event: dict[str, Any]) -> dict[str, Any]:
    """A batch event for the run-level log: no command lines, the batch's job as ``grid``."""
    payload = {k: v for k, v in event.items() if k not in ("ts", "cmd", "env_extra", "group")}
    payload["grid"] = payload.pop("job", None)
    return {**payload, "source": "batch"}


def _synthetic_listener(state: RunState, group: str, keys: dict[int, str]) -> Callable[[dict[str, Any]], None]:
    """Mirror a batch's events into the run state (job key ``<group>/<bridge stem>``)."""

    def listener(event: dict[str, Any]) -> None:
        index = event.get("candidate_index")
        key = keys.get(int(index)) if index is not None else None
        state.event(**_mirrored(event), group=group, job=key)
        if key is None:
            return
        kind = event.get("event")
        if kind == "start":
            state.job(key, status="running", step=event.get("stage"))
        elif kind == "candidate_done" or (kind == "pilot_finish" and event.get("status") == "done"):
            state.job(key, status="done", step="complete", seconds=event.get("seconds"))
        elif kind == "candidate_failed":
            state.job(key, status="failed", step=event.get("stage"), message=event.get("message"),
                      seconds=event.get("seconds"))
        elif kind == "candidate_cancelled":
            state.job(key, status="cancelled", step=event.get("stage"))
        elif kind == "candidate_skipped_resume_done":
            state.job(key, status="done", message="done in an earlier invocation")
        elif kind == "candidate_skipped_resume_failed":
            state.job(key, status="failed", message="failed in an earlier invocation (use rerun_failed)")

    return listener


def run_synthetic(ctx: RunContext) -> int:
    from gridexpand.scenario import synthetic_ags_runner as runner

    run: SyntheticRun = ctx.run  # type: ignore[assignment]
    state = ctx.state
    groups = execution_groups(run.model_cases)
    settings = {group.materialization_case: synthetic_settings(ctx, group) for group in groups}
    for group_settings in settings.values():
        runner.check_settings(group_settings)
    with state.stage_context("prepare"):
        first = next(iter(settings.values()))
        candidates = runner.load_candidates(first)
        write_json_atomic(ctx.run_dir / "candidates.json", candidates)
        if not candidates:
            raise ValueError(f"No candidate grids for AGS {run.ags} (filters plz={run.plz}, kcid={run.kcid}, "
                             f"bcid={run.bcid}, min_buildings={run.min_buildings}, pylovo {run.pylovo_version_id}).")
        post = [s for s in settings.values() if s.profiles != "status_quo"]
        if post:
            prepare_settings = post[0].replace(run_dir=ctx.run_dir / "prepare", resume=True)
            (prepare_settings.run_dir / "logs").mkdir(parents=True, exist_ok=True)
            status = runner.StatusLog(prepare_settings.run_dir,
                                      listener=lambda event: state.event(**_mirrored(event), group="prepare"))
            runner.ensure_electrification_assignment(prepare_settings, candidates, status)
        selected = candidates
        if run.start_index is not None:
            selected = [c for c in selected if int(c["candidate_index"]) >= run.start_index]
        if run.limit is not None:
            selected = selected[: run.limit]
        jobs = [
            {"job": f"{group}/{runner.job_key(c)}", "group": group, "candidate_index": int(c["candidate_index"]),
             "plz": c["plz"], "kcid": c["kcid"], "bcid": c["bcid"], "n_buildings": c["n_buildings"],
             "log": f"{group}/logs/candidate_{int(c['candidate_index']):03d}_{runner.step2_filename(c, s)}.log"}
            for group, s in settings.items() for c in selected
        ]
        # The batches resume by grid themselves (status.tsv of each group); the state keeps done records.
        state.plan(jobs, resume=ctx.resume)
        write_json_atomic(ctx.run_dir / "plan.json", {
            "pipeline": run.pipeline, "groups": [g.__dict__ for g in groups],
            "commands": {group: synthetic_equivalent_command(s) for group, s in settings.items()},
            "jobs": [job["job"] for job in jobs],
        })
    if not ctx.wants("execute"):
        return EXIT_OK

    codes: dict[str, int] = {}
    with state.stage_context("execute") as record:
        for group, group_settings in settings.items():
            keys = {int(c["candidate_index"]): f"{group}/{runner.job_key(c)}" for c in candidates}
            print(f"[execute] {group}: {' '.join(synthetic_equivalent_command(group_settings))}", flush=True)
            codes[group] = runner.run_batch(
                group_settings, candidates=candidates, listener=_synthetic_listener(state, group, keys),
                refresh_qgis_views=False,
            )
            ctx.groups[group] = _read_json(group_settings.run_dir / "summary.json")
            if CANCEL.is_set() or codes[group] == runner.EXIT_CANCELLED:
                raise Cancelled(f"cancelled during {group}")
            if codes[group] == 1:
                record["result"] = "failed"
                break
    if any(code == 1 for code in codes.values()):
        return EXIT_FAILED
    if ctx.wants("postprocess"):
        with state.stage_context("postprocess"):
            if any((summary or {}).get("materialized_expansion") for summary in ctx.groups.values()):
                from gridexpand.db import refresh_qgis_views

                refresh_qgis_views()
                state.event(event="qgis_views_refreshed")
    return EXIT_JOB_FAILURES if any(code == 2 for code in codes.values()) else EXIT_OK


# Dry run ----------------------------------------------------------------------------------


def plan_header(ctx: RunContext) -> list[str]:
    """The first two dry-run lines: run, pipeline, hashes, scenario, pylovo version, run directory."""
    run = ctx.run
    return [
        f"run {run.run_id} · pipeline {run.pipeline} · run_hash {ctx.run_hash[:12]} · "
        f"scenario {ctx.scenario.scenario_id} ({ctx.scenario_hash[:12]}) · pylovo {run.pylovo_version_id}",
        f"run directory: {ctx.run_dir}",
    ]


def plan_lines(ctx: RunContext) -> list[str]:
    """Human-readable plan: groups and the commands each stage would run (no DB, no files)."""
    run: SyntheticRun = ctx.run
    lines = plan_header(ctx)
    groups = execution_groups(run.model_cases)
    region = f"AGS {run.ags}" + (f", PLZ {run.plz}" if run.plz is not None else "") + (
        f", grid {run.kcid}/{run.bcid}" if run.kcid is not None else "")
    lines.append(f"[prepare] candidate grids of {region} (min {run.min_buildings} buildings); "
                 "regional electrification assignment" if any(g.materialization_case != "pre" for g in groups)
                 else f"[prepare] candidate grids of {region} (min {run.min_buildings} buildings)")
    for group in groups:
        lines.append(f"[execute:{group.materialization_case}] "
                     + " ".join(synthetic_equivalent_command(synthetic_settings(ctx, group))))
    lines.append("[postprocess] refresh the QGIS views")
    return lines


# Main ---------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_context(args: argparse.Namespace) -> RunContext:
    """Parse and validate everything (no database); raises ValueError on invalid input."""
    run_yaml = args.run_yaml or args.run_config
    if run_yaml is None:
        raise ValueError("Give the run YAML: gridexpand run config/runs/<run>.yaml")
    run, run_hash = load_run_config(run_yaml)
    if args.model_case:
        run = with_cases(run, tuple(args.model_case))
    run_dir = (args.run_dir or RUNS_DIR / run.run_id).resolve()
    raw = run.scenario_raw()
    scenario, scenario_hash = ScenarioConfig.from_dict(raw), configuration_hash(raw)
    if run.scenario_overrides:
        # Steps 2-4 read the merged scenario from the run directory (written by execute).
        run = dataclasses.replace(run, scenario_path=run_dir / "inputs" / "scenario.yaml")
    ctx = RunContext(
        run=run,
        run_hash=run_hash,
        run_yaml=Path(run_yaml).resolve(),
        scenario=scenario,
        scenario_hash=scenario_hash,
        run_dir=run_dir,
        resume=bool(args.resume or run.resume),
        until="prepare" if args.prepare_only else args.until,
        scenario_raw=raw if run.scenario_overrides else None,
    )
    return ctx


def write_merged_scenario(path: Path, raw: dict[str, Any], scenario_hash: str) -> None:
    """Write the merged scenario of a run with overrides; its hash must stay the one the run was planned with."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# The run's scenario: the named scenario with the run YAML's scenario_overrides merged in.\n"
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if configuration_hash(_read_yaml(path)) != scenario_hash:
        raise ValueError(f"The merged scenario {path} does not reproduce its hash; check scenario_overrides.")


def run_identity(ctx: RunContext) -> dict[str, Any]:
    return {
        **ctx.run.identity(),
        "scenario_id": ctx.scenario.scenario_id,
        "scenario_hash": ctx.scenario_hash,
        "scenario_key": scenario_identity_key(ctx.scenario.scenario_id, ctx.scenario_hash),
    }


def execute(ctx: RunContext, runner: Callable[[RunContext], int] | None = None) -> int:
    """Run the stages of ``ctx`` in its run directory with ``runner`` (default: the synthetic pipeline)."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    check_identity(ctx.run_dir, run_identity(ctx), info={"run_hash": ctx.run_hash, "git": git_revision()})
    if ctx.scenario_raw is not None:
        write_merged_scenario(ctx.run.scenario_path, ctx.scenario_raw, ctx.scenario_hash)
    freeze_inputs(ctx.run_dir, ctx.run_yaml, ctx.run.scenario_path)
    ctx.state = RunState(ctx.run_dir, ctx.run.run_id, ctx.run.pipeline)
    if ctx.run.optimizer:
        # Every Step 3 job of the run inherits the optimizer (like GRIDEXPAND_SOLVER).
        os.environ[OPTIMIZER_ENV] = ctx.run.optimizer
    ctx.state.event(event="run_start", run_hash=ctx.run_hash, scenario_hash=ctx.scenario_hash,
                    model_cases=list(ctx.run.model_cases), resume=ctx.resume, until=ctx.until,
                    optimizer=os.environ.get(OPTIMIZER_ENV) or "urbs")
    try:
        code = (runner or run_synthetic)(ctx)
        status = {EXIT_OK: "done", EXIT_JOB_FAILURES: "completed_with_failures"}.get(code, "failed")
    except Cancelled:
        code, status = EXIT_CANCELLED, "cancelled"
    except Exception as exc:  # recorded in state.json and summary.json, then re-raised
        ctx.state.event(event="run_error", message=str(exc))
        _finish(ctx, "failed", EXIT_FAILED, error=str(exc))
        raise
    if CANCEL.is_set():
        code, status = EXIT_CANCELLED, "cancelled"
    _finish(ctx, status, code)
    return code


def _finish(ctx: RunContext, status: str, code: int, error: str | None = None) -> None:
    ctx.state.finish(status, code)
    write_json_atomic(ctx.run_dir / "summary.json", {
        "run_id": ctx.run.run_id,
        "pipeline": ctx.run.pipeline,
        "status": status,
        "exit_code": code,
        "error": error,
        "jobs": ctx.state.counts(),
        "stages": ctx.state.stages,
        "readiness": ctx.readiness,
        "groups": ctx.groups,
        "finished_at": utc_now(),
    })
    print(f"[run] {ctx.run.run_id}: {status} (exit {code}) · jobs {ctx.state.counts()} · {ctx.run_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctx = load_context(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"gridexpand run: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if args.dry_run:
        print("\n".join(plan_lines(ctx)))
        return EXIT_OK
    install_cancel_handlers()
    try:
        return execute(ctx)
    except IdentityMismatch as exc:
        print(f"gridexpand run: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
