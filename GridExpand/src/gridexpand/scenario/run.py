#!/usr/bin/env python3
"""``gridexpand run <run.yaml>``: prepare, execute and postprocess one run YAML.

``grids: pylovo`` (synthetic grids) and ``grids: files`` (own grids) run the same
scenario method (:mod:`gridexpand.scenario.grid_run`): prepare = the buildings of
the grids, the scenario dataset, weather, heat and PV libraries; execute = the
per-grid runner per execution group; postprocess = the expansion analyses and one
QGIS view refresh. The real-vs-synthetic validation (``GridExpand/validation``)
reuses the run context, :func:`execute` and :func:`plan_header` for ``pipeline:
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
from gridexpand.optimization.solver import OPTIMIZER_ENV, SOLVER_ENV
from gridexpand.paths import RUNS_DIR
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash, scenario_identity_key
from gridexpand.scenario.run_config import load_run_config, with_cases
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
    from gridexpand.scenario import grid_run

    return plan_header(ctx) + grid_run.plan_lines(ctx)


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
    """Run the stages of ``ctx`` in its run directory with ``runner`` (default: :func:`grid_run.run_grids`)."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    check_identity(ctx.run_dir, run_identity(ctx), info={"run_hash": ctx.run_hash, "git": git_revision()})
    if ctx.scenario_raw is not None:
        write_merged_scenario(ctx.run.scenario_path, ctx.scenario_raw, ctx.scenario_hash)
    freeze_inputs(ctx.run_dir, ctx.run_yaml, ctx.run.scenario_path)
    ctx.state = RunState(ctx.run_dir, ctx.run.run_id, ctx.run.pipeline)
    if ctx.run.optimizer:
        # Every Step 3 job of the run inherits the optimizer and the solver.
        os.environ[OPTIMIZER_ENV] = ctx.run.optimizer
    if getattr(ctx.run, "solver", None):
        os.environ[SOLVER_ENV] = ctx.run.solver
    ctx.state.event(event="run_start", run_hash=ctx.run_hash, scenario_hash=ctx.scenario_hash,
                    model_cases=list(ctx.run.model_cases), resume=ctx.resume, until=ctx.until,
                    optimizer=os.environ.get(OPTIMIZER_ENV) or "urbs")
    try:
        if runner is None:
            from gridexpand.scenario.grid_run import run_grids

            runner = run_grids
        code = runner(ctx)
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
