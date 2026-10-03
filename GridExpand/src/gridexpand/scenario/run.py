#!/usr/bin/env python3
"""``gridexpand run <run.yaml>``: prepare, execute and postprocess one run YAML.

Pipelines (``run.pipeline``):

- ``synthetic``: prepare = candidate grids + one regional electrification
  assignment; execute = one ``gridexpand synthetic`` batch per execution group
  (:func:`gridexpand.scenario.synthetic_ags_runner.run_batch`, including its
  expansion analyses); postprocess = one QGIS view refresh.
- ``paired_validation`` / ``paired_aligned``: prepare = build (unless
  ``resume``/``--skip-prepare``) and validate the paired datasets; execute = one
  paired-runner process per execution group (and provider); postprocess =
  expansion analyses when every job succeeded.

The run directory (default ``WORK_DIR/runs/<run.id>``) holds frozen inputs,
``identity.json``, ``state.json`` and ``events.jsonl`` (see
:mod:`gridexpand.scenario.rundir`). ``--resume`` skips jobs recorded as done;
SIGTERM/SIGINT cancel the running steps. Exit codes: 0 done, 1 failed,
2 finished with failed jobs, 3 invalid configuration, 143 cancelled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gridexpand.common.orchestration import CANCEL, Cancelled, install_cancel_handlers, run_step, utc_now
from gridexpand.optimization.solver import OPTIMIZER_ENV
from gridexpand.paths import RUNS_DIR
from gridexpand.scenario import commands
from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key
from gridexpand.scenario.model_cases import MODEL_CASES, ExecutionGroup, execution_groups
from gridexpand.scenario.run_config import (
    AlignedRun,
    PairedRun,
    ProviderResources,
    RunConfig,
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
    """One ``gridexpand run`` invocation."""

    run: RunConfig
    run_hash: str
    run_yaml: Path
    scenario: ScenarioConfig
    scenario_hash: str
    run_dir: Path
    resume: bool = False
    until: str = "postprocess"
    skip_prepare: bool = False
    provider: str | None = None
    target_grid_id: int | None = None
    pre_only: bool = False
    state: RunState | None = None
    readiness: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)

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
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Paired pipelines: use the prepared datasets as they are (validated).")
    parser.add_argument("--provider", choices=("swf", "uzw"), help="paired_aligned: one provider only.")
    parser.add_argument("--target-grid-id", type=int,
                        help="Paired pipelines: diagnostic single-grid filter (own run sub-directory).")
    parser.add_argument("--pre-only", action="store_true",
                        help="paired_aligned: Step 2 + Step-4 pre only, no URBS.")
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


# Paired (SWF) -----------------------------------------------------------------------------


def _group_dir(base: Path, group: str, target_grid_id: int | None) -> Path:
    """Run directory of one paired group; diagnostic single-grid runs get their own."""
    return base / (group if target_grid_id is None else f"{group}-grid{target_grid_id}")


def paired_jobs(ctx: RunContext) -> list[tuple[str, list[str], Path]]:
    """``(job key, argv, runner run dir)`` of a ``paired_validation`` run."""
    run: PairedRun = ctx.run  # type: ignore[assignment]
    target_grid_id = ctx.target_grid_id if ctx.target_grid_id is not None else run.target_grid_id
    jobs = []
    for group in execution_groups(run.model_cases):
        run_dir = _group_dir(ctx.run_dir, group.name, ctx.target_grid_id)
        jobs.append((group.name, commands.paired_runner_command(
            paired_dataset_id=run.paired_dataset_id,
            pylovo_version_id=run.pylovo_version_id,
            scenario_config=run.scenario_path,
            target=run.target_network,
            workers=run.workers,
            step3_cpus=run.step3_cpus,
            step3_cluster_concurrency=run.step3_cluster_concurrency,
            step4_cpus=run.step4_cpus,
            powerflow_grid_scope=run.powerflow_grid_scope,
            profile_seed=run.profile_seed,
            scenario_label=run.run_id,
            run_name_prefix=run.run_id,
            run_dir=run_dir,
            materialization_case=group.materialization_case,
            result_cases=group.result_cases,
            target_grid_id=target_grid_id,
            cleanup_intermediates=run.cleanup_intermediates,
            resume=ctx.resume,
            skip_pre=not group.emits_pre,
        ), run_dir))
    return jobs


def paired_expansion_commands(run: PairedRun, target_grid_id: int | None = None) -> list[list[str]]:
    """Expansion analyses ``<run_id>[_real]_<analysis suffix>`` of a paired run (full grid set only)."""
    if not run.materialize_expansion or run.target_grid_id is not None or target_grid_id is not None:
        return []
    targets = {"synthetic": ("synthetic",), "real_swf": ("real_swf",)}.get(run.target_network, ("synthetic", "real_swf"))
    argvs = []
    for target in targets:
        source_suffix = "" if target == "synthetic" else "_real"
        for case_name in ("pre", *run.model_cases):
            case = MODEL_CASES[case_name]
            argvs.append(commands.expansion_command(
                f"{run.run_id}_{target}_{case_name}",
                data_source=target,
                stage=case.stage,
                analysis_key=f"{run.run_id}{source_suffix}_{case.analysis_suffix}",
                ags=run.ags if target == "synthetic" else None,
                plz=None if target == "synthetic" else run.plz,
                exclude_real_lv_ids=() if target == "synthetic" else run.excluded_real_lv_ids,
            ))
    return argvs


def paired_preparation(ctx: RunContext) -> list[tuple[str, list[str]]]:
    run: PairedRun = ctx.run  # type: ignore[assignment]
    return commands.paired_preparation_commands(
        ags=run.ags, plz=run.plz, milestone_year=ctx.scenario.milestone_year,
        pylovo_version_id=run.pylovo_version_id, min_buildings=run.min_buildings,
        scenario_config=run.scenario_path, profile_seed=run.profile_seed, paired_dir=run.paired_dir,
        heat_library=run.heat_library, weather_hdf=run.weather_hdf,
        reference_year=ctx.scenario.mobility.reference_year,
    )


def _run_prepare_commands(ctx: RunContext, steps: list[tuple[str, list[str]]], prefix: str = "",
                          env_extra: dict[str, str] | None = None) -> None:
    for stage, argv in steps:
        name = f"{prefix}{stage}"
        print(f"[prepare:{name}] {' '.join(argv)}", flush=True)
        ctx.state.event(event="prepare_step_start", step=name, cmd=argv)
        code, seconds = run_step(argv, log_path=ctx.run_dir / "logs" / f"prepare_{name}.log",
                                 env_extra=env_extra, echo=True, header=name)
        ctx.state.event(event="prepare_step_finish", step=name, returncode=code, seconds=seconds)
        if code != 0:
            raise RuntimeError(f"Preparation step {name} failed with return code {code}.")


def _run_job(ctx: RunContext, key: str, argv: list[str], env_extra: dict[str, str] | None = None) -> int:
    print(f"[execute:{key}] {' '.join(argv)}", flush=True)
    ctx.state.job(key, status="running", step="paired_runner")
    try:
        code, seconds = run_step(argv, log_path=ctx.run_dir / "jobs" / key.replace("/", "_") / "log.txt",
                                 env_extra=env_extra, echo=True, header=key)
    except Cancelled:
        ctx.state.job(key, status="cancelled")
        raise
    ctx.state.job(key, status="done" if code == 0 else "failed", exit_code=code, seconds=seconds,
                  message=None if code == 0 else f"exit code {code}")
    return code


def _postprocess_commands(ctx: RunContext, argvs: list[list[str]], *, refresh: bool) -> None:
    for argv in argvs:
        print(f"[postprocess] {' '.join(argv)}", flush=True)
        code, _ = run_step(argv, log_path=ctx.run_dir / "logs" / "postprocess_expansion.log", echo=True,
                           header="expansion")
        if code != 0:
            raise RuntimeError(f"Expansion materialization failed with return code {code}: {' '.join(argv)}")
    if argvs and refresh:
        from gridexpand.db import refresh_qgis_views

        refresh_qgis_views()
        ctx.state.event(event="qgis_views_refreshed")


def run_paired(ctx: RunContext) -> int:
    from gridexpand.paired.datasets import validate_prepared_dataset

    run: PairedRun = ctx.run  # type: ignore[assignment]
    state = ctx.state
    jobs = paired_jobs(ctx)
    with state.stage_context("prepare"):
        if not (ctx.resume or ctx.skip_prepare):
            _run_prepare_commands(ctx, paired_preparation(ctx))
        ctx.readiness["paired"] = validate_prepared_dataset(
            run.paired_dataset_id, pylovo_version_id=run.pylovo_version_id, scenario_hash=ctx.scenario_hash,
            ags=run.ags, plz=run.plz, electrification=ctx.scenario.electrification,
            profile_seed=run.profile_seed, require_publication_ready=True,
        )
        print(f"[validate] {json.dumps(ctx.readiness['paired'], sort_keys=True)}", flush=True)
        state.event(event="dataset_validated", readiness=ctx.readiness["paired"])
        todo = set(state.plan([{"job": key, "group": key, "run_dir": str(run_dir)} for key, _, run_dir in jobs],
                              resume=ctx.resume))
    if not ctx.wants("execute"):
        return EXIT_OK
    with state.stage_context("execute"):
        for key, argv, _ in jobs:
            if key not in todo:
                continue
            if _run_job(ctx, key, argv) != 0:
                return EXIT_FAILED
    if ctx.wants("postprocess"):
        with state.stage_context("postprocess"):
            _postprocess_commands(ctx, paired_expansion_commands(run, ctx.target_grid_id), refresh=True)
    return EXIT_OK


# Paired aligned (SWF + ÜZW) ------------------------------------------------------------------


def aligned_providers(ctx: RunContext) -> list[ProviderResources]:
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    return [p for p in run.providers if ctx.provider in (None, p.provider)]


def aligned_preparation(ctx: RunContext, provider: ProviderResources) -> list[tuple[str, list[str]]]:
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    return commands.aligned_preparation_commands(
        provider=provider.provider, alignment_dir=run.alignment_dir, population=run.population,
        uzw_grids_dir=run.uzw_grids_dir, pylovo_version_id=run.pylovo_version_id,
        scenario_config=run.scenario_path, profile_seed=run.profile_seed, paired_dir=provider.paired_dir,
        weather_hdf=provider.weather_hdf, heat_sources=provider.heat_sources, heat_library=provider.heat_library,
        heat_profile_set_id=provider.heat_profile_set_id, heat_workers=run.heat_workers,
        reference_year=ctx.scenario.mobility.reference_year, weather_year=run.weather_year,
    )


def aligned_jobs(ctx: RunContext, provider: ProviderResources, subset_path: Path | None) -> list[tuple[str, list[str], Path]]:
    """``(job key, argv, runner run dir)`` of one provider of a ``paired_aligned`` run."""
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    root = ctx.run_dir / provider.provider
    common = dict(
        paired_dataset_id=provider.paired_dataset_id,
        pylovo_version_id=run.pylovo_version_id,
        provider=provider.provider,
        scenario_config=run.scenario_path,
        target=run.target_for(provider),
        workers=provider.workers or run.workers,
        step3_cpus=run.step3_cpus,
        step3_cluster_concurrency=run.step3_cluster_concurrency,
        step4_cpus=run.step4_cpus,
        powerflow_grid_scope=run.powerflow_grid_scope,
        profile_seed=run.profile_seed,
        scenario_label=f"{run.run_id}_{provider.provider}",
        run_name_prefix=f"{run.run_id}_{provider.provider}",
        max_timesteps=run.powerflow_max_timesteps,
        job_subset=subset_path,
        target_grid_id=ctx.target_grid_id,
        cleanup_intermediates=run.cleanup_intermediates,
        resume=ctx.resume,
    )
    if ctx.pre_only:
        run_dir = _group_dir(root, "pre-only", ctx.target_grid_id)
        return [(f"{provider.provider}/pre-only", commands.paired_runner_command(**common, pre_only=True,
                                                                              run_dir=run_dir), run_dir)]
    jobs = []
    for group in execution_groups(run.model_cases):
        run_dir = _group_dir(root, group.name, ctx.target_grid_id)
        jobs.append((f"{provider.provider}/{group.name}", commands.paired_runner_command(
            **common, run_dir=run_dir, materialization_case=group.materialization_case,
            result_cases=group.result_cases, skip_pre=not group.emits_pre,
        ), run_dir))
    return jobs


def run_aligned(ctx: RunContext) -> int:
    from gridexpand.paired.aligned import select_grid_subset
    from gridexpand.paired.datasets import validate_prepared_dataset

    run: AlignedRun = ctx.run  # type: ignore[assignment]
    state = ctx.state
    providers = aligned_providers(ctx)
    env = {"PYLOVO_VERSION_ID": run.pylovo_version_id}
    jobs: dict[str, list[tuple[str, list[str], Path]]] = {}
    with state.stage_context("prepare"):
        for provider in providers:
            if not (ctx.resume or ctx.skip_prepare):
                _run_prepare_commands(ctx, aligned_preparation(ctx, provider), f"{provider.provider}:", env)
            ctx.readiness[provider.provider] = validate_prepared_dataset(
                provider.paired_dataset_id, pylovo_version_id=run.pylovo_version_id,
                scenario_hash=ctx.scenario_hash, provider=provider.provider, require_exact_heat=True,
                weather_hdf=provider.weather_hdf,
            )
            print(f"[{provider.provider}:validate] {json.dumps(ctx.readiness[provider.provider], sort_keys=True)}",
                  flush=True)
            subset_path = None
            if run.grid_subset is not None:
                # After preparation (review-orch B3): the subset needs the registered grids.
                subset_path = ctx.run_dir / provider.provider / "grid_subset.json"
                write_json_atomic(subset_path, select_grid_subset(
                    population=run.population, provider=provider.provider, paired_dir=provider.paired_dir,
                    grid_subset=run.grid_subset,
                ))
            jobs[provider.provider] = aligned_jobs(ctx, provider, subset_path)
        todo = set(state.plan(
            [{"job": key, "group": key, "run_dir": str(run_dir)} for items in jobs.values() for key, _, run_dir in items],
            resume=ctx.resume,
        ))
    if not ctx.wants("execute"):
        return EXIT_OK

    def execute(provider: ProviderResources) -> int:
        code = 0
        for key, argv, _ in jobs[provider.provider]:
            if key in todo:
                code = max(code, _run_job(ctx, key, argv, env))
        return code

    with state.stage_context("execute"):
        if run.parallel_providers and len(providers) > 1:
            with ThreadPoolExecutor(max_workers=len(providers)) as pool:
                codes = list(pool.map(execute, providers))
        else:
            codes = [execute(provider) for provider in providers]
    if any(codes):
        return EXIT_FAILED
    if ctx.wants("postprocess") and run.materialize_expansion and not ctx.pre_only and ctx.target_grid_id is None:
        with state.stage_context("postprocess"):
            # aligned_expansion refreshes the QGIS views once itself.
            _postprocess_commands(ctx, [commands.aligned_expansion_command(
                run.run_id, providers=[p.provider for p in providers], cases=("pre", *run.model_cases),
                pylovo_version_id=run.pylovo_version_id,
            )], refresh=False)
    return EXIT_OK


# Dry run ----------------------------------------------------------------------------------


def plan_lines(ctx: RunContext) -> list[str]:
    """Human-readable plan: groups and the commands each stage would run (no DB, no files)."""
    run = ctx.run
    lines = [
        f"run {run.run_id} · pipeline {run.pipeline} · run_hash {ctx.run_hash[:12]} · "
        f"scenario {ctx.scenario.scenario_id} ({ctx.scenario_hash[:12]}) · pylovo {run.pylovo_version_id}",
        f"run directory: {ctx.run_dir}",
    ]
    if isinstance(run, SyntheticRun):
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
    elif isinstance(run, PairedRun):
        if not (ctx.resume or ctx.skip_prepare):
            lines += [f"[prepare:{stage}] {' '.join(argv)}" for stage, argv in paired_preparation(ctx)]
        lines.append(f"[prepare:validate] dataset {run.paired_dataset_id}")
        lines += [f"[execute:{key}] {' '.join(argv)}" for key, argv, _ in paired_jobs(ctx)]
        lines += [f"[postprocess] {' '.join(argv)}" for argv in paired_expansion_commands(run, ctx.target_grid_id)]
    else:
        for provider in aligned_providers(ctx):
            if not (ctx.resume or ctx.skip_prepare):
                lines += [f"[prepare:{provider.provider}:{stage}] {' '.join(argv)}"
                          for stage, argv in aligned_preparation(ctx, provider)]
            lines.append(f"[prepare:{provider.provider}:validate] dataset {provider.paired_dataset_id}")
            subset = None
            if run.grid_subset is not None:
                subset = ctx.run_dir / provider.provider / "grid_subset.json"
                lines.append(f"[prepare:{provider.provider}:grid_subset] {subset} (computed after preparation)")
            lines += [f"[execute:{key}] {' '.join(argv)}" for key, argv, _ in aligned_jobs(ctx, provider, subset)]
        if run.materialize_expansion and not ctx.pre_only and ctx.target_grid_id is None:
            lines.append("[postprocess] " + " ".join(commands.aligned_expansion_command(
                run.run_id, providers=[p.provider for p in aligned_providers(ctx)],
                cases=("pre", *run.model_cases), pylovo_version_id=run.pylovo_version_id)))
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
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    if args.provider is not None and not isinstance(run, AlignedRun):
        raise ValueError("--provider applies to paired_aligned runs only.")
    if args.pre_only and not isinstance(run, AlignedRun):
        raise ValueError("--pre-only applies to paired_aligned runs only.")
    if args.target_grid_id is not None and isinstance(run, SyntheticRun):
        raise ValueError("--target-grid-id applies to paired runs; synthetic runs select grids in resources.")
    if args.skip_prepare and isinstance(run, SyntheticRun):
        raise ValueError("--skip-prepare applies to paired runs only.")
    if isinstance(run, AlignedRun) and args.provider is not None:
        run.provider(args.provider)
    ctx = RunContext(
        run=run,
        run_hash=run_hash,
        run_yaml=Path(run_yaml).resolve(),
        scenario=scenario,
        scenario_hash=scenario_hash,
        run_dir=(args.run_dir or RUNS_DIR / run.run_id).resolve(),
        resume=bool(args.resume or run.resume),
        until="prepare" if args.prepare_only else args.until,
        skip_prepare=args.skip_prepare,
        provider=args.provider,
        target_grid_id=args.target_grid_id,
        pre_only=args.pre_only,
    )
    return ctx


def run_identity(ctx: RunContext) -> dict[str, Any]:
    return {
        **ctx.run.identity(),
        "scenario_id": ctx.scenario.scenario_id,
        "scenario_hash": ctx.scenario_hash,
        "scenario_key": scenario_identity_key(ctx.scenario.scenario_id, ctx.scenario_hash),
    }


def execute(ctx: RunContext) -> int:
    """Run the stages of ``ctx`` in its run directory; returns the exit code."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    check_identity(ctx.run_dir, run_identity(ctx), info={"run_hash": ctx.run_hash, "git": git_revision()})
    freeze_inputs(ctx.run_dir, ctx.run_yaml, ctx.run.scenario_path)
    ctx.state = RunState(ctx.run_dir, ctx.run.run_id, ctx.run.pipeline)
    if ctx.run.optimizer:
        # Every Step 3 job of the run inherits the optimizer (like GRIDEXPAND_SOLVER).
        os.environ[OPTIMIZER_ENV] = ctx.run.optimizer
    ctx.state.event(event="run_start", run_hash=ctx.run_hash, scenario_hash=ctx.scenario_hash,
                    model_cases=list(ctx.run.model_cases), resume=ctx.resume, until=ctx.until,
                    optimizer=os.environ.get(OPTIMIZER_ENV) or "urbs")
    runner = {"synthetic": run_synthetic, "paired_validation": run_paired, "paired_aligned": run_aligned}
    try:
        code = runner[ctx.run.pipeline](ctx)
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
