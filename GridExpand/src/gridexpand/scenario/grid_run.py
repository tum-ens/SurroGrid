"""``gridexpand run``: the scenario method on pylovo grids (``grids: pylovo``) or own grid files (``grids: files``).

- prepare: the buildings of the grids and their buses (pylovo grids: :mod:`gridexpand.grids.pylovo`; own grids:
  ``gridexpand match``, connection points → InfDB buildings), the scenario dataset (:mod:`gridexpand.grids.dataset`),
  weather, heat and PV libraries, then the dataset check;
- execute: one per-grid runner process (:mod:`gridexpand.runner.grid_runner`, target ``synthetic`` or ``real``) per
  execution group; the first group also runs the ``pre`` stage;
- postprocess: one expansion analysis per case (data source ``Synthetic`` or ``Real``) and one QGIS refresh.

Run names: ``<run>_<network>_<case>``; analysis keys ``<run>_<network>_<pre|post_inflex|post|post_hems_optimized>``
with network ``synthetic`` or ``real``.
"""

from __future__ import annotations

import json
from pathlib import Path

from gridexpand.common.orchestration import Cancelled, run_step
from gridexpand.runner.datasets import HEAT_LIBRARY_ROOT, PAIRED_DATASET_ROOT, WEATHER_RESULT_ROOT
from gridexpand.scenario import commands
from gridexpand.scenario.model_cases import execution_groups
from gridexpand.scenario.run_config import RealRun, SyntheticRun


def network(run) -> str:
    """``synthetic`` (pylovo grids) or ``real`` (own grids): target, provider and data source of the run."""
    return run.pipeline


def dataset_id(run) -> str:
    return f"{run.run_id}_{network(run)}"


def paths(ctx) -> dict[str, Path]:
    """Where the run's grid selection, dataset, weather and heat files go."""
    run = ctx.run
    tag = ctx.scenario_hash[:12]
    paired_dir = PAIRED_DATASET_ROOT / dataset_id(run)
    heat_set = f"{dataset_id(run)}_heat_{tag}"
    return {
        "supply": ctx.run_dir / "prepare" / ("grids" if isinstance(run, SyntheticRun) else "matching"),
        "paired_dir": paired_dir,
        "weather": WEATHER_RESULT_ROOT / f"weather_{dataset_id(run)}.h5",
        "heat_sources": paired_dir / "heat_sources" / tag,
        "heat_library": HEAT_LIBRARY_ROOT / f"{heat_set}.h5",
        "heat_set": Path(heat_set),
    }


def supply_stage(ctx) -> tuple[str, list[str]]:
    """The buildings of the grids and their buses: pylovo selection or matching of own grids."""
    run = ctx.run
    output = paths(ctx)["supply"]
    if isinstance(run, SyntheticRun):
        argv = commands.module_command("gridexpand.grids.pylovo", "--version", run.pylovo_version_id,
                                       "--ags", run.ags, "--min-buildings", run.min_buildings, "--output", output)
        for name in ("plz", "kcid", "bcid", "start_index", "limit"):
            if getattr(run, name) is not None:
                argv += [f"--{name.replace('_', '-')}", str(getattr(run, name))]
        return "grids", argv
    argv = commands.module_command("gridexpand.grids.match", run.grid_dir, "--output", output, "--crs", run.crs)
    if run.points is not None:
        argv += ["--points", str(run.points)]
    if run.matching:
        argv += ["--config", str(ctx.run_yaml)]
    return "match", argv


def preparation(ctx) -> list[tuple[str, list[str]]]:
    """``(stage, argv)`` of the prepare stage."""
    run = ctx.run
    where = paths(ctx)
    grids = (["--pylovo-version-id", run.pylovo_version_id] if isinstance(run, SyntheticRun)
             else ["--grids", run.grid_dir])
    dataset = commands.module_command(
        "gridexpand.grids.dataset",
        "--supply", where["supply"] / "building_supply.csv",
        *grids,
        "--scenario-config", run.scenario_path,
        "--output-dir", where["paired_dir"],
        "--profile-seed", run.profile_seed,
        "--network", network(run),
    )
    return [supply_stage(ctx), ("dataset", dataset), *commands.dataset_preparation_commands(
        paired_dir=where["paired_dir"], scenario_config=run.scenario_path, weather_hdf=where["weather"],
        heat_sources=where["heat_sources"], heat_library=where["heat_library"],
        heat_profile_set_id=str(where["heat_set"]), heat_workers=run.heat_workers,
        reference_year=ctx.scenario.mobility.reference_year, weather_year=run.weather_year,
        per_building_heat=True,
    )]


def jobs(ctx) -> list[tuple[str, list[str], Path]]:
    """``(job key, argv, runner run directory)`` per execution group."""
    run = ctx.run
    result = []
    if run.model_cases == ("pre",):  # status quo only
        run_dir = ctx.run_dir / "pre-only"
        return [("pre-only", commands.paired_runner_command(
            paired_dataset_id=dataset_id(run), pylovo_version_id=run.pylovo_version_id, provider=network(run),
            scenario_config=run.scenario_path, target=network(run), workers=run.workers, step3_cpus=run.step3_cpus,
            step3_cluster_concurrency=run.step3_cluster_concurrency, step4_cpus=run.step4_cpus,
            powerflow_grid_scope=run.powerflow_grid_scope, profile_seed=run.profile_seed,
            scenario_label=run.run_id, run_name_prefix=run.run_id, run_dir=run_dir, pre_only=True,
            max_timesteps=run.powerflow_max_timesteps, cleanup_intermediates=run.cleanup_intermediates,
            resume=ctx.resume,
        ), run_dir)]
    for group in execution_groups(run.model_cases):
        run_dir = ctx.run_dir / group.name
        result.append((group.name, commands.paired_runner_command(
            paired_dataset_id=dataset_id(run), pylovo_version_id=run.pylovo_version_id, provider=network(run),
            scenario_config=run.scenario_path, target=network(run), workers=run.workers, step3_cpus=run.step3_cpus,
            step3_cluster_concurrency=run.step3_cluster_concurrency, step4_cpus=run.step4_cpus,
            powerflow_grid_scope=run.powerflow_grid_scope, profile_seed=run.profile_seed,
            scenario_label=run.run_id, run_name_prefix=run.run_id, run_dir=run_dir,
            materialization_case=group.materialization_case, result_cases=group.result_cases,
            skip_pre=not group.emits_pre, max_timesteps=run.powerflow_max_timesteps,
            cleanup_intermediates=run.cleanup_intermediates, resume=ctx.resume,
        ), run_dir))
    return result


def expansion_commands(ctx) -> list[list[str]]:
    """One expansion analysis per case, the ``pre`` stage first."""
    from gridexpand.analysis.expansion.cases import CASE_STAGES

    run = ctx.run
    label = network(run)
    ags = run.ags if isinstance(run, SyntheticRun) else None
    argvs = []
    for case in ("pre", *(case for case in run.model_cases if case != "pre")):
        stage, suffix = CASE_STAGES[case]
        argvs.append(commands.expansion_command(
            f"{run.run_id}_{label}_{case}", stage=stage, analysis_key=f"{run.run_id}_{label}_{suffix}",
            data_source=label, ags=ags, note=f"{label} run {run.run_id}",
        ))
    return argvs


def plan_lines(ctx) -> list[str]:
    lines = []
    for stage, argv in preparation(ctx):
        lines.append(f"[prepare:{stage}] {' '.join(map(str, argv))}")
    lines.append(f"[prepare:check] dataset {dataset_id(ctx.run)}")
    for key, argv, _ in jobs(ctx):
        lines.append(f"[execute:{key}] {' '.join(argv)}")
    if ctx.run.materialize_expansion:
        for argv in expansion_commands(ctx):
            lines.append(f"[postprocess] {' '.join(argv)}")
        lines.append("[postprocess] refresh the QGIS views")
    return lines


def _step(ctx, kind: str, name: str, argv: list[str], log: Path) -> int:
    print(f"[{kind}:{name}] {' '.join(map(str, argv))}", flush=True)
    code, seconds = run_step([str(part) for part in argv], log_path=log, echo=True, header=name)
    ctx.state.event(event=f"{kind}_step_finish", step=name, returncode=code, seconds=seconds)
    return code


def run_grids(ctx) -> int:
    """The stages of a run (``ctx`` is a :class:`gridexpand.scenario.run.RunContext`)."""
    from gridexpand.runner.datasets import validate_prepared_dataset
    from gridexpand.scenario.run import EXIT_FAILED, EXIT_OK

    run: SyntheticRun | RealRun = ctx.run
    label = network(run)
    state = ctx.state
    where = paths(ctx)

    def check() -> dict:
        return validate_prepared_dataset(
            dataset_id(run), pylovo_version_id=run.pylovo_version_id, scenario_hash=ctx.scenario_hash,
            provider=label, require_exact_heat=True, weather_hdf=where["weather"],
        )

    with state.stage_context("prepare"):
        ready = None
        if ctx.resume:  # a prepared dataset that passes the check is reused
            try:
                ready = check()
            except (ValueError, FileNotFoundError, KeyError, OSError):
                ready = None
        if ready is None:
            for stage, argv in preparation(ctx):
                if _step(ctx, "prepare", stage, argv, ctx.run_dir / "logs" / f"prepare_{stage}.log") != 0:
                    raise RuntimeError(f"Preparation step {stage} failed; see logs/prepare_{stage}.log.")
            ready = check()
        ctx.readiness[label] = ready
        print(f"[prepare:check] {json.dumps(ctx.readiness[label], sort_keys=True)}", flush=True)
        planned = jobs(ctx)
        todo = set(state.plan([{"job": key, "group": key, "run_dir": str(run_dir)} for key, _, run_dir in planned],
                              resume=ctx.resume))
    if not ctx.wants("execute"):
        return EXIT_OK
    with state.stage_context("execute"):
        failed = False
        for key, argv, _ in planned:
            if key not in todo:
                continue
            state.job(key, status="running", step="grid_runner")
            try:
                code = _step(ctx, "execute", key, argv, ctx.run_dir / "jobs" / key / "log.txt")
            except Cancelled:
                state.job(key, status="cancelled")
                raise
            state.job(key, status="done" if code == 0 else "failed", exit_code=code,
                      message=None if code == 0 else f"exit code {code}")
            failed = failed or code != 0
    if failed:
        return EXIT_FAILED
    if ctx.wants("postprocess") and run.materialize_expansion:
        with state.stage_context("postprocess"):
            for argv in expansion_commands(ctx):
                if _step(ctx, "postprocess", "expansion", argv, ctx.run_dir / "logs" / "postprocess_expansion.log"):
                    raise RuntimeError(f"Expansion materialization failed: {' '.join(argv)}")
            from gridexpand.db import refresh_qgis_views

            refresh_qgis_views()
            state.event(event="qgis_views_refreshed")
    return EXIT_OK
