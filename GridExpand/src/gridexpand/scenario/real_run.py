"""``gridexpand run`` with ``grids: files``: own grid files through the scenario method.

- prepare: ``gridexpand match`` (connection points → InfDB buildings), the scenario dataset
  (:mod:`gridexpand.grids.dataset`), weather, heat and PV libraries, then the dataset check;
- execute: one per-grid runner process (:mod:`gridexpand.runner.grid_runner`, target ``real``) per execution group;
  the first group also runs the ``pre`` stage;
- postprocess: one expansion analysis per case (data source ``Real``) and one QGIS refresh.

Run names: ``<run>_real_<case>``; analysis keys ``<run>_real_<pre|post_inflex|post|post_hems_optimized>``.
"""

from __future__ import annotations

import json
from pathlib import Path

from gridexpand.common.orchestration import Cancelled, run_step
from gridexpand.runner.datasets import HEAT_LIBRARY_ROOT, PAIRED_DATASET_ROOT, WEATHER_RESULT_ROOT
from gridexpand.scenario import commands
from gridexpand.scenario.model_cases import execution_groups
from gridexpand.scenario.run_config import RealRun

NETWORK = "real"


def dataset_id(run: RealRun) -> str:
    return f"{run.run_id}_real"


def paths(ctx) -> dict[str, Path]:
    """Where the run's matching, dataset, weather and heat files go."""
    run: RealRun = ctx.run
    tag = ctx.scenario_hash[:12]
    paired_dir = PAIRED_DATASET_ROOT / dataset_id(run)
    heat_set = f"{dataset_id(run)}_teaser_heat_{tag}"
    return {
        "matching": ctx.run_dir / "prepare" / "matching",
        "paired_dir": paired_dir,
        "weather": WEATHER_RESULT_ROOT / f"weather_{dataset_id(run)}.h5",
        "heat_sources": paired_dir / "heat_sources" / tag,
        "heat_library": HEAT_LIBRARY_ROOT / f"{heat_set}.h5",
        "heat_set": Path(heat_set),
    }


def preparation(ctx) -> list[tuple[str, list[str]]]:
    """``(stage, argv)`` of the prepare stage."""
    run: RealRun = ctx.run
    where = paths(ctx)
    match = commands.module_command("gridexpand.grids.match", run.grid_dir, "--output", where["matching"],
                                    "--crs", run.crs)
    if run.points is not None:
        match += ["--points", str(run.points)]
    if run.matching:
        match += ["--config", str(ctx.run_yaml)]
    dataset = commands.module_command(
        "gridexpand.grids.dataset",
        "--supply", where["matching"] / "building_supply.csv",
        "--grids", run.grid_dir,
        "--scenario-config", run.scenario_path,
        "--output-dir", where["paired_dir"],
        "--profile-seed", run.profile_seed,
        "--network", NETWORK,
    )
    return [("match", match), ("dataset", dataset), *commands.dataset_preparation_commands(
        paired_dir=where["paired_dir"], scenario_config=run.scenario_path, weather_hdf=where["weather"],
        heat_sources=where["heat_sources"], heat_library=where["heat_library"],
        heat_profile_set_id=str(where["heat_set"]), heat_workers=run.heat_workers,
        reference_year=ctx.scenario.mobility.reference_year, weather_year=run.weather_year,
        per_building_heat=True,
    )]


def jobs(ctx) -> list[tuple[str, list[str], Path]]:
    """``(job key, argv, runner run directory)`` per execution group."""
    run: RealRun = ctx.run
    result = []
    for group in execution_groups(run.model_cases):
        run_dir = ctx.run_dir / group.name
        result.append((group.name, commands.paired_runner_command(
            paired_dataset_id=dataset_id(run), pylovo_version_id=run.pylovo_version_id, provider=NETWORK,
            scenario_config=run.scenario_path, target=NETWORK, workers=run.workers, step3_cpus=run.step3_cpus,
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

    run: RealRun = ctx.run
    argvs = []
    for case in ("pre", *run.model_cases):
        stage, suffix = CASE_STAGES[case]
        argvs.append(commands.expansion_command(
            f"{run.run_id}_{NETWORK}_{case}", stage=stage, analysis_key=f"{run.run_id}_{NETWORK}_{suffix}",
            data_source=NETWORK, note=f"real run {run.run_id}",
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


def run_real(ctx) -> int:
    """The stages of a ``grids: files`` run (``ctx`` is a :class:`gridexpand.scenario.run.RunContext`)."""
    from gridexpand.runner.datasets import validate_prepared_dataset
    from gridexpand.scenario.run import EXIT_FAILED, EXIT_OK

    run: RealRun = ctx.run
    state = ctx.state
    where = paths(ctx)
    def check() -> dict:
        return validate_prepared_dataset(
            dataset_id(run), pylovo_version_id=run.pylovo_version_id, scenario_hash=ctx.scenario_hash,
            provider=NETWORK, require_exact_heat=True, weather_hdf=where["weather"],
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
        ctx.readiness[NETWORK] = ready
        print(f"[prepare:check] {json.dumps(ctx.readiness[NETWORK], sort_keys=True)}", flush=True)
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
