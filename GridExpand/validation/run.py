"""``python -m validation run <run.yaml>``: one ``pipeline: paired_aligned`` run.

Every provider (DSO A, DSO B) of a pylovo alignment bundle against its synthetic counterpart:

- prepare: build (unless ``resume``/``--skip-prepare``) and validate the paired dataset of each provider;
- execute: one paired-runner process (:mod:`gridexpand.paired.runner`) per execution group and provider;
- postprocess: the expansion analyses of every provider group when every job succeeded.

Run directories, state, resume, cancellation and exit codes are those of ``gridexpand run``
(:mod:`gridexpand.scenario.run`).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gridexpand.common.orchestration import Cancelled, install_cancel_handlers, run_step
from gridexpand.paired.datasets import HEAT_LIBRARY_ROOT, PAIRED_DATASET_ROOT, WEATHER_RESULT_ROOT
from gridexpand.paths import RUNS_DIR
from gridexpand.scenario import commands as core_commands
from gridexpand.scenario import run as core_run
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash, load_scenario_config
from gridexpand.scenario.model_cases import POST_MODEL_CASES, execution_groups, validate_cases
from gridexpand.scenario.run_config import (
    _concurrency,
    _grid_scope,
    _optimizer,
    _pylovo_version,
    _resolve,
    _seed,
    choice,
    directory_safe,
    flag,
    parse_run_blocks,
    positive_int,
    with_cases,
)
from gridexpand.scenario.rundir import IdentityMismatch, write_json_atomic
from gridexpand.scenario.scenario_config import _mapping as mapping
from gridexpand.scenario.scenario_config import _only as only

from . import commands
from .grid_subsets import select_grid_subset

PIPELINE = "paired_aligned"


# Run YAML ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderResources:
    """One DSO of an aligned run and the paths of its paired dataset."""

    provider: str
    paired_dataset_id: str
    weather_source_hdf: str
    workers: int | None = None
    # First 12 characters of the scenario hash: heat profiles depend on the
    # scenario (e.g. TEASER retrofit level), so their library does too.
    scenario_tag: str = ""
    # The provider's name inside the pylovo alignment bundle and the comparison
    # JSON; empty: the provider key.
    bundle_name: str = ""

    @property
    def bundle_key(self) -> str:
        return self.bundle_name or self.provider

    @property
    def paired_dir(self) -> Path:
        return PAIRED_DATASET_ROOT / self.paired_dataset_id

    @property
    def heat_profile_set_id(self) -> str:
        return f"{self.paired_dataset_id}_teaser_heat_{self.scenario_tag}"

    @property
    def heat_library(self) -> Path:
        return HEAT_LIBRARY_ROOT / f"{self.heat_profile_set_id}.h5"

    @property
    def heat_sources(self) -> Path:
        return self.paired_dir / "heat_sources" / self.scenario_tag

    @property
    def weather_hdf(self) -> Path:
        return WEATHER_RESULT_ROOT / self.weather_source_hdf


@dataclass(frozen=True)
class AlignedRun:
    """``pipeline: paired_aligned``: every provider of a pylovo alignment bundle."""

    run_id: str
    scenario_path: Path
    pylovo_version_id: str
    alignment_dir: Path
    population: Path
    dso_b_grids_dir: Path | None
    providers: tuple[ProviderResources, ...]
    target_network: str
    model_cases: tuple[str, ...]
    workers: int
    step3_cpus: int
    step3_cluster_concurrency: int | None
    step4_cpus: int
    heat_workers: int
    powerflow_grid_scope: str
    powerflow_max_timesteps: int | None
    profile_seed: int
    cleanup_intermediates: bool
    resume: bool
    parallel_providers: bool
    materialize_expansion: bool
    grid_subset: dict[str, Any] | None
    optimizer: str | None = None
    # None: PVGIS TMY per provider; a year: that real calendar year for every provider.
    weather_year: int | None = None
    pipeline: str = PIPELINE

    @classmethod
    def from_blocks(
        cls,
        run_id: str,
        scenario_path: Path,
        resources: dict[str, Any],
        execution: dict[str, Any],
        *,
        base_dir: Path,
    ) -> "AlignedRun":
        only(resources, {"pylovo_version_id", "alignment_dir", "population", "dso_b_grids_dir", "providers",
                         "target_network", "weather_year"}, "paired_aligned resources")
        only(execution, {"model_cases", "workers", "step3_cpus", "step3_cluster_concurrency", "step4_cpus",
                         "heat_workers", "powerflow_grid_scope", "powerflow_max_timesteps", "profile_seed",
                         "cleanup_intermediates", "resume", "parallel_providers", "materialize_expansion",
                         "grid_subset", "optimizer"}, "paired_aligned execution")
        scenario_tag = load_scenario_config(scenario_path)[1][:12]
        providers = []
        for name, block in mapping(resources.get("providers"), "resources.providers").items():
            if name not in {"dso_a", "dso_b"}:
                raise ValueError(f"Unknown provider {name!r}.")
            block = mapping(block, f"resources.providers.{name}")
            only(block, {"paired_dataset_id", "weather_source_hdf", "workers", "bundle_name"}, f"provider {name}")
            providers.append(ProviderResources(
                provider=name,
                paired_dataset_id=directory_safe(block.get("paired_dataset_id"),
                                                 f"resources.providers.{name}.paired_dataset_id"),
                weather_source_hdf=str(block["weather_source_hdf"]),
                bundle_name=str(block.get("bundle_name") or ""),
                workers=None if block.get("workers") is None
                else positive_int(block["workers"], f"provider {name} workers"),
                scenario_tag=scenario_tag,
            ))
        if not providers:
            raise ValueError("resources.providers must name at least one provider.")
        if any(p.provider == "dso_b" for p in providers) and resources.get("dso_b_grids_dir") in (None, ""):
            raise ValueError("resources.dso_b_grids_dir is required for the dso_b provider.")
        max_steps = execution.get("powerflow_max_timesteps")
        return cls(
            run_id=run_id,
            scenario_path=scenario_path,
            pylovo_version_id=_pylovo_version(resources),
            alignment_dir=_resolve(resources["alignment_dir"], base_dir),
            population=_resolve(resources["population"], base_dir),
            dso_b_grids_dir=_resolve(resources["dso_b_grids_dir"], base_dir) if resources.get("dso_b_grids_dir") else None,
            providers=tuple(providers),
            target_network=choice(resources.get("target_network", "both"), "resources.target_network",
                                  ("both", "real", "synthetic")),
            model_cases=validate_cases(execution.get("model_cases") or (), allowed=POST_MODEL_CASES),
            workers=positive_int(execution.get("workers", 1), "execution.workers"),
            step3_cpus=positive_int(execution.get("step3_cpus", 1), "execution.step3_cpus"),
            step3_cluster_concurrency=_concurrency(execution),
            step4_cpus=positive_int(execution.get("step4_cpus", 1), "execution.step4_cpus"),
            heat_workers=positive_int(execution.get("heat_workers", 4), "execution.heat_workers"),
            powerflow_grid_scope=_grid_scope(execution),
            powerflow_max_timesteps=(
                None if max_steps is None else positive_int(max_steps, "execution.powerflow_max_timesteps")
            ),
            profile_seed=_seed(execution),
            cleanup_intermediates=flag(execution.get("cleanup_intermediates", False), "execution.cleanup_intermediates"),
            resume=flag(execution.get("resume", False), "execution.resume"),
            parallel_providers=flag(execution.get("parallel_providers", False), "execution.parallel_providers"),
            materialize_expansion=flag(execution.get("materialize_expansion", False), "execution.materialize_expansion"),
            grid_subset=_grid_subset(execution.get("grid_subset")),
            optimizer=_optimizer(execution),
            weather_year=(None if resources.get("weather_year") is None
                          else positive_int(resources["weather_year"], "resources.weather_year")),
        )

    def provider(self, name: str) -> ProviderResources:
        for provider in self.providers:
            if provider.provider == name:
                return provider
        raise ValueError(f"Provider {name!r} is not part of run {self.run_id}.")

    def target_for(self, provider: ProviderResources) -> str:
        """Paired-runner ``--target`` of one provider."""
        return {"both": "both", "synthetic": "synthetic", "real": f"real_{provider.provider}"}[self.target_network]

    def identity(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "pylovo_version_id": self.pylovo_version_id,
            "providers": {p.provider: p.paired_dataset_id for p in self.providers},
            "target_network": self.target_network,
            "profile_seed": self.profile_seed,
            "powerflow_grid_scope": self.powerflow_grid_scope,
            "powerflow_max_timesteps": self.powerflow_max_timesteps,
            "grid_subset": self.grid_subset,
        }


def _grid_subset(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    raw = mapping(raw, "execution.grid_subset")
    only(raw, {"method", "seed", "real_grids_per_provider"}, "execution.grid_subset")
    limit = raw.get("real_grids_per_provider")
    return {
        "method": choice(raw.get("method", "components"), "grid_subset.method", ("components", "islands")),
        "seed": positive_int(raw["seed"], "grid_subset.seed", allow_zero=True),
        "real_grids_per_provider": None if limit is None else positive_int(limit, "grid_subset.real_grids_per_provider"),
    }


def aligned_run_from_dict(raw: dict[str, Any], *, base_dir: Path) -> AlignedRun:
    """Parse a ``paired_aligned`` run YAML mapping (``base_dir``: directory of relative paths)."""
    run_id, pipeline, scenario_path, resources, execution = parse_run_blocks(raw, base_dir=base_dir)
    if pipeline != PIPELINE:
        raise ValueError(f"run.pipeline must be {PIPELINE} here (got {pipeline!r}); run other pipelines with "
                         "gridexpand run.")
    return AlignedRun.from_blocks(run_id, scenario_path, resources, execution, base_dir=base_dir)


def load_aligned_run(path: str | Path) -> tuple[AlignedRun, str]:
    """Load a ``paired_aligned`` run YAML; returns the run and ``run_hash`` (SHA-256 of the parsed YAML)."""
    resolved = Path(path).resolve()
    raw = _read_yaml(resolved)
    return aligned_run_from_dict(raw, base_dir=resolved.parent), configuration_hash(raw)


# Steps -------------------------------------------------------------------------------------


@dataclass
class AlignedContext(core_run.RunContext):
    """One ``python -m validation run`` invocation."""

    skip_prepare: bool = False
    provider: str | None = None
    target_grid_id: int | None = None
    pre_only: bool = False


def _group_dir(base: Path, group: str, target_grid_id: int | None) -> Path:
    """Run directory of one paired group; diagnostic single-grid runs get their own."""
    return base / (group if target_grid_id is None else f"{group}-grid{target_grid_id}")


def _run_prepare_commands(ctx: AlignedContext, steps: list[tuple[str, list[str]]], prefix: str = "",
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


def _run_job(ctx: AlignedContext, key: str, argv: list[str], env_extra: dict[str, str] | None = None) -> int:
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


def _postprocess_commands(ctx: AlignedContext, argvs: list[list[str]]) -> None:
    for argv in argvs:
        print(f"[postprocess] {' '.join(argv)}", flush=True)
        code, _ = run_step(argv, log_path=ctx.run_dir / "logs" / "postprocess_expansion.log", echo=True,
                           header="expansion")
        if code != 0:
            raise RuntimeError(f"Expansion materialization failed with return code {code}: {' '.join(argv)}")


def aligned_providers(ctx: AlignedContext) -> list[ProviderResources]:
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    return [p for p in run.providers if ctx.provider in (None, p.provider)]


def aligned_preparation(ctx: AlignedContext, provider: ProviderResources) -> list[tuple[str, list[str]]]:
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    return commands.aligned_preparation_commands(
        provider=provider.provider, bundle_name=provider.bundle_key, alignment_dir=run.alignment_dir,
        population=run.population,
        dso_b_grids_dir=run.dso_b_grids_dir, pylovo_version_id=run.pylovo_version_id,
        scenario_config=run.scenario_path, profile_seed=run.profile_seed, paired_dir=provider.paired_dir,
        weather_hdf=provider.weather_hdf, heat_sources=provider.heat_sources, heat_library=provider.heat_library,
        heat_profile_set_id=provider.heat_profile_set_id, heat_workers=run.heat_workers,
        reference_year=ctx.scenario.mobility.reference_year, weather_year=run.weather_year,
    )


def aligned_jobs(ctx: AlignedContext, provider: ProviderResources, subset_path: Path | None) -> list[tuple[str, list[str], Path]]:
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
        return [(f"{provider.provider}/pre-only", core_commands.paired_runner_command(**common, pre_only=True,
                                                                                   run_dir=run_dir), run_dir)]
    jobs = []
    for group in execution_groups(run.model_cases):
        run_dir = _group_dir(root, group.name, ctx.target_grid_id)
        jobs.append((f"{provider.provider}/{group.name}", core_commands.paired_runner_command(
            **common, run_dir=run_dir, materialization_case=group.materialization_case,
            result_cases=group.result_cases, skip_pre=not group.emits_pre,
        ), run_dir))
    return jobs


def run_aligned(ctx: AlignedContext) -> int:
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
                    population=run.population, provider=provider.provider, bundle_name=provider.bundle_key,
                    paired_dir=provider.paired_dir,
                    grid_subset=run.grid_subset,
                ))
            jobs[provider.provider] = aligned_jobs(ctx, provider, subset_path)
        todo = set(state.plan(
            [{"job": key, "group": key, "run_dir": str(run_dir)} for items in jobs.values() for key, _, run_dir in items],
            resume=ctx.resume,
        ))
    if not ctx.wants("execute"):
        return core_run.EXIT_OK

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
        return core_run.EXIT_FAILED
    if ctx.wants("postprocess") and run.materialize_expansion and not ctx.pre_only and ctx.target_grid_id is None:
        with state.stage_context("postprocess"):
            # aligned_expansion refreshes the QGIS views once itself.
            _postprocess_commands(ctx, [commands.aligned_expansion_command(
                run.run_id, providers=[p.provider for p in providers], cases=("pre", *run.model_cases),
                pylovo_version_id=run.pylovo_version_id,
            )])
    return core_run.EXIT_OK


# Dry run and command line ------------------------------------------------------------------


def plan_lines(ctx: AlignedContext) -> list[str]:
    """Human-readable plan: the commands each stage would run (no DB, no files)."""
    run: AlignedRun = ctx.run  # type: ignore[assignment]
    lines = core_run.plan_header(ctx)
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


def build_parser() -> argparse.ArgumentParser:
    parser = core_run.build_parser()
    parser.prog = "python -m validation run"
    parser.description = "Run one paired_aligned run YAML (validation/config/*.yaml): prepare, execute and postprocess."
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Use the prepared datasets as they are (validated).")
    parser.add_argument("--provider", choices=("dso_a", "dso_b"), help="One provider only.")
    parser.add_argument("--target-grid-id", type=int,
                        help="Diagnostic single-grid filter (own run sub-directory, no expansion analyses).")
    parser.add_argument("--pre-only", action="store_true", help="Step 2 + Step-4 pre only, no URBS.")
    return parser


def load_context(args: argparse.Namespace) -> AlignedContext:
    """Parse and validate everything (no database); raises ValueError on invalid input."""
    run_yaml = args.run_yaml or args.run_config
    if run_yaml is None:
        raise ValueError("Give the run YAML: python -m validation run validation/config/<run>.yaml")
    run, run_hash = load_aligned_run(run_yaml)
    if args.model_case:
        run = with_cases(run, tuple(args.model_case))
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    if args.provider is not None:
        run.provider(args.provider)
    return AlignedContext(
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctx = load_context(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"python -m validation run: invalid configuration: {exc}", file=sys.stderr)
        return core_run.EXIT_INVALID
    if args.dry_run:
        print("\n".join(plan_lines(ctx)))
        return core_run.EXIT_OK
    install_cancel_handlers()
    try:
        return core_run.execute(ctx, run_aligned)
    except IdentityMismatch as exc:
        print(f"python -m validation run: {exc}", file=sys.stderr)
        return core_run.EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
