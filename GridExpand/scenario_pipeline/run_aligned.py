#!/usr/bin/env python3
"""Prepare and run the paired comparison of every aligned DSO provider.

One run YAML (``run.pipeline: paired_aligned``) and one scenario YAML drive
all providers of a pylovo alignment directory, so their assumptions cannot
drift. Each provider gets its own paired dataset (weather, PV library, heat
library, selection scope) and its own paired-runner invocation over its real
grids and the synthetic grids of the same pylovo version.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from scenario_pipeline.config_loader import configuration_hash  # noqa: E402
from scenario_pipeline.model_cases import POST_MODEL_CASES  # noqa: E402
from scenario_pipeline.scenario_config import _mapping, _only, _positive  # noqa: E402

GRIDALLOC_DIR = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
DATASET_ROOT = GRIDALLOC_DIR / "outputs" / "scenario_calibration"
HEAT_LIBRARY_ROOT = DATASET_ROOT / "profile_libraries"
WEATHER_ROOT = GRIDALLOC_DIR / "results"
HEURISTIC_CASES = ("post-inflex-heuristic", "post-hems-heuristic")


@dataclass(frozen=True)
class ProviderResources:
    provider: str
    paired_dataset_id: str
    weather_source_hdf: str
    workers: int | None = None
    # First 12 characters of the scenario hash: heat profiles depend on the
    # scenario (e.g. TEASER retrofit level), so their library does too.
    scenario_tag: str = ""

    @property
    def paired_dir(self) -> Path:
        return DATASET_ROOT / self.paired_dataset_id

    @property
    def heat_profile_set_id(self) -> str:
        return f"{self.paired_dataset_id}_teaser_heat_{self.scenario_tag}"

    @property
    def heat_library(self) -> Path:
        return HEAT_LIBRARY_ROOT / f"{self.heat_profile_set_id}.h5"

    @property
    def weather_hdf(self) -> Path:
        return WEATHER_ROOT / self.weather_source_hdf


@dataclass(frozen=True)
class AlignedRunConfig:
    run_id: str
    scenario_path: Path
    pylovo_version_id: str
    alignment_dir: Path
    population: Path
    uzw_grids_dir: Path | None
    providers: tuple[ProviderResources, ...]
    target_network: str
    model_cases: tuple[str, ...]
    workers: int
    step3_cpus: int
    step3_cluster_concurrency: int
    step4_cpus: int
    heat_workers: int
    powerflow_grid_scope: str
    powerflow_max_timesteps: int | None
    seed: int
    cleanup_intermediates: bool
    resume: bool
    parallel_providers: bool
    materialize_expansion: bool
    grid_subset: dict[str, int] | None


def load_aligned_run_config(path: Path) -> tuple[AlignedRunConfig, str]:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _only(raw, {"run", "resources", "execution"}, "top-level run")
    run = _mapping(raw["run"], "run")
    resources = _mapping(raw["resources"], "resources")
    execution = _mapping(raw["execution"], "execution")
    _only(run, {"id", "scenario", "pipeline"}, "run")
    if run["pipeline"] != "paired_aligned":
        raise ValueError("run.pipeline must be paired_aligned.")
    _only(
        resources,
        {"pylovo_version_id", "alignment_dir", "population", "uzw_grids_dir",
         "providers", "target_network"},
        "paired_aligned resources",
    )
    _only(
        execution,
        {"model_cases", "workers", "step3_cpus", "step3_cluster_concurrency",
         "step4_cpus", "heat_workers", "powerflow_grid_scope",
         "powerflow_max_timesteps", "profile_seed", "cleanup_intermediates",
         "resume", "parallel_providers", "materialize_expansion", "grid_subset"},
        "paired_aligned execution",
    )
    providers = []
    for name, block in _mapping(resources["providers"], "resources.providers").items():
        if name not in {"swf", "uzw"}:
            raise ValueError(f"Unknown provider {name!r}.")
        block = _mapping(block, f"resources.providers.{name}")
        _only(block, {"paired_dataset_id", "weather_source_hdf", "workers"}, f"provider {name}")
        providers.append(
            ProviderResources(
                name,
                str(block["paired_dataset_id"]),
                str(block["weather_source_hdf"]),
                None if block.get("workers") is None
                else int(_positive(block["workers"], f"provider {name} workers")),
            )
        )
    cases = tuple(str(case) for case in execution["model_cases"])
    if not cases or set(cases).difference(POST_MODEL_CASES) or len(set(cases)) != len(cases):
        raise ValueError(f"execution.model_cases must be distinct post cases: {cases}")
    target = str(resources.get("target_network", "both"))
    if target not in {"both", "real", "synthetic"}:
        raise ValueError("resources.target_network must be both, real or synthetic.")
    scope = str(execution.get("powerflow_grid_scope", "full"))
    if scope not in {"full", "backbone"}:
        raise ValueError("execution.powerflow_grid_scope must be full or backbone.")
    max_steps = execution.get("powerflow_max_timesteps")
    scenario = Path(str(run["scenario"]))
    scenario_file = (scenario if scenario.is_absolute() else path.parent / scenario).resolve()
    from scenario_pipeline.config_loader import load_scenario_config

    scenario_tag = load_scenario_config(scenario_file)[1][:12]
    providers = [dataclasses.replace(item, scenario_tag=scenario_tag) for item in providers]
    config = AlignedRunConfig(
        run_id=str(run["id"]),
        scenario_path=(scenario if scenario.is_absolute() else path.parent / scenario).resolve(),
        pylovo_version_id=str(resources["pylovo_version_id"]),
        alignment_dir=Path(str(resources["alignment_dir"])).expanduser().resolve(),
        population=Path(str(resources["population"])).expanduser().resolve(),
        uzw_grids_dir=(
            Path(str(resources["uzw_grids_dir"])).expanduser().resolve()
            if resources.get("uzw_grids_dir") else None
        ),
        providers=tuple(providers),
        target_network=target,
        model_cases=cases,
        workers=int(_positive(execution.get("workers", 1), "execution.workers")),
        step3_cpus=int(_positive(execution.get("step3_cpus", 1), "execution.step3_cpus")),
        step3_cluster_concurrency=int(_positive(
            execution.get("step3_cluster_concurrency", 1), "execution.step3_cluster_concurrency"
        )),
        step4_cpus=int(_positive(execution.get("step4_cpus", 1), "execution.step4_cpus")),
        heat_workers=int(_positive(execution.get("heat_workers", 4), "execution.heat_workers")),
        powerflow_grid_scope=scope,
        powerflow_max_timesteps=(
            None if max_steps is None
            else int(_positive(max_steps, "execution.powerflow_max_timesteps"))
        ),
        seed=int(_positive(execution.get("profile_seed", 481527), "execution.profile_seed", allow_zero=True)),
        cleanup_intermediates=bool(execution.get("cleanup_intermediates", False)),
        resume=bool(execution.get("resume", False)),
        parallel_providers=bool(execution.get("parallel_providers", False)),
        materialize_expansion=bool(execution.get("materialize_expansion", False)),
        grid_subset=_grid_subset(execution.get("grid_subset")),
    )
    return config, configuration_hash(raw)


def _grid_subset(raw: Any) -> dict[str, int] | None:
    if raw is None:
        return None
    raw = _mapping(raw, "execution.grid_subset")
    _only(raw, {"method", "seed", "real_grids_per_provider"}, "execution.grid_subset")
    method = str(raw.get("method", "components"))
    if method not in {"components", "islands"}:
        raise ValueError("grid_subset.method must be components or islands.")
    limit = raw.get("real_grids_per_provider")
    return {
        "method": method,
        "seed": int(_positive(raw["seed"], "grid_subset.seed", allow_zero=True)),
        "real_grids_per_provider": (
            None if limit is None
            else int(_positive(limit, "grid_subset.real_grids_per_provider"))
        ),
    }


def select_grid_subset(run: AlignedRunConfig, provider: ProviderResources) -> dict[str, Any]:
    """Pick whole overlap components so real and synthetic keep the same buildings.

    pylovo's metric cohort groups real and synthetic grids into components that
    share buildings. ``components`` takes them in a seeded random order while the
    provider's real-grid count stays at or below the requested number;
    ``islands`` keeps only one-real-one-synthetic components (identical building
    sets, directly comparable grid by grid), all of them unless a number is set.
    """
    import random

    import pandas as pd

    document = json.loads(run.population.read_text(encoding="utf-8"))
    components = list(document["metric_cohort"][provider.provider]["successful_components"])
    if run.grid_subset["method"] == "islands":
        components = [c for c in components if len(c["real"]) == 1 and len(c["synthetic"]) == 1]
    random.Random(run.grid_subset["seed"]).shuffle(components)
    target = run.grid_subset["real_grids_per_provider"] or sum(len(c["real"]) for c in components)
    registered = pd.read_csv(provider.paired_dir / "paired_registered_synthetic_grids.csv")
    case_by_result = dict(zip(registered["grid_result_id"].astype(str), registered["grid_case_id"].astype(int)))
    chosen, real, synthetic = [], [], []
    for component in components:
        if len(real) + len(component["real"]) > target:
            continue
        chosen.append(component)
        for name in component["real"]:
            number = name.split(":")[-1] if provider.provider == "uzw" else name.split("__")[0]
            real.append(int(number.removeprefix("area-").removeprefix("LV_")))
        synthetic.extend(case_by_result[str(value)] for value in component["synthetic"])
        if len(real) == target:
            break
    return {
        "seed": run.grid_subset["seed"],
        "method": f"{run.grid_subset['method']}: whole pylovo overlap components, seeded random order",
        "components": chosen,
        f"real_{provider.provider}": sorted(real),
        "synthetic": sorted(synthetic),
    }


def preparation_commands(run: AlignedRunConfig, provider: ProviderResources) -> list[tuple[str, list[str]]]:
    """Commands, run in gridalloc, that build one provider's paired dataset."""
    module = ["uv", "run", "--project", "..", "python", "-m"]
    paired_dir = str(provider.paired_dir)
    allocation = [
        "src.scenario_calibration.allocation.aligned_allocation",
        "--provider", provider.provider,
        "--alignment-dir", str(run.alignment_dir),
        "--population", str(run.population),
        "--pylovo-version-id", run.pylovo_version_id,
        "--scenario-config", str(run.scenario_path),
        "--profile-seed", str(run.seed),
        "--output-dir", paired_dir,
    ]
    if provider.provider == "uzw":
        allocation += ["--uzw-grids-dir", str(run.uzw_grids_dir)]
    heat_sources = str(provider.paired_dir / "heat_sources" / provider.scenario_tag)
    return [
        ("allocation", module + allocation),
        ("weather", module + [
            "src.scenario_calibration.profiles.aligned_weather",
            "--paired-dir", paired_dir,
            "--plz", _weather_postcode(provider),
            "--output", str(provider.weather_hdf),
        ]),
        ("heat_regeneration", module + [
            "src.scenario_calibration.profiles.paired_heat_profile_regeneration",
            "--paired-dir", paired_dir, "--refresh-catalog", "--resume",
            "--workers", str(run.heat_workers), "--n-cpu", "1",
            "--scenario-config", str(run.scenario_path),
            "--output-directory", str(provider.paired_dir / "heat_regeneration"),
            "--synthetic-library", heat_sources,
        ]),
        ("heat_readiness_sources", module + [
            "src.scenario_calibration.profiles.paired_profile_readiness",
            "--paired-dir", paired_dir, "--synthetic-input-dir", heat_sources,
        ]),
        ("heat_library", module + [
            "src.scenario_calibration.profiles.physical_heat_profile_library",
            "--source-catalog", str(provider.paired_dir / "paired_heat_profile_catalog.csv"),
            "--source-hdf-dir", heat_sources, "--source-mode", "exact",
            "--output", str(provider.heat_library),
            "--profile-set-id", provider.heat_profile_set_id,
        ]),
        ("heat_readiness_library", module + [
            "src.scenario_calibration.profiles.paired_profile_readiness",
            "--paired-dir", paired_dir, "--synthetic-input-dir", heat_sources,
            "--heat-profile-library", str(provider.heat_library),
        ]),
        ("pv_library", module + [
            "src.scenario_calibration.profiles.pv_profile_library",
            "--roof-catalog", str(provider.paired_dir / "paired_roof_sections.csv"),
            "--weather-source-hdf", str(provider.weather_hdf),
            "--output", str(provider.paired_dir / "paired_pv_profile_library.h5"),
            "--reference-year", "2009",
        ]),
    ]


def _weather_postcode(provider: ProviderResources) -> str:
    import re

    match = re.search(r"_(\d{5})_", provider.weather_source_hdf)
    if match is None:
        raise ValueError(f"{provider.weather_source_hdf} must contain a five-digit postcode.")
    return match.group(1)


def validate_prepared(run: AlignedRunConfig, provider: ProviderResources, scenario_hash: str) -> dict[str, Any]:
    import pandas as pd

    from common.electrification import assignment_manifest_hash
    from paired_validation.datasets import resolve_paired_dataset

    dataset = resolve_paired_dataset(
        provider.paired_dataset_id, expected_pylovo_version_id=run.pylovo_version_id
    )
    paired_dir = dataset.paired_dir
    metadata = json.loads((paired_dir / "paired_scenario_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("provider") != provider.provider:
        raise ValueError(f"{paired_dir} belongs to provider {metadata.get('provider')!r}.")
    if metadata.get("scenario_hash") != scenario_hash:
        raise ValueError(f"{paired_dir} was prepared with another scenario YAML.")
    real = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    synthetic = pd.read_csv(paired_dir / "paired_synthetic_bus_allocation_plan.csv")
    heat = pd.read_csv(paired_dir / "paired_heat_profile_catalog.csv")
    assignment = pd.read_csv(paired_dir / "paired_electrification_assignment.csv")
    if metadata["electrification_assignment_hash"] != assignment_manifest_hash(assignment):
        raise ValueError("Prepared assignment differs from its metadata hash.")
    if set(real["building_objectid"].astype(str)) != set(synthetic["building_objectid"].astype(str)):
        raise ValueError("Prepared real and synthetic plans contain different buildings.")
    not_exact = heat["profile_method"].ne("exact_physical_building") | heat["profile_source_kind"].ne(
        "physical_heat_library"
    )
    if not_exact.any():
        raise ValueError(f"{int(not_exact.sum())} heat profiles are not exact library profiles.")
    if dataset.weather_source_hdf.resolve() != provider.weather_hdf.resolve():
        raise ValueError(f"PV library was built from {dataset.weather_source_hdf}, not {provider.weather_hdf}.")
    return {
        "provider": provider.provider,
        "paired_dataset_id": provider.paired_dataset_id,
        "buildings": int(real["building_objectid"].nunique()),
        "real_grids": int(real["target_grid_id"].nunique()),
        "synthetic_grids": int(synthetic["target_grid_id"].nunique()),
        "heat_profiles": int(heat["building_objectid"].nunique()),
        "electrification_assignment_hash": metadata["electrification_assignment_hash"],
        "electrification_selected": assignment.groupby("technology")["selected"].sum().astype(int).to_dict(),
    }


def runner_commands(
    run: AlignedRunConfig,
    provider: ProviderResources,
    *,
    pre_only: bool = False,
    target_grid_id: int | None = None,
) -> list[list[str]]:
    target = {"both": "both", "synthetic": "synthetic", "real": f"real_{provider.provider}"}[
        run.target_network
    ]
    base = [
        "uv", "run", "--project", "GridExpand/2.demand_allocation", "python",
        "GridExpand/paired_validation/runner.py",
        "--repo-root", str(GRIDEXPAND_DIR.parent),
        "--paired-dataset-id", provider.paired_dataset_id,
        "--pylovo-version-id", run.pylovo_version_id,
        "--provider", provider.provider,
        "--scenario-config", str(run.scenario_path),
        "--target", target,
        "--workers", str(provider.workers or run.workers),
        "--step3-cpus", str(run.step3_cpus),
        "--step3-cluster-concurrency", str(run.step3_cluster_concurrency),
        "--step4-cpus", str(run.step4_cpus),
        "--powerflow-grid-scope", run.powerflow_grid_scope,
        "--profile-seed", str(run.seed),
        "--scenario-label", f"{run.run_id}_{provider.provider}",
        "--run-name-prefix", f"{run.run_id}_{provider.provider}",
    ]
    if run.powerflow_max_timesteps is not None:
        base += ["--max-timesteps", str(run.powerflow_max_timesteps)]
    run_root = GRIDEXPAND_DIR / "run_logs" / run.run_id / provider.provider
    if run.grid_subset is not None:
        subset_path = run_root / "grid_subset.json"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text(
            json.dumps(select_grid_subset(run, provider), indent=2), encoding="utf-8"
        )
        base += ["--job-subset", str(subset_path)]
    if target_grid_id is not None:
        base += ["--target-grid-id", str(target_grid_id)]
    if run.cleanup_intermediates:
        base.append("--cleanup-intermediates")
    if run.resume:
        base.append("--resume")
    if pre_only:
        return [base + ["--pre-only", "--run-dir", str(run_root / "pre-only")]]
    commands = []
    heuristic = tuple(case for case in HEURISTIC_CASES if case in run.model_cases)
    if heuristic:
        commands.append(base + [
            "--model-case", "post-hems-heuristic", "--result-cases", *heuristic,
            "--run-dir", str(run_root / "heuristic-assets"),
        ])
    if "post-hems-optimized" in run.model_cases:
        commands.append(base + [
            "--model-case", "post-hems-optimized", "--result-cases", "post-hems-optimized",
            "--run-dir", str(run_root / "post-hems-optimized"),
            *(["--skip-pre"] if heuristic else []),
        ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing prepared datasets.")
    parser.add_argument("--pre-only", action="store_true", help="Step 2 + Step-4 pre only, no URBS.")
    parser.add_argument("--provider", choices=("swf", "uzw"), default=None, help="Limit to one provider.")
    parser.add_argument("--target-grid-id", type=int, default=None, help="Diagnostic single-grid filter.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from scenario_pipeline.config_loader import load_scenario_config

    run, run_hash = load_aligned_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    providers = [p for p in run.providers if args.provider in (None, p.provider)]
    manifest_path = GRIDEXPAND_DIR / "run_logs" / run.run_id / "run_manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run.run_id,
        "run_hash": run_hash,
        "run_config": str(args.run_config.resolve()),
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "pylovo_version_id": run.pylovo_version_id,
        "powerflow_max_timesteps": run.powerflow_max_timesteps,
        "providers": {},
    }
    prepare = not (args.skip_prepare or run.resume)
    for provider in providers:
        steps = preparation_commands(run, provider) if prepare else []
        manifest["providers"][provider.provider] = {
            "preparation": [{"stage": stage, "command": command} for stage, command in steps],
            "execution": [] if args.prepare_only else runner_commands(
                run, provider, pre_only=args.pre_only, target_grid_id=args.target_grid_id
            ),
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        for name, entry in manifest["providers"].items():
            for step in entry["preparation"]:
                print(f"[{name}:{step['stage']}] {' '.join(step['command'])}")
            for command in entry["execution"]:
                print(f"[{name}:execute] {' '.join(command)}")
        return

    for provider in providers:
        for stage, command in (preparation_commands(run, provider) if prepare else []):
            print(f"[{provider.provider}:{stage}] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=GRIDALLOC_DIR, check=True, env=_env(run))
        readiness = validate_prepared(run, provider, scenario_hash)
        manifest["providers"][provider.provider]["readiness"] = readiness
        print(f"[{provider.provider}:validate] {json.dumps(readiness, sort_keys=True)}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.prepare_only:
        return

    def execute(provider: ProviderResources) -> int:
        code = 0
        for command in manifest["providers"][provider.provider]["execution"]:
            print(f"[{provider.provider}:execute] {' '.join(command)}", flush=True)
            code = max(code, subprocess.run(command, cwd=GRIDEXPAND_DIR.parent, env=_env(run)).returncode)
        return code

    if run.parallel_providers:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            codes = list(pool.map(execute, providers))
    else:
        codes = [execute(provider) for provider in providers]
    if any(codes):
        raise SystemExit(1)
    if run.materialize_expansion and not args.pre_only and args.target_grid_id is None:
        command = [
            "uv", "run", "python", "-m", "expansion.aligned_expansion",
            "--run-id", run.run_id,
            "--providers", *(provider.provider for provider in providers),
            "--cases", "pre", *run.model_cases,
            "--pylovo-version-id", run.pylovo_version_id,
        ]
        print(f"[postprocess_expansion] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=GRIDEXPAND_DIR / "5.postprocessing", check=True)


def _env(run: AlignedRunConfig) -> dict[str, str]:
    import os

    return {**os.environ, "PYLOVO_VERSION_ID": run.pylovo_version_id}


if __name__ == "__main__":
    main()
