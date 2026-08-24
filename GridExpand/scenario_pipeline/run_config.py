"""Typed execution/resource configuration with pipeline-specific options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .scenario_config import _mapping, _only, _positive

POST_MODEL_CASES = {
    "post-inflex-heuristic",
    "post-hems-optimized",
    "post-hems-heuristic",
}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    scenario_path: Path
    pipeline: str
    pylovo_version_id: str

    # Ordinary synthetic scenario selection.
    inputfile_id: str | None
    storage: str
    output_directory: Path | None
    n_cpu: int
    mobility_source: str
    demand_scope: str
    timeframe_mode: str

    # Paired-validation selection.
    paired_dataset_id: str | None
    target_network: str | None
    target_grid_id: int | None
    model_cases: tuple[str, ...]
    workers: int
    step3_cpus: int
    step3_cluster_concurrency: int
    step4_cpus: int
    seed: int
    cleanup_intermediates: bool
    resume: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path) -> "RunConfig":
        raw = _mapping(raw, "run configuration")
        _only(raw, {"run", "resources", "execution"}, "top-level run")
        run = _mapping(raw["run"], "run")
        resources = _mapping(raw["resources"], "resources")
        execution = _mapping(raw["execution"], "execution")
        _only(run, {"id", "scenario", "pipeline"}, "run")
        run_id = str(run["id"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
            raise ValueError("run.id must be a directory-safe identifier.")
        pipeline = str(run["pipeline"])
        if pipeline not in {"scenario", "paired_validation"}:
            raise ValueError("run.pipeline must be scenario or paired_validation.")

        scenario_path = Path(str(run["scenario"]))
        if not scenario_path.is_absolute():
            scenario_path = (base_dir / scenario_path).resolve()

        def path_or_none(value: Any) -> Path | None:
            if value in (None, ""):
                return None
            path = Path(str(value))
            return path if path.is_absolute() else (base_dir / path).resolve()

        pylovo_version_id = str(resources.get("pylovo_version_id", "")).strip()
        if not pylovo_version_id:
            raise ValueError("resources.pylovo_version_id must be set explicitly.")

        if pipeline == "scenario":
            _only(
                resources,
                {"inputfile_id", "storage", "output_directory", "pylovo_version_id"},
                "scenario resources",
            )
            _only(
                execution,
                {
                    "n_cpu", "mobility_source", "demand_scope", "timeframe_mode",
                    "model_cases", "profile_seed",
                },
                "scenario execution",
            )
            storage = str(resources["storage"])
            if storage not in {"db", "h5"}:
                raise ValueError("resources.storage must be db or h5.")
            cases = tuple(str(value) for value in execution.get(
                "model_cases", ("post-hems-heuristic",)
            ))
            if not cases:
                raise ValueError("execution.model_cases cannot be empty.")
            unknown = set(cases).difference({"pre", *POST_MODEL_CASES})
            if unknown:
                raise ValueError(f"Unknown scenario model cases: {sorted(unknown)}")
            if len(cases) != len(set(cases)):
                raise ValueError("execution.model_cases contains duplicates.")
            return cls(
                run_id=run_id,
                scenario_path=scenario_path,
                pipeline=pipeline,
                pylovo_version_id=pylovo_version_id,
                inputfile_id=str(resources["inputfile_id"]),
                storage=storage,
                output_directory=path_or_none(resources.get("output_directory")),
                n_cpu=int(_positive(execution["n_cpu"], "execution.n_cpu")),
                mobility_source=str(execution["mobility_source"]),
                demand_scope=str(execution["demand_scope"]),
                timeframe_mode=str(execution["timeframe_mode"]),
                paired_dataset_id=None,
                target_network=None,
                target_grid_id=None,
                model_cases=cases,
                workers=1,
                step3_cpus=1,
                step3_cluster_concurrency=1,
                step4_cpus=1,
                seed=int(_positive(
                    execution.get("profile_seed", 481527),
                    "execution.profile_seed",
                    allow_zero=True,
                )),
                cleanup_intermediates=False,
                resume=False,
            )

        _only(
            resources,
            {"paired_dataset_id", "pylovo_version_id", "target_network", "target_grid_id"},
            "paired-validation resources",
        )
        _only(
            execution,
            {
                "model_cases", "workers", "step3_cpus",
                "step3_cluster_concurrency", "step4_cpus", "profile_seed",
                "cleanup_intermediates", "resume",
            },
            "paired-validation execution",
        )
        dataset_id = str(resources["paired_dataset_id"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset_id) is None:
            raise ValueError("resources.paired_dataset_id must be a directory-safe ID.")
        target = str(resources.get("target_network", "both"))
        if target not in {"both", "real_swf", "synthetic"}:
            raise ValueError("resources.target_network must be both, real_swf, or synthetic.")
        cases = tuple(str(value) for value in execution["model_cases"])
        if not cases:
            raise ValueError("execution.model_cases cannot be empty.")
        unknown = set(cases).difference(POST_MODEL_CASES)
        if unknown:
            raise ValueError(f"Unknown paired model cases: {sorted(unknown)}")
        if len(cases) != len(set(cases)):
            raise ValueError("execution.model_cases contains duplicates.")
        for name in ("cleanup_intermediates", "resume"):
            if name in execution and not isinstance(execution[name], bool):
                raise ValueError(f"execution.{name} must be true or false.")

        return cls(
            run_id=run_id,
            scenario_path=scenario_path,
            pipeline=pipeline,
            pylovo_version_id=pylovo_version_id,
            inputfile_id=None,
            storage="db",
            output_directory=None,
            n_cpu=1,
            mobility_source="pool",
            demand_scope="all",
            timeframe_mode="full_year",
            paired_dataset_id=dataset_id,
            target_network=target,
            target_grid_id=(
                int(resources["target_grid_id"])
                if resources.get("target_grid_id") is not None else None
            ),
            model_cases=cases,
            workers=int(_positive(execution.get("workers", 1), "execution.workers")),
            step3_cpus=int(_positive(execution.get("step3_cpus", 1), "execution.step3_cpus")),
            step3_cluster_concurrency=int(_positive(
                execution.get("step3_cluster_concurrency", 1),
                "execution.step3_cluster_concurrency",
            )),
            step4_cpus=int(_positive(execution.get("step4_cpus", 1), "execution.step4_cpus")),
            seed=int(_positive(
                execution.get("profile_seed", 481527),
                "execution.profile_seed",
                allow_zero=True,
            )),
            cleanup_intermediates=bool(execution.get("cleanup_intermediates", False)),
            resume=bool(execution.get("resume", False)),
        )
