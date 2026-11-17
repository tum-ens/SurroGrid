"""Run YAMLs (``config/runs/*.yaml``): the loader of ``pipeline: synthetic``.

A run YAML has three blocks: ``run`` (``id``, ``scenario``, ``pipeline``),
``resources`` (what to run on) and ``execution`` (how). ``pipeline: synthetic``
runs Steps 2-4 and the expansion analyses for the synthetic grids of one AGS
(optionally one PLZ or one grid), one batch per execution group
(:class:`SyntheticRun`). The real-vs-synthetic validation (``pipeline:
paired_aligned``) has its own loader in ``GridExpand/validation``, built on
:func:`parse_run_blocks` and the helpers here.

Parsing never touches a database. Relative paths resolve against the YAML's
directory. :func:`load_run_config` returns the run and the SHA-256 of the
parsed YAML (``run_hash``).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gridexpand.common.reproducibility import DEFAULT_PROFILE_SEED
from gridexpand.common.timeframe import TIMEFRAME_MODES
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash
from gridexpand.scenario.model_cases import (
    POST_MODEL_CASES,
    SYNTHETIC_UNSUPPORTED_CASES,
    validate_cases,
)
from gridexpand.scenario.scenario_config import _mapping as mapping
from gridexpand.scenario.scenario_config import _only as only
from gridexpand.scenario.scenario_config import _positive

PIPELINES = ("synthetic",)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# Validation helpers ----------------------------------------------------------------


def positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    """An integer >= 1 (>= 0 with ``allow_zero``); YAML floats like 2.0 are rejected."""
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(_positive(value, label, allow_zero=allow_zero))


def flag(value: Any, label: str) -> bool:
    """A YAML boolean (the string ``"false"`` is an error, not True)."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false.")
    return value


def choice(value: Any, label: str, choices: tuple[str, ...]) -> str:
    text = str(value)
    if text not in choices:
        raise ValueError(f"{label} must be one of {list(choices)}, got {value!r}.")
    return text


def directory_safe(value: Any, label: str) -> str:
    text = str(value)
    if _SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{label} must be a directory-safe identifier, got {value!r}.")
    return text


def optional_int(value: Any, label: str) -> int | None:
    """Optional non-negative integer selector; null, '' or '-' mean 'any'."""
    if value in (None, "", "-"):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer, null, or '-'.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, null, or '-'.") from exc
    if result < 0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def signed_optional_int(value: Any, label: str) -> int | None:
    """Optional integer that may be negative (pylovo ``bcid`` is -1 for some grids)."""
    if value in (None, "", "-"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer or null.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer or null.") from exc


def _resolve(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _pylovo_version(resources: dict[str, Any]) -> str:
    version = str(resources.get("pylovo_version_id", "") or "").strip()
    if not version:
        raise ValueError("resources.pylovo_version_id must be set explicitly.")
    return version


def _grid_scope(execution: dict[str, Any]) -> str:
    return choice(execution.get("powerflow_grid_scope", "full"), "execution.powerflow_grid_scope", ("full", "backbone"))


def _optimizer(execution: dict[str, Any]) -> str | None:
    """``execution.optimizer`` (Step 3 optimizer) or None for ``$GRIDEXPAND_OPTIMIZER``/urbs."""
    optimizer = execution.get("optimizer")
    if optimizer is None:
        return None
    from gridexpand.optimization.solver import SUPPORTED_OPTIMIZERS

    return choice(optimizer, "execution.optimizer", SUPPORTED_OPTIMIZERS)


def _concurrency(execution: dict[str, Any]) -> int | None:
    """``execution.step3_cluster_concurrency``; None (unset): the optimizer's default."""
    value = execution.get("step3_cluster_concurrency")
    return None if value is None else positive_int(value, "execution.step3_cluster_concurrency")


def _seed(execution: dict[str, Any]) -> int:
    return positive_int(execution.get("profile_seed", DEFAULT_PROFILE_SEED), "execution.profile_seed", allow_zero=True)


# Synthetic ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRun:
    """``pipeline: synthetic``: the synthetic grids of one region, Steps 2-4 + expansion.

    Resources select the region (``ags``; optional ``plz`` or ``plz/kcid/bcid``,
    which is also the scope of the electrification assignment) and optionally a
    candidate range (``start_index``/``limit``, like ``gridexpand synthetic``).
    Execution keys are the ``gridexpand synthetic`` flags; output is always
    case-qualified.
    """

    run_id: str
    scenario_path: Path
    pylovo_version_id: str
    ags: int
    model_cases: tuple[str, ...]
    plz: int | None = None
    kcid: int | None = None
    bcid: int | None = None
    min_buildings: int = 5
    start_index: int | None = None
    limit: int | None = None
    profile_seed: int = DEFAULT_PROFILE_SEED
    timeframe_mode: str = "full_year"
    demand_scope: str = "all"
    workers: int = 1
    step2_cpus: int = 4
    step2_timeseries_storage: str = "temp"
    step3_cpus: int = 16
    step3_max_cpus: int = 32
    step3_target_columns: int = 35
    step3_cluster_concurrency: int | None = None  # None: 1 for urbs, automatic for pypsa
    dynamic_step3: bool = True
    step4_cpus: int = 4
    solver: str | None = None
    optimizer: str | None = None
    powerflow_output: str = "summary"
    powerflow_grid_scope: str = "full"
    cleanup_intermediates: str = "never"
    materialize_expansion: bool = True
    pilot_gate: bool = True
    pilot_index: int = 0
    resume: bool = False
    rerun_failed: bool = False
    pipeline: str = "synthetic"

    RESOURCE_KEYS = frozenset({
        "pylovo_version_id", "ags", "plz", "kcid", "bcid", "min_buildings", "start_index", "limit",
        "storage", "output_directory",
    })
    EXECUTION_KEYS = frozenset({
        "model_cases", "profile_seed", "timeframe_mode", "demand_scope", "mobility_source", "n_cpu",
        "workers", "step2_cpus", "step2_timeseries_storage", "step3_cpus", "step3_max_cpus",
        "step3_target_columns", "step3_cluster_concurrency", "dynamic_step3", "step4_cpus", "solver",
        "optimizer", "powerflow_output", "powerflow_grid_scope", "cleanup_intermediates", "materialize_expansion",
        "pilot_gate", "pilot_index", "resume", "rerun_failed",
    })

    @classmethod
    def from_blocks(
        cls, run_id: str, scenario_path: Path, resources: dict[str, Any], execution: dict[str, Any]
    ) -> "SyntheticRun":
        only(resources, set(cls.RESOURCE_KEYS), "synthetic resources")
        only(execution, set(cls.EXECUTION_KEYS), "synthetic execution")
        if str(resources.get("storage", "db")) != "db":
            raise ValueError("The synthetic pipeline reads grids from the database: resources.storage must be db.")
        if resources.get("output_directory") not in (None, ""):
            raise ValueError("resources.output_directory is not supported by the synthetic pipeline; use null.")
        if str(execution.get("mobility_source", "pool")) != "pool":
            raise ValueError("The synthetic pipeline uses the mobility profile pool: execution.mobility_source must be pool.")
        ags = optional_int(resources.get("ags"), "resources.ags")
        if ags is None:
            raise ValueError("resources.ags is required for the synthetic pipeline.")
        plz = optional_int(resources.get("plz"), "resources.plz")
        kcid = optional_int(resources.get("kcid"), "resources.kcid")
        bcid = signed_optional_int(resources.get("bcid"), "resources.bcid")
        if (kcid is None) != (bcid is None):
            raise ValueError("resources.kcid and resources.bcid must be set together.")
        if kcid is not None and plz is None:
            raise ValueError("resources.plz is required when kcid/bcid are set.")
        if "model_cases" not in execution:
            raise ValueError("execution.model_cases is required.")
        cases = validate_cases(execution["model_cases"])
        for case in cases:
            if case in SYNTHETIC_UNSUPPORTED_CASES:
                raise ValueError(
                    f"execution.model_cases: {case} is not available for the synthetic pipeline. "
                    f"{SYNTHETIC_UNSUPPORTED_CASES[case]}"
                )
        if "n_cpu" in execution and "step2_cpus" in execution:
            raise ValueError("Use execution.step2_cpus or its alias execution.n_cpu, not both.")
        step2_cpus = execution.get("step2_cpus", execution.get("n_cpu", cls.step2_cpus))

        def number(name: str, default: int, *, allow_zero: bool = False) -> int:
            return positive_int(execution.get(name, default), f"execution.{name}", allow_zero=allow_zero)

        def boolean(name: str, default: bool) -> bool:
            return flag(execution.get(name, default), f"execution.{name}")

        solver = execution.get("solver")
        if solver is not None:
            from gridexpand.optimization.solver import SUPPORTED_SOLVERS

            solver = choice(solver, "execution.solver", SUPPORTED_SOLVERS)
        return cls(
            run_id=run_id,
            scenario_path=scenario_path,
            pylovo_version_id=_pylovo_version(resources),
            ags=ags,
            plz=plz,
            kcid=kcid,
            bcid=bcid,
            min_buildings=positive_int(resources.get("min_buildings", cls.min_buildings), "resources.min_buildings"),
            start_index=optional_int(resources.get("start_index"), "resources.start_index"),
            limit=(
                None if resources.get("limit") is None
                else positive_int(resources["limit"], "resources.limit")
            ),
            model_cases=cases,
            profile_seed=_seed(execution),
            timeframe_mode=choice(execution.get("timeframe_mode", cls.timeframe_mode), "execution.timeframe_mode",
                                  TIMEFRAME_MODES),
            demand_scope=choice(execution.get("demand_scope", cls.demand_scope), "execution.demand_scope",
                                ("all", "residential")),
            workers=number("workers", cls.workers),
            step2_cpus=positive_int(step2_cpus, "execution.step2_cpus"),
            step2_timeseries_storage=choice(
                execution.get("step2_timeseries_storage", cls.step2_timeseries_storage),
                "execution.step2_timeseries_storage", ("db", "temp", "both"),
            ),
            step3_cpus=number("step3_cpus", cls.step3_cpus),
            step3_max_cpus=number("step3_max_cpus", cls.step3_max_cpus),
            step3_target_columns=number("step3_target_columns", cls.step3_target_columns),
            step3_cluster_concurrency=_concurrency(execution),
            dynamic_step3=boolean("dynamic_step3", cls.dynamic_step3),
            step4_cpus=number("step4_cpus", cls.step4_cpus),
            solver=solver,
            optimizer=_optimizer(execution),
            powerflow_output=choice(execution.get("powerflow_output", cls.powerflow_output),
                                    "execution.powerflow_output", ("raw", "summary", "both")),
            powerflow_grid_scope=_grid_scope(execution),
            cleanup_intermediates=choice(execution.get("cleanup_intermediates", cls.cleanup_intermediates),
                                         "execution.cleanup_intermediates", ("never", "success")),
            materialize_expansion=boolean("materialize_expansion", cls.materialize_expansion),
            pilot_gate=boolean("pilot_gate", cls.pilot_gate),
            pilot_index=number("pilot_index", cls.pilot_index, allow_zero=True),
            resume=boolean("resume", cls.resume),
            rerun_failed=boolean("rerun_failed", cls.rerun_failed),
        )

    def identity(self) -> dict[str, Any]:
        """What a resumed run must share with the recorded one (not resources or cases)."""
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "pylovo_version_id": self.pylovo_version_id,
            "region": {"ags": self.ags, "plz": self.plz, "kcid": self.kcid, "bcid": self.bcid},
            "min_buildings": self.min_buildings,
            "demand_scope": self.demand_scope,
            "timeframe_mode": self.timeframe_mode,
            "profile_seed": self.profile_seed,
            "powerflow_output": self.powerflow_output,
            "powerflow_grid_scope": self.powerflow_grid_scope,
        }


# Loader --------------------------------------------------------------------------------------

RunConfig = SyntheticRun


def parse_run_blocks(raw: dict[str, Any], *, base_dir: Path) -> tuple[str, str, Path, dict[str, Any], dict[str, Any]]:
    """``(run id, pipeline, scenario path, resources, execution)`` of a run YAML mapping; the pipeline is not checked."""
    raw = mapping(raw, "run configuration")
    only(raw, {"run", "resources", "execution"}, "top-level run")
    for block in ("run", "resources", "execution"):
        if block not in raw:
            raise ValueError(f"The run YAML needs a {block!r} block.")
    run = mapping(raw["run"], "run")
    resources = mapping(raw["resources"], "resources")
    execution = mapping(raw["execution"], "execution")
    only(run, {"id", "scenario", "pipeline"}, "run")
    run_id = directory_safe(run.get("id"), "run.id")
    pipeline = str(run.get("pipeline"))
    if not run.get("scenario"):
        raise ValueError("run.scenario (the scenario YAML) is required.")
    scenario_path = _resolve(run["scenario"], base_dir)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario YAML not found: {scenario_path}")
    return run_id, pipeline, scenario_path, resources, execution


def run_config_from_dict(raw: dict[str, Any], *, base_dir: Path) -> RunConfig:
    """Parse a run YAML mapping (``base_dir``: directory of relative paths)."""
    run_id, pipeline, scenario_path, resources, execution = parse_run_blocks(raw, base_dir=base_dir)
    if pipeline == "scenario":
        raise ValueError(
            "run.pipeline 'scenario' was renamed to 'synthetic', which runs Steps 2-4 and the "
            "expansion analyses (not Step 2 only). Rename it in the run YAML."
        )
    if pipeline == "paired_aligned":
        raise ValueError(
            "run.pipeline paired_aligned is the real-vs-synthetic validation: run it with "
            "'uv run python -m validation run <run.yaml>' (GridExpand/validation)."
        )
    choice(pipeline, "run.pipeline", PIPELINES)
    return SyntheticRun.from_blocks(run_id, scenario_path, resources, execution)


def load_run_config(path: str | Path) -> tuple[RunConfig, str]:
    """Load a run YAML; returns the run and ``run_hash`` (SHA-256 of the parsed YAML)."""
    resolved = Path(path).resolve()
    raw = _read_yaml(resolved)
    return run_config_from_dict(raw, base_dir=resolved.parent), configuration_hash(raw)


def with_cases(run: RunConfig, cases: tuple[str, ...]) -> RunConfig:
    """``run`` restricted to ``cases`` (validated with the pipeline's rules)."""
    allowed = POST_MODEL_CASES if run.pipeline != "synthetic" else None
    cases = validate_cases(cases) if allowed is None else validate_cases(cases, allowed=allowed)
    if run.pipeline == "synthetic":
        for case in cases:
            if case in SYNTHETIC_UNSUPPORTED_CASES:
                raise ValueError(f"{case} is not available for the synthetic pipeline. {SYNTHETIC_UNSUPPORTED_CASES[case]}")
    return dataclasses.replace(run, model_cases=cases)
