"""Run YAMLs (``config/runs/*.yaml``): one loader, one dataclass per pipeline.

A run YAML has three blocks: ``run`` (``id``, ``scenario``, ``pipeline``),
``resources`` (what to run on) and ``execution`` (how). ``run.pipeline`` is

- ``synthetic``: Steps 2-4 and the expansion analyses for the synthetic grids
  of one AGS (optionally one PLZ or one grid), one batch per execution group
  (:class:`SyntheticRun`);
- ``paired_validation``: SWF real vs synthetic grids of one prepared paired
  dataset (:class:`PairedRun`);
- ``paired_aligned``: every provider of a pylovo alignment bundle
  (:class:`AlignedRun`).

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
from gridexpand.paths import ALLOCATION_RESULTS_DIR, SCENARIO_CALIBRATION_OUTPUT_DIR
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash, load_scenario_config
from gridexpand.scenario.model_cases import (
    POST_MODEL_CASES,
    SYNTHETIC_UNSUPPORTED_CASES,
    validate_cases,
)
from gridexpand.scenario.scenario_config import _mapping as mapping
from gridexpand.scenario.scenario_config import _only as only
from gridexpand.scenario.scenario_config import _positive

PIPELINES = ("synthetic", "paired_validation", "paired_aligned")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PAIRED_DATASET_ROOT = SCENARIO_CALIBRATION_OUTPUT_DIR
HEAT_LIBRARY_ROOT = PAIRED_DATASET_ROOT / "profile_libraries"
WEATHER_RESULT_ROOT = ALLOCATION_RESULTS_DIR


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


# Paired validation (SWF) --------------------------------------------------------------------


@dataclass(frozen=True)
class PairedRun:
    """``pipeline: paired_validation``: SWF real and synthetic grids of one prepared dataset."""

    run_id: str
    scenario_path: Path
    pylovo_version_id: str
    ags: int
    plz: int
    min_buildings: int
    heat_profile_set_id: str
    weather_source_hdf: str
    excluded_real_lv_ids: tuple[int, ...]
    paired_dataset_id: str
    target_network: str
    target_grid_id: int | None
    model_cases: tuple[str, ...]
    workers: int
    step3_cpus: int
    step3_cluster_concurrency: int | None
    step4_cpus: int
    powerflow_grid_scope: str
    profile_seed: int
    cleanup_intermediates: bool
    resume: bool
    materialize_expansion: bool
    optimizer: str | None = None
    pipeline: str = "paired_validation"

    @classmethod
    def from_blocks(
        cls, run_id: str, scenario_path: Path, resources: dict[str, Any], execution: dict[str, Any]
    ) -> "PairedRun":
        only(resources, {
            "paired_dataset_id", "pylovo_version_id", "target_network", "target_grid_id", "ags", "plz",
            "min_buildings", "heat_profile_set_id", "weather_source_hdf", "excluded_real_lv_ids",
        }, "paired-validation resources")
        only(execution, {
            "model_cases", "workers", "step3_cpus", "step3_cluster_concurrency", "step4_cpus",
            "powerflow_grid_scope", "profile_seed", "cleanup_intermediates", "resume", "materialize_expansion",
            "optimizer",
        }, "paired-validation execution")
        missing = [name for name in ("ags", "plz", "heat_profile_set_id", "weather_source_hdf")
                   if resources.get(name) in (None, "")]
        if missing:
            raise ValueError("Paired preparation requires resources: " + ", ".join(missing))
        return cls(
            run_id=run_id,
            scenario_path=scenario_path,
            pylovo_version_id=_pylovo_version(resources),
            ags=int(resources["ags"]),
            plz=int(resources["plz"]),
            min_buildings=positive_int(resources.get("min_buildings", 5), "resources.min_buildings"),
            heat_profile_set_id=str(resources["heat_profile_set_id"]),
            weather_source_hdf=str(resources["weather_source_hdf"]),
            excluded_real_lv_ids=tuple(int(value) for value in resources.get("excluded_real_lv_ids") or ()),
            paired_dataset_id=directory_safe(resources.get("paired_dataset_id"), "resources.paired_dataset_id"),
            target_network=choice(resources.get("target_network", "both"), "resources.target_network",
                                  ("both", "real_swf", "synthetic")),
            target_grid_id=optional_int(resources.get("target_grid_id"), "resources.target_grid_id"),
            model_cases=validate_cases(execution.get("model_cases") or (), allowed=POST_MODEL_CASES),
            workers=positive_int(execution.get("workers", 1), "execution.workers"),
            step3_cpus=positive_int(execution.get("step3_cpus", 1), "execution.step3_cpus"),
            step3_cluster_concurrency=_concurrency(execution),
            step4_cpus=positive_int(execution.get("step4_cpus", 1), "execution.step4_cpus"),
            powerflow_grid_scope=_grid_scope(execution),
            profile_seed=_seed(execution),
            cleanup_intermediates=flag(execution.get("cleanup_intermediates", False), "execution.cleanup_intermediates"),
            resume=flag(execution.get("resume", False), "execution.resume"),
            materialize_expansion=flag(execution.get("materialize_expansion", True), "execution.materialize_expansion"),
            optimizer=_optimizer(execution),
        )

    @property
    def paired_dir(self) -> Path:
        return PAIRED_DATASET_ROOT / self.paired_dataset_id

    @property
    def heat_library(self) -> Path:
        return HEAT_LIBRARY_ROOT / f"{self.heat_profile_set_id}.h5"

    @property
    def weather_hdf(self) -> Path:
        return WEATHER_RESULT_ROOT / self.weather_source_hdf

    def identity(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "pylovo_version_id": self.pylovo_version_id,
            "paired_dataset_id": self.paired_dataset_id,
            "region": {"ags": self.ags, "plz": self.plz},
            "target_network": self.target_network,
            "target_grid_id": self.target_grid_id,
            "profile_seed": self.profile_seed,
            "powerflow_grid_scope": self.powerflow_grid_scope,
        }


# Paired aligned (SWF + ÜZW) ------------------------------------------------------------------


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
    uzw_grids_dir: Path | None
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
    pipeline: str = "paired_aligned"

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
        only(resources, {"pylovo_version_id", "alignment_dir", "population", "uzw_grids_dir", "providers",
                         "target_network", "weather_year"}, "paired_aligned resources")
        only(execution, {"model_cases", "workers", "step3_cpus", "step3_cluster_concurrency", "step4_cpus",
                         "heat_workers", "powerflow_grid_scope", "powerflow_max_timesteps", "profile_seed",
                         "cleanup_intermediates", "resume", "parallel_providers", "materialize_expansion",
                         "grid_subset", "optimizer"}, "paired_aligned execution")
        scenario_tag = load_scenario_config(scenario_path)[1][:12]
        providers = []
        for name, block in mapping(resources.get("providers"), "resources.providers").items():
            if name not in {"swf", "uzw"}:
                raise ValueError(f"Unknown provider {name!r}.")
            block = mapping(block, f"resources.providers.{name}")
            only(block, {"paired_dataset_id", "weather_source_hdf", "workers"}, f"provider {name}")
            providers.append(ProviderResources(
                provider=name,
                paired_dataset_id=directory_safe(block.get("paired_dataset_id"),
                                                 f"resources.providers.{name}.paired_dataset_id"),
                weather_source_hdf=str(block["weather_source_hdf"]),
                workers=None if block.get("workers") is None
                else positive_int(block["workers"], f"provider {name} workers"),
                scenario_tag=scenario_tag,
            ))
        if not providers:
            raise ValueError("resources.providers must name at least one provider.")
        if any(p.provider == "uzw" for p in providers) and resources.get("uzw_grids_dir") in (None, ""):
            raise ValueError("resources.uzw_grids_dir is required for the uzw provider.")
        max_steps = execution.get("powerflow_max_timesteps")
        return cls(
            run_id=run_id,
            scenario_path=scenario_path,
            pylovo_version_id=_pylovo_version(resources),
            alignment_dir=_resolve(resources["alignment_dir"], base_dir),
            population=_resolve(resources["population"], base_dir),
            uzw_grids_dir=_resolve(resources["uzw_grids_dir"], base_dir) if resources.get("uzw_grids_dir") else None,
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


# Loader --------------------------------------------------------------------------------------

RunConfig = SyntheticRun | PairedRun | AlignedRun


def run_config_from_dict(raw: dict[str, Any], *, base_dir: Path) -> RunConfig:
    """Parse a run YAML mapping (``base_dir``: directory of relative paths)."""
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
    if pipeline == "scenario":
        raise ValueError(
            "run.pipeline 'scenario' was renamed to 'synthetic', which runs Steps 2-4 and the "
            "expansion analyses (not Step 2 only). Rename it in the run YAML."
        )
    pipeline = choice(pipeline, "run.pipeline", PIPELINES)
    if not run.get("scenario"):
        raise ValueError("run.scenario (the scenario YAML) is required.")
    scenario_path = _resolve(run["scenario"], base_dir)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario YAML not found: {scenario_path}")
    if pipeline == "synthetic":
        return SyntheticRun.from_blocks(run_id, scenario_path, resources, execution)
    if pipeline == "paired_validation":
        return PairedRun.from_blocks(run_id, scenario_path, resources, execution)
    return AlignedRun.from_blocks(run_id, scenario_path, resources, execution, base_dir=base_dir)


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
