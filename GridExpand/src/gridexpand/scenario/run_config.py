"""Run YAMLs (``config/runs/*.yaml``): what to run, on which grids, with which scenario.

A run YAML is one short file::

    run: my_town_2045                 # run id (directory-safe)
    scenario: default_2045            # config/scenarios/<name>.yaml, or a path to a scenario YAML
    scenario_overrides: {...}         # optional: single scenario values, merged into the scenario
    grids:
      pylovo: {version: "3", ags: 9184137}   # synthetic grids of a pylovo version (optional plz, kcid/bcid, ...)
    model_cases: [post-inflex-heuristic, post-hems-heuristic, post-hems-optimized]   # optional, the defaults
    execution: {...}                  # optional: CPUs, solver, outputs, seed (keys in docs/configuration.md)

The scenario is the scientific part: it is shared by many runs and its hash names the results. With
``scenario_overrides`` the run uses the merged scenario, written to ``<run dir>/inputs/scenario.yaml``; its hash
is taken after the merge. ``grids: files`` runs own grid files (:class:`RealRun`, ``pipeline: real``); both
grid sources run the same scenario method (:mod:`gridexpand.scenario.grid_run`), the pre stage always included.

Parsing never touches a database. Relative paths resolve against the YAML's directory. :func:`load_run_config`
returns the run and the SHA-256 of the parsed YAML (``run_hash``). The real-vs-synthetic validation has its own
run format in ``GridExpand/validation``.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gridexpand.common.reproducibility import DEFAULT_PROFILE_SEED
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import _read_yaml, configuration_hash
from gridexpand.scenario.model_cases import POST_MODEL_CASES, validate_cases
from gridexpand.scenario.scenario_config import _mapping as mapping
from gridexpand.scenario.scenario_config import _only as only
from gridexpand.scenario.scenario_config import _positive

PIPELINES = ("synthetic", "real")
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


def _pylovo_version(pylovo: dict[str, Any]) -> str:
    version = str(pylovo.get("version", "") or "").strip()
    if not version:
        raise ValueError("grids.pylovo.version (the pylovo version id) must be set explicitly.")
    return version


def resolve_scenario(value: Any, base_dir: Path) -> Path:
    """A shipped scenario by name (``config/scenarios/<name>.yaml``) or a path to a scenario YAML."""
    if not value:
        raise ValueError("scenario is required: a scenario name of config/scenarios/ or a path to a scenario YAML.")
    text = str(value)
    if "/" in text or text.endswith((".yaml", ".yml")):
        path = _resolve(text, base_dir)
    else:
        path = SCENARIO_CONFIG_DIR / f"{text}.yaml"
    if not path.is_file():
        names = sorted(item.stem for item in SCENARIO_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"Scenario {text!r} not found ({path}); shipped scenarios: {names}.")
    return path


def deep_merge(base: Any, overrides: Any) -> Any:
    """``base`` with ``overrides`` merged in: mappings key by key, everything else replaced."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return copy.deepcopy(overrides)
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        merged[key] = deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
    return merged


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


# Runs -----------------------------------------------------------------------------------------

# Every run computes the pre stage once (with the first execution group); ``[pre]`` alone runs only the pre stage.
DEFAULT_MODEL_CASES = ("post-inflex-heuristic", "post-hems-heuristic", "post-hems-optimized")
EXECUTION_KEYS = frozenset({
    "profile_seed", "weather_year", "workers", "step3_cpus", "step3_cluster_concurrency", "step4_cpus",
    "heat_workers", "solver", "optimizer", "powerflow_grid_scope", "powerflow_max_timesteps",
    "cleanup_intermediates", "materialize_expansion", "resume",
})


def _cases(model_cases: Any) -> tuple[str, ...]:
    """The model cases; ``pre`` is implied by any post case and kept only alone (a status-quo run)."""
    if isinstance(model_cases, str) or not isinstance(model_cases, (list, tuple)):
        raise ValueError("model_cases must be a list of model cases.")
    cases = validate_cases(model_cases)
    posts = tuple(case for case in cases if case in POST_MODEL_CASES)
    return posts or ("pre",)


def _execution(execution: dict[str, Any]) -> dict[str, Any]:
    """The execution fields shared by both grid sources (``execution`` keys of the run YAML)."""
    only(execution, set(EXECUTION_KEYS), "execution")

    def number(name: str, default: int) -> int:
        return positive_int(execution.get(name, default), f"execution.{name}")

    weather_year = execution.get("weather_year")
    max_timesteps = execution.get("powerflow_max_timesteps")
    solver = execution.get("solver")
    if solver is not None:
        from gridexpand.optimization.solver import SUPPORTED_SOLVERS

        solver = choice(solver, "execution.solver", SUPPORTED_SOLVERS)
    return {
        "profile_seed": _seed(execution),
        "weather_year": None if weather_year is None else positive_int(weather_year, "execution.weather_year"),
        "workers": number("workers", 1),
        "step3_cpus": number("step3_cpus", 16),
        "step3_cluster_concurrency": _concurrency(execution),
        "step4_cpus": number("step4_cpus", 4),
        "heat_workers": number("heat_workers", 4),
        "solver": solver,
        "optimizer": _optimizer(execution),
        "powerflow_grid_scope": _grid_scope(execution),
        "powerflow_max_timesteps": None if max_timesteps is None else positive_int(
            max_timesteps, "execution.powerflow_max_timesteps"),
        "cleanup_intermediates": flag(execution.get("cleanup_intermediates", False), "execution.cleanup_intermediates"),
        "materialize_expansion": flag(execution.get("materialize_expansion", True), "execution.materialize_expansion"),
        "resume": flag(execution.get("resume", False), "execution.resume"),
    }


@dataclass(frozen=True)
class _GridRun:
    """What both grid sources share: scenario, model cases and execution (:mod:`gridexpand.scenario.grid_run`).

    ``scenario_path`` is the scenario file the steps read: the named scenario, or with overrides the merged copy in
    the run directory (set by ``gridexpand run``).
    """

    run_id: str
    scenario_path: Path
    model_cases: tuple[str, ...]
    profile_seed: int = DEFAULT_PROFILE_SEED
    weather_year: int | None = None
    workers: int = 1
    step3_cpus: int = 16
    step3_cluster_concurrency: int | None = None
    step4_cpus: int = 4
    heat_workers: int = 4
    solver: str | None = None
    optimizer: str | None = None
    powerflow_grid_scope: str = "full"
    powerflow_max_timesteps: int | None = None
    cleanup_intermediates: bool = False
    materialize_expansion: bool = True
    resume: bool = False
    scenario_source: Path | None = None
    scenario_overrides: dict[str, Any] = field(default_factory=dict)

    def scenario_raw(self) -> dict[str, Any]:
        """The scenario mapping of the run: the named scenario with the overrides merged in."""
        raw = _read_yaml(self.scenario_source or self.scenario_path)
        return deep_merge(raw, self.scenario_overrides) if self.scenario_overrides else raw

    def _identity(self) -> dict[str, Any]:
        """What a resumed run must share with the recorded one (not resources or cases)."""
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "profile_seed": self.profile_seed,
            "weather_year": self.weather_year,
            "powerflow_grid_scope": self.powerflow_grid_scope,
            "powerflow_max_timesteps": self.powerflow_max_timesteps,
        }


@dataclass(frozen=True)
class SyntheticRun(_GridRun):
    """The pylovo grids of one region (``grids: pylovo``) through the scenario method.

    ``grids.pylovo`` selects the region (``ags``; optional ``plz`` or ``plz/kcid/bcid``) and optionally a candidate
    range (``start_index``/``limit``), numbered as ``gridexpand grids`` lists them.
    """

    pylovo_version_id: str = ""
    ags: int = 0
    plz: int | None = None
    kcid: int | None = None
    bcid: int | None = None
    min_buildings: int = 5
    start_index: int | None = None
    limit: int | None = None
    pipeline: str = "synthetic"

    PYLOVO_KEYS = frozenset({"version", "ags", "plz", "kcid", "bcid", "min_buildings", "start_index", "limit"})

    @classmethod
    def from_blocks(cls, run_id: str, scenario_source: Path, scenario_overrides: dict[str, Any],
                    pylovo: dict[str, Any], model_cases: Any, execution: dict[str, Any]) -> "SyntheticRun":
        only(pylovo, set(cls.PYLOVO_KEYS), "grids.pylovo")
        ags = optional_int(pylovo.get("ags"), "grids.pylovo.ags")
        if ags is None:
            raise ValueError("grids.pylovo.ags (the municipality key of the region) is required.")
        plz = optional_int(pylovo.get("plz"), "grids.pylovo.plz")
        kcid = optional_int(pylovo.get("kcid"), "grids.pylovo.kcid")
        bcid = signed_optional_int(pylovo.get("bcid"), "grids.pylovo.bcid")
        if (kcid is None) != (bcid is None):
            raise ValueError("grids.pylovo.kcid and grids.pylovo.bcid must be set together.")
        if kcid is not None and plz is None:
            raise ValueError("grids.pylovo.plz is required when kcid/bcid are set.")
        return cls(
            run_id=run_id,
            scenario_path=scenario_source,
            scenario_source=scenario_source,
            scenario_overrides=dict(scenario_overrides),
            model_cases=_cases(model_cases),
            pylovo_version_id=_pylovo_version(pylovo),
            ags=ags,
            plz=plz,
            kcid=kcid,
            bcid=bcid,
            min_buildings=positive_int(pylovo.get("min_buildings", cls.min_buildings), "grids.pylovo.min_buildings"),
            start_index=optional_int(pylovo.get("start_index"), "grids.pylovo.start_index"),
            limit=None if pylovo.get("limit") is None else positive_int(pylovo["limit"], "grids.pylovo.limit"),
            **_execution(execution),
        )

    def identity(self) -> dict[str, Any]:
        return {
            **self._identity(),
            "pylovo_version_id": self.pylovo_version_id,
            "region": {"ags": self.ags, "plz": self.plz, "kcid": self.kcid, "bcid": self.bcid},
            "min_buildings": self.min_buildings,
            "start_index": self.start_index,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class RealRun(_GridRun):
    """Own grid files (``grids: files``): matching, then the scenario method like pylovo grids."""

    grid_dir: Path = Path()
    points: Path | None = None
    crs: str = "EPSG:25832"
    matching: dict[str, Any] = field(default_factory=dict)
    pipeline: str = "real"
    pylovo_version_id: str = "none"  # own grids read no pylovo result

    FILES_KEYS = frozenset({"dir", "points", "crs"})

    @classmethod
    def from_blocks(cls, run_id: str, scenario_source: Path, scenario_overrides: dict[str, Any], files: Any,
                    matching: dict[str, Any], model_cases: Any, execution: dict[str, Any], base_dir: Path) -> "RealRun":
        if isinstance(files, str):
            files = {"dir": files}
        files = mapping(files, "grids.files")
        only(files, set(cls.FILES_KEYS), "grids.files")
        if not files.get("dir"):
            raise ValueError("grids.files.dir (the directory of the pandapower JSON grid files) is required.")
        grid_dir = _resolve(files["dir"], base_dir)
        if not grid_dir.is_dir():
            raise FileNotFoundError(f"grids.files.dir is no directory: {grid_dir}")
        points = None if files.get("points") in (None, "") else _resolve(files["points"], base_dir)
        if points is not None and not points.is_file():
            raise FileNotFoundError(f"grids.files.points not found: {points}")
        from gridexpand.grids.matching import MatchingConfig

        MatchingConfig.from_dict(matching)  # validated here, applied by gridexpand match
        return cls(
            run_id=run_id,
            scenario_path=scenario_source,
            scenario_source=scenario_source,
            scenario_overrides=dict(scenario_overrides),
            model_cases=_cases(model_cases),
            grid_dir=grid_dir,
            points=points,
            crs=str(files.get("crs", cls.crs)),
            matching=dict(matching),
            **_execution(execution),
        )

    def identity(self) -> dict[str, Any]:
        return {
            **self._identity(),
            "grid_dir": str(self.grid_dir),
            "points": None if self.points is None else str(self.points),
            "crs": self.crs,
            "matching": self.matching,
        }


# Loader --------------------------------------------------------------------------------------

RunConfig = SyntheticRun | RealRun


TOP_LEVEL_KEYS = {"run", "scenario", "scenario_overrides", "grids", "matching", "model_cases", "execution"}


def run_config_from_dict(raw: dict[str, Any], *, base_dir: Path) -> RunConfig:
    """Parse a run YAML mapping (``base_dir``: directory of relative paths)."""
    raw = mapping(raw, "run configuration")
    if "resources" in raw or isinstance(raw.get("run"), dict):
        raise ValueError(
            "This is a run YAML of the former format (run/resources/execution blocks). The current format is "
            "one short file: run, scenario, grids (pylovo: version, ags, ...), optional model_cases and execution; "
            "see config/runs/00_run_template.yaml. Validation runs use 'python -m validation run'."
        )
    only(raw, TOP_LEVEL_KEYS, "run YAML")
    run_id = directory_safe(raw.get("run"), "run")
    scenario_source = resolve_scenario(raw.get("scenario"), base_dir)
    overrides = mapping(raw.get("scenario_overrides") or {}, "scenario_overrides")
    grids = mapping(raw.get("grids"), "grids") if raw.get("grids") is not None else {}
    only(grids, {"pylovo", "files"}, "grids")
    if len(grids) != 1:
        raise ValueError("grids needs exactly one source: pylovo (synthetic grids) or files (own grid files).")
    execution = mapping(raw.get("execution") or {}, "execution")
    matching = mapping(raw.get("matching") or {}, "matching")
    if "files" in grids:
        return RealRun.from_blocks(run_id, scenario_source, overrides, grids["files"], matching,
                                   raw.get("model_cases", list(DEFAULT_MODEL_CASES)), execution, base_dir)
    if matching:
        raise ValueError("matching applies to own grid files (grids.files) only.")
    return SyntheticRun.from_blocks(run_id, scenario_source, overrides, mapping(grids["pylovo"], "grids.pylovo"),
                                    raw.get("model_cases", list(DEFAULT_MODEL_CASES)), execution)


def load_run_config(path: str | Path) -> tuple[RunConfig, str]:
    """Load a run YAML; returns the run and ``run_hash`` (SHA-256 of the parsed YAML)."""
    resolved = Path(path).resolve()
    raw = _read_yaml(resolved)
    return run_config_from_dict(raw, base_dir=resolved.parent), configuration_hash(raw)


def with_cases(run: RunConfig, cases: tuple[str, ...]) -> RunConfig:
    """``run`` restricted to ``cases``. Grid runs: ``pre`` alone is a status-quo run, else implied. Other runs (the
    validation): post cases only, the pre stage comes with the first group."""
    if isinstance(run, _GridRun):
        return dataclasses.replace(run, model_cases=_cases(list(cases)))
    return dataclasses.replace(run, model_cases=validate_cases(cases, allowed=POST_MODEL_CASES))
