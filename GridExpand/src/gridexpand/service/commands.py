"""Build the ``gridexpand`` commands of a pipeline job.

:func:`pipeline_steps` is the only place that knows how a pipeline request becomes command
lines. Today every model case is one ``gridexpand synthetic`` run; when
``gridexpand run <run.yaml>`` (with ``state.json``) lands, only this module changes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from gridexpand.service.jobs import JobStep

# Model cases the service offers. ``post-inflex-heuristic`` is left out: the synthetic
# Step 2 writes no ``urbs_in/ev_sessions``, so the synthetic INFLEX power flow cannot run.
MODEL_CASES = ("pre", "post-hems-heuristic", "post-hems-optimized")
POST_CASES = tuple(case for case in MODEL_CASES if case != "pre")


@dataclass(frozen=True)
class PipelineSpec:
    """A validated pipeline request.

    Attributes:
        ags: Municipality key (integer, as stored by GridExpand).
        pylovo_version_id: pylovo topology version.
        scenario_config: Scenario YAML.
        model_cases: Cases to run, in this order.
        timeframe_mode: One of :data:`gridexpand.common.timeframe.TIMEFRAME_MODES`.
        min_buildings: Candidate filter of the runner (changes the candidate numbering).
        start_index: First candidate index to run (``None``: from the first candidate).
        limit: Number of candidates from ``start_index`` (``None``: all).
        workers: Grids processed in parallel by the runner.
        powerflow_output: ``summary`` (compact metrics) or ``both`` (plus raw time series).
    """

    ags: int
    pylovo_version_id: str
    scenario_config: Path
    model_cases: tuple[str, ...]
    timeframe_mode: str
    min_buildings: int = 5
    start_index: int | None = None
    limit: int | None = None
    workers: int = 1
    powerflow_output: str = "summary"


def profiles_for_case(case: str) -> str:
    """Demand profile scope of a model case (``pre`` = status quo)."""
    return "status_quo" if case == "pre" else "all"


def synthetic_command(spec: PipelineSpec, case: str, run_dir: Path, python: str = sys.executable) -> list[str]:
    """``gridexpand synthetic`` for one model case."""
    argv = [
        python, "-m", "gridexpand", "synthetic",
        "--ags", str(spec.ags),
        "--pylovo-version-id", str(spec.pylovo_version_id),
        "--min-buildings", str(spec.min_buildings),
        "--scenario-config", str(spec.scenario_config),
        "--model-case", case,
        "--profiles", profiles_for_case(case),
        "--case-qualified-output",
        "--timeframe-mode", spec.timeframe_mode,
        "--powerflow-output", spec.powerflow_output,
        "--workers", str(spec.workers),
        "--run-dir", str(run_dir),
    ]
    if spec.start_index is not None:
        # The pilot grid (default: candidate 0) gates the batch; make it the first selected grid.
        argv += ["--start-index", str(spec.start_index), "--pilot-index", str(spec.start_index)]
    if spec.limit is not None:
        argv += ["--limit", str(spec.limit)]
    return argv


def pipeline_steps(spec: PipelineSpec, job_run_dir: Path, python: str = sys.executable) -> list[JobStep]:
    """One job step per model case, each with its own run directory ``<job_run_dir>/<case>``."""
    steps = []
    for case in spec.model_cases:
        run_dir = Path(job_run_dir) / case
        steps.append(JobStep(name=case, argv=synthetic_command(spec, case, run_dir, python), run_dir=str(run_dir)))
    return steps


def contiguous_range(indexes: list[int]) -> tuple[int, int]:
    """``(start_index, limit)`` of a list of candidate indexes without gaps.

    Raises:
        ValueError: If the list is empty or has gaps (the runner selects one range).
    """
    values = sorted({int(i) for i in indexes})
    if not values:
        raise ValueError("No grid selected")
    if values[-1] - values[0] + 1 != len(values):
        raise ValueError("Select grids with consecutive candidate numbers (the runner runs one range)")
    return values[0], len(values)
