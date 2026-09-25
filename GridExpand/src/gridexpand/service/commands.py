"""Build the ``gridexpand`` commands of a pipeline job.

:func:`pipeline_steps` is the only place that knows how a pipeline request becomes command
lines. A request becomes one ``pipeline: synthetic`` run YAML in the job's run directory,
and every model case is one ``gridexpand run <run.yaml> --model-case <case>`` step in that
directory: the steps share the run's identity, ``state.json`` and the regional
electrification assignment; each case's batch writes ``<run dir>/<case>/``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gridexpand.scenario.model_cases import MODEL_CASES as ALL_MODEL_CASES
from gridexpand.service.jobs import JobStep

# Model cases the service offers. ``post-inflex-heuristic`` is left out: the synthetic
# Step 2 writes no ``urbs_in/ev_sessions``, so the synthetic INFLEX power flow cannot run.
MODEL_CASES = ("pre", "post-hems-heuristic", "post-hems-optimized")
POST_CASES = tuple(case for case in MODEL_CASES if case != "pre")
RUN_YAML = "run.yaml"


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
        plz: Only the grids of this PLZ (``resources.plz``).
        kcid: With ``plz`` and ``bcid``: exactly one grid (also the scope of the
            electrification assignment).
        bcid: See ``kcid``.
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
    plz: int | None = None
    kcid: int | None = None
    bcid: int | None = None


def profiles_for_case(case: str) -> str:
    """Demand profile scope of a model case (``pre`` = status quo)."""
    return ALL_MODEL_CASES[case].profiles


def run_yaml(spec: PipelineSpec, run_id: str, scenario: str | None = None) -> dict[str, Any]:
    """The ``pipeline: synthetic`` run YAML of a request.

    ``scenario`` replaces the absolute scenario path (e.g. ``../scenarios/<file>`` for a
    run YAML that is copied to another checkout's ``config/runs/``).
    """
    resources: dict[str, Any] = {
        "pylovo_version_id": str(spec.pylovo_version_id),
        "ags": int(spec.ags),
        "min_buildings": int(spec.min_buildings),
    }
    if spec.plz is not None:
        resources["plz"] = int(spec.plz)
    if spec.kcid is not None:
        resources["kcid"] = int(spec.kcid)
        resources["bcid"] = int(spec.bcid)
    execution: dict[str, Any] = {
        "model_cases": list(spec.model_cases),
        "timeframe_mode": spec.timeframe_mode,
        "powerflow_output": spec.powerflow_output,
        "workers": int(spec.workers),
    }
    if spec.start_index is not None:
        resources["start_index"] = int(spec.start_index)
        # The pilot grid (default: candidate 0) gates the batch; make it the first selected grid.
        execution["pilot_index"] = int(spec.start_index)
    if spec.limit is not None:
        resources["limit"] = int(spec.limit)
    return {
        "run": {"id": run_id, "scenario": scenario or str(Path(spec.scenario_config).resolve()), "pipeline": "synthetic"},
        "resources": resources,
        "execution": execution,
    }


def run_yaml_text(spec: PipelineSpec, run_id: str, header: str, scenario: str | None = None) -> str:
    """The run YAML as text with a comment header."""
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    return comment + yaml.safe_dump(run_yaml(spec, run_id, scenario), sort_keys=False)


def write_run_yaml(spec: PipelineSpec, job_run_dir: Path) -> Path:
    """Write the run YAML of a request into its run directory; returns its path."""
    job_run_dir = Path(job_run_dir)
    job_run_dir.mkdir(parents=True, exist_ok=True)
    path = job_run_dir / RUN_YAML
    run_id = f"service_{job_run_dir.name}"
    path.write_text(run_yaml_text(spec, run_id, "Written by the GridExpand service for one pipeline job."),
                    encoding="utf-8")
    return path


def terminal_commands(run_yaml_path: Path, run_id: str, *, in_container: bool, project_dir: Path) -> list[dict[str, str]]:
    """Shell commands that run a prepared run YAML in a ``tmux`` session and follow it.

    Long runs (a whole PLZ, full years) belong in a terminal: they survive a closed browser
    and a restarted service. In a container (GridPlanner) the commands go through
    ``docker compose exec`` from the GridPlanner folder.
    """
    session = re.sub(r"[^\w-]", "-", run_id)[:40]
    if in_container:
        run = f"docker compose exec gridexpand gridexpand run {run_yaml_path}"
        status = f"docker compose exec gridexpand gridexpand status {run_id}"
        where = "in the GridPlanner folder"
    else:
        run = f"uv run gridexpand run {run_yaml_path}"
        status = f"uv run gridexpand status {run_id}"
        where = f"in {project_dir}"
    return [
        {"label": f"Start in tmux ({where})", "command": f"tmux new-session -d -s {session} '{run}; exec bash'"},
        {"label": "Watch the live log", "command": f"tmux attach -t {session}"},
        {"label": "Status (also shown in this panel)", "command": status},
        {"label": "Resume after an interruption", "command": f"{run} --resume"},
    ]


def run_command(run_yaml_path: Path, job_run_dir: Path, case: str, python: str = sys.executable) -> list[str]:
    """``gridexpand run`` of one model case of the job's run YAML."""
    return [python, "-m", "gridexpand", "run", str(run_yaml_path), "--run-dir", str(job_run_dir),
            "--model-case", case]


def pipeline_steps(spec: PipelineSpec, job_run_dir: Path, python: str = sys.executable) -> list[JobStep]:
    """One job step per model case; each follows the case's batch directory ``<job_run_dir>/<case>``.

    Writes ``<job_run_dir>/run.yaml``.
    """
    yaml_path = write_run_yaml(spec, job_run_dir)
    return [
        JobStep(name=case, argv=run_command(yaml_path, job_run_dir, case, python),
                run_dir=str(Path(job_run_dir) / case))
        for case in spec.model_cases
    ]


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
