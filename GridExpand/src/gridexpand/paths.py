"""Directory layout of GridExpand; the only module that knows where files live.

Code, configuration and static inputs are located relative to the project
directory (``GridExpand/``). Static input data, runtime artifacts and the
``.env`` file can be relocated with environment variables:

- ``GRIDEXPAND_DATA_DIR``: static input data (default ``GridExpand/data``)
- ``GRIDEXPAND_WORK_DIR``: runtime artifacts (default ``GridExpand/work``)
- ``GRIDEXPAND_ENV_FILE``: database credentials (default ``GridExpand/.env``)

The variables are read from the process environment when this module is first
imported (not from ``.env``). Child processes started by the orchestrators
inherit them.
"""

from __future__ import annotations

import os
from pathlib import Path


def _from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]

CONFIG_DIR = PROJECT_DIR / "config"
SCENARIO_CONFIG_DIR = CONFIG_DIR / "scenarios"
RUN_CONFIG_DIR = CONFIG_DIR / "runs"
SQL_DIR = PACKAGE_DIR / "db" / "sql"

DATA_DIR = _from_env("GRIDEXPAND_DATA_DIR", PROJECT_DIR / "data")
WORK_DIR = _from_env("GRIDEXPAND_WORK_DIR", PROJECT_DIR / "work")
ENV_FILE = _from_env("GRIDEXPAND_ENV_FILE", PROJECT_DIR / ".env")

# Static inputs
STATISTICS_DIR = DATA_DIR / "statistics"
SAMPLING_DATA_DIR = DATA_DIR / "sampling"

# Runtime artifacts (hand-off files between steps, logs, caches, run folders)
SAMPLING_RESULTS_DIR = WORK_DIR / "sampling" / "results"
ALLOCATION_GRIDS_DIR = WORK_DIR / "allocation" / "grids"
ALLOCATION_RESULTS_DIR = WORK_DIR / "allocation" / "results"
ALLOCATION_OUTPUTS_DIR = WORK_DIR / "allocation" / "outputs"
ALLOCATION_LOGS_DIR = WORK_DIR / "allocation" / "logs"
OPTIMIZATION_INPUT_DIR = WORK_DIR / "optimization" / "input"
OPTIMIZATION_RESULT_DIR = WORK_DIR / "optimization" / "result"
OPTIMIZATION_LOGS_DIR = WORK_DIR / "optimization" / "logs"
POWERFLOW_INPUT_DIR = WORK_DIR / "powerflow" / "input"
POWERFLOW_OUTPUT_DIR = WORK_DIR / "powerflow" / "output"
POWERFLOW_ANALYSIS_DIR = WORK_DIR / "powerflow" / "analysis"
POWERFLOW_PLOTS_DIR = WORK_DIR / "powerflow" / "plots"
ANALYSIS_OUTPUT_DIR = WORK_DIR / "analysis" / "output"
RUNS_DIR = WORK_DIR / "runs"

# Paired/aligned scenario datasets written by Step 2 scenario calibration
SCENARIO_CALIBRATION_OUTPUT_DIR = ALLOCATION_OUTPUTS_DIR / "scenario_calibration"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
