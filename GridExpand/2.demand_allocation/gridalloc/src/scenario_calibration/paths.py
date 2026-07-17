"""Filesystem locations shared by scenario-calibration modules."""

from pathlib import Path

SCENARIO_CALIBRATION_DIR = Path(__file__).resolve().parent
SRC_DIR = SCENARIO_CALIBRATION_DIR.parent
GRIDALLOC_DIR = SRC_DIR.parent
STEP2_DIR = GRIDALLOC_DIR.parent
GRIDEXPAND_DIR = STEP2_DIR.parent
ENV_PATH = GRIDEXPAND_DIR / ".env"
DEMAND_STATISTICS_DIR = GRIDALLOC_DIR / "data" / "statistics"
SYNTHETIC_INPUT_DIR = GRIDEXPAND_DIR / "3.urbs" / "Input"
