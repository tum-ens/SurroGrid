"""Filesystem locations and environment settings for scenario calibration."""

import os
from pathlib import Path

from dotenv import load_dotenv

SCENARIO_CALIBRATION_DIR = Path(__file__).resolve().parent
SRC_DIR = SCENARIO_CALIBRATION_DIR.parent
GRIDALLOC_DIR = SRC_DIR.parent
STEP2_DIR = GRIDALLOC_DIR.parent
GRIDEXPAND_DIR = STEP2_DIR.parent
ENV_PATH = GRIDEXPAND_DIR / ".env"
DEMAND_STATISTICS_DIR = GRIDALLOC_DIR / "data" / "statistics"
SYNTHETIC_INPUT_DIR = GRIDEXPAND_DIR / "3.urbs" / "Input"


def configured_pylovo_version_id() -> str:
    """Return the pylovo version selected in ``GridExpand/.env``."""
    load_dotenv(ENV_PATH, override=True)
    value = os.getenv("PYLOVO_VERSION_ID", "").strip().strip(chr(34)).strip(chr(39))
    if not value:
        raise ValueError(
            f"PYLOVO_VERSION_ID must be set in {ENV_PATH} for paired scenarios."
        )
    return value
