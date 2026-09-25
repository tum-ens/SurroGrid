"""Filesystem locations and environment settings for scenario calibration.

All directories come from :mod:`gridexpand.paths`; the aliases below keep the
names used throughout ``scenario_calibration``.
"""

from gridexpand.paths import (
    ALLOCATION_RESULTS_DIR,
    ENV_FILE,
    OPTIMIZATION_INPUT_DIR,
    SCENARIO_CALIBRATION_OUTPUT_DIR,
    SCENARIO_CONFIG_DIR,
    STATISTICS_DIR,
)

ENV_PATH = ENV_FILE
DEMAND_STATISTICS_DIR = STATISTICS_DIR
# Step 3 input directory: paired Step 2 inputs and synthetic heat sources.
SYNTHETIC_INPUT_DIR = OPTIMIZATION_INPUT_DIR
# Prepared paired/aligned datasets and profile libraries.
OUTPUT_DIR = SCENARIO_CALIBRATION_OUTPUT_DIR
# Step 2 result directory (per-scenario HDF5 outputs, weather sources).
RESULTS_DIR = ALLOCATION_RESULTS_DIR

__all__ = [
    "DEMAND_STATISTICS_DIR",
    "ENV_PATH",
    "OUTPUT_DIR",
    "RESULTS_DIR",
    "SCENARIO_CONFIG_DIR",
    "SYNTHETIC_INPUT_DIR",
]
