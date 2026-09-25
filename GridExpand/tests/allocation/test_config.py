"""Step 2 configuration: no data reads at import, same tables on access."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd

from gridexpand.allocation.config import config
from gridexpand.paths import STATISTICS_DIR


def test_import_reads_no_data_files():
    code = (
        "import pandas as pd\n"
        "calls = []\n"
        "original = pd.read_csv\n"
        "pd.read_csv = lambda *a, **k: calls.append(a) or original(*a, **k)\n"
        "import gridexpand.allocation.config\n"
        "print(len(calls))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "0"


def test_tables_match_direct_reads_and_are_cached():
    stat = str(STATISTICS_DIR)
    expected = {
        "ELEC_BY_HHSIZE_CDFS_NOHEAT": pd.read_csv(f"{stat}/inhabited_buildings/elec_by_hhsize_cdfs_noheat.csv", header=[0, 1]),
        "HH_SIZE_DISTRIBUTION": pd.read_csv(f"{stat}/inhabited_buildings/hh_size_distribution.csv", header=[0], skiprows=1),
        "TYPE_GHD_DISTRIBUTION": pd.read_csv(f"{stat}/uninhabited_buildings/nonresbuilding_usetype_distribution.csv", header=[0], skiprows=1),
        "AGE_GHD_DISTRIBUTION": pd.read_csv(f"{stat}/uninhabited_buildings/nonresbuilding_age_distribution.csv", header=[0], skiprows=1),
        "CARS_PER_HH_BY_REGION": pd.read_csv(
            f"{stat}/general/cars_per_household_by_region.csv",
            dtype={"region": int, "hh_size": int, "vehicle_count": int, "probability": float}, skiprows=1),
        "CAR_MODEL_DISTRIBUTION": pd.read_csv(
            f"{stat}/general/cars_by_model.csv", dtype={"model": str, "probability": float}, skiprows=1),
    }
    for name, frame in expected.items():
        actual = getattr(config, name)
        pd.testing.assert_frame_equal(actual, frame, check_exact=True)
        assert getattr(config, name) is actual
