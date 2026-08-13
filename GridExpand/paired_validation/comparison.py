"""Network-independent equivalence checks for paired validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_tsam_signature(result_hdf: Path) -> dict[str, Any]:
    prefix = "/urbs_out/tsam"

    def values(name: str) -> list[Any]:
        frame = pd.read_hdf(result_hdf, f"{prefix}/{name}")
        return frame.to_numpy().reshape(-1).tolist()

    return {
        "selection_variables": ["Tamb", "Irradiation"],
        "cluster_center_indices": [
            int(value) for value in values("clusterCenterIndices")
        ],
        "cluster_order": [int(value) for value in values("clusterOrder")],
        "hours_per_period": int(values("hoursPerPeriod")[0]),
        "number_of_typical_periods": int(values("noTypicalPeriods")[0]),
        "extreme_period_method": str(values("extremePeriodMethod")[0]),
    }


def validate_shared_tsam(
    result_hdf: Path,
    expected: dict[str, Any] | None,
) -> None:
    if expected is None:
        return
    actual = read_tsam_signature(result_hdf)
    expected_core = {key: expected[key] for key in actual}
    if actual != expected_core:
        raise ValueError(
            "TSAM mapping differs from the shared weather-based reference; "
            "power flow was not started for this grid."
        )
