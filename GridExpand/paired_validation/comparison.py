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


def read_temporal_method(result_hdf: Path) -> dict[str, Any]:
    """Read the Step-3 temporal-method audit written for every run."""
    try:
        audit = pd.read_hdf(result_hdf, "/urbs_out/temporal_method")
    except (FileNotFoundError, KeyError) as exc:
        raise ValueError(
            f"{result_hdf.name} has no '/urbs_out/temporal_method' record. It "
            "predates temporal provenance and cannot be accepted."
        ) from exc
    return dict(audit)


def _result_time_axis(result_hdf: Path) -> "tuple[int, list[int]] | None":
    """Return (operating-hour count, sorted distinct hour indices) of a result.

    Declared metadata is what a producer *claims*; this is what the file actually
    contains. The demand table is used because Step 3 writes it in both temporal
    modes and it carries the model hour index.
    """
    for key in ("urbs_out/reduced_data/demand", "urbs_in/demand"):
        try:
            frame = pd.read_hdf(result_hdf, key)
        except (FileNotFoundError, KeyError):
            continue
        index = frame.index
        if isinstance(index, pd.MultiIndex) and "t" in (index.names or []):
            hours = index.get_level_values("t")
        else:
            hours = index
        values = sorted({int(value) for value in hours})
        # t = 0 is the storage initialization row and carries no energy.
        operating = [value for value in values if value > 0]
        return len(operating), operating
    return None


def validate_full_year_result(
    result_hdf: Path,
    *,
    expected_operating_hours: int,
    expected_scenario_key: str | None = None,
    expected_scenario_hash: str | None = None,
    expected_delta_t_hours: float = 1.0,
    expected_source_year: int | None = None,
) -> dict[str, Any]:
    """Reject any result that is not the requested full-year chronological run.

    Neither the file name nor the presence of the 'reduced_data' group proves
    anything: Step 3 writes that group in both modes. Identity is decided by the
    temporal-method record, the operating-hour count, the timestep occurrence
    weight, the EV boundary policy *and* the time axis the file actually
    contains. A missing field is never treated as agreement.
    """
    audit = read_temporal_method(result_hdf)
    method = str(audit.get("temporal_method"))
    if method != "full_year_no_tsam":
        raise ValueError(
            f"{result_hdf.name} was produced with temporal_method={method!r}; a "
            "full-year chronological reference run was requested. Stale "
            "representative-period results are never reused."
        )
    operating_hours = int(audit.get("operating_hours", -1))
    if operating_hours != int(expected_operating_hours):
        raise ValueError(
            f"{result_hdf.name} has {operating_hours} operating hours, expected "
            f"{expected_operating_hours}."
        )
    annual_weight = float(audit.get("annual_weight", 0.0))
    if abs(annual_weight - 1.0) > 1e-9:
        raise ValueError(
            f"{result_hdf.name} uses annual weight {annual_weight}, but a "
            "full-year run must weight every operating hour exactly once."
        )
    delta_t = audit.get("delta_t_hours")
    if delta_t is None:
        raise ValueError(f"{result_hdf.name} does not record delta_t_hours.")
    if abs(float(delta_t) - float(expected_delta_t_hours)) > 1e-9:
        raise ValueError(
            f"{result_hdf.name} uses delta_t_hours={delta_t}, expected "
            f"{expected_delta_t_hours}."
        )
    if str(audit.get("storage_boundary_policy")) != "annual_equality":
        raise ValueError(
            f"{result_hdf.name} does not use the annual storage-closure "
            f"equality (found {audit.get('storage_boundary_policy')!r})."
        )
    if str(audit.get("ev_boundary_policy")) != "dedicated_sessions_annual_wrap":
        raise ValueError(
            f"{result_hdf.name} does not use the dedicated EV session contract "
            f"(found {audit.get('ev_boundary_policy')!r})."
        )
    if expected_source_year is not None:
        found_year = audit.get("source_reference_year")
        if found_year is None or int(found_year) != int(expected_source_year):
            raise ValueError(
                f"{result_hdf.name} maps to source year {found_year!r}, expected "
                f"{expected_source_year}."
            )
    if expected_scenario_key is not None:
        found_key = str(audit.get("scenario_key", ""))
        if found_key != str(expected_scenario_key):
            raise ValueError(
                f"{result_hdf.name} belongs to scenario_key={found_key!r}, but "
                f"{expected_scenario_key!r} was requested."
            )
    if expected_scenario_hash is not None:
        found_hash = str(audit.get("scenario_hash", ""))
        if found_hash != str(expected_scenario_hash):
            raise ValueError(
                f"{result_hdf.name} was built from scenario_hash={found_hash!r}, "
                f"but {expected_scenario_hash!r} was requested."
            )

    # The declared horizon must match the axis the file actually contains.
    axis = _result_time_axis(result_hdf)
    if axis is None:
        raise ValueError(
            f"{result_hdf.name} has no readable demand table, so its operating "
            "hours cannot be verified against its declared metadata."
        )
    actual_hours, operating_indices = axis
    if actual_hours != int(expected_operating_hours):
        raise ValueError(
            f"{result_hdf.name} declares {expected_operating_hours} operating "
            f"hours but its demand axis contains {actual_hours}."
        )
    expected_axis = list(range(1, int(expected_operating_hours) + 1))
    if operating_indices != expected_axis:
        first_gap = next(
            (
                position
                for position, value in enumerate(operating_indices)
                if value != expected_axis[position]
            ),
            None,
        )
        raise ValueError(
            f"{result_hdf.name} has a non-contiguous or misnumbered operating-hour "
            f"axis; first disagreement at position {first_gap}."
        )

    with pd.HDFStore(result_hdf, mode="r") as store:
        keys = set(store.keys())
    if "/urbs_out/tsam/clusterOrder" in keys:
        raise ValueError(
            f"{result_hdf.name} contains TSAM cluster metadata; it is not a "
            "full-year chronological result."
        )
    return audit
