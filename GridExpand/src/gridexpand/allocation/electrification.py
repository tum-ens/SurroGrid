"""Electrification eligibility of physical buildings for the synthetic Step 2.

The heat, mobility and PV/battery adopters are selected among eligible
buildings by :func:`gridexpand.common.electrification.build_electrification_assignment`.
The eligibility inventory is derived here once for both synthetic entry
points: one grid (``Grid.prepare_electrification_assignment``) and one region
(``electrification_preparation``). The paired and aligned pipelines keep their
own copies, because their assignment hashes are stored in prepared datasets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridexpand.allocation.functions.infdb_ro_heat import validate_space_heat_coverage
from gridexpand.common.electrification import TECHNOLOGIES, assignment_manifest_hash

INVENTORY_COLUMNS = (
    "heat_eligible",
    "heat_exclusion_reason",
    "mobility_eligible",
    "mobility_exclusion_reason",
    "pv_battery_eligible",
    "pv_battery_exclusion_reason",
)


def has_household(occupancy: pd.Series) -> pd.Series:
    """Return True where a building's occupancy list has at least one household."""
    return occupancy.map(
        lambda value: isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
    ).astype(bool)


def electrification_inventory(
    building_ids: pd.Series,
    *,
    residential: pd.Series,
    has_household: pd.Series,
    vehicle_count: pd.Series,
    roof_capacity_kw: pd.Series,
    annual_electricity_kwh: pd.Series,
) -> pd.DataFrame:
    """Return per-technology eligibility and the first failing exclusion reason.

    All inputs are aligned on the index of ``building_ids``. Reasons are
    checked in a fixed order: heat needs a residential component; mobility a
    residential component, a household and at least one sampled vehicle;
    PV/battery a usable (genuine) LoD2 roof first and positive base electricity
    second.

    Args:
        building_ids: Physical building object ids.
        residential: Whether the building has an included residential component.
        has_household: Whether the building has at least one household.
        vehicle_count: Sampled vehicles of the building (missing counts as 0).
        roof_capacity_kw: Usable LoD2 roof capacity (missing counts as 0).
        annual_electricity_kwh: Annual base electricity (missing counts as 0).

    Returns:
        ``building_objectid`` plus the columns in :data:`INVENTORY_COLUMNS`.
    """
    residential = residential.astype(bool)
    household = has_household.astype(bool)
    vehicles = pd.to_numeric(vehicle_count, errors="coerce").fillna(0.0)
    roof = pd.to_numeric(roof_capacity_kw, errors="coerce").fillna(0.0)
    annual = pd.to_numeric(annual_electricity_kwh, errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "building_objectid": building_ids.astype(str),
            "heat_eligible": residential,
            "heat_exclusion_reason": np.where(residential, None, "no_residential_component"),
            "mobility_eligible": residential & household & vehicles.gt(0.0),
            "mobility_exclusion_reason": np.select(
                [~residential, ~household, vehicles.le(0.0)],
                ["no_residential_component", "no_household", "no_vehicle_inventory"],
                default=None,
            ),
            "pv_battery_eligible": roof.gt(0.0) & annual.gt(0.0),
            "pv_battery_exclusion_reason": np.select(
                [roof.le(0.0), annual.le(0.0)],
                ["no_usable_lod2_roof", "no_base_electricity"],
                default=None,
            ),
        },
        index=building_ids.index,
    )


def check_heat_profile_source(source: str, heat_buildings: pd.DataFrame, *, engine=None) -> None:
    """Fail before adopter selection if the space-heat source cannot cover a building.

    TEASER generates a profile for every residential building. For INFDB
    ``ro_heat`` the coverage is checked against the database.

    Args:
        source: ``asset_sizing.heat.space_heat_source`` of the scenario.
        heat_buildings: Residential buildings with ``residential_effective_floor_area_m2``.
        engine: Optional SQLAlchemy engine for ``ro_heat``.
    """
    if source == "teaser":
        return
    if source != "infdb_ro_heat":
        raise ValueError(f"Unknown space heat source {source!r}.")
    if heat_buildings.empty:
        return
    validate_space_heat_coverage(heat_buildings, engine=engine)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepared_assignment(
    path: Path,
    building_ids: Iterable[object],
    *,
    scenario_hash: str,
    profile_seed: int,
) -> tuple[pd.DataFrame, str | None, list[dict[str, Any]] | None]:
    """Return the rows of one grid from a prepared (regional) assignment.

    The sidecar ``<path>.json`` written by ``electrification_preparation``
    pins the scenario, the profile seed and the content: its
    ``assignment_file_sha256`` is compared with the file (one pass over the
    bytes). Sidecars without that key fall back to recomputing the manifest
    hash of all rows, which costs a Python loop over the whole region.

    Args:
        path: Assignment CSV (or HDF with ``raw_data/electrification_assignment``).
        building_ids: Physical buildings of the current grid.
        scenario_hash: Hash of the active scenario YAML.
        profile_seed: Seed of the active run.

    Returns:
        The grid's assignment rows (one per building and technology), the
        sidecar's ``assignment_hash`` and its ``assignment_summary``.

    Raises:
        ValueError: If the sidecar is missing, the content, scenario or seed
            differ, or the rows do not cover the grid exactly.
    """
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        existing = pd.read_csv(path)
    else:
        existing = pd.read_hdf(path, key="raw_data/electrification_assignment")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(
            f"Prepared electrification assignment is missing sidecar: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_hash = metadata.get("assignment_hash")
    expected_file_hash = metadata.get("assignment_file_sha256")
    if expected_file_hash is not None:
        if file_sha256(path) != expected_file_hash:
            raise ValueError(
                "Prepared electrification assignment file differs from the file "
                "recorded in its sidecar."
            )
    elif source_hash != assignment_manifest_hash(existing):
        raise ValueError(
            "Prepared electrification assignment sidecar hash does not "
            "match the assignment rows."
        )
    ids = {str(value) for value in building_ids}
    existing["building_objectid"] = existing["building_objectid"].astype(str)
    existing = existing.loc[existing["building_objectid"].isin(ids)].copy()
    if len(existing) != len(ids) * len(TECHNOLOGIES) or existing.duplicated(
        ["building_objectid", "technology"]
    ).any():
        raise ValueError(
            "The supplied electrification assignment is not one row per "
            "current physical building and technology."
        )
    if metadata.get("scenario_hash") != scenario_hash:
        raise ValueError(
            "Prepared electrification assignment scenario_hash differs "
            "from the active Step-2 scenario."
        )
    if int(metadata.get("profile_seed", -1)) != int(profile_seed):
        raise ValueError(
            "Prepared electrification assignment profile_seed differs "
            "from the active Step-2 run."
        )
    return existing.reset_index(drop=True), source_hash, metadata.get("assignment_summary")
