"""Electrification eligibility of physical buildings for the synthetic Step 2.

The heat, mobility and PV/battery adopters are selected among eligible
buildings by :func:`gridexpand.common.electrification.build_electrification_assignment`.
The eligibility inventory is derived here once for both synthetic entry
points: one grid (``Grid.prepare_electrification_assignment``) and one region
(``electrification_preparation``). The paired and aligned pipelines keep their
own copies, because their assignment hashes are stored in prepared datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridexpand.allocation.functions.infdb_ro_heat import validate_space_heat_coverage

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
