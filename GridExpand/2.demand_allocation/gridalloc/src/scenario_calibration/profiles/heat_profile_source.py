"""Resolve physical heat profiles independently of target-network buses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .physical_heat_profile_library import PhysicalHeatProfileLibrary


def load_physical_heat_profile(
    *,
    building_objectid: object,
    group: pd.DataFrame,
    hours: int,
    synthetic_input_dir: Path,
    library: PhysicalHeatProfileLibrary | None,
    catalog_by_building: dict[object, dict[str, Any]],
    source_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    allow_diagnostic_fallback: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, str, int | None, str, float]:
    """Return one building's heat series and auditable source metadata."""
    building_id = str(building_objectid)
    catalog_row = catalog_by_building.get(building_objectid)
    if catalog_row is None:
        catalog_row = catalog_by_building.get(building_id)

    if library is not None and building_id in library:
        space, water, cop = library.read(building_id, hours=hours)
        return (
            space,
            water,
            cop,
            str(library.path),
            None,
            "exact_physical_building",
            1.0,
        )

    if catalog_row is not None and not bool(
        catalog_row.get("exact_profile_available", True)
    ):
        if not allow_diagnostic_fallback:
            raise ValueError(
                "No exact physical heat profile is available for "
                f"{building_id}. Add it to the physical heat-profile library "
                "or explicitly enable the diagnostic fallback."
            )
        profile_method = str(catalog_row["profile_method"])
        profile_scale = float(catalog_row["profile_scale"])
        source_kind = str(catalog_row.get("profile_source_kind", "legacy_grid_hdf"))
        if source_kind == "physical_heat_library":
            if library is None:
                raise ValueError(
                    f"Diagnostic profile for {building_id} requires a physical "
                    "heat-profile library."
                )
            source_building = str(
                catalog_row["profile_source_building_objectid"]
            )
            space, water, cop = library.read(source_building, hours=hours)
            return (
                space * profile_scale,
                water * profile_scale,
                cop,
                str(library.path),
                None,
                profile_method,
                profile_scale,
            )
        source_name = str(catalog_row["profile_source_hdf"])
        source_bus = int(catalog_row["profile_source_bus"])
    else:
        source_files = group["synthetic_bridge_filename"].dropna().unique()
        source_buses = (
            pd.to_numeric(group["synthetic_bus"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(source_files) != 1 or len(source_buses) != 1:
            raise ValueError(
                f"Physical building {building_id} has ambiguous legacy heat "
                "profile sources."
            )
        source_name = str(source_files[0])
        source_bus = int(source_buses[0])
        profile_method = "exact_physical_building"
        profile_scale = 1.0

    if source_name not in source_cache:
        source_path = synthetic_input_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing physical-building profile source {source_path}."
            )
        source_cache[source_name] = (
            pd.read_hdf(source_path, key="urbs_in/demand"),
            pd.read_hdf(source_path, key="urbs_in/eff_factor"),
        )
    source_demand, source_eff = source_cache[source_name]
    needed_demand = [
        (source_bus, "space_heat"),
        (source_bus, "water_heat"),
    ]
    needed_eff = (source_bus, "heatpump_air")
    if any(column not in source_demand for column in needed_demand):
        raise ValueError(
            f"Missing heat demand for physical building {building_id} "
            f"at {source_name}:{source_bus}."
        )
    if needed_eff not in source_eff:
        raise ValueError(
            f"Missing heat-pump COP for physical building {building_id} "
            f"at {source_name}:{source_bus}."
        )
    space = (
        pd.to_numeric(source_demand[needed_demand[0]], errors="coerce")
        .fillna(0.0)
        .iloc[:hours]
        .reset_index(drop=True)
        * profile_scale
    )
    water = (
        pd.to_numeric(source_demand[needed_demand[1]], errors="coerce")
        .fillna(0.0)
        .iloc[:hours]
        .reset_index(drop=True)
        * profile_scale
    )
    cop = (
        pd.to_numeric(source_eff[needed_eff], errors="coerce")
        .fillna(1.0)
        .iloc[:hours]
        .reset_index(drop=True)
        .clip(lower=0.1)
    )
    return (
        space,
        water,
        cop,
        source_name,
        source_bus,
        profile_method,
        profile_scale,
    )
