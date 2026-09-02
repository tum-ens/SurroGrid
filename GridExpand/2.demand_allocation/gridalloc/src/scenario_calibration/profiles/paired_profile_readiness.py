"""Audit physical heat-profile coverage for the paired SWF scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import GRIDALLOC_DIR, SYNTHETIC_INPUT_DIR
from .physical_heat_profile_library import PhysicalHeatProfileLibrary

DEFAULT_PAIRED_DIR = (
    GRIDALLOC_DIR
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_paired_v5_91301_station_hybrid_v2"
)


def build_heat_profile_catalog(
    allocation: pd.DataFrame,
    building_plan: pd.DataFrame,
    *,
    synthetic_input_dir: Path = SYNTHETIC_INPUT_DIR,
    heat_profile_library: Path | None = None,
) -> pd.DataFrame:
    """Map each SWF heat-pump building to an exact or area-matched profile."""
    library = (
        PhysicalHeatProfileLibrary(heat_profile_library)
        if heat_profile_library is not None
        else None
    )
    # Paired heat materialization currently models residential_wp_rows only.
    wp_count = pd.to_numeric(
        allocation.get("residential_wp_rows", 0.0),
        errors="coerce",
    ).fillna(0.0)
    selected = allocation.loc[wp_count.gt(0.0)].copy()
    selected = selected.drop_duplicates("building_objectid")
    attributes = (
        building_plan[
            [
                "building_objectid",
                "residential_effective_floor_area_m2",
                "building_use",
                "building_type",
                "building_floor_area",
            ]
        ]
        .drop_duplicates("building_objectid")
        .copy()
    )
    selected = selected.merge(
        attributes,
        on="building_objectid",
        how="left",
        suffixes=("", "_building"),
    )
    if "building_use_building" in selected:
        selected["building_use"] = selected["building_use"].fillna(
            selected["building_use_building"]
        )
    if "residential_effective_floor_area_m2" in selected:
        selected["building_floor_area"] = selected[
            "residential_effective_floor_area_m2"
        ].fillna(selected["building_floor_area"])

    column_cache: dict[str, tuple[pd.Index, pd.Index]] = {}
    records = []
    for row in selected.to_dict("records"):
        building_id = str(row["building_objectid"])
        source_name = str(row["synthetic_bridge_filename"])
        source_bus = int(row["synthetic_bus"])
        library_exact = library is not None and building_id in library
        library_metadata = (
            library.profile_metadata(building_id) if library_exact else None
        )
        exact = False
        if not library_exact:
            source_path = synthetic_input_dir / source_name
            if source_name not in column_cache:
                if not source_path.exists():
                    demand_columns = pd.Index([])
                    efficiency_columns = pd.Index([])
                else:
                    demand_columns = pd.read_hdf(
                        source_path,
                        key="urbs_in/demand",
                        start=0,
                        stop=1,
                    ).columns
                    efficiency_columns = pd.read_hdf(
                        source_path,
                        key="urbs_in/eff_factor",
                        start=0,
                        stop=1,
                    ).columns
                column_cache[source_name] = (
                    demand_columns,
                    efficiency_columns,
                )
            demand_columns, efficiency_columns = column_cache[source_name]
            exact = (
                (source_bus, "space_heat") in demand_columns
                and (source_bus, "water_heat") in demand_columns
                and (source_bus, "heatpump_air") in efficiency_columns
            )
        records.append(
            {
                "building_objectid": building_id,
                "building_use": row.get("building_use"),
                "building_type": row.get("building_type"),
                "building_floor_area": float(row.get("building_floor_area") or np.nan),
                "exact_source_hdf": source_name,
                "exact_source_bus": source_bus,
                "exact_profile_available": bool(library_exact or exact),
                "exact_profile_source": (
                    "physical_heat_library"
                    if library_exact
                    else "legacy_grid_hdf"
                    if exact
                    else "missing"
                ),
                "library_profile_method": (
                    library_metadata["profile_method"]
                    if library_metadata is not None
                    else None
                ),
                "library_profile_source_building_objectid": (
                    library_metadata["profile_source_building_objectid"]
                    if library_metadata is not None
                    else None
                ),
                "library_profile_scale": (
                    library_metadata["profile_scale"]
                    if library_metadata is not None
                    else None
                ),
                "library_fallback_match_scope": (
                    library_metadata["fallback_match_scope"]
                    if library_metadata is not None
                    else None
                ),
            }
        )
    catalog = pd.DataFrame(records)
    if catalog.empty:
        return catalog

    available = catalog[
        catalog["exact_profile_available"] & catalog["building_floor_area"].gt(0.0)
    ].copy()
    if available.empty:
        raise ValueError("No exact physical heat profiles are available.")

    output = []
    for row in catalog.to_dict("records"):
        if row["exact_profile_available"]:
            source = row
            library_method = row.get("library_profile_method")
            method = (
                str(library_method)
                if pd.notna(library_method) and str(library_method)
                else "exact_physical_building"
            )
            fallback_scope = row.get("library_fallback_match_scope")
        else:
            area = float(row["building_floor_area"])
            same_source = available["exact_source_hdf"].astype(str).eq(
                str(row["exact_source_hdf"])
            )
            same_type = available["building_type"].astype(str).eq(
                str(row.get("building_type"))
            )
            same_use = available["building_use"].astype(str).eq(
                str(row.get("building_use"))
            )
            tiers = (
                ("same_grid_and_type", same_source & same_type),
                ("same_grid_and_use", same_source & same_use),
                ("same_grid", same_source),
                ("same_type", same_type),
                ("same_use", same_use),
                ("regional", pd.Series(True, index=available.index)),
            )
            for fallback_scope, mask in tiers:
                candidates = available.loc[mask]
                if not candidates.empty:
                    break
            if np.isfinite(area) and area > 0.0:
                distances = (
                    np.log(candidates["building_floor_area"].astype(float))
                    - np.log(area)
                ).abs()
                source = candidates.loc[distances.idxmin()].to_dict()
            else:
                source = candidates.sort_values("building_objectid").iloc[0].to_dict()
                fallback_scope = f"{fallback_scope}_missing_target_area"
            method = "nearest_floor_area_scaled_approved"
        source_area = float(source["building_floor_area"])
        target_area = float(row["building_floor_area"])
        output.append(
            {
                **row,
                "profile_method": method,
                "profile_source_building_objectid": (
                    source.get("library_profile_source_building_objectid")
                    if pd.notna(
                        source.get("library_profile_source_building_objectid")
                    )
                    else source["building_objectid"]
                ),
                "profile_source_kind": source["exact_profile_source"],
                "profile_source_hdf": (
                    source["exact_source_hdf"]
                    if source["exact_profile_source"] == "legacy_grid_hdf"
                    else None
                ),
                "profile_source_bus": (
                    int(source["exact_source_bus"])
                    if source["exact_profile_source"] == "legacy_grid_hdf"
                    else None
                ),
                "profile_library_path": (
                    str(library.path)
                    if source["exact_profile_source"] == "physical_heat_library"
                    and library is not None
                    else None
                ),
                "profile_set_id": (
                    library.profile_set_id
                    if source["exact_profile_source"] == "physical_heat_library"
                    and library is not None
                    else None
                ),
                "profile_source_floor_area": source_area,
                "profile_scale": (
                    float(source["library_profile_scale"])
                    if pd.notna(source.get("library_profile_scale"))
                    else target_area / source_area
                    if method != "exact_physical_building"
                    and np.isfinite(target_area)
                    and target_area > 0.0
                    else 1.0
                ),
                "fallback_match_scope": (
                    fallback_scope
                    if method != "exact_physical_building"
                    and pd.notna(fallback_scope)
                    else None
                ),
                "publication_ready": bool(
                    method
                    in {
                        "exact_physical_building",
                        "nearest_floor_area_scaled_approved",
                    }
                ),
            }
        )
    return pd.DataFrame(output)


def profile_readiness_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()
    return (
        catalog.groupby(
            ["profile_method", "publication_ready"],
            dropna=False,
        )
        .agg(
            physical_buildings=("building_objectid", "nunique"),
            floor_area_m2=("building_floor_area", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=DEFAULT_PAIRED_DIR)
    parser.add_argument(
        "--synthetic-input-dir",
        type=Path,
        default=SYNTHETIC_INPUT_DIR,
    )
    parser.add_argument("--heat-profile-library", type=Path)
    args = parser.parse_args()
    paired_dir = args.paired_dir.resolve()
    allocation = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    buildings = pd.read_csv(paired_dir / "paired_building_scenario_plan.csv")
    catalog = build_heat_profile_catalog(
        allocation,
        buildings,
        synthetic_input_dir=args.synthetic_input_dir.resolve(),
        heat_profile_library=(
            args.heat_profile_library.resolve()
            if args.heat_profile_library is not None
            else None
        ),
    )
    output = paired_dir / "paired_heat_profile_catalog.csv"
    catalog.to_csv(output, index=False)
    summary = profile_readiness_summary(catalog)
    summary_output = paired_dir / "paired_heat_profile_readiness_summary.csv"
    summary.to_csv(summary_output, index=False)
    print(summary.to_string(index=False))
    print(output)
    print(summary_output)


if __name__ == "__main__":
    main()
