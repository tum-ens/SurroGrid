"""Audit physical heat-profile coverage for the paired SWF scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import GRIDALLOC_DIR, SYNTHETIC_INPUT_DIR

DEFAULT_PAIRED_DIR = (
    GRIDALLOC_DIR / "outputs" / "scenario_calibration" / "swf_2045_paired_v3_91301_station_hybrid_v2"
)


def build_heat_profile_catalog(
    allocation: pd.DataFrame,
    building_plan: pd.DataFrame,
    *,
    synthetic_input_dir: Path = SYNTHETIC_INPUT_DIR,
) -> pd.DataFrame:
    """Map each SWF heat-pump building to an exact or area-matched profile."""
    wp_count = pd.to_numeric(
        allocation.get("residential_wp_rows", 0.0),
        errors="coerce",
    ).fillna(0.0) + pd.to_numeric(
        allocation.get("ghd_wp_rows", 0.0),
        errors="coerce",
    ).fillna(0.0)
    selected = allocation.loc[wp_count.gt(0.0)].copy()
    selected = selected.drop_duplicates("building_objectid")
    attributes = (
        building_plan[["building_objectid", "building_use", "building_floor_area"]]
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

    column_cache: dict[str, tuple[pd.Index, pd.Index]] = {}
    records = []
    for row in selected.to_dict("records"):
        source_name = str(row["synthetic_bridge_filename"])
        source_bus = int(row["synthetic_bus"])
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
                "building_objectid": row["building_objectid"],
                "building_use": row.get("building_use"),
                "building_floor_area": float(row.get("building_floor_area") or np.nan),
                "exact_source_hdf": source_name,
                "exact_source_bus": source_bus,
                "exact_profile_available": bool(exact),
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
            method = "exact_physical_building"
        else:
            area = float(row["building_floor_area"])
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError(
                    "Area-matched heat fallback requires a positive floor area "
                    f"for {row['building_objectid']}."
                )
            distances = (
                np.log(available["building_floor_area"].astype(float)) - np.log(area)
            ).abs()
            source = available.loc[distances.idxmin()].to_dict()
            method = "nearest_floor_area_scaled_diagnostic"
        source_area = float(source["building_floor_area"])
        target_area = float(row["building_floor_area"])
        output.append(
            {
                **row,
                "profile_method": method,
                "profile_source_building_objectid": source["building_objectid"],
                "profile_source_hdf": source["exact_source_hdf"],
                "profile_source_bus": int(source["exact_source_bus"]),
                "profile_source_floor_area": source_area,
                "profile_scale": (
                    target_area / source_area
                    if method != "exact_physical_building"
                    else 1.0
                ),
                "publication_ready": bool(method == "exact_physical_building"),
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
    args = parser.parse_args()
    paired_dir = args.paired_dir.resolve()
    allocation = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    buildings = pd.read_csv(paired_dir / "paired_building_scenario_plan.csv")
    catalog = build_heat_profile_catalog(
        allocation,
        buildings,
        synthetic_input_dir=args.synthetic_input_dir.resolve(),
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
