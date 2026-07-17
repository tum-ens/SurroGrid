"""Build one calibrated physical-building scenario for real and synthetic grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from .allocation_plan import build_scenario_allocation_plan
from .ghd_calibration import build_synthetic_ghd_calibration
from ..profiles.profile_contract import assert_paired_plan_equivalence
from .scope_filters import build_grid_scope_summary
from .sector_asset_calibration import build_sector_asset_calibration
from .swf_2045_building_match import (
    ENV_PATH,
    GRIDALLOC_DIR,
    MatchConfig,
    _database_engine,
    _expected_ghd_kwh_per_m2_by_building_use,
    _read_pylovo_buildings,
    add_comparison_asset_class,
    load_swf_2045_assets,
    match_assets_to_buildings,
)

DEFAULT_OUTPUT_DIR = (
    GRIDALLOC_DIR / "outputs" / "scenario_calibration" / "swf_2045_paired_v3_91301"
)
WP_IDENTITY_COLUMNS = ["lv_id", "bus", "name", "Baujahr"]
SCENARIO_UNIT_COLUMNS = [
    "source_lv_id",
    "source_allocation_bus",
    "building_objectid",
]


def _deduplicate_exact_wp_identities(
    assets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove repeated SWF WP marker rows representing one identical asset."""
    wp_mask = assets["asset_type"].eq("WP")
    wp = assets.loc[wp_mask].copy()
    identity_columns = [
        column for column in WP_IDENTITY_COLUMNS if column in wp.columns
    ]
    if not identity_columns:
        raise ValueError("SWF WP deduplication requires at least one identity column.")
    duplicate_mask = wp.duplicated(identity_columns, keep="first")
    kept_wp = wp.loc[~duplicate_mask]
    result = pd.concat([assets.loc[~wp_mask], kept_wp], ignore_index=True, sort=False)
    return result, {
        "wp_rows_before": int(len(wp)),
        "wp_unique_identities": int(len(kept_wp)),
        "wp_exact_duplicate_rows_removed": int(duplicate_mask.sum()),
    }


def _synthetic_building_mapping(
    *,
    plz: int,
    pylovo_version_id: str,
) -> pd.DataFrame:
    query = text(
        """
        SELECT
            gbb.objectid::text AS building_objectid,
            gbb.bus AS synthetic_bus,
            gc.grid_case_id AS synthetic_grid_case_id,
            gc.cell_id || '_' || gc.plz || '_' || gc.kcid || '_' || gc.bcid || '.h5'
                AS synthetic_bridge_filename,
            gc.kcid AS synthetic_kcid,
            gc.bcid AS synthetic_bcid
        FROM surrogrid.grid_building_bus gbb
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE gc.plz = :plz
          AND gc.pylovo_version_id::text = :pylovo_version_id
        """
    )
    with _database_engine().connect() as conn:
        mapping = pd.read_sql_query(
            query,
            conn,
            params={"plz": int(plz), "pylovo_version_id": str(pylovo_version_id)},
        )
    if mapping.empty:
        raise ValueError(
            f"No synthetic building mapping found for PLZ={plz}, "
            f"pylovo_version_id={pylovo_version_id!r}."
        )
    duplicates = mapping["building_objectid"].duplicated(keep=False)
    if duplicates.any():
        examples = mapping.loc[duplicates, "building_objectid"].head(10).tolist()
        raise ValueError(
            "The selected synthetic version maps physical buildings more than once: "
            f"{examples}"
        )
    return mapping


def _add_scenario_unit_ids(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> None:
    """Assign stable sites before either plan is projected onto target buses."""
    real_keys = real[SCENARIO_UNIT_COLUMNS].astype(str).agg("|".join, axis=1)
    synthetic_keys = (
        synthetic[SCENARIO_UNIT_COLUMNS]
        .astype(str)
        .agg(
            "|".join,
            axis=1,
        )
    )
    if real_keys.duplicated().any() or synthetic_keys.duplicated().any():
        raise ValueError("Scenario-unit identity is not unique within a paired plan.")
    if set(real_keys) != set(synthetic_keys):
        raise ValueError("Real and synthetic plans contain different scenario units.")
    unit_ids = {key: unit_id for unit_id, key in enumerate(sorted(set(real_keys)))}
    real["scenario_unit_id"] = real_keys.map(unit_ids).astype(int)
    synthetic["scenario_unit_id"] = synthetic_keys.map(unit_ids).astype(int)


def _paired_plans(
    bus_plan: pd.DataFrame,
    synthetic_mapping: pd.DataFrame,
    *,
    min_buildings: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = bus_plan[bus_plan["include_full_local_demand_scenario"]].copy()
    full["building_objectid"] = full["building_objectid"].astype("string")
    mapped = full.merge(
        synthetic_mapping, on="building_objectid", how="left", indicator=True
    )

    mapped_buildings = set(
        mapped.loc[mapped["_merge"].eq("both"), "building_objectid"].dropna()
    )
    real = full[full["building_objectid"].isin(mapped_buildings)].copy()
    real["source_lv_id"] = real["lv_id"].astype(int)
    real["source_allocation_bus"] = real["allocation_bus"].astype(int)
    real = real.merge(synthetic_mapping, on="building_objectid", how="left")
    synthetic = mapped[mapped["_merge"].eq("both")].drop(columns="_merge").copy()
    synthetic["source_lv_id"] = synthetic["lv_id"].astype(int)
    synthetic["source_allocation_bus"] = synthetic["allocation_bus"].astype(int)
    synthetic["allocation_bus"] = synthetic["synthetic_bus"].astype(int)

    while True:
        previous_buildings = set(real["building_objectid"].dropna())
        synthetic_counts = synthetic.groupby(
            "synthetic_grid_case_id",
            observed=True,
        )["building_objectid"].nunique()
        selected_synthetic_grids = set(
            synthetic_counts[synthetic_counts.ge(int(min_buildings))].index
        )
        synthetic = synthetic[
            synthetic["synthetic_grid_case_id"].isin(selected_synthetic_grids)
        ].copy()

        retained_buildings = set(synthetic["building_objectid"].dropna())
        real = real[real["building_objectid"].isin(retained_buildings)].copy()
        real_counts = real.groupby("lv_id", observed=True)[
            "building_objectid"
        ].nunique()
        retained_real_grids = set(real_counts[real_counts.ge(int(min_buildings))].index)
        real = real[real["lv_id"].isin(retained_real_grids)].copy()

        retained_buildings = set(real["building_objectid"].dropna())
        synthetic = synthetic[
            synthetic["building_objectid"].isin(retained_buildings)
        ].copy()
        if retained_buildings == previous_buildings:
            break

    _add_scenario_unit_ids(real, synthetic)
    real["target_network"] = "real_swf"
    real["target_grid_id"] = real["lv_id"].astype(int)
    synthetic["target_network"] = "synthetic"
    synthetic["target_grid_id"] = synthetic["synthetic_grid_case_id"].astype(int)
    for frame in (real, synthetic):
        frame["scenario_scope"] = "paired_full_local_demand"
    assert_paired_plan_equivalence(real, synthetic)

    audit = pd.DataFrame(
        [
            _plan_summary("real_swf", real),
            _plan_summary("synthetic", synthetic),
            {
                "target_network": "unmapped_from_synthetic_version",
                "target_grids": 0,
                "physical_buildings": int(
                    full.loc[
                        ~full["building_objectid"].isin(mapped_buildings),
                        "building_objectid",
                    ].nunique()
                ),
                "plan_rows": int(
                    (~full["building_objectid"].isin(mapped_buildings)).sum()
                ),
                "hh_rows": float(
                    full.loc[
                        ~full["building_objectid"].isin(mapped_buildings),
                        "residential_equivalent_hh_rows",
                    ].sum()
                ),
                "hh_annual_kwh": float(
                    full.loc[
                        ~full["building_objectid"].isin(mapped_buildings),
                        "residential_equivalent_hh_annual_kwh",
                    ].sum()
                ),
                "ghd_annual_kwh": float(
                    full.loc[
                        ~full["building_objectid"].isin(mapped_buildings),
                        "calibrated_annual_ghd_kwh",
                    ].sum()
                ),
            },
        ]
    )
    return real, synthetic, audit


def _plan_summary(label: str, plan: pd.DataFrame) -> dict[str, Any]:
    return {
        "target_network": label,
        "target_grids": int(plan["target_grid_id"].nunique()),
        "physical_buildings": int(plan["building_objectid"].nunique()),
        "plan_rows": int(len(plan)),
        "hh_rows": float(plan["residential_equivalent_hh_rows"].sum()),
        "hh_annual_kwh": float(plan["residential_equivalent_hh_annual_kwh"].sum()),
        "ghd_annual_kwh": float(plan["calibrated_annual_ghd_kwh"].sum()),
    }


def build_paired_allocation(
    *,
    plz: int,
    final_year: int,
    pylovo_version_id: str,
    output_dir: Path,
    grid_data_path: Path | None = None,
    max_match_distance_m: float = 100.0,
    min_buildings: int = 5,
) -> dict[str, pd.DataFrame]:
    load_dotenv(ENV_PATH, override=True)
    os.environ["PYLOVO_VERSION_ID"] = str(pylovo_version_id)
    grid_root = grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    config = MatchConfig(
        plz=int(plz),
        final_year=int(final_year),
        building_scope="all",
        max_match_distance_m=float(max_match_distance_m),
    )

    buildings = _read_pylovo_buildings(config)
    assets_raw = load_swf_2045_assets(
        grid_data_path=grid_root,
        config=config,
    )
    assets, wp_dedup = _deduplicate_exact_wp_identities(assets_raw)
    matches = add_comparison_asset_class(
        match_assets_to_buildings(assets, buildings, config)
    )
    ghd_calibration, _ = build_synthetic_ghd_calibration(
        buildings,
        matches,
        _expected_ghd_kwh_per_m2_by_building_use(),
    )
    grid_scope = build_grid_scope_summary(
        matches,
        ghd_calibration,
        min_residential_equivalent_hh_buildings=int(min_buildings),
    )
    sector_calibration, _ = build_sector_asset_calibration(
        matches,
        ghd_calibration,
    )
    bus_plan, building_plan, street_plan, scope_totals = build_scenario_allocation_plan(
        matches,
        ghd_calibration,
        sector_calibration,
        grid_scope,
    )
    synthetic_mapping = _synthetic_building_mapping(
        plz=plz,
        pylovo_version_id=pylovo_version_id,
    )
    real_plan, synthetic_plan, paired_audit = _paired_plans(
        bus_plan,
        synthetic_mapping,
        min_buildings=min_buildings,
    )

    metadata = {
        "plz": int(plz),
        "final_year": int(final_year),
        "pylovo_version_id": str(pylovo_version_id),
        "max_match_distance_m": float(max_match_distance_m),
        "min_physical_buildings_per_target_grid": int(min_buildings),
        "scenario_scope": "paired_full_local_demand",
        **wp_dedup,
    }
    outputs = {
        "paired_real_bus_allocation_plan": real_plan,
        "paired_synthetic_bus_allocation_plan": synthetic_plan,
        "paired_scope_audit": paired_audit,
        "paired_building_scenario_plan": building_plan,
        "paired_street_scenario_plan": street_plan,
        "paired_scenario_scope_totals": scope_totals,
        "paired_asset_building_matches": matches,
        "paired_grid_scope_audit": grid_scope,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "paired_scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--final-year", type=int, default=2045)
    parser.add_argument("--pylovo-version-id", required=True)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-match-distance-m", type=float, default=100.0)
    parser.add_argument("--min-buildings", type=int, default=5)
    args = parser.parse_args()
    outputs = build_paired_allocation(
        plz=args.plz,
        final_year=args.final_year,
        pylovo_version_id=args.pylovo_version_id,
        output_dir=args.output_dir,
        grid_data_path=args.grid_data_path,
        max_match_distance_m=args.max_match_distance_m,
        min_buildings=args.min_buildings,
    )
    print(outputs["paired_scope_audit"].to_string(index=False))


if __name__ == "__main__":
    main()
