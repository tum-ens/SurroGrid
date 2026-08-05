"""Build one calibrated physical-building scenario for real and synthetic grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from .allocation_plan import build_scenario_allocation_plan
from .ghd_calibration import build_synthetic_ghd_calibration
from ..profiles.profile_contract import assert_paired_plan_equivalence
from .scope_filters import build_grid_scope_summary
from .sector_asset_calibration import build_sector_asset_calibration
from .pv_roof_potential import building_roof_capacity, load_lod2_roof_catalog
from ..paths import configured_pylovo_version_id
from .swf_2045_building_match import (
    GRIDALLOC_DIR,
    MatchConfig,
    _database_engine,
    _expected_ghd_kwh_per_m2_by_building_use,
    _read_pylovo_buildings,
    add_comparison_asset_class,
    load_swf_2045_assets,
    match_assets_to_buildings,
)

WP_IDENTITY_COLUMNS = ["lv_id", "bus", "name", "Baujahr"]
SCENARIO_UNIT_COLUMNS = [
    "source_lv_id",
    "source_allocation_bus",
    "building_objectid",
]
PV_LOCATION_MODES = ("swf", "all_buildings")


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


def _pv_scenario_unit_assignments(
    real_plan: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    location_mode: str,
) -> pd.DataFrame:
    """Choose exactly one source connection for each eligible physical roof."""
    if location_mode not in PV_LOCATION_MODES:
        raise ValueError(f"pv_location_mode must be one of {PV_LOCATION_MODES}.")
    candidates = real_plan.copy()
    candidates["building_objectid"] = candidates["building_objectid"].astype(str)
    if location_mode == "swf":
        pv = matches[
            matches["matched"].fillna(False)
            & matches["asset_type"].eq("Photovoltaik")
        ].copy()
        counts = (
            pv.groupby(["building_objectid", "lv_id", "bus"], dropna=False)
            .size()
            .rename("swf_pv_rows_at_connection")
            .reset_index()
            .rename(
                columns={
                    "lv_id": "source_lv_id",
                    "bus": "source_allocation_bus",
                }
            )
        )
        counts["building_objectid"] = counts["building_objectid"].astype(str)
        candidates = candidates.merge(
            counts,
            on=[
                "building_objectid",
                "source_lv_id",
                "source_allocation_bus",
            ],
            how="inner",
        )
        candidates = candidates.sort_values(
            [
                "building_objectid",
                "swf_pv_rows_at_connection",
                "source_lv_id",
                "source_allocation_bus",
                "scenario_unit_id",
            ],
            ascending=[True, False, True, True, True],
        )
        method = "swf_cumulative_pv_location"
    else:
        candidates["_base_annual_kwh"] = (
            pd.to_numeric(
                candidates["residential_equivalent_hh_annual_kwh"], errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                candidates["calibrated_annual_ghd_kwh"], errors="coerce"
            ).fillna(0.0)
        )
        candidates = candidates.sort_values(
            ["building_objectid", "_base_annual_kwh", "scenario_unit_id"],
            ascending=[True, False, True],
        )
        method = "all_buildings_primary_demand_connection"
    chosen = candidates.drop_duplicates("building_objectid", keep="first").copy()
    chosen["pv_roof_assignment_method"] = method
    return chosen[
        ["building_objectid", "scenario_unit_id", "pv_roof_assignment_method"]
    ].reset_index(drop=True)


def _add_pv_roof_assignment(
    plan: pd.DataFrame,
    assignments: pd.DataFrame,
    capacity_by_building: pd.Series,
) -> pd.DataFrame:
    result = plan.copy()
    result["building_objectid"] = result["building_objectid"].astype(str)
    result = result.merge(
        assignments,
        on=["building_objectid", "scenario_unit_id"],
        how="left",
    )
    result["pv_roof_eligible"] = result["pv_roof_assignment_method"].notna()
    result["pv_roof_capacity_kw"] = (
        result["building_objectid"].map(capacity_by_building).fillna(0.0)
        * result["pv_roof_eligible"].astype(float)
    )
    return result


def build_paired_allocation(
    *,
    plz: int,
    final_year: int,
    output_dir: Path | None = None,
    grid_data_path: Path | None = None,
    max_match_distance_m: float = 100.0,
    min_buildings: int = 5,
    pv_location_mode: str = "swf",
) -> dict[str, pd.DataFrame]:
    pylovo_version_id = configured_pylovo_version_id()
    output_dir = output_dir or (
        GRIDALLOC_DIR
        / "outputs"
        / "scenario_calibration"
        / f"swf_2045_paired_v{pylovo_version_id}_{plz}"
    )
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
    ghd_calibration, ghd_calibration_summary = build_synthetic_ghd_calibration(
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
    retained_buildings = sorted(
        real_plan["building_objectid"].dropna().astype(str).unique()
    )
    roof_catalog = load_lod2_roof_catalog(_database_engine(), retained_buildings)
    roof_capacity = building_roof_capacity(roof_catalog)
    pv_assignments = _pv_scenario_unit_assignments(
        real_plan,
        matches,
        location_mode=pv_location_mode,
    )
    real_plan = _add_pv_roof_assignment(real_plan, pv_assignments, roof_capacity)
    synthetic_plan = _add_pv_roof_assignment(
        synthetic_plan,
        pv_assignments,
        roof_capacity,
    )
    building_plan["building_objectid"] = building_plan["building_objectid"].astype(str)
    building_plan["pv_roof_capacity_kw"] = (
        building_plan["building_objectid"].map(roof_capacity).fillna(0.0)
    )
    eligible_buildings = set(pv_assignments["building_objectid"])
    building_plan["pv_roof_eligible"] = building_plan["building_objectid"].isin(
        eligible_buildings
    )
    assert_paired_plan_equivalence(real_plan, synthetic_plan)

    metadata = {
        "plz": int(plz),
        "final_year": int(final_year),
        "pylovo_version_id": str(pylovo_version_id),
        "max_match_distance_m": float(max_match_distance_m),
        "min_physical_buildings_per_target_grid": int(min_buildings),
        "scenario_scope": "paired_full_local_demand",
        "pv_location_mode": pv_location_mode,
        "pv_capacity_source": "citydb_lod2_roof_surfaces",
        "pv_fallback_capacity_kw": 14.5,
        "pv_eligible_buildings": int(len(eligible_buildings)),
        "pv_available_capacity_kw": float(
            roof_capacity.reindex(sorted(eligible_buildings)).fillna(0.0).sum()
        ),
        "pv_fallback_buildings": int(
            roof_catalog.loc[
                roof_catalog["quality_flag"].eq("fallback_14_5_kw"),
                "building_objectid",
            ].nunique()
        ),
        "unmatched_ghd_policy": "exclude_row_retain_grid",
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
        "paired_synthetic_ghd_calibration": ghd_calibration,
        "paired_synthetic_ghd_calibration_summary": ghd_calibration_summary,
        "paired_roof_sections": roof_catalog,
        "paired_pv_roof_assignments": pv_assignments,
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
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-match-distance-m", type=float, default=100.0)
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument(
        "--pv-location-mode",
        choices=PV_LOCATION_MODES,
        default="swf",
        help=(
            "Use cumulative SWF PV locations or make every retained physical "
            "building PV-eligible."
        ),
    )
    args = parser.parse_args()
    outputs = build_paired_allocation(
        plz=args.plz,
        final_year=args.final_year,
        output_dir=args.output_dir,
        grid_data_path=args.grid_data_path,
        max_match_distance_m=args.max_match_distance_m,
        min_buildings=args.min_buildings,
        pv_location_mode=args.pv_location_mode,
    )
    print(outputs["paired_scope_audit"].to_string(index=False))


if __name__ == "__main__":
    main()
