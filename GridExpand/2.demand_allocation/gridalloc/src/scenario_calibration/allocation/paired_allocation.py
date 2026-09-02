"""Build one calibrated physical-building scenario for real and synthetic grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from ..paths import ENV_PATH, GRIDEXPAND_DIR

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402
from common.building_components import build_building_components  # noqa: E402
from common.reproducibility import stable_seed  # noqa: E402

from .allocation_plan import build_scenario_allocation_plan
from .ghd_calibration import build_synthetic_ghd_calibration
from ..profiles.profile_contract import (
    assert_paired_component_plan_equivalence,
    assert_paired_plan_equivalence,
)
from .scope_filters import build_grid_scope_summary
from .sector_asset_calibration import build_sector_asset_calibration
from .pv_roof_potential import building_roof_capacity, load_lod2_roof_catalog
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


def _register_synthetic_grid_cases(
    *,
    ags: int,
    pylovo_version_id: str,
    min_buildings: int,
) -> dict[str, int]:
    """Register the complete eligible PyLoVo region before using SurroGrid views."""
    database = SurroGridDatabase()
    database.ensure_schema()
    candidate_query = text(
        """
        WITH ags_plz AS (
            SELECT DISTINCT plz
            FROM pylovo.municipal_register
            WHERE ags = :ags
        ),
        building_counts AS (
            SELECT grid_result_id, version_id, COUNT(*) AS n_buildings
            FROM pylovo.buildings_result
            GROUP BY grid_result_id, version_id
        ),
        selected AS (
            SELECT DISTINCT ON (gr.plz, gr.kcid, gr.bcid)
                gr.grid_result_id,
                gr.version_id,
                gr.plz,
                gr.kcid,
                gr.bcid,
                bc.n_buildings
            FROM pylovo.grid_result gr
            JOIN ags_plz ap ON ap.plz = gr.plz
            JOIN building_counts bc
              ON bc.grid_result_id = gr.grid_result_id
             AND bc.version_id = gr.version_id
            WHERE gr.version_id::text = :pylovo_version_id
              AND bc.n_buildings >= :min_buildings
            ORDER BY gr.plz, gr.kcid, gr.bcid, gr.version_id DESC
        )
        SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY plz, kcid, bcid) - 1 AS candidate_index
        FROM selected
        ORDER BY candidate_index
        """
    )
    with database.engine.connect() as conn:
        candidates = [
            dict(row)
            for row in conn.execute(
                candidate_query,
                {
                    "ags": int(ags),
                    "pylovo_version_id": str(pylovo_version_id),
                    "min_buildings": int(min_buildings),
                },
            ).mappings()
        ]
    if not candidates:
        raise ValueError(
            "No eligible PyLoVo grids found for "
            f"AGS={ags}, version={pylovo_version_id!r}."
        )

    grid_refs = [
        {
            "ags": int(ags),
            "plz": int(row["plz"]),
            "kcid": int(row["kcid"]),
            "bcid": int(row["bcid"]),
            "grid_result_id": int(row["grid_result_id"]),
            "version_id": str(row["version_id"]),
            "cell_id": f"{int(ags)}-{int(row['candidate_index']):02d}",
        }
        for row in candidates
    ]
    insert_query = text(
        """
        INSERT INTO surrogrid.grid_case (
            ags, plz, kcid, bcid, pylovo_grid_result_id,
            pylovo_version_id, cell_id
        )
        VALUES (
            :ags, :plz, :kcid, :bcid, :grid_result_id,
            :version_id, :cell_id
        )
        ON CONFLICT (ags, plz, kcid, bcid, pylovo_grid_result_id)
        DO UPDATE SET
            pylovo_version_id = EXCLUDED.pylovo_version_id,
            cell_id = EXCLUDED.cell_id
        """
    )
    with database.engine.begin() as conn:
        conn.execute(insert_query, grid_refs)
    return {
        "eligible_grid_cases": len(grid_refs),
        "eligible_buildings": sum(int(row["n_buildings"]) for row in candidates),
    }


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


def _scenario_unit_keys(plan: pd.DataFrame) -> pd.Series:
    """Return the stable source-connection identity for every plan row."""
    return plan[SCENARIO_UNIT_COLUMNS].astype(str).agg("|".join, axis=1)


def _add_scenario_unit_ids(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> None:
    """Assign stable sites before either plan is projected onto target buses."""
    real_keys = _scenario_unit_keys(real)
    synthetic_keys = _scenario_unit_keys(synthetic)
    if real_keys.duplicated().any() or synthetic_keys.duplicated().any():
        raise ValueError("Scenario-unit identity is not unique within a paired plan.")
    if set(real_keys) != set(synthetic_keys):
        raise ValueError("Real and synthetic plans contain different scenario units.")
    unit_ids = {key: unit_id for unit_id, key in enumerate(sorted(set(real_keys)))}
    real["scenario_unit_id"] = real_keys.map(unit_ids).astype(int)
    synthetic["scenario_unit_id"] = synthetic_keys.map(unit_ids).astype(int)


PAIRED_COMPONENT_COLUMNS = [
    "scenario_unit_id", "building_objectid", "component_id", "component_kind",
    "component_category", "source_component_category", "effective_floor_area_m2",
    "gross_floor_area_m2", "households", "occupants", "installed_peak_kw",
    "source_pylovo_grid_result_id", "source_pylovo_version_id", "source_building_use",
    "source_building_use_id", "source_building_type", "mix_score", "mix_rule",
    "mix_confidence", "source_lv_id", "source_allocation_bus", "real_target_grid_id",
    "real_target_bus", "synthetic_target_grid_case_id", "synthetic_target_bus",
    "synthetic_bridge_filename", "included_in_lv", "mv_direct", "annual_energy_kwh",
    "profile_method", "profile_hash", "stable_seed", "profile_seed",
    "source_asset_count", "matched_swf_asset_count", "source_asset_ids",
    "source_use_conflict", "suppression_reason",
]


def _paired_profile_hash(
    component_id: str,
    category: str,
    annual_energy_kwh: float,
    profile_method: str,
    stable_profile_seed: int,
    asset_ids: list[str],
) -> str:
    payload = "|".join([
        str(component_id), str(category), f"{float(annual_energy_kwh):.12f}",
        str(profile_method), str(int(stable_profile_seed)), ",".join(sorted(asset_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paired_component_plan(
    buildings: pd.DataFrame,
    matches: pd.DataFrame,
    real_plan: pd.DataFrame,
    synthetic_plan: pd.DataFrame,
    *,
    profile_seed: int,
) -> pd.DataFrame:
    """Reconcile matched SWF demand evidence with explicit PyLoVo components."""
    real_units = real_plan[[
        "scenario_unit_id", "building_objectid", "source_lv_id",
        "source_allocation_bus", "target_grid_id", "allocation_bus",
    ]].copy()
    synthetic_units = synthetic_plan[[
        "scenario_unit_id", "building_objectid", "target_grid_id", "allocation_bus",
        "synthetic_bridge_filename",
    ]].copy()
    for frame in (real_units, synthetic_units):
        frame["building_objectid"] = frame["building_objectid"].astype(str)
    if real_units["building_objectid"].duplicated().any() or synthetic_units["building_objectid"].duplicated().any():
        raise ValueError("A paired physical building must map to exactly one scenario unit.")
    if set(real_units["building_objectid"]) != set(synthetic_units["building_objectid"]):
        raise ValueError("Paired component planning received different physical buildings.")

    retained_ids = set(real_units["building_objectid"])
    physical = buildings.copy()
    physical["objectid"] = physical["objectid"].astype(str)
    physical = physical[physical["objectid"].isin(retained_ids)].copy()
    source_bus = real_units.set_index("building_objectid")["source_allocation_bus"]
    physical["bus"] = physical["objectid"].map(source_bus)
    if physical["bus"].isna().any():
        raise ValueError("Every paired physical building requires one source allocation bus.")
    source_components = build_building_components(physical)
    source_components["component_kind"] = "pylovo"
    source_components["source_component_category"] = source_components["component_category"]
    source_categories = source_components.groupby("objectid")["component_category"].agg(set).to_dict()

    demand_assets = matches[
        matches["matched"].fillna(False)
        & matches["asset_type"].isin(["HH", "GHD"])
        & matches["building_objectid"].notna()
        & matches["building_objectid"].astype(str).isin(retained_ids)
    ].copy()
    demand_assets["building_objectid"] = demand_assets["building_objectid"].astype(str)
    demand_assets["asset_id"] = demand_assets["asset_id"].astype(str)
    demand_assets["swf_annual_demand_kwh"] = pd.to_numeric(
        demand_assets["swf_annual_demand_kwh"], errors="coerce"
    ).fillna(0.0)

    rows: list[dict[str, Any]] = []
    for component in source_components.to_dict("records"):
        object_id = str(component["objectid"])
        category = str(component["component_category"])
        asset_type = "HH" if category == "Residential" else "GHD"
        evidence = demand_assets[
            demand_assets["building_objectid"].eq(object_id)
            & demand_assets["asset_type"].eq(asset_type)
        ]
        asset_ids = sorted(evidence["asset_id"].tolist())
        annual = float(evidence["swf_annual_demand_kwh"].sum())
        direct = bool(component["mv_direct"])
        if direct:
            included, suppression, method, annual = False, "outside_paired_lv_scope", "suppressed_mv_direct", 0.0
        elif evidence.empty or annual <= 0.0:
            included, suppression, method = False, "no_matched_swf_evidence", "suppressed_no_swf_evidence"
        else:
            included, suppression = True, None
            method = "paired_swf_hh_profile_v2" if asset_type == "HH" else f"paired_swf_ghd_{category.lower()}_shape_v2"
        stable_profile_seed = stable_seed(
            int(profile_seed), object_id, category, "electricity", "paired"
        )
        real = real_units.loc[real_units["building_objectid"].eq(object_id)].iloc[0]
        synthetic = synthetic_units.loc[synthetic_units["building_objectid"].eq(object_id)].iloc[0]
        rows.append({
            **component,
            "building_objectid": object_id,
            "scenario_unit_id": int(real["scenario_unit_id"]),
            "source_pylovo_grid_result_id": component.get("pylovo_grid_result_id"),
            "source_pylovo_version_id": component.get("pylovo_version_id"),
            "source_lv_id": int(real["source_lv_id"]),
            "source_allocation_bus": int(real["source_allocation_bus"]),
            "real_target_grid_id": int(real["target_grid_id"]),
            "real_target_bus": int(real["allocation_bus"]),
            "synthetic_target_grid_case_id": int(synthetic["target_grid_id"]),
            "synthetic_target_bus": int(synthetic["allocation_bus"]),
            "synthetic_bridge_filename": synthetic["synthetic_bridge_filename"],
            "included_in_lv": bool(included), "mv_direct": direct,
            "annual_energy_kwh": annual, "profile_method": method,
            "profile_hash": _paired_profile_hash(component["component_id"], category, annual, method, stable_profile_seed, asset_ids),
            "stable_seed": int(stable_profile_seed), "profile_seed": int(profile_seed),
            "source_asset_count": int(len(evidence)), "matched_swf_asset_count": int(len(evidence)),
            "source_asset_ids": ",".join(asset_ids), "source_use_conflict": None,
            "suppression_reason": suppression,
        })

    def add_proxy(object_id: str, asset_type: str, evidence: pd.DataFrame) -> None:
        if evidence.empty:
            return
        physical_row = physical.loc[physical["objectid"].eq(object_id)].iloc[0]
        category = "Residential" if asset_type == "HH" else "Commercial"
        component_id = f"{object_id}::swf_evidence_proxy::{asset_type.lower()}"
        asset_ids = sorted(evidence["asset_id"].tolist())
        annual = float(evidence["swf_annual_demand_kwh"].sum())
        stable_profile_seed = stable_seed(
            int(profile_seed), object_id, category, "electricity", "paired_proxy"
        )
        method = f"swf_evidence_proxy_{asset_type.lower()}_v2"
        real = real_units.loc[real_units["building_objectid"].eq(object_id)].iloc[0]
        synthetic = synthetic_units.loc[synthetic_units["building_objectid"].eq(object_id)].iloc[0]
        rows.append({
            "scenario_unit_id": int(real["scenario_unit_id"]), "building_objectid": object_id,
            "component_id": component_id, "component_kind": "swf_evidence_proxy",
            "component_category": category, "source_component_category": "unmatched_upstream_category",
            "effective_floor_area_m2": 0.0,
            "gross_floor_area_m2": float(physical_row["floor_area"] * physical_row["floor_number"]),
            "households": float(pd.to_numeric(pd.Series([physical_row["households"]]), errors="coerce").fillna(0.0).iloc[0]) if asset_type == "HH" else pd.NA,
            "occupants": float(pd.to_numeric(pd.Series([physical_row["occupants"]]), errors="coerce").fillna(0.0).iloc[0]) if asset_type == "HH" else pd.NA,
            "installed_peak_kw": 0.0,
            "source_pylovo_grid_result_id": physical_row.get("pylovo_grid_result_id"),
            "source_pylovo_version_id": physical_row.get("pylovo_version_id"),
            "source_building_use": physical_row["building_use"], "source_building_use_id": physical_row["building_use_id"],
            "source_building_type": physical_row["building_type"], "mix_score": physical_row["mix_score"],
            "mix_rule": physical_row["mix_rule"], "mix_confidence": physical_row["mix_confidence"],
            "source_lv_id": int(real["source_lv_id"]), "source_allocation_bus": int(real["source_allocation_bus"]),
            "real_target_grid_id": int(real["target_grid_id"]), "real_target_bus": int(real["allocation_bus"]),
            "synthetic_target_grid_case_id": int(synthetic["target_grid_id"]), "synthetic_target_bus": int(synthetic["allocation_bus"]),
            "synthetic_bridge_filename": synthetic["synthetic_bridge_filename"],
            "included_in_lv": True, "mv_direct": False, "annual_energy_kwh": annual,
            "profile_method": method,
            "profile_hash": _paired_profile_hash(component_id, category, annual, method, stable_profile_seed, asset_ids),
            "stable_seed": int(stable_profile_seed), "profile_seed": int(profile_seed),
            "source_asset_count": int(len(evidence)), "matched_swf_asset_count": int(len(evidence)),
            "source_asset_ids": ",".join(asset_ids),
            "source_use_conflict": "swf_evidence_on_nonmatching_upstream_category",
            "suppression_reason": None,
        })

    for object_id in sorted(retained_ids):
        evidence = demand_assets[demand_assets["building_objectid"].eq(object_id)]
        categories = source_categories.get(object_id, set())
        if "Residential" not in categories:
            add_proxy(object_id, "HH", evidence[evidence["asset_type"].eq("HH")])
        if not categories.intersection({"Commercial", "Public"}):
            add_proxy(object_id, "GHD", evidence[evidence["asset_type"].eq("GHD")])

    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("Paired allocation produced no physical or SWF-evidence components.")
    plan = plan[PAIRED_COMPONENT_COLUMNS].sort_values(
        ["scenario_unit_id", "building_objectid", "component_id"]
    ).reset_index(drop=True)
    if plan["component_id"].duplicated().any():
        raise ValueError("Paired component IDs must be unique.")
    return plan


def _refresh_paired_plan_demand(
    real_plan: pd.DataFrame,
    synthetic_plan: pd.DataFrame,
    component_plan: pd.DataFrame,
) -> None:
    """Make physical projection totals equal the included component evidence."""
    included = component_plan[component_plan["included_in_lv"].astype(bool)].copy()
    residential_area = component_plan.loc[
        component_plan["component_category"].eq("Residential")
    ].groupby("scenario_unit_id")["effective_floor_area_m2"].sum()
    by_unit = included.assign(
        _hh_kwh=included["annual_energy_kwh"].where(included["component_category"].eq("Residential"), 0.0),
        _ghd_kwh=included["annual_energy_kwh"].where(included["component_category"].isin(["Commercial", "Public"]), 0.0),
        _hh_rows=included["source_asset_count"].where(included["component_category"].eq("Residential"), 0.0),
    ).groupby("scenario_unit_id").agg(
        _hh_kwh=("_hh_kwh", "sum"), _ghd_kwh=("_ghd_kwh", "sum"), _hh_rows=("_hh_rows", "sum")
    )
    for frame in (real_plan, synthetic_plan):
        frame["residential_effective_floor_area_m2"] = frame["scenario_unit_id"].map(residential_area).fillna(0.0)
        frame["residential_equivalent_hh_annual_kwh"] = frame["scenario_unit_id"].map(by_unit["_hh_kwh"]).fillna(0.0)
        frame["calibrated_annual_ghd_kwh"] = frame["scenario_unit_id"].map(by_unit["_ghd_kwh"]).fillna(0.0)
        frame["residential_equivalent_hh_rows"] = frame["scenario_unit_id"].map(by_unit["_hh_rows"]).fillna(0.0)

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
        previous_scenario_units = set(_scenario_unit_keys(real))
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

        retained_scenario_units = set(_scenario_unit_keys(synthetic))
        real = real[
            _scenario_unit_keys(real).isin(retained_scenario_units)
        ].copy()
        real_counts = real.groupby("lv_id", observed=True)[
            "building_objectid"
        ].nunique()
        retained_real_grids = set(real_counts[real_counts.ge(int(min_buildings))].index)
        real = real[real["lv_id"].isin(retained_real_grids)].copy()

        retained_scenario_units = set(_scenario_unit_keys(real))
        synthetic = synthetic[
            _scenario_unit_keys(synthetic).isin(retained_scenario_units)
        ].copy()
        if retained_scenario_units == previous_scenario_units:
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
            matches["matched"].fillna(False) & matches["asset_type"].eq("Photovoltaik")
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
        candidates["_base_annual_kwh"] = pd.to_numeric(
            candidates["residential_equivalent_hh_annual_kwh"], errors="coerce"
        ).fillna(0.0) + pd.to_numeric(
            candidates["calibrated_annual_ghd_kwh"], errors="coerce"
        ).fillna(0.0)
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
    result["pv_roof_capacity_kw"] = result["building_objectid"].map(
        capacity_by_building
    ).fillna(0.0) * result["pv_roof_eligible"].astype(float)
    return result


def build_paired_allocation(
    *,
    ags: int,
    plz: int,
    final_year: int,
    pylovo_version_id: str,
    output_dir: Path | None = None,
    grid_data_path: Path | None = None,
    max_match_distance_m: float = 100.0,
    min_buildings: int = 5,
    pv_location_mode: str = "swf",
    profile_seed: int = 481527,
) -> dict[str, pd.DataFrame]:
    load_dotenv(ENV_PATH, override=True)
    os.environ["PYLOVO_VERSION_ID"] = str(pylovo_version_id)
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

    registration = _register_synthetic_grid_cases(
        ags=int(ags),
        pylovo_version_id=str(pylovo_version_id),
        min_buildings=int(min_buildings),
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
    component_plan = _paired_component_plan(
        buildings, matches, real_plan, synthetic_plan, profile_seed=int(profile_seed)
    )
    residential_area_by_building = component_plan.loc[
        component_plan["component_category"].eq("Residential")
    ].groupby("building_objectid")["effective_floor_area_m2"].sum()
    building_plan["residential_effective_floor_area_m2"] = building_plan[
        "building_objectid"
    ].astype(str).map(residential_area_by_building).fillna(0.0)
    _refresh_paired_plan_demand(real_plan, synthetic_plan, component_plan)
    assert_paired_component_plan_equivalence(component_plan)
    assert_paired_plan_equivalence(real_plan, synthetic_plan)

    metadata = {
        "ags": int(ags),
        "plz": int(plz),
        "final_year": int(final_year),
        "pylovo_version_id": str(pylovo_version_id),
        "max_match_distance_m": float(max_match_distance_m),
        "min_physical_buildings_per_target_grid": int(min_buildings),
        "scenario_scope": "paired_full_local_demand",
        "pv_location_mode": pv_location_mode,
        "profile_seed": int(profile_seed),
        "component_contract": "physical_building_component_v1",
        "paired_contract": "physical_building_component_paired_v2",
        "component_rows": int(len(component_plan)),
        "included_component_rows": int(component_plan["included_in_lv"].astype(bool).sum()),
        "suppressed_component_rows": int((~component_plan["included_in_lv"].astype(bool)).sum()),
        "mixed_use_physical_buildings": int(
            component_plan.groupby("building_objectid")["component_category"].nunique().gt(1).sum()
        ),
        "heat_area_method": "residential_effective_floor_area_v1",
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
        "registered_pylovo_grid_cases": registration["eligible_grid_cases"],
        "registered_pylovo_buildings": registration["eligible_buildings"],
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
        "paired_component_scenario_plan": component_plan,
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
    parser.add_argument("--ags", type=int, required=True)
    parser.add_argument("--plz", type=int, required=True)
    parser.add_argument("--final-year", type=int, default=2045)
    parser.add_argument("--pylovo-version-id", required=True)
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
    parser.add_argument(
        "--profile-seed", type=int, default=481527,
        help="Stable seed used by the paired component profile realization.",
    )
    args = parser.parse_args()
    outputs = build_paired_allocation(
        ags=args.ags,
        plz=args.plz,
        final_year=args.final_year,
        pylovo_version_id=args.pylovo_version_id,
        output_dir=args.output_dir,
        grid_data_path=args.grid_data_path,
        max_match_distance_m=args.max_match_distance_m,
        min_buildings=args.min_buildings,
        pv_location_mode=args.pv_location_mode,
        profile_seed=args.profile_seed,
    )
    print(outputs["paired_scope_audit"].to_string(index=False))


if __name__ == "__main__":
    main()
