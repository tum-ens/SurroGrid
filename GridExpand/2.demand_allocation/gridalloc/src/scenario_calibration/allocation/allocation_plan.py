"""Scenario allocation plan outputs for SWF 2045 calibration audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

RESIDENTIAL_EQUIVALENT_HH_CLASSES = {"residential_hh", "mixed_residential_proxy"}
RESIDENTIAL_EQUIVALENT_SECTOR_SCOPE = "residential_equivalent_sector_asset"
CALIBRATED_GHD_SECTOR_SCOPE = "calibrated_ghd_sector_asset"


def _build_bus_plan(
    matches: pd.DataFrame,
    sector_asset_calibration: pd.DataFrame,
    grid_scope_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create the electrical-bus handover table for real-grid scenario allocation.

    The building and street tables are useful for audits, but the real SWF
    pandapower model needs loads at electrical buses. This table keeps the
    calibrated scenario quantities at the original SWF bus while retaining the
    matched physical building id for transparent comparison.
    """
    matched = (
        matches[matches["matched"]].dropna(subset=["building_match_id", "bus"]).copy()
    )
    if matched.empty:
        return pd.DataFrame()

    demand_assets = matched[
        matched["asset_type"].isin(["HH", "GHD"])
        & (
            matched["comparison_asset_class"].isin(RESIDENTIAL_EQUIVALENT_HH_CLASSES)
            | matched["asset_type"].eq("GHD")
        )
    ].copy()
    sector_assets = sector_asset_calibration.dropna(
        subset=["building_match_id", "bus"]
    ).copy()

    base_frames = []
    if not demand_assets.empty:
        base_frames.append(demand_assets)
    if not sector_assets.empty:
        base_frames.append(sector_assets)
    if not base_frames:
        return pd.DataFrame()

    base_assets = pd.concat(base_frames, ignore_index=True, sort=False)
    base = (
        base_assets.groupby(["lv_id", "bus", "building_match_id"], dropna=False)
        .agg(
            building_objectid=("building_objectid", "first"),
            building_use=("building_use", "first"),
            building_type=("building_type", "first"),
            building_street=("building_street", "first"),
            building_house_number=("building_house_number", "first"),
            building_is_residential=("building_is_residential", "first"),
        )
        .reset_index()
        .rename(columns={"bus": "allocation_bus"})
    )

    hh = demand_assets[
        demand_assets["asset_type"].eq("HH")
        & demand_assets["comparison_asset_class"].isin(
            RESIDENTIAL_EQUIVALENT_HH_CLASSES
        )
    ].copy()
    if not hh.empty:
        hh["hh_kwh_num"] = pd.to_numeric(
            hh["swf_annual_demand_kwh"], errors="coerce"
        ).fillna(0.0)
        hh_summary = (
            hh.groupby(["lv_id", "bus", "building_match_id"], dropna=False)
            .agg(
                residential_equivalent_hh_rows=("asset_id", "count"),
                residential_equivalent_hh_annual_kwh=("hh_kwh_num", "sum"),
            )
            .reset_index()
            .rename(columns={"bus": "allocation_bus"})
        )
    else:
        hh_summary = pd.DataFrame(
            columns=["lv_id", "allocation_bus", "building_match_id"]
        )

    ghd = demand_assets[demand_assets["asset_type"].eq("GHD")].copy()
    if not ghd.empty:
        ghd["ghd_kwh_num"] = pd.to_numeric(
            ghd["swf_annual_demand_kwh"], errors="coerce"
        ).fillna(0.0)
        ghd_summary = (
            ghd.groupby(["lv_id", "bus", "building_match_id"], dropna=False)
            .agg(
                swf_ghd_rows=("asset_id", "count"),
                calibrated_annual_ghd_kwh=("ghd_kwh_num", "sum"),
            )
            .reset_index()
            .rename(columns={"bus": "allocation_bus"})
        )
    else:
        ghd_summary = pd.DataFrame(
            columns=["lv_id", "allocation_bus", "building_match_id"]
        )

    sector_summary = _bus_sector_summary(sector_assets)

    plan = base.merge(
        hh_summary, on=["lv_id", "allocation_bus", "building_match_id"], how="left"
    )
    plan = plan.merge(
        ghd_summary, on=["lv_id", "allocation_bus", "building_match_id"], how="left"
    )
    plan = plan.merge(
        sector_summary, on=["lv_id", "allocation_bus", "building_match_id"], how="left"
    )
    plan = plan.merge(_grid_flags(grid_scope_summary), on="lv_id", how="left")

    numeric_columns = [
        column for column in plan.columns if column.endswith(("_rows", "_kwh", "_kw"))
    ]
    for column in numeric_columns:
        plan[column] = pd.to_numeric(plan[column], errors="coerce").fillna(0.0)

    bool_columns = [
        "recommended_for_residential_equivalent_scope",
        "recommended_for_full_local_demand_scope",
        "passes_min_residential_equivalent_hh_buildings",
        "ghd_heavy_warning",
        "ghd_heavy_extreme",
        "has_unmatched_ghd",
    ]
    for column in bool_columns:
        if column in plan.columns:
            plan[column] = plan[column].fillna(False).astype(bool)

    plan["include_residential_equivalent_scope"] = plan[
        "recommended_for_residential_equivalent_scope"
    ] & plan["residential_equivalent_hh_rows"].gt(0)
    plan["include_full_local_demand_scope"] = plan[
        "recommended_for_full_local_demand_scope"
    ] & (
        plan["residential_equivalent_hh_rows"].gt(0)
        | plan["calibrated_annual_ghd_kwh"].gt(0)
    )
    plan["include_residential_equivalent_sector_assets"] = plan[
        "recommended_for_residential_equivalent_scope"
    ] & plan[
        [
            "residential_ev_charger_kw",
            "residential_pv_kw",
            "residential_battery_kwh",
            "residential_wp_rows",
        ]
    ].sum(axis=1).gt(0)
    plan["include_calibrated_ghd_sector_assets"] = plan[
        "recommended_for_full_local_demand_scope"
    ] & plan[["ghd_ev_charger_kw", "ghd_pv_kw", "ghd_battery_kwh", "ghd_wp_rows"]].sum(
        axis=1
    ).gt(0)
    plan["include_residential_equivalent_scenario"] = (
        plan["include_residential_equivalent_scope"]
        | plan["include_residential_equivalent_sector_assets"]
    )
    plan["include_full_local_demand_scenario"] = plan[
        "include_full_local_demand_scope"
    ] | (
        plan["recommended_for_full_local_demand_scope"]
        & (
            plan[
                [
                    "residential_ev_charger_kw",
                    "residential_pv_kw",
                    "residential_battery_kwh",
                    "residential_wp_rows",
                ]
            ]
            .sum(axis=1)
            .gt(0)
            | plan[["ghd_ev_charger_kw", "ghd_pv_kw", "ghd_battery_kwh", "ghd_wp_rows"]]
            .sum(axis=1)
            .gt(0)
        )
    )

    return plan.sort_values(
        ["lv_id", "allocation_bus", "building_match_id"], na_position="last"
    ).reset_index(drop=True)


def build_scenario_allocation_plan(
    matches: pd.DataFrame,
    ghd_calibration: pd.DataFrame,
    sector_asset_calibration: pd.DataFrame,
    grid_scope_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create bus-, building-, street-, and scope-level scenario allocation tables."""
    bus_plan = _build_bus_plan(matches, sector_asset_calibration, grid_scope_summary)
    building_plan = _build_building_plan(
        matches, ghd_calibration, sector_asset_calibration, grid_scope_summary
    )
    street_plan = _build_street_plan(building_plan)
    scope_totals = _build_scope_totals(building_plan)
    return bus_plan, building_plan, street_plan, scope_totals


def _build_building_plan(
    matches: pd.DataFrame,
    ghd_calibration: pd.DataFrame,
    sector_asset_calibration: pd.DataFrame,
    grid_scope_summary: pd.DataFrame,
) -> pd.DataFrame:
    matched = matches[matches["matched"]].dropna(subset=["building_match_id"]).copy()
    if matched.empty:
        return pd.DataFrame()

    base = (
        matched.groupby(["lv_id", "building_match_id"], dropna=False)
        .agg(
            building_objectid=("building_objectid", "first"),
            building_use=("building_use", "first"),
            building_type=("building_type", "first"),
            building_street=("building_street", "first"),
            building_house_number=("building_house_number", "first"),
            building_is_residential=("building_is_residential", "first"),
            building_households=("building_households", "first"),
            building_floor_area=("building_floor_area", "first"),
        )
        .reset_index()
    )

    hh = matched[
        matched["asset_type"].eq("HH")
        & matched["comparison_asset_class"].isin(RESIDENTIAL_EQUIVALENT_HH_CLASSES)
    ].copy()
    if not hh.empty:
        hh["hh_kwh_num"] = pd.to_numeric(
            hh["swf_annual_demand_kwh"], errors="coerce"
        ).fillna(0.0)
        hh_summary = (
            hh.groupby(["lv_id", "building_match_id"], dropna=False)
            .agg(
                residential_equivalent_hh_rows=("asset_id", "count"),
                residential_equivalent_hh_annual_kwh=("hh_kwh_num", "sum"),
            )
            .reset_index()
        )
    else:
        hh_summary = pd.DataFrame(columns=["lv_id", "building_match_id"])

    ghd_summary = _building_ghd_summary(matches, ghd_calibration)
    sector_summary = _building_sector_summary(sector_asset_calibration)

    plan = base.merge(hh_summary, on=["lv_id", "building_match_id"], how="left")
    plan = plan.merge(ghd_summary, on=["lv_id", "building_match_id"], how="left")
    plan = plan.merge(sector_summary, on=["lv_id", "building_match_id"], how="left")
    plan = plan.merge(_grid_flags(grid_scope_summary), on="lv_id", how="left")

    numeric_columns = [
        column
        for column in plan.columns
        if column.endswith(("_rows", "_kwh", "_kw", "_kwh_default"))
    ]
    for column in numeric_columns:
        plan[column] = pd.to_numeric(plan[column], errors="coerce").fillna(0.0)

    bool_columns = [
        "recommended_for_residential_equivalent_scope",
        "recommended_for_full_local_demand_scope",
        "passes_min_residential_equivalent_hh_buildings",
        "ghd_heavy_warning",
        "ghd_heavy_extreme",
        "has_unmatched_ghd",
    ]
    for column in bool_columns:
        if column in plan.columns:
            plan[column] = plan[column].fillna(False).astype(bool)

    plan["include_residential_equivalent_scope"] = plan[
        "recommended_for_residential_equivalent_scope"
    ] & plan["residential_equivalent_hh_rows"].gt(0)
    plan["include_full_local_demand_scope"] = plan[
        "recommended_for_full_local_demand_scope"
    ] & (
        plan["residential_equivalent_hh_rows"].gt(0)
        | plan["calibrated_annual_ghd_kwh"].gt(0)
    )
    plan["include_residential_equivalent_sector_assets"] = plan[
        "recommended_for_residential_equivalent_scope"
    ] & plan[
        [
            "residential_ev_charger_kw",
            "residential_pv_kw",
            "residential_battery_kwh",
            "residential_wp_rows",
        ]
    ].sum(axis=1).gt(0)
    plan["include_calibrated_ghd_sector_assets"] = plan[
        "recommended_for_full_local_demand_scope"
    ] & plan[["ghd_ev_charger_kw", "ghd_pv_kw", "ghd_battery_kwh", "ghd_wp_rows"]].sum(
        axis=1
    ).gt(0)

    return plan.sort_values(
        ["lv_id", "building_street", "building_house_number"], na_position="last"
    ).reset_index(drop=True)


def _building_ghd_summary(
    matches: pd.DataFrame, ghd_calibration: pd.DataFrame
) -> pd.DataFrame:
    if ghd_calibration.empty:
        return pd.DataFrame(columns=["lv_id", "building_match_id"])

    matched = matches[matches["matched"]].dropna(subset=["building_match_id"]).copy()
    ghd = matched[matched["asset_type"].eq("GHD")].copy()
    ghd["swf_ghd_annual_kwh_num"] = pd.to_numeric(
        ghd["swf_annual_demand_kwh"], errors="coerce"
    ).fillna(0.0)
    if not ghd.empty:
        ghd_by_building = (
            ghd.groupby(["lv_id", "building_match_id"], dropna=False)
            .agg(
                swf_ghd_rows=("asset_id", "count"),
                swf_ghd_annual_kwh=("swf_ghd_annual_kwh_num", "sum"),
            )
            .reset_index()
        )
    else:
        ghd_by_building = pd.DataFrame(
            columns=["lv_id", "building_match_id", "swf_ghd_rows", "swf_ghd_annual_kwh"]
        )

    direct_building_lv = (
        ghd[["building_match_id", "lv_id"]]
        .drop_duplicates("building_match_id")
        .reset_index(drop=True)
    )
    fallback_building_lv = (
        matched[["building_match_id", "lv_id"]]
        .drop_duplicates("building_match_id")
        .reset_index(drop=True)
    )
    building_lv = direct_building_lv.merge(
        fallback_building_lv,
        on="building_match_id",
        how="outer",
        suffixes=("_direct", "_fallback"),
    )
    building_lv["lv_id"] = building_lv["lv_id_direct"].combine_first(
        building_lv["lv_id_fallback"]
    )
    building_lv = building_lv[["building_match_id", "lv_id"]]

    calibration = ghd_calibration.drop_duplicates("building_match_id").copy()
    calibration = calibration.merge(building_lv, on="building_match_id", how="left")
    calibration = calibration.merge(
        ghd_by_building,
        on=["lv_id", "building_match_id"],
        how="left",
        suffixes=("_calibration", "_direct"),
    )

    for column in ["swf_ghd_rows", "swf_ghd_annual_kwh"]:
        direct_column = f"{column}_direct"
        calibration[column] = pd.to_numeric(
            calibration.get(direct_column), errors="coerce"
        ).fillna(0.0)

    calibration["calibrated_annual_ghd_kwh"] = calibration["swf_ghd_annual_kwh"]
    direct_mask = calibration["swf_ghd_rows"].gt(0)
    calibration["excluded_default_annual_ghd_kwh"] = np.where(
        direct_mask,
        0.0,
        pd.to_numeric(
            calibration["synthetic_default_annual_ghd_kwh"], errors="coerce"
        ).fillna(0.0),
    )

    return calibration[
        [
            "lv_id",
            "building_match_id",
            "ghd_calibration_class",
            "swf_ghd_rows",
            "swf_ghd_annual_kwh",
            "synthetic_default_annual_ghd_kwh",
            "calibrated_annual_ghd_kwh",
            "excluded_default_annual_ghd_kwh",
        ]
    ].dropna(subset=["lv_id"])


def _bus_sector_summary(sector_asset_calibration: pd.DataFrame) -> pd.DataFrame:
    if sector_asset_calibration.empty:
        return pd.DataFrame(columns=["lv_id", "allocation_bus", "building_match_id"])
    sector = sector_asset_calibration.dropna(subset=["building_match_id", "bus"]).copy()
    if sector.empty:
        return pd.DataFrame(columns=["lv_id", "allocation_bus", "building_match_id"])

    rows = []
    for (lv_id, bus, building_id), group in sector.groupby(
        ["lv_id", "bus", "building_match_id"], dropna=False
    ):
        row = {
            "lv_id": lv_id,
            "allocation_bus": int(bus),
            "building_match_id": building_id,
        }
        for prefix, scope in [
            ("residential", RESIDENTIAL_EQUIVALENT_SECTOR_SCOPE),
            ("ghd", CALIBRATED_GHD_SECTOR_SCOPE),
            ("unsupported_nonres", "nonresidential_sector_asset_without_ghd_evidence"),
        ]:
            subset = group[group["sector_calibration_scope"].eq(scope)]
            row.update(_sector_values(prefix, subset))
        rows.append(row)
    return pd.DataFrame(rows)


def _building_sector_summary(sector_asset_calibration: pd.DataFrame) -> pd.DataFrame:
    if sector_asset_calibration.empty:
        return pd.DataFrame(columns=["lv_id", "building_match_id"])
    sector = sector_asset_calibration.dropna(subset=["building_match_id"]).copy()
    if sector.empty:
        return pd.DataFrame(columns=["lv_id", "building_match_id"])

    rows = []
    for (lv_id, building_id), group in sector.groupby(
        ["lv_id", "building_match_id"], dropna=False
    ):
        row = {"lv_id": lv_id, "building_match_id": building_id}
        for prefix, scope in [
            ("residential", RESIDENTIAL_EQUIVALENT_SECTOR_SCOPE),
            ("ghd", CALIBRATED_GHD_SECTOR_SCOPE),
            ("unsupported_nonres", "nonresidential_sector_asset_without_ghd_evidence"),
        ]:
            subset = group[group["sector_calibration_scope"].eq(scope)]
            row.update(_sector_values(prefix, subset))
        rows.append(row)
    return pd.DataFrame(rows)


def _sector_values(prefix: str, group: pd.DataFrame) -> dict[str, float | int]:
    if group.empty:
        return {
            f"{prefix}_ev_charger_rows": 0,
            f"{prefix}_ev_charger_kw": 0.0,
            f"{prefix}_wp_rows": 0,
            f"{prefix}_pv_rows": 0,
            f"{prefix}_pv_kw": 0.0,
            f"{prefix}_battery_rows": 0,
            f"{prefix}_battery_kwh": 0.0,
            f"{prefix}_heat_storage_rows": 0,
        }
    return {
        f"{prefix}_ev_charger_rows": int(group["asset_type"].eq("Ladestation").sum()),
        f"{prefix}_ev_charger_kw": float(
            group.loc[group["asset_type"].eq("Ladestation"), "sector_capacity_kw"].sum()
        ),
        f"{prefix}_wp_rows": int(group["asset_type"].eq("WP").sum()),
        f"{prefix}_pv_rows": int(group["asset_type"].eq("Photovoltaik").sum()),
        f"{prefix}_pv_kw": float(
            group.loc[
                group["asset_type"].eq("Photovoltaik"), "sector_capacity_kw"
            ].sum()
        ),
        f"{prefix}_battery_rows": int(group["asset_type"].eq("Batterie").sum()),
        f"{prefix}_battery_kwh": float(
            group.loc[group["asset_type"].eq("Batterie"), "sector_battery_kwh"].sum()
        ),
        f"{prefix}_heat_storage_rows": int(
            group["asset_type"].eq("Wärmespeicher").sum()
        ),
    }


def _grid_flags(grid_scope_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "lv_id",
        "passes_min_residential_equivalent_hh_buildings",
        "ghd_heavy_warning",
        "ghd_heavy_extreme",
        "has_unmatched_ghd",
        "recommended_for_residential_equivalent_scope",
        "recommended_for_full_local_demand_scope",
    ]
    return grid_scope_summary[
        [column for column in columns if column in grid_scope_summary.columns]
    ].copy()


def _build_street_plan(building_plan: pd.DataFrame) -> pd.DataFrame:
    if building_plan.empty:
        return pd.DataFrame()
    numeric_columns = [
        column
        for column in building_plan.columns
        if column.endswith(("_rows", "_kwh", "_kw", "_kwh_default"))
    ]
    named_aggregations = {column: (column, "sum") for column in numeric_columns}
    named_aggregations.update(
        buildings=("building_match_id", "nunique"),
        included_residential_equivalent_buildings=(
            "include_residential_equivalent_scope",
            "sum",
        ),
        included_full_local_demand_buildings=("include_full_local_demand_scope", "sum"),
    )
    return (
        building_plan.groupby(["lv_id", "building_street"], dropna=False)
        .agg(**named_aggregations)
        .reset_index()
        .sort_values(["lv_id", "building_street"], na_position="last")
    )


def _build_scope_totals(building_plan: pd.DataFrame) -> pd.DataFrame:
    if building_plan.empty:
        return pd.DataFrame()
    scopes = [
        ("all_matched_buildings", pd.Series(True, index=building_plan.index)),
        (
            "residential_equivalent_recommended",
            building_plan["recommended_for_residential_equivalent_scope"],
        ),
        (
            "full_local_demand_recommended",
            building_plan["recommended_for_full_local_demand_scope"],
        ),
    ]
    rows = []
    for scope, mask in scopes:
        group = building_plan[mask.fillna(False)]
        rows.append(
            {
                "scenario_scope": scope,
                "lv_grids": int(group["lv_id"].nunique()),
                "buildings": int(group["building_match_id"].nunique()),
                "residential_equivalent_hh_rows": int(
                    group["residential_equivalent_hh_rows"].sum()
                ),
                "residential_equivalent_hh_annual_kwh": float(
                    group["residential_equivalent_hh_annual_kwh"].sum()
                ),
                "calibrated_annual_ghd_kwh": float(
                    group["calibrated_annual_ghd_kwh"].sum()
                ),
                "residential_ev_charger_kw": float(
                    group["residential_ev_charger_kw"].sum()
                ),
                "ghd_ev_charger_kw": float(group["ghd_ev_charger_kw"].sum()),
                "residential_pv_kw": float(group["residential_pv_kw"].sum()),
                "ghd_pv_kw": float(group["ghd_pv_kw"].sum()),
                "residential_battery_kwh": float(
                    group["residential_battery_kwh"].sum()
                ),
                "ghd_battery_kwh": float(group["ghd_battery_kwh"].sum()),
            }
        )
    return pd.DataFrame(rows)
