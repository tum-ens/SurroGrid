"""Sector-coupling asset calibration helpers for SWF scenario audits."""

from __future__ import annotations


import numpy as np
import pandas as pd

SECTOR_ASSET_TYPES = {"WP", "Ladestation", "Photovoltaik", "Batterie", "Wärmespeicher"}
RESIDENTIAL_EQUIVALENT_CLASSES = {
    "sector_asset_on_residential",
    "sector_asset_on_mixed_proxy",
}


def build_sector_asset_calibration(
    matches: pd.DataFrame,
    ghd_calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify sector assets by residential-equivalent and calibrated GHD evidence."""
    if matches.empty:
        empty = pd.DataFrame()
        return empty, empty

    sector_assets = matches[matches["asset_type"].isin(SECTOR_ASSET_TYPES)].copy()
    if sector_assets.empty:
        empty = pd.DataFrame()
        return empty, empty

    calibrated_ghd_building_ids = set(
        ghd_calibration.loc[
            pd.to_numeric(ghd_calibration["swf_ghd_rows"], errors="coerce")
            .fillna(0.0)
            .gt(0),
            "building_match_id",
        ]
        .dropna()
        .astype(int)
    )

    sector_assets["sector_calibration_scope"] = sector_assets.apply(
        lambda row: _sector_scope(row, calibrated_ghd_building_ids),
        axis=1,
    )
    sector_assets["sector_capacity_kw"] = pd.to_numeric(
        sector_assets["asset_capacity_kw"], errors="coerce"
    ).fillna(0.0)
    sector_assets["sector_battery_kwh"] = pd.to_numeric(
        sector_assets["battery_capacity_kwh"], errors="coerce"
    ).fillna(0.0)

    columns = [
        "lv_id",
        "asset_id",
        "asset_table",
        "asset_type",
        "name",
        "bus",
        "bus_name",
        "building_match_id",
        "building_objectid",
        "building_use",
        "building_type",
        "building_street",
        "building_house_number",
        "comparison_asset_class",
        "sector_calibration_scope",
        "sector_capacity_kw",
        "sector_battery_kwh",
        "matched",
        "match_method",
        "match_distance_m",
    ]
    existing_columns = [column for column in columns if column in sector_assets.columns]
    sector_assets = sector_assets[existing_columns].sort_values(
        ["sector_calibration_scope", "asset_type", "lv_id", "building_match_id"],
        na_position="last",
    )

    summary = _summarize_sector_assets(sector_assets)
    return sector_assets.reset_index(drop=True), summary


def _sector_scope(row: pd.Series, calibrated_ghd_building_ids: set[int]) -> str:
    if not bool(row.get("matched", False)):
        return "unmatched_sector_asset"
    if row.get("comparison_asset_class") in RESIDENTIAL_EQUIVALENT_CLASSES:
        return "residential_equivalent_sector_asset"

    building_id = row.get("building_match_id")
    if not pd.isna(building_id) and int(building_id) in calibrated_ghd_building_ids:
        return "calibrated_ghd_sector_asset"
    return "nonresidential_sector_asset_without_ghd_evidence"


def _summarize_sector_assets(sector_assets: pd.DataFrame) -> pd.DataFrame:
    if sector_assets.empty:
        return pd.DataFrame()

    summary = (
        sector_assets.groupby(["sector_calibration_scope", "asset_type"], dropna=False)
        .agg(
            asset_rows=("asset_id", "count"),
            lv_grids=("lv_id", "nunique"),
            matched_buildings=("building_match_id", "nunique"),
            capacity_kw=("sector_capacity_kw", "sum"),
            battery_kwh=("sector_battery_kwh", "sum"),
        )
        .reset_index()
    )
    total_by_type = summary.groupby("asset_type")["asset_rows"].transform("sum")
    summary["asset_type_row_share_percent"] = np.where(
        total_by_type.gt(0),
        100.0 * summary["asset_rows"] / total_by_type,
        np.nan,
    )
    return summary.sort_values(["sector_calibration_scope", "asset_type"]).reset_index(
        drop=True
    )


def summarize_sector_scope_filters(sector_summary: pd.DataFrame) -> pd.DataFrame:
    if sector_summary.empty:
        return pd.DataFrame()
    return (
        sector_summary.groupby("sector_calibration_scope", dropna=False)
        .agg(
            asset_rows=("asset_rows", "sum"),
            lv_grid_asset_rows=("lv_grids", "sum"),
            matched_building_asset_rows=("matched_buildings", "sum"),
            capacity_kw=("capacity_kw", "sum"),
            battery_kwh=("battery_kwh", "sum"),
        )
        .reset_index()
    )
