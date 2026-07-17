"""GHD calibration helpers for SWF-to-pylovo scenario audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_synthetic_ghd_calibration(
    buildings: pd.DataFrame,
    matches: pd.DataFrame,
    expected_ghd_kwh_per_m2: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a no-scaling GHD calibration audit.

    Synthetic Commercial/Public buildings keep GHD demand only where SWF has
    directly matched GHD evidence. The calibrated annual GHD demand is the SWF
    annual demand at that building, not a scaled synthetic default.
    """
    if buildings.empty:
        empty = pd.DataFrame()
        return empty, empty

    matched = matches[matches["matched"]].copy()
    ghd = matched[matched["asset_type"].eq("GHD")].copy()
    hh = matched[matched["asset_type"].eq("HH")].copy()
    ghd_building_ids = set(ghd["building_match_id"].dropna().astype(int))

    candidates = buildings[
        buildings["building_use"].isin(expected_ghd_kwh_per_m2)
        | buildings["building_match_id"].isin(ghd_building_ids)
    ].copy()
    if candidates.empty:
        empty = pd.DataFrame()
        return empty, empty

    candidates["synthetic_default_ghd_kwh_per_m2"] = (
        candidates["building_use"].map(expected_ghd_kwh_per_m2).fillna(0.0)
    )
    candidates["synthetic_default_annual_ghd_kwh"] = pd.to_numeric(
        candidates["floor_area"], errors="coerce"
    ).fillna(0.0) * candidates["synthetic_default_ghd_kwh_per_m2"].fillna(0.0)

    ghd_by_building = (
        ghd.groupby("building_match_id", dropna=False)
        .agg(
            swf_ghd_rows=("asset_id", "count"),
            swf_ghd_annual_kwh=(
                "swf_annual_demand_kwh",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
        )
        .reset_index()
    )
    hh_by_building = (
        hh.groupby("building_match_id", dropna=False)
        .agg(
            swf_hh_rows=("asset_id", "count"),
            swf_hh_annual_kwh=(
                "swf_annual_demand_kwh",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
        )
        .reset_index()
    )

    calibration = candidates.merge(ghd_by_building, on="building_match_id", how="left")
    calibration = calibration.merge(hh_by_building, on="building_match_id", how="left")
    for column in [
        "swf_ghd_rows",
        "swf_ghd_annual_kwh",
        "swf_hh_rows",
        "swf_hh_annual_kwh",
    ]:
        calibration[column] = pd.to_numeric(
            calibration[column], errors="coerce"
        ).fillna(0.0)

    def _classify(row: pd.Series) -> str:
        has_default_ghd = row["synthetic_default_ghd_kwh_per_m2"] > 0
        if row["swf_ghd_rows"] > 0 and row["swf_hh_rows"] > 0 and has_default_ghd:
            return "direct_swf_ghd_match_with_hh_proxy"
        if row["swf_ghd_rows"] > 0 and row["swf_hh_rows"] > 0:
            return "direct_swf_ghd_match_on_residential_with_hh_proxy"
        if row["swf_ghd_rows"] > 0 and has_default_ghd:
            return "direct_swf_ghd_match"
        if row["swf_ghd_rows"] > 0:
            return "direct_swf_ghd_match_on_residential"
        if row["swf_hh_rows"] > 0:
            return "mixed_residential_proxy_no_ghd"
        return "no_swf_ghd_evidence"

    calibration["ghd_calibration_class"] = calibration.apply(_classify, axis=1)
    direct_mask = calibration["swf_ghd_rows"].gt(0)
    calibration["calibrated_annual_ghd_kwh"] = np.where(
        direct_mask,
        calibration["swf_ghd_annual_kwh"],
        0.0,
    )
    calibration["excluded_default_annual_ghd_kwh"] = np.where(
        direct_mask,
        0.0,
        calibration["synthetic_default_annual_ghd_kwh"],
    )

    columns = [
        "building_match_id",
        "objectid",
        "feature_id",
        "building_use",
        "building_type",
        "type",
        "street",
        "house_number",
        "floor_area",
        "households",
        "ghd_calibration_class",
        "swf_ghd_rows",
        "swf_ghd_annual_kwh",
        "swf_hh_rows",
        "swf_hh_annual_kwh",
        "synthetic_default_ghd_kwh_per_m2",
        "synthetic_default_annual_ghd_kwh",
        "calibrated_annual_ghd_kwh",
        "excluded_default_annual_ghd_kwh",
    ]
    calibration = calibration[columns].sort_values(
        ["ghd_calibration_class", "building_use", "street", "house_number"],
        na_position="last",
    )

    summary = _summarize_calibration(calibration, matches)
    return calibration.reset_index(drop=True), summary


def _summarize_calibration(
    calibration: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not calibration.empty:
        grouped = calibration.groupby(
            ["ghd_calibration_class", "building_use"], dropna=False
        )
        for (calibration_class, building_use), group in grouped:
            rows.append(
                {
                    "ghd_calibration_class": calibration_class,
                    "building_use": building_use,
                    "buildings": int(len(group)),
                    "floor_area_m2": float(
                        pd.to_numeric(group["floor_area"], errors="coerce").sum()
                    ),
                    "synthetic_default_annual_ghd_kwh": float(
                        group["synthetic_default_annual_ghd_kwh"].sum()
                    ),
                    "swf_ghd_rows": int(group["swf_ghd_rows"].sum()),
                    "swf_ghd_annual_kwh": float(group["swf_ghd_annual_kwh"].sum()),
                    "calibrated_annual_ghd_kwh": float(
                        group["calibrated_annual_ghd_kwh"].sum()
                    ),
                    "excluded_default_annual_ghd_kwh": float(
                        group["excluded_default_annual_ghd_kwh"].sum()
                    ),
                }
            )

    unmatched_ghd = matches[
        matches["comparison_asset_class"].eq("unmatched")
        & matches["asset_type"].eq("GHD")
    ]
    if not unmatched_ghd.empty:
        rows.append(
            {
                "ghd_calibration_class": "unmatched_swf_ghd_not_allocated",
                "building_use": "unmatched",
                "buildings": 0,
                "floor_area_m2": 0.0,
                "synthetic_default_annual_ghd_kwh": 0.0,
                "swf_ghd_rows": int(len(unmatched_ghd)),
                "swf_ghd_annual_kwh": float(
                    pd.to_numeric(
                        unmatched_ghd["swf_annual_demand_kwh"], errors="coerce"
                    ).sum()
                ),
                "calibrated_annual_ghd_kwh": 0.0,
                "excluded_default_annual_ghd_kwh": 0.0,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["ghd_calibration_class", "building_use"])
        .reset_index(drop=True)
    )
