"""Comparison-scope flags for SWF scenario calibration audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

RESIDENTIAL_EQUIVALENT_HH_CLASSES = {"residential_hh", "mixed_residential_proxy"}


def build_grid_scope_summary(
    matches: pd.DataFrame,
    ghd_calibration: pd.DataFrame,
    *,
    min_residential_equivalent_hh_buildings: int = 5,
    ghd_to_hh_warning_ratio: float = 1.0,
    ghd_to_hh_extreme_ratio: float = 2.0,
) -> pd.DataFrame:
    """Summarize which LV grids are suitable for comparison scopes."""
    if matches.empty:
        return pd.DataFrame()

    lv_ids = pd.Series(
        sorted(matches["lv_id"].dropna().astype(int).unique()), name="lv_id"
    ).to_frame()
    hh = matches[
        matches["asset_type"].eq("HH")
        & matches["comparison_asset_class"].isin(RESIDENTIAL_EQUIVALENT_HH_CLASSES)
    ].copy()
    hh_summary = _summarize_hh(hh)
    ghd_summary = _summarize_swf_ghd(matches)
    calibration_summary = _summarize_calibrated_ghd_by_grid(matches, ghd_calibration)

    summary = lv_ids.merge(hh_summary, on="lv_id", how="left")
    summary = summary.merge(ghd_summary, on="lv_id", how="left")
    summary = summary.merge(calibration_summary, on="lv_id", how="left")
    numeric_columns = [column for column in summary.columns if column != "lv_id"]
    summary[numeric_columns] = summary[numeric_columns].fillna(0.0)

    summary["ghd_to_hh_annual_demand_ratio"] = summary["swf_ghd_annual_kwh"] / summary[
        "residential_equivalent_hh_annual_kwh"
    ].where(summary["residential_equivalent_hh_annual_kwh"].gt(0))
    summary["passes_min_residential_equivalent_hh_buildings"] = summary[
        "residential_equivalent_hh_buildings"
    ] >= int(min_residential_equivalent_hh_buildings)
    summary["ghd_heavy_warning"] = summary["ghd_to_hh_annual_demand_ratio"].gt(
        float(ghd_to_hh_warning_ratio)
    )
    summary["ghd_heavy_extreme"] = summary["ghd_to_hh_annual_demand_ratio"].gt(
        float(ghd_to_hh_extreme_ratio)
    )
    summary["has_unmatched_ghd"] = summary["unmatched_swf_ghd_rows"].gt(0)
    summary["recommended_for_residential_equivalent_scope"] = summary[
        "passes_min_residential_equivalent_hh_buildings"
    ]
    summary["recommended_for_full_local_demand_scope"] = (
        summary["passes_min_residential_equivalent_hh_buildings"]
        & ~summary["has_unmatched_ghd"]
    )
    return summary.sort_values("lv_id").reset_index(drop=True)


def summarize_grid_scope_filters(grid_scope_summary: pd.DataFrame) -> pd.DataFrame:
    if grid_scope_summary.empty:
        return pd.DataFrame()

    rows = []
    for label, mask in [
        ("all_grids", pd.Series(True, index=grid_scope_summary.index)),
        (
            "residential_equivalent_recommended",
            grid_scope_summary["recommended_for_residential_equivalent_scope"],
        ),
        (
            "full_local_demand_recommended",
            grid_scope_summary["recommended_for_full_local_demand_scope"],
        ),
        (
            "filtered_min_hh_buildings",
            ~grid_scope_summary["passes_min_residential_equivalent_hh_buildings"],
        ),
        ("flagged_ghd_heavy_warning", grid_scope_summary["ghd_heavy_warning"]),
        ("flagged_ghd_heavy_extreme", grid_scope_summary["ghd_heavy_extreme"]),
        ("flagged_unmatched_ghd", grid_scope_summary["has_unmatched_ghd"]),
    ]:
        group = grid_scope_summary[mask.fillna(False)]
        rows.append(_scope_row(label, group))
    return pd.DataFrame(rows)


def _summarize_hh(hh: pd.DataFrame) -> pd.DataFrame:
    if hh.empty:
        return pd.DataFrame(columns=["lv_id"])
    return (
        hh.groupby("lv_id")
        .agg(
            residential_equivalent_hh_buildings=("building_match_id", "nunique"),
            residential_equivalent_hh_rows=("asset_id", "count"),
            residential_equivalent_hh_annual_kwh=(
                "swf_annual_demand_kwh",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
        )
        .reset_index()
    )


def _summarize_swf_ghd(matches: pd.DataFrame) -> pd.DataFrame:
    ghd = matches[matches["asset_type"].eq("GHD")].copy()
    if ghd.empty:
        return pd.DataFrame(columns=["lv_id"])
    ghd["swf_annual_demand_kwh_num"] = pd.to_numeric(
        ghd["swf_annual_demand_kwh"], errors="coerce"
    ).fillna(0.0)
    ghd["unmatched_ghd_kwh_num"] = np.where(
        ghd["comparison_asset_class"].eq("unmatched"),
        ghd["swf_annual_demand_kwh_num"],
        0.0,
    )
    return (
        ghd.groupby("lv_id")
        .agg(
            swf_ghd_rows=("asset_id", "count"),
            swf_ghd_annual_kwh=("swf_annual_demand_kwh_num", "sum"),
            unmatched_swf_ghd_rows=(
                "comparison_asset_class",
                lambda s: int((s == "unmatched").sum()),
            ),
            unmatched_swf_ghd_annual_kwh=("unmatched_ghd_kwh_num", "sum"),
        )
        .reset_index()
    )


def _summarize_calibrated_ghd_by_grid(
    matches: pd.DataFrame, ghd_calibration: pd.DataFrame
) -> pd.DataFrame:
    if ghd_calibration.empty:
        return pd.DataFrame(columns=["lv_id"])

    ghd = matches[matches["asset_type"].eq("GHD")].copy()
    ghd = ghd[ghd["comparison_asset_class"].ne("unmatched")].copy()
    if ghd.empty:
        calibrated = pd.DataFrame(columns=["lv_id", "calibrated_annual_ghd_kwh"])
    else:
        ghd["swf_annual_demand_kwh_num"] = pd.to_numeric(
            ghd["swf_annual_demand_kwh"],
            errors="coerce",
        ).fillna(0.0)
        calibrated = (
            ghd.groupby("lv_id")
            .agg(calibrated_annual_ghd_kwh=("swf_annual_demand_kwh_num", "sum"))
            .reset_index()
        )

    building_lv = (
        matches[matches["matched"]]
        .dropna(subset=["building_match_id"])[["building_match_id", "lv_id"]]
        .drop_duplicates()
    )
    calibration = ghd_calibration.merge(building_lv, on="building_match_id", how="left")
    calibration = calibration.dropna(subset=["lv_id"]).copy()
    if calibration.empty:
        defaults = pd.DataFrame(
            columns=[
                "lv_id",
                "synthetic_default_annual_ghd_kwh",
                "excluded_default_annual_ghd_kwh",
            ]
        )
    else:
        calibration = calibration.drop_duplicates(["building_match_id", "lv_id"])
        defaults = (
            calibration.groupby("lv_id")
            .agg(
                synthetic_default_annual_ghd_kwh=(
                    "synthetic_default_annual_ghd_kwh",
                    "sum",
                ),
                excluded_default_annual_ghd_kwh=(
                    "excluded_default_annual_ghd_kwh",
                    "sum",
                ),
            )
            .reset_index()
        )

    return defaults.merge(calibrated, on="lv_id", how="outer")


def _scope_row(label: str, group: pd.DataFrame) -> dict[str, float | int | str]:
    hh_kwh = (
        float(group["residential_equivalent_hh_annual_kwh"].sum())
        if not group.empty
        else 0.0
    )
    ghd_kwh = float(group["swf_ghd_annual_kwh"].sum()) if not group.empty else 0.0
    return {
        "scope_filter": label,
        "grids": int(len(group)),
        "residential_equivalent_hh_buildings": int(
            group["residential_equivalent_hh_buildings"].sum()
        )
        if not group.empty
        else 0,
        "residential_equivalent_hh_rows": int(
            group["residential_equivalent_hh_rows"].sum()
        )
        if not group.empty
        else 0,
        "residential_equivalent_hh_annual_kwh": hh_kwh,
        "swf_ghd_rows": int(group["swf_ghd_rows"].sum()) if not group.empty else 0,
        "swf_ghd_annual_kwh": ghd_kwh,
        "ghd_to_hh_annual_demand_ratio": ghd_kwh / hh_kwh if hh_kwh else np.nan,
        "unmatched_swf_ghd_rows": int(group["unmatched_swf_ghd_rows"].sum())
        if not group.empty
        else 0,
        "unmatched_swf_ghd_annual_kwh": float(
            group["unmatched_swf_ghd_annual_kwh"].sum()
        )
        if not group.empty
        else 0.0,
        "synthetic_default_annual_ghd_kwh": float(
            group["synthetic_default_annual_ghd_kwh"].sum()
        )
        if not group.empty
        else 0.0,
        "calibrated_annual_ghd_kwh": float(group["calibrated_annual_ghd_kwh"].sum())
        if not group.empty
        else 0.0,
        "excluded_default_annual_ghd_kwh": float(
            group["excluded_default_annual_ghd_kwh"].sum()
        )
        if not group.empty
        else 0.0,
    }
