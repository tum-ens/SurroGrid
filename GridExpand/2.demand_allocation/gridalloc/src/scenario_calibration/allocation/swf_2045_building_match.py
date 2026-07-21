"""Audit SWF 2045 electrification assets against pylovo building geometries.

The SWF real-grid model stores household loads, heat-pump markers, EV chargers,
PV plants, batteries, and heat-storage markers on pandapower buses. For a fair
post-electrification demand-allocation comparison we want to understand whether those electrical assets
can be reallocated to the physical building layer used by pylovo, instead of
treating the electrical connection point itself as a building.

This module is intentionally read-only. It produces CSV audits that can be
inspected before any demand-allocation or power-flow pipeline is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandapower as pp
from dotenv import load_dotenv
from scipy.spatial import cKDTree
from sqlalchemy import create_engine, text

from ..paths import (
    DEMAND_STATISTICS_DIR,
    ENV_PATH,
    GRIDALLOC_DIR,
    GRIDEXPAND_DIR,
)

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import get_pylovo_version_id  # noqa: E402
from .ghd_calibration import build_synthetic_ghd_calibration  # noqa: E402
from .scope_filters import build_grid_scope_summary, summarize_grid_scope_filters  # noqa: E402
from .sector_asset_calibration import (
    build_sector_asset_calibration,
    summarize_sector_scope_filters,
)  # noqa: E402
from .allocation_plan import build_scenario_allocation_plan  # noqa: E402

HH_MIN_ANNUAL_DEMAND_KWH = 500.0
HH_NAME_PATTERN = re.compile(r"NS_(?:Er)?Last", re.IGNORECASE)
ANNUAL_DEMAND_PATTERN = re.compile(
    r"2022:\s*([0-9]+(?:[.,][0-9]+)?)\s*kWh", re.IGNORECASE
)
RESIDENTIAL_TYPES = {"AB", "MFH", "SFH", "TH"}
SECTOR_ASSET_TYPES = {"WP", "Ladestation", "Photovoltaik", "Batterie", "Wärmespeicher"}


@dataclass(frozen=True)
class MatchConfig:
    plz: int
    final_year: int
    building_scope: str
    max_match_distance_m: float


def _database_engine():
    load_dotenv(ENV_PATH, override=True)
    return create_engine(
        "postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}"
    )


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).casefold()
    value = value.replace("straße", "strasse")
    value = value.replace("str.", "strasse")
    return re.sub(r"[^a-z0-9]+", "", value)


def _street_from_address(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text_value = str(value).strip()
    return re.sub(
        r"\s+\d+[a-zA-Z]?(?:\s*[-,/]\s*\d+[a-zA-Z]?)*\s*$", "", text_value
    ).strip()


def _parse_annual_demand_kwh(description: Any) -> float:
    match = ANNUAL_DEMAND_PATTERN.search(str(description))
    if not match:
        return float("nan")
    return float(match.group(1).replace(",", "."))


def _parse_bus_geo(value: Any) -> tuple[float, float]:
    if value is None or pd.isna(value):
        return (float("nan"), float("nan"))
    if isinstance(value, str):
        payload = json.loads(value)
    else:
        payload = value
    coordinates = payload.get("coordinates", [])
    if len(coordinates) < 2:
        return (float("nan"), float("nan"))
    return (float(coordinates[0]), float(coordinates[1]))


def _scenario_year_mask(df: pd.DataFrame, final_year: int) -> pd.Series:
    """Keep all assets present by final_year.

    Missing Baujahr values are retained and audited separately. In the current
    SWF 2045 data this is expected to include all scenario assets while avoiding
    silent loss of legacy rows with incomplete year metadata.
    """
    if "Baujahr" not in df.columns:
        return pd.Series(True, index=df.index)
    years = pd.to_numeric(df["Baujahr"], errors="coerce")
    return years.isna() | (years <= int(final_year))


def _building_is_residential(use: Any, building_type: Any, fallback_type: Any) -> bool:
    if str(use).strip().casefold() == "residential":
        return True
    return str(building_type or fallback_type).strip().upper() in RESIDENTIAL_TYPES


def _read_pylovo_buildings(config: MatchConfig) -> pd.DataFrame:
    version_id = get_pylovo_version_id()
    scope_filter = ""
    if config.building_scope == "residential":
        scope_filter = (
            "AND (lower(coalesce(building_use, '')) = 'residential' "
            "OR upper(coalesce(building_type, type, '')) = ANY(:residential_types))"
        )
    elif config.building_scope != "all":
        raise ValueError("building_scope must be 'residential' or 'all'.")

    query = text(
        f"""
        SELECT
            objectid,
            id,
            feature_id,
            building_use,
            building_use_id,
            building_type,
            type,
            households,
            occupants,
            floor_area,
            floor_number,
            construction_year,
            postcode,
            street,
            house_number,
            ST_X(centroid) AS x,
            ST_Y(centroid) AS y,
            ST_SRID(centroid) AS srid
        FROM pylovo.buildings_result
        WHERE postcode = :plz
          AND centroid IS NOT NULL
          AND (:version_id IS NULL OR version_id::text = :version_id)
          {scope_filter}
        """
    )
    params = {
        "plz": int(config.plz),
        "version_id": version_id,
        "residential_types": list(RESIDENTIAL_TYPES),
    }
    with _database_engine().connect() as conn:
        buildings = pd.read_sql_query(query, conn, params=params)

    if buildings.empty:
        raise ValueError(
            f"No pylovo building candidates found for PLZ={config.plz}, "
            f"version={version_id!r}, scope={config.building_scope!r}."
        )

    buildings = buildings.reset_index(drop=True)
    buildings["building_match_id"] = np.arange(len(buildings), dtype=int)
    buildings["building_address"] = (
        buildings["street"].fillna("").astype(str).str.strip()
        + " "
        + buildings["house_number"].fillna("").astype(str).str.strip()
    ).str.strip()
    buildings["building_address_key"] = buildings["building_address"].map(
        _normalize_text
    )
    buildings["building_street_key"] = buildings["street"].map(_normalize_text)
    buildings["building_is_residential"] = buildings.apply(
        lambda row: _building_is_residential(
            row["building_use"], row["building_type"], row["type"]
        ),
        axis=1,
    )
    return buildings


def _read_pylovo_building_summary(config: MatchConfig) -> pd.DataFrame:
    version_id = get_pylovo_version_id()
    query = text(
        """
        SELECT
            building_use,
            type,
            count(*) AS buildings,
            coalesce(sum(households), 0) AS households,
            coalesce(sum(occupants), 0) AS occupants,
            coalesce(sum(floor_area), 0) AS floor_area_m2
        FROM pylovo.buildings_result
        WHERE postcode = :plz
          AND (:version_id IS NULL OR version_id::text = :version_id)
        GROUP BY building_use, type
        ORDER BY building_use, type
        """
    )
    with _database_engine().connect() as conn:
        return pd.read_sql_query(
            query, conn, params={"plz": int(config.plz), "version_id": version_id}
        )


def _expected_ghd_kwh_per_m2_by_building_use() -> dict[str, float]:
    profiles = pd.read_csv(
        DEMAND_STATISTICS_DIR / "uninhabited_buildings" / "elec_ghd_per_m2.csv",
        skiprows=1,
    )
    type_distribution = pd.read_csv(
        DEMAND_STATISTICS_DIR
        / "uninhabited_buildings"
        / "nonresbuilding_usetype_distribution.csv",
        skiprows=1,
    )
    annual_kwh_per_m2 = profiles.drop(columns=["Unnamed: 0"], errors="ignore").sum()

    expected: dict[str, float] = {}
    for building_use, probability_col in [
        ("Public", "public_prob"),
        ("Commercial", "commercial_prob"),
    ]:
        weights = (
            type_distribution.set_index("type")[probability_col]
            .reindex(annual_kwh_per_m2.index)
            .fillna(0.0)
        )
        expected[building_use] = float((annual_kwh_per_m2 * weights).sum())
    return expected


def summarize_base_demand_definitions(
    matches: pd.DataFrame, building_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _swf_row(label: str, group: pd.DataFrame) -> None:
        rows.append(
            {
                "system": "real_swf",
                "scope": label,
                "buildings_or_connection_points": int(
                    group["building_match_id"].nunique()
                ),
                "load_or_household_rows": int(len(group)),
                "households": np.nan,
                "floor_area_m2": np.nan,
                "annual_electricity_kwh": float(
                    pd.to_numeric(group["swf_annual_demand_kwh"], errors="coerce").sum()
                ),
                "note": "Measured SWF annual demand parsed from load descriptions where available.",
            }
        )

    _swf_row(
        "HH on residential buildings",
        matches[matches["comparison_asset_class"].eq("residential_hh")],
    )
    _swf_row(
        "HH mixed proxy",
        matches[matches["comparison_asset_class"].eq("mixed_residential_proxy")],
    )
    _swf_row(
        "HH residential-equivalent total",
        matches[
            matches["comparison_asset_class"].isin(
                ["residential_hh", "mixed_residential_proxy"]
            )
        ],
    )
    _swf_row(
        "GHD", matches[matches["comparison_asset_class"].eq("true_nonresidential")]
    )
    _swf_row(
        "unmatched load rows",
        matches[
            matches["comparison_asset_class"].eq("unmatched")
            & matches["asset_type"].isin(["HH", "GHD"])
        ],
    )

    expected_ghd_kwh_per_m2 = _expected_ghd_kwh_per_m2_by_building_use()
    for building_use, group in building_summary.groupby("building_use", dropna=False):
        building_use_text = str(building_use)
        floor_area = float(pd.to_numeric(group["floor_area_m2"], errors="coerce").sum())
        annual_kwh = np.nan
        note = "Structural pylovo building stock. Residential annual electricity is sampled during Step 2."
        if building_use_text in expected_ghd_kwh_per_m2:
            annual_kwh = floor_area * expected_ghd_kwh_per_m2[building_use_text]
            note = "Expected GHD electricity from Step-2 per-m2 profiles and public/commercial type probabilities."
        rows.append(
            {
                "system": "synthetic_pylovo",
                "scope": building_use_text,
                "buildings_or_connection_points": int(group["buildings"].sum()),
                "load_or_household_rows": np.nan,
                "households": float(
                    pd.to_numeric(group["households"], errors="coerce").sum()
                ),
                "floor_area_m2": floor_area,
                "annual_electricity_kwh": annual_kwh,
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def _select_manifest_rows(
    root: Path, lv_ids: list[str] | None, limit: int | None
) -> pd.DataFrame:
    legacy_path = root / "split_manifest.csv"
    station_split_path = root / "station_split_manifest.csv"
    station_radial_path = root / "station_radialization_manifest.csv"
    if legacy_path.exists():
        manifest = pd.read_csv(legacy_path)
        selected = manifest[
            (manifest["variant"] == "radialized")
            & (manifest["status"] == "exported")
            & (manifest["category"] == "regular")
            & (manifest["load_status"] == "lvload")
        ].copy()
        selected["source_file"] = selected["file"]
    elif station_split_path.exists() and station_radial_path.exists():
        split = pd.read_csv(station_split_path)
        radial = pd.read_csv(station_radial_path).rename(
            columns={"grid": "station_id", "file": "radialized_file"}
        )
        manifest = split.merge(
            radial[["station_id", "radialized_file", "status"]],
            on="station_id",
            how="inner",
            suffixes=("_split", "_radial"),
        )
        selected = manifest[
            manifest["status_radial"].eq("ok")
            & manifest["category"].eq("regular")
            & manifest["load_status"].eq("lvload")
        ].copy()
        selected["lv_id"] = selected["station_id"].str.extract(r"(\d+)")[0].astype(int)
        selected["source_file"] = selected["radialized_file"]
    else:
        raise FileNotFoundError(
            f"Expected split_manifest.csv or both station split manifests in {root}."
        )
    if lv_ids:
        wanted = {int(str(value).removeprefix("LV_")) for value in lv_ids}
        selected = selected[selected["lv_id"].astype(int).isin(wanted)]
    selected = selected.sort_values("lv_id")
    if limit is not None:
        selected = selected.head(int(limit))
    selected["source_file"] = selected["source_file"].map(
        lambda value: Path(str(value))
    )
    selected["source_file"] = selected["source_file"].map(
        lambda value: value if value.is_absolute() else root / value
    )
    return selected


def _bus_lookup(net: pp.pandapowerNet) -> pd.DataFrame:
    buses = net.bus.copy()
    coords = buses["geo"].map(_parse_bus_geo).apply(pd.Series)
    coords.columns = ["asset_x", "asset_y"]
    buses = pd.concat([buses, coords], axis=1)
    buses = buses.rename(
        columns={
            "name": "bus_name",
            "description": "bus_description",
            "Baujahr": "bus_baujahr",
            "chr_name": "bus_chr_name",
        }
    )
    buses["bus"] = buses.index.astype(int)
    buses["bus_address"] = buses["bus_description"].where(
        buses["bus_name"].astype(str).str.contains("HaAn", case=False, na=False),
        "",
    )
    buses["bus_address_key"] = buses["bus_address"].map(_normalize_text)
    buses["bus_street"] = buses["bus_address"].map(_street_from_address)
    buses["bus_street_key"] = buses["bus_street"].map(_normalize_text)
    return buses[
        [
            "bus",
            "bus_name",
            "bus_description",
            "bus_address",
            "bus_address_key",
            "bus_street",
            "bus_street_key",
            "asset_x",
            "asset_y",
            "bus_chr_name",
            "bus_baujahr",
        ]
    ]


def _load_assets_from_grid(row: pd.Series, config: MatchConfig) -> pd.DataFrame:
    net = pp.from_excel(str(row["source_file"]))
    buses = _bus_lookup(net)
    frames: list[pd.DataFrame] = []

    loads = net.load.copy()
    if not loads.empty:
        if "in_service" in loads.columns:
            loads = loads[loads["in_service"].fillna(True).astype(bool)].copy()
        loads = loads[_scenario_year_mask(loads, config.final_year)].copy()
        loads["asset_table"] = "load"
        loads["asset_type"] = loads["type"].astype(str)
        loads["swf_annual_demand_kwh"] = loads["description"].map(
            _parse_annual_demand_kwh
        )
        hh_mask = (
            loads["asset_type"].eq("HH")
            & loads["name"].astype(str).str.contains(HH_NAME_PATTERN, na=False)
            & (
                loads["swf_annual_demand_kwh"].isna()
                | (loads["swf_annual_demand_kwh"] >= HH_MIN_ANNUAL_DEMAND_KWH)
            )
        )
        asset_loads = loads[
            hh_mask | loads["asset_type"].isin(["WP", "Ladestation", "GHD"])
        ].copy()
        frames.append(asset_loads)

    sgens = net.sgen.copy()
    if not sgens.empty:
        if "in_service" in sgens.columns:
            sgens = sgens[sgens["in_service"].fillna(True).astype(bool)].copy()
        sgens = sgens[_scenario_year_mask(sgens, config.final_year)].copy()
        sgens["asset_table"] = "sgen"
        sgens["asset_type"] = sgens["type"].astype(str)
        sgens["swf_annual_demand_kwh"] = np.nan
        sgens = sgens[
            sgens["asset_type"].isin(["Photovoltaik", "Batterie", "Wärmespeicher"])
        ].copy()
        frames.append(sgens)

    if not frames:
        return pd.DataFrame()

    assets = pd.concat(frames, ignore_index=False).reset_index(names="source_index")
    assets["lv_id"] = int(row["lv_id"])
    assets["source_file"] = str(row["source_file"])
    assets["asset_id"] = (
        assets["asset_table"] + ":" + assets["source_index"].astype(str)
    )
    assets["asset_capacity_kw"] = (
        pd.to_numeric(assets.get("p_mw"), errors="coerce") * 1000.0
    )
    assets.loc[assets["asset_type"].eq("Batterie"), "asset_capacity_kw"] = np.nan
    assets["battery_capacity_kwh"] = (
        pd.to_numeric(assets.get("capacity"), errors="coerce") * 1000.0
    )
    assets = assets.merge(buses, on="bus", how="left")
    return assets


def load_swf_2045_assets(
    *,
    grid_data_path: Path,
    config: MatchConfig,
    lv_ids: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    selected = _select_manifest_rows(grid_data_path, lv_ids, limit)
    frames = [_load_assets_from_grid(row, config) for _, row in selected.iterrows()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    assets = pd.concat(frames, ignore_index=True)
    return assets


def _build_address_index(buildings: pd.DataFrame) -> dict[str, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    for key, group in buildings.groupby("building_address_key"):
        if key:
            grouped[key] = group.index.to_numpy(dtype=int)
    return grouped


def match_assets_to_buildings(
    assets: pd.DataFrame, buildings: pd.DataFrame, config: MatchConfig
) -> pd.DataFrame:
    if assets.empty:
        return assets.copy()
    valid_buildings = buildings.dropna(subset=["x", "y"]).copy()
    if valid_buildings.empty:
        raise ValueError("Building candidates have no valid coordinates.")

    tree = cKDTree(valid_buildings[["x", "y"]].to_numpy(dtype=float))
    address_index = _build_address_index(valid_buildings)
    matched_rows: list[dict[str, Any]] = []
    valid_lookup = valid_buildings.reset_index().rename(
        columns={"index": "_building_row"}
    )

    for asset in assets.to_dict("records"):
        x = asset.get("asset_x")
        y = asset.get("asset_y")
        if pd.isna(x) or pd.isna(y):
            match = {
                "match_method": "unmatched_no_asset_coordinates",
                "match_distance_m": np.nan,
            }
            building_row = None
        else:
            asset_xy = np.array([float(x), float(y)])
            address_candidates = address_index.get(
                str(asset.get("bus_address_key", "")), np.array([], dtype=int)
            )
            if len(address_candidates):
                candidate_buildings = valid_buildings.loc[address_candidates]
                distances = np.linalg.norm(
                    candidate_buildings[["x", "y"]].to_numpy(dtype=float) - asset_xy,
                    axis=1,
                )
                best_position = int(np.argmin(distances))
                building_row = candidate_buildings.iloc[best_position]
                match = {
                    "match_method": "exact_address_nearest",
                    "match_distance_m": float(distances[best_position]),
                }
            else:
                distance, position = tree.query(asset_xy, k=1)
                building_row = valid_lookup.iloc[int(position)]
                match = {
                    "match_method": "nearest_building",
                    "match_distance_m": float(distance),
                }

        if (
            building_row is None
            or match["match_distance_m"] > config.max_match_distance_m
        ):
            building_payload = {
                "matched": False,
                "building_match_id": np.nan,
                "building_objectid": None,
                "building_street": None,
                "building_house_number": None,
                "building_address": None,
                "building_use": None,
                "building_use_id": None,
                "building_type": None,
                "building_is_residential": False,
                "building_households": np.nan,
                "building_floor_area": np.nan,
                "building_x": np.nan,
                "building_y": np.nan,
            }
            if building_row is not None:
                match["match_method"] = f"{match['match_method']}_beyond_threshold"
        else:
            building_payload = {
                "matched": True,
                "building_match_id": int(building_row["building_match_id"]),
                "building_objectid": building_row["objectid"],
                "building_street": building_row["street"],
                "building_house_number": building_row["house_number"],
                "building_address": building_row["building_address"],
                "building_use": building_row["building_use"],
                "building_use_id": building_row["building_use_id"],
                "building_type": building_row["building_type"],
                "building_is_residential": bool(
                    building_row["building_is_residential"]
                ),
                "building_households": building_row["households"],
                "building_floor_area": building_row["floor_area"],
                "building_x": building_row["x"],
                "building_y": building_row["y"],
            }

        if not building_payload["matched"]:
            comparison_scope = "unmatched"
        elif building_payload["building_is_residential"]:
            comparison_scope = "matched_residential"
        else:
            comparison_scope = "matched_nonresidential"
        matched_rows.append(
            {**asset, **match, **building_payload, "comparison_scope": comparison_scope}
        )

    return pd.DataFrame(matched_rows)


def add_comparison_asset_class(matches: pd.DataFrame) -> pd.DataFrame:
    """Classify SWF assets for transparent real/synthetic scenario comparison."""
    if matches.empty:
        return matches.copy()

    classified = matches.copy()
    matched_hh = classified[classified["matched"] & classified["asset_type"].eq("HH")]
    hh_buildings = set(matched_hh["building_match_id"].dropna().astype(int))

    def _classify(row: pd.Series) -> str:
        if not row["matched"]:
            return "unmatched"

        asset_type = str(row["asset_type"])
        is_residential = bool(row["building_is_residential"])
        building_id = row["building_match_id"]
        has_hh_at_building = (
            not pd.isna(building_id) and int(building_id) in hh_buildings
        )

        if asset_type == "HH":
            return "residential_hh" if is_residential else "mixed_residential_proxy"
        if asset_type == "GHD":
            return "true_nonresidential"
        if asset_type in SECTOR_ASSET_TYPES:
            if is_residential:
                return "sector_asset_on_residential"
            if has_hh_at_building:
                return "sector_asset_on_mixed_proxy"
            return "sector_asset_on_nonresidential"
        return "other"

    classified["comparison_asset_class"] = classified.apply(_classify, axis=1)
    return classified


def summarize_comparison_classes(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    summary = (
        matches.groupby(["comparison_asset_class", "asset_type"], dropna=False)
        .agg(
            asset_rows=("asset_id", "count"),
            matched_buildings=("building_match_id", "nunique"),
            hh_annual_demand_kwh=(
                "swf_annual_demand_kwh",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
            asset_capacity_kw=(
                "asset_capacity_kw",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
            battery_capacity_kwh=(
                "battery_capacity_kwh",
                lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
            ),
        )
        .reset_index()
    )
    total_rows = float(summary["asset_rows"].sum())
    summary["asset_row_share_percent"] = (
        100.0 * summary["asset_rows"] / total_rows if total_rows else np.nan
    )
    return summary.sort_values(["comparison_asset_class", "asset_type"]).reset_index(
        drop=True
    )


def summarize_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    def _asset_count(asset_type: str) -> tuple[str, str]:
        return (f"{asset_type}_count", f"{asset_type}_matched")

    summary_rows = []
    for lv_id, group in matches.groupby("lv_id"):
        row: dict[str, Any] = {
            "lv_id": lv_id,
            "asset_rows": int(len(group)),
            "matched_asset_rows": int(group["matched"].sum()),
            "match_rate_percent": 100.0 * float(group["matched"].mean()),
            "median_match_distance_m": float(
                group.loc[group["matched"], "match_distance_m"].median()
            ),
            "p95_match_distance_m": float(
                group.loc[group["matched"], "match_distance_m"].quantile(0.95)
            ),
            "max_match_distance_m": float(
                group.loc[group["matched"], "match_distance_m"].max()
            ),
            "matched_buildings": int(
                group.loc[group["matched"], "building_match_id"].nunique()
            ),
            "matched_residential_asset_rows": int(
                group["comparison_scope"].eq("matched_residential").sum()
            ),
            "matched_nonresidential_asset_rows": int(
                group["comparison_scope"].eq("matched_nonresidential").sum()
            ),
            "unmatched_asset_rows": int(
                group["comparison_scope"].eq("unmatched").sum()
            ),
        }
        for asset_type in [
            "HH",
            "WP",
            "Ladestation",
            "Photovoltaik",
            "Batterie",
            "Wärmespeicher",
            "GHD",
        ]:
            count_col, matched_col = _asset_count(asset_type)
            subset = group[group["asset_type"].eq(asset_type)]
            row[count_col] = int(len(subset))
            row[matched_col] = int(subset["matched"].sum()) if not subset.empty else 0
        row["hh_annual_demand_kwh"] = float(
            pd.to_numeric(
                group.loc[group["asset_type"].eq("HH"), "swf_annual_demand_kwh"],
                errors="coerce",
            ).sum()
        )
        row["ev_charger_kw"] = float(
            pd.to_numeric(
                group.loc[group["asset_type"].eq("Ladestation"), "asset_capacity_kw"],
                errors="coerce",
            ).sum()
        )
        row["pv_kw"] = float(
            pd.to_numeric(
                group.loc[group["asset_type"].eq("Photovoltaik"), "asset_capacity_kw"],
                errors="coerce",
            ).sum()
        )
        row["battery_kwh"] = float(
            pd.to_numeric(
                group.loc[group["asset_type"].eq("Batterie"), "battery_capacity_kwh"],
                errors="coerce",
            ).sum()
        )
        summary_rows.append(row)

    matched = matches[matches["matched"]].copy()
    matched["hh_annual_demand_kwh_sum"] = np.where(
        matched["asset_type"].eq("HH"),
        pd.to_numeric(matched["swf_annual_demand_kwh"], errors="coerce").fillna(0.0),
        0.0,
    )
    matched["ev_charger_kw_sum"] = np.where(
        matched["asset_type"].eq("Ladestation"),
        pd.to_numeric(matched["asset_capacity_kw"], errors="coerce").fillna(0.0),
        0.0,
    )
    matched["pv_kw_sum"] = np.where(
        matched["asset_type"].eq("Photovoltaik"),
        pd.to_numeric(matched["asset_capacity_kw"], errors="coerce").fillna(0.0),
        0.0,
    )
    matched["battery_kwh_sum"] = np.where(
        matched["asset_type"].eq("Batterie"),
        pd.to_numeric(matched["battery_capacity_kwh"], errors="coerce").fillna(0.0),
        0.0,
    )
    building_allocations = (
        matched.groupby(
            ["lv_id", "building_match_id", "building_objectid"], dropna=False
        )
        .agg(
            building_street=("building_street", "first"),
            building_house_number=("building_house_number", "first"),
            building_address=("building_address", "first"),
            building_use=("building_use", "first"),
            building_type=("building_type", "first"),
            building_is_residential=("building_is_residential", "first"),
            comparison_scope=("comparison_scope", "first"),
            comparison_asset_classes=(
                "comparison_asset_class",
                lambda s: ",".join(sorted(set(map(str, s)))),
            ),
            building_households=("building_households", "first"),
            building_floor_area=("building_floor_area", "first"),
            matched_asset_rows=("asset_id", "count"),
            hh_rows=("asset_type", lambda s: int((s == "HH").sum())),
            wp_rows=("asset_type", lambda s: int((s == "WP").sum())),
            ev_charger_rows=("asset_type", lambda s: int((s == "Ladestation").sum())),
            pv_rows=("asset_type", lambda s: int((s == "Photovoltaik").sum())),
            battery_rows=("asset_type", lambda s: int((s == "Batterie").sum())),
            heat_storage_rows=(
                "asset_type",
                lambda s: int((s == "Wärmespeicher").sum()),
            ),
            hh_annual_demand_kwh=("hh_annual_demand_kwh_sum", "sum"),
            ev_charger_kw=("ev_charger_kw_sum", "sum"),
            pv_kw=("pv_kw_sum", "sum"),
            battery_kwh=("battery_kwh_sum", "sum"),
            max_match_distance_m=("match_distance_m", "max"),
        )
        .reset_index()
    )

    street_allocations = (
        building_allocations.groupby(
            ["lv_id", "comparison_scope", "building_street"], dropna=False
        )
        .agg(
            matched_buildings=("building_match_id", "nunique"),
            hh_rows=("hh_rows", "sum"),
            wp_rows=("wp_rows", "sum"),
            ev_charger_rows=("ev_charger_rows", "sum"),
            pv_rows=("pv_rows", "sum"),
            battery_rows=("battery_rows", "sum"),
            heat_storage_rows=("heat_storage_rows", "sum"),
            hh_annual_demand_kwh=("hh_annual_demand_kwh", "sum"),
            ev_charger_kw=("ev_charger_kw", "sum"),
            pv_kw=("pv_kw", "sum"),
            battery_kwh=("battery_kwh", "sum"),
        )
        .reset_index()
    )

    return (
        pd.DataFrame(summary_rows).sort_values("lv_id"),
        building_allocations,
        street_allocations,
    )


def _scenario_intensity_row(label: str, group: pd.DataFrame) -> dict[str, Any]:
    buildings = group["building_match_id"].nunique()
    hh_rows = int(group["asset_type"].eq("HH").sum())
    hh_kwh = float(
        pd.to_numeric(
            group.loc[group["asset_type"].eq("HH"), "swf_annual_demand_kwh"],
            errors="coerce",
        ).sum()
    )
    ev_rows = int(group["asset_type"].eq("Ladestation").sum())
    ev_kw = float(
        pd.to_numeric(
            group.loc[group["asset_type"].eq("Ladestation"), "asset_capacity_kw"],
            errors="coerce",
        ).sum()
    )
    wp_rows = int(group["asset_type"].eq("WP").sum())
    pv_rows = int(group["asset_type"].eq("Photovoltaik").sum())
    pv_kw = float(
        pd.to_numeric(
            group.loc[group["asset_type"].eq("Photovoltaik"), "asset_capacity_kw"],
            errors="coerce",
        ).sum()
    )
    battery_rows = int(group["asset_type"].eq("Batterie").sum())
    battery_kwh = float(
        pd.to_numeric(
            group.loc[group["asset_type"].eq("Batterie"), "battery_capacity_kwh"],
            errors="coerce",
        ).sum()
    )
    heat_storage_rows = int(group["asset_type"].eq("Wärmespeicher").sum())

    return {
        "scenario_scope": label,
        "matched_buildings": int(buildings),
        "hh_rows": hh_rows,
        "hh_annual_demand_kwh": hh_kwh,
        "ev_charger_rows": ev_rows,
        "ev_charger_kw": ev_kw,
        "wp_rows": wp_rows,
        "pv_rows": pv_rows,
        "pv_kw": pv_kw,
        "battery_rows": battery_rows,
        "battery_kwh": battery_kwh,
        "heat_storage_rows": heat_storage_rows,
        "ev_chargers_per_hh_row": ev_rows / hh_rows if hh_rows else np.nan,
        "ev_kw_per_hh_row": ev_kw / hh_rows if hh_rows else np.nan,
        "wp_rows_per_hh_row": wp_rows / hh_rows if hh_rows else np.nan,
        "pv_kw_per_hh_row": pv_kw / hh_rows if hh_rows else np.nan,
        "battery_kwh_per_pv_kw": battery_kwh / pv_kw if pv_kw else np.nan,
        "hh_rows_per_building": hh_rows / buildings if buildings else np.nan,
        "ev_chargers_per_building": ev_rows / buildings if buildings else np.nan,
        "wp_rows_per_building": wp_rows / buildings if buildings else np.nan,
        "pv_kw_per_building": pv_kw / buildings if buildings else np.nan,
    }


def summarize_scenario_intensities(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    strict_classes = {"residential_hh", "sector_asset_on_residential"}
    residential_equivalent_classes = {
        "residential_hh",
        "mixed_residential_proxy",
        "sector_asset_on_residential",
        "sector_asset_on_mixed_proxy",
    }
    rows = []
    for label, classes in [
        ("strict_residential_buildings", strict_classes),
        ("residential_equivalent_with_mixed_proxy", residential_equivalent_classes),
    ]:
        group = matches[matches["comparison_asset_class"].isin(classes)].copy()
        if not group.empty:
            rows.append(_scenario_intensity_row(label, group))
    return pd.DataFrame(rows)


def run_audit(
    *,
    plz: int,
    final_year: int,
    output_dir: Path,
    grid_data_path: Path | None = None,
    lv_ids: list[str] | None = None,
    limit: int | None = None,
    building_scope: str = "all",
    max_match_distance_m: float = 100.0,
    min_residential_equivalent_hh_buildings: int = 5,
    ghd_to_hh_warning_ratio: float = 1.0,
    ghd_to_hh_extreme_ratio: float = 2.0,
) -> dict[str, pd.DataFrame]:
    load_dotenv(ENV_PATH, override=True)
    root = grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    config = MatchConfig(
        plz=int(plz),
        final_year=int(final_year),
        building_scope=building_scope,
        max_match_distance_m=float(max_match_distance_m),
    )
    buildings = _read_pylovo_buildings(config)
    building_summary = _read_pylovo_building_summary(config)
    assets = load_swf_2045_assets(
        grid_data_path=root, config=config, lv_ids=lv_ids, limit=limit
    )
    matches = match_assets_to_buildings(assets, buildings, config)
    matches = add_comparison_asset_class(matches)
    grid_summary, building_allocations, street_allocations = summarize_matches(matches)
    comparison_class_summary = summarize_comparison_classes(matches)
    base_demand_definitions = summarize_base_demand_definitions(
        matches, building_summary
    )
    expected_ghd_kwh_per_m2 = _expected_ghd_kwh_per_m2_by_building_use()
    ghd_calibration, ghd_calibration_summary = build_synthetic_ghd_calibration(
        buildings,
        matches,
        expected_ghd_kwh_per_m2,
    )
    grid_scope_summary = build_grid_scope_summary(
        matches,
        ghd_calibration,
        min_residential_equivalent_hh_buildings=min_residential_equivalent_hh_buildings,
        ghd_to_hh_warning_ratio=ghd_to_hh_warning_ratio,
        ghd_to_hh_extreme_ratio=ghd_to_hh_extreme_ratio,
    )
    grid_scope_filter_summary = summarize_grid_scope_filters(grid_scope_summary)
    sector_asset_calibration, sector_asset_calibration_summary = (
        build_sector_asset_calibration(
            matches,
            ghd_calibration,
        )
    )
    sector_scope_filter_summary = summarize_sector_scope_filters(
        sector_asset_calibration_summary
    )
    (
        bus_scenario_plan,
        building_scenario_plan,
        street_scenario_plan,
        scenario_scope_totals,
    ) = build_scenario_allocation_plan(
        matches,
        ghd_calibration,
        sector_asset_calibration,
        grid_scope_summary,
    )
    scenario_intensities = summarize_scenario_intensities(matches)

    output_dir.mkdir(parents=True, exist_ok=True)
    full_local_bus_plan = bus_scenario_plan[
        bus_scenario_plan["include_full_local_demand_scenario"]
    ].copy()
    full_local_bus_plan["scenario_scope"] = "full_local_demand_recommended"
    residential_equivalent_bus_plan = bus_scenario_plan[
        bus_scenario_plan["include_residential_equivalent_scenario"]
    ].copy()
    residential_equivalent_bus_plan["scenario_scope"] = (
        "residential_equivalent_recommended"
    )
    residential_fallback_zero_columns = [
        "swf_ghd_rows",
        "calibrated_annual_ghd_kwh",
        "ghd_ev_charger_rows",
        "ghd_ev_charger_kw",
        "ghd_wp_rows",
        "ghd_pv_rows",
        "ghd_pv_kw",
        "ghd_battery_rows",
        "ghd_battery_kwh",
        "ghd_heat_storage_rows",
    ]
    for column in residential_fallback_zero_columns:
        if column in residential_equivalent_bus_plan.columns:
            residential_equivalent_bus_plan[column] = 0.0

    outputs = {
        "swf_2045_assets": assets,
        "swf_2045_asset_building_matches": matches,
        "swf_2045_match_grid_summary": grid_summary,
        "swf_2045_building_allocations": building_allocations,
        "swf_2045_street_allocations": street_allocations,
        "swf_2045_comparison_class_summary": comparison_class_summary,
        "swf_2045_base_demand_definitions": base_demand_definitions,
        "swf_2045_synthetic_ghd_calibration": ghd_calibration,
        "swf_2045_synthetic_ghd_calibration_summary": ghd_calibration_summary,
        "swf_2045_grid_scope_summary": grid_scope_summary,
        "swf_2045_grid_scope_filter_summary": grid_scope_filter_summary,
        "swf_2045_sector_asset_calibration": sector_asset_calibration,
        "swf_2045_sector_asset_calibration_summary": sector_asset_calibration_summary,
        "swf_2045_sector_scope_filter_summary": sector_scope_filter_summary,
        "swf_2045_bus_scenario_plan": bus_scenario_plan,
        "swf_2045_full_local_demand_bus_allocation_plan": full_local_bus_plan,
        "swf_2045_residential_equivalent_bus_allocation_plan": residential_equivalent_bus_plan,
        "swf_2045_building_scenario_plan": building_scenario_plan,
        "swf_2045_street_scenario_plan": street_scenario_plan,
        "swf_2045_scenario_scope_totals": scenario_scope_totals,
        "swf_2045_scenario_intensities": scenario_intensities,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--final-year", type=int, default=2045)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GRIDALLOC_DIR
        / "outputs"
        / "scenario_calibration"
        / "swf_2045_building_match",
    )
    parser.add_argument(
        "--lv-id",
        action="append",
        dest="lv_ids",
        help="Optional LV id filter. Can be passed repeatedly.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--building-scope", choices=["residential", "all"], default="all"
    )
    parser.add_argument("--max-match-distance-m", type=float, default=100.0)
    parser.add_argument(
        "--min-residential-equivalent-hh-buildings", type=int, default=5
    )
    parser.add_argument("--ghd-to-hh-warning-ratio", type=float, default=1.0)
    parser.add_argument("--ghd-to-hh-extreme-ratio", type=float, default=2.0)
    args = parser.parse_args()

    outputs = run_audit(
        plz=args.plz,
        final_year=args.final_year,
        output_dir=args.output_dir,
        grid_data_path=args.grid_data_path,
        lv_ids=args.lv_ids,
        limit=args.limit,
        building_scope=args.building_scope,
        max_match_distance_m=args.max_match_distance_m,
        min_residential_equivalent_hh_buildings=args.min_residential_equivalent_hh_buildings,
        ghd_to_hh_warning_ratio=args.ghd_to_hh_warning_ratio,
        ghd_to_hh_extreme_ratio=args.ghd_to_hh_extreme_ratio,
    )
    summary = outputs["swf_2045_match_grid_summary"]
    print(f"Wrote SWF 2045 building-match audit to {args.output_dir}")
    if not summary.empty:
        print(summary.describe(include="all").to_string())


if __name__ == "__main__":
    main()
