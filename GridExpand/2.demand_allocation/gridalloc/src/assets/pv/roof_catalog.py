"""LoD2 roof-surface catalog used by every active PV workflow."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text


DEFAULT_MODULE_CAPACITY_KW_PER_M2 = 0.202
DEFAULT_FLAT_ROOF_UTILIZATION = 0.27
DEFAULT_SLANTED_ROOF_UTILIZATION = 0.58
DEFAULT_FALLBACK_CAPACITY_KW = 14.5
DEFAULT_TILT_BIN_DEG = 5.0
DEFAULT_AZIMUTH_BIN_DEG = 15.0
FALLBACK_TILT_DEG = 45.0
FALLBACK_AZIMUTH_DEG = 180.0
FLAT_TILT_TOLERANCE_DEG = 0.01


ROOF_SURFACE_QUERY = text(
    """
    SELECT
        building.objectid::text AS building_objectid,
        roof.id::text AS roof_surface_id,
        max(CASE WHEN attribute.name = 'Dachneigung'
                 THEN attribute.val_string::double precision END) AS dachneigung,
        max(CASE WHEN attribute.name = 'Dachorientierung'
                 THEN attribute.val_string::double precision END) AS dachorientierung,
        max(CASE WHEN attribute.name = 'Flaeche'
                 THEN attribute.val_string::double precision END) AS roof_area_m2
    FROM citydb.feature AS building
    JOIN citydb.property AS boundary
      ON boundary.feature_id = building.id
     AND boundary.name = 'boundary'
    JOIN citydb.feature AS roof
      ON roof.id = boundary.val_feature_id
    JOIN citydb.property AS attribute
      ON attribute.feature_id = roof.id
     AND attribute.name IN ('Dachneigung', 'Dachorientierung', 'Flaeche')
    WHERE building.objectid IN :building_objectids
    GROUP BY building.objectid, roof.id
    HAVING count(*) FILTER (WHERE attribute.name = 'Dachneigung') > 0
    """
).bindparams(bindparam("building_objectids", expanding=True))


def load_lod2_roof_catalog(
    engine,
    building_objectids: Iterable[object],
    **options,
) -> pd.DataFrame:
    """Load and normalize CityDB LoD2 sections for the requested buildings."""
    building_ids = sorted({str(value) for value in building_objectids if pd.notna(value)})
    if not building_ids:
        return empty_roof_catalog()
    with engine.connect() as connection:
        raw = pd.read_sql_query(
            ROOF_SURFACE_QUERY,
            connection,
            params={"building_objectids": building_ids},
        )
    catalog = normalize_lod2_roof_sections(raw, **options)
    return add_missing_building_fallbacks(catalog, building_ids, **_bin_fallback_options(options))


def read_lod2_roof_catalog_hdf(
    path: str | Path,
    building_objectids: Iterable[object],
    **options,
) -> pd.DataFrame:
    """Read embedded LoD2 sections; HDF inputs never synthesize roof geometry."""
    try:
        raw = pd.read_hdf(path, key="raw_data/pv_roof_sections")
    except KeyError as exc:
        raise ValueError(
            "HDF grid input lacks raw_data/pv_roof_sections. Regenerate the input "
            "from CityDB LoD2 data; sampled roof geometry is no longer supported."
        ) from exc
    ids = {str(value) for value in building_objectids if pd.notna(value)}
    raw = raw[raw["building_objectid"].astype(str).isin(ids)].copy()
    catalog = normalize_lod2_roof_sections(raw, **options)
    return add_missing_building_fallbacks(catalog, ids, **_bin_fallback_options(options))


def normalize_lod2_roof_sections(
    raw: pd.DataFrame,
    *,
    tilt_bin_deg: float = DEFAULT_TILT_BIN_DEG,
    azimuth_bin_deg: float = DEFAULT_AZIMUTH_BIN_DEG,
    module_capacity_kw_per_m2: float = DEFAULT_MODULE_CAPACITY_KW_PER_M2,
    flat_roof_utilization: float = DEFAULT_FLAT_ROOF_UTILIZATION,
    slanted_roof_utilization: float = DEFAULT_SLANTED_ROOF_UTILIZATION,
    fallback_capacity_kw: float = DEFAULT_FALLBACK_CAPACITY_KW,
) -> pd.DataFrame:
    """Convert CityDB angles to pvlib convention and calculate section potential."""
    del fallback_capacity_kw
    required = {"building_objectid", "roof_surface_id", "dachneigung", "dachorientierung", "roof_area_m2"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"LoD2 roof data is missing columns: {sorted(missing)}")
    if min(tilt_bin_deg, azimuth_bin_deg, module_capacity_kw_per_m2) <= 0:
        raise ValueError("PV profile bins and module capacity density must be positive.")
    if not 0 < flat_roof_utilization <= 1 or not 0 < slanted_roof_utilization <= 1:
        raise ValueError("PV roof utilization factors must be in (0, 1].")

    result = raw.copy()
    result["building_objectid"] = result["building_objectid"].astype(str)
    for column in ("dachneigung", "dachorientierung", "roof_area_m2"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["surface_tilt_deg"] = 90.0 - result["dachneigung"]
    flat = result["surface_tilt_deg"].abs().le(FLAT_TILT_TOLERANCE_DEG)
    orientation_valid = result["dachorientierung"].between(0.0, 360.0, inclusive="both")
    tilt_valid = result["surface_tilt_deg"].between(0.0, 90.0, inclusive="both")
    area_valid = result["roof_area_m2"].gt(0.0)
    result["surface_azimuth_deg"] = result["dachorientierung"].mod(360.0)
    result.loc[flat, "surface_azimuth_deg"] = 0.0
    result["profile_usable"] = area_valid & tilt_valid & (flat | orientation_valid)
    result["quality_flag"] = np.select(
        [~area_valid, ~tilt_valid, ~flat & ~orientation_valid],
        ["invalid_area", "invalid_tilt", "undefined_nonflat_orientation"],
        default="lod2",
    )
    utilization = np.where(flat, flat_roof_utilization, slanted_roof_utilization)
    result["available_pv_kw"] = np.where(
        result["profile_usable"],
        result["roof_area_m2"] * utilization * module_capacity_kw_per_m2,
        0.0,
    )
    result["profile_tilt_deg"] = (result["surface_tilt_deg"] / tilt_bin_deg).round() * tilt_bin_deg
    result["profile_azimuth_deg"] = ((result["surface_azimuth_deg"] / azimuth_bin_deg).round() * azimuth_bin_deg).mod(360.0)
    result.loc[~result["profile_usable"], ["profile_tilt_deg", "profile_azimuth_deg"]] = np.nan
    return result[_catalog_columns()].sort_values(["building_objectid", "roof_surface_id"]).reset_index(drop=True)


def add_missing_building_fallbacks(
    catalog: pd.DataFrame,
    building_objectids: Iterable[object],
    *,
    tilt_bin_deg: float = DEFAULT_TILT_BIN_DEG,
    azimuth_bin_deg: float = DEFAULT_AZIMUTH_BIN_DEG,
    fallback_capacity_kw: float = DEFAULT_FALLBACK_CAPACITY_KW,
) -> pd.DataFrame:
    """Add one explicit fallback only where a building has no usable LoD2 section."""
    requested = {str(value) for value in building_objectids if pd.notna(value)}
    usable = set(catalog.loc[catalog["profile_usable"], "building_objectid"].astype(str))
    missing = sorted(requested - usable)
    if not missing:
        return catalog.reset_index(drop=True)
    fallback = pd.DataFrame({
        "building_objectid": missing,
        "roof_surface_id": ["fallback"] * len(missing),
        "dachneigung": [np.nan] * len(missing),
        "dachorientierung": [np.nan] * len(missing),
        "roof_area_m2": [np.nan] * len(missing),
        "surface_tilt_deg": [FALLBACK_TILT_DEG] * len(missing),
        "surface_azimuth_deg": [FALLBACK_AZIMUTH_DEG] * len(missing),
        "profile_tilt_deg": [round(FALLBACK_TILT_DEG / tilt_bin_deg) * tilt_bin_deg] * len(missing),
        "profile_azimuth_deg": [(round(FALLBACK_AZIMUTH_DEG / azimuth_bin_deg) * azimuth_bin_deg) % 360.0] * len(missing),
        "available_pv_kw": [float(fallback_capacity_kw)] * len(missing),
        "profile_usable": [True] * len(missing),
        "quality_flag": ["fallback_14_5_kw"] * len(missing),
    })
    return pd.concat([catalog, fallback[_catalog_columns()]], ignore_index=True).sort_values(
        ["building_objectid", "roof_surface_id"]
    ).reset_index(drop=True)


def building_roof_capacity(catalog: pd.DataFrame) -> pd.Series:
    return catalog[catalog["profile_usable"]].groupby("building_objectid")["available_pv_kw"].sum()


def assert_fallback_share(catalog: pd.DataFrame, building_ids: Iterable[object], maximum_share: float) -> float:
    ids = {str(value) for value in building_ids if pd.notna(value)}
    fallback_ids = set(catalog.loc[catalog["quality_flag"].eq("fallback_14_5_kw"), "building_objectid"].astype(str)) & ids
    share = len(fallback_ids) / len(ids) if ids else 0.0
    if share > maximum_share + 1e-12:
        raise ValueError(f"LoD2 PV fallback share {share:.3%} exceeds configured maximum {maximum_share:.3%}.")
    return share


def _bin_fallback_options(options: dict) -> dict:
    return {key: options[key] for key in ("tilt_bin_deg", "azimuth_bin_deg", "fallback_capacity_kw") if key in options}


def _catalog_columns() -> list[str]:
    return [
        "building_objectid", "roof_surface_id", "dachneigung", "dachorientierung",
        "roof_area_m2", "surface_tilt_deg", "surface_azimuth_deg", "profile_tilt_deg",
        "profile_azimuth_deg", "available_pv_kw", "profile_usable", "quality_flag",
    ]


def empty_roof_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=_catalog_columns())
