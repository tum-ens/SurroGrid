"""LoD2 roof-surface data and available rooftop-PV capacity."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text


PV_AREA_FACTOR_KW_PER_M2 = 0.202
FLAT_ROOF_UTILIZATION = 0.27
SLANTED_ROOF_UTILIZATION = 0.58
FALLBACK_PV_CAPACITY_KW = 14.5
FALLBACK_TILT_DEG = 45.0
FALLBACK_AZIMUTH_DEG = 180.0
PROFILE_TILT_BIN_DEG = 1.0
PROFILE_AZIMUTH_BIN_DEG = 5.0
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
    *,
    tilt_bin_deg: float = PROFILE_TILT_BIN_DEG,
    azimuth_bin_deg: float = PROFILE_AZIMUTH_BIN_DEG,
) -> pd.DataFrame:
    """Return normalized LoD2 roof sections, adding one fallback when necessary."""
    building_ids = sorted({str(value) for value in building_objectids if pd.notna(value)})
    if not building_ids:
        return _empty_catalog()
    with engine.connect() as connection:
        raw = pd.read_sql_query(
            ROOF_SURFACE_QUERY,
            connection,
            params={"building_objectids": building_ids},
        )
    catalog = normalize_lod2_roof_sections(
        raw,
        tilt_bin_deg=tilt_bin_deg,
        azimuth_bin_deg=azimuth_bin_deg,
    )
    return add_missing_building_fallbacks(
        catalog,
        building_ids,
        tilt_bin_deg=tilt_bin_deg,
        azimuth_bin_deg=azimuth_bin_deg,
    )


def normalize_lod2_roof_sections(
    raw: pd.DataFrame,
    *,
    tilt_bin_deg: float = PROFILE_TILT_BIN_DEG,
    azimuth_bin_deg: float = PROFILE_AZIMUTH_BIN_DEG,
) -> pd.DataFrame:
    """Convert source angles to pvlib convention and calculate section potential."""
    required = {
        "building_objectid",
        "roof_surface_id",
        "dachneigung",
        "dachorientierung",
        "roof_area_m2",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"LoD2 roof data is missing columns: {sorted(missing)}")
    if tilt_bin_deg <= 0 or azimuth_bin_deg <= 0:
        raise ValueError("PV profile angle bins must be positive.")

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
    utilization = np.where(flat, FLAT_ROOF_UTILIZATION, SLANTED_ROOF_UTILIZATION)
    result["available_pv_kw"] = np.where(
        result["profile_usable"],
        result["roof_area_m2"] * utilization * PV_AREA_FACTOR_KW_PER_M2,
        0.0,
    )
    result["profile_tilt_deg"] = (
        result["surface_tilt_deg"] / float(tilt_bin_deg)
    ).round() * float(tilt_bin_deg)
    result["profile_azimuth_deg"] = (
        (result["surface_azimuth_deg"] / float(azimuth_bin_deg)).round()
        * float(azimuth_bin_deg)
    ).mod(360.0)
    result.loc[~result["profile_usable"], ["profile_tilt_deg", "profile_azimuth_deg"]] = np.nan
    return result[_catalog_columns()].sort_values(
        ["building_objectid", "roof_surface_id"]
    ).reset_index(drop=True)


def add_missing_building_fallbacks(
    catalog: pd.DataFrame,
    building_objectids: Iterable[object],
    *,
    tilt_bin_deg: float = PROFILE_TILT_BIN_DEG,
    azimuth_bin_deg: float = PROFILE_AZIMUTH_BIN_DEG,
) -> pd.DataFrame:
    """Add a 14.5 kW, 45°/180° section when no usable LoD2 section exists."""
    requested = {str(value) for value in building_objectids if pd.notna(value)}
    usable = set(
        catalog.loc[catalog["profile_usable"], "building_objectid"].astype(str)
    )
    missing = sorted(requested - usable)
    if not missing:
        return catalog.reset_index(drop=True)
    fallback = pd.DataFrame(
        {
            "building_objectid": missing,
            "roof_surface_id": ["fallback"] * len(missing),
            "dachneigung": [np.nan] * len(missing),
            "dachorientierung": [np.nan] * len(missing),
            "roof_area_m2": [np.nan] * len(missing),
            "surface_tilt_deg": [FALLBACK_TILT_DEG] * len(missing),
            "surface_azimuth_deg": [FALLBACK_AZIMUTH_DEG] * len(missing),
            "profile_tilt_deg": [
                round(FALLBACK_TILT_DEG / tilt_bin_deg) * tilt_bin_deg
            ]
            * len(missing),
            "profile_azimuth_deg": [
                (round(FALLBACK_AZIMUTH_DEG / azimuth_bin_deg) * azimuth_bin_deg)
                % 360.0
            ]
            * len(missing),
            "available_pv_kw": [FALLBACK_PV_CAPACITY_KW] * len(missing),
            "profile_usable": [True] * len(missing),
            "quality_flag": ["fallback_14_5_kw"] * len(missing),
        }
    )
    return (
        pd.concat([catalog, fallback[_catalog_columns()]], ignore_index=True)
        .sort_values(["building_objectid", "roof_surface_id"])
        .reset_index(drop=True)
    )


def building_roof_capacity(catalog: pd.DataFrame) -> pd.Series:
    """Return available capacity once per physical building."""
    usable = catalog[catalog["profile_usable"]].copy()
    return usable.groupby("building_objectid")["available_pv_kw"].sum()


def _catalog_columns() -> list[str]:
    return [
        "building_objectid",
        "roof_surface_id",
        "dachneigung",
        "dachorientierung",
        "roof_area_m2",
        "surface_tilt_deg",
        "surface_azimuth_deg",
        "profile_tilt_deg",
        "profile_azimuth_deg",
        "available_pv_kw",
        "profile_usable",
        "quality_flag",
    ]


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=_catalog_columns())
