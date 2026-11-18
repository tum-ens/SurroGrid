"""Attributes of given buildings from InfDB, derived with pylovo's rules.

The buildings of a given grid take their attributes directly from InfDB ``basedata.buildings``; what GridExpand's
demand model needs on top (household fallback by building type, non-residential use, component peaks, MV-direct
share above 100 kW, missing occupants) is derived with pylovo's rules (pylovo 8afedcc), so a building gets the same
attributes as in a pylovo grid without any pylovo result being read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# pylovo 8afedcc values (config_generation.yaml, preprocessing_mixin._household_fallback_parameters).
PYLOVO_PEAK_LOAD_HOUSEHOLD_KW = 16.825
PYLOVO_PEAK_W_PER_M2 = {"Commercial": 79.0, "Public": 29.0, "Unknown": 29.0}
PYLOVO_MV_DIRECT_THRESHOLD_KW = 100.0
PYLOVO_AREA_PER_HOUSEHOLD_M2 = {"MFH": 181.0, "untyped": 181.0, "AB": 146.0}
PYLOVO_MINIMUM_HOUSEHOLDS = {"MFH": 2, "untyped": 1, "AB": 5}

INFDB_SELECT = """
SELECT objectid::text AS objectid, id, feature_id, height, floor_area, floor_number,
       residential_floor_area, nonresidential_floor_area, mix_rule, building_use, building_use_id, building_type,
       occupants, households, CAST(construction_year AS VARCHAR) AS construction_year, postcode,
       address_street_id, street, house_number, gemeindeschluessel, assigned_way_id,
       CASE
         WHEN COALESCE(nonresidential_floor_area, 0) <= 0 THEN NULL
         WHEN building_use = 'Residential' THEN 'Commercial'
         WHEN building_use = 'Mixed' THEN
           CASE {schema}.classify_building_use(building_use_id)
             WHEN 'Residential' THEN 'Commercial'
             ELSE {schema}.classify_building_use(building_use_id)
           END
         WHEN building_use IN ('Commercial', 'Public', 'Unknown') THEN building_use
       END AS nonresidential_use,
       COALESCE(building_type, building_use) AS type,
       ST_Y(ST_Transform(centroid, 4326)) AS lat,
       ST_X(ST_Transform(centroid, 4326)) AS lon
FROM {schema}.buildings
"""
INFDB_BUILDINGS_SQL = INFDB_SELECT + "WHERE objectid = ANY(:ids)\n"

# Columns of the dataset contract that only pylovo grids fill.
PYLOVO_ONLY_COLUMNS = ("pylovo_grid_result_id", "pylovo_version_id", "vertice_id", "connection_point",
                       "consumer_vertex")


def read_infdb_buildings(engine, objectids: list[str], schema: str = "basedata") -> pd.DataFrame:
    """InfDB building rows with pylovo's ``nonresidential_use`` and ``type`` (read-only)."""
    from sqlalchemy import text

    if not schema.isidentifier():
        raise ValueError(f"Invalid schema name {schema!r}.")
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = pd.read_sql_query(text(INFDB_BUILDINGS_SQL.format(schema=schema)), conn,
                                 params={"ids": list(objectids)})
        conn.rollback()
    rows["objectid"] = rows["objectid"].astype(str)
    missing = sorted(set(map(str, objectids)) - set(rows["objectid"]))
    if missing:
        raise ValueError(f"{len(missing)} buildings are missing from {schema}.buildings: {missing[:10]}")
    return rows


def pylovo_equivalent_buildings(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill what pylovo derives before GridExpand reads a building (8afedcc rules).

    Missing households by type, the non-residential use of plain Residential/typed
    buildings, component peaks, MV-direct above 100 kW and missing occupants
    (households x 2.03, as ``gridexpand.db.grids.read_buildings``).
    """
    from gridexpand.sampling.db_read import impute_missing_occupants

    b = rows.copy()
    b["objectid"] = b["objectid"].astype(str)
    kind = b["type"].astype(str)
    res = pd.to_numeric(b["residential_floor_area"], errors="coerce")
    nonres = pd.to_numeric(b["nonresidential_floor_area"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(b["floor_area"], errors="coerce") * pd.to_numeric(b["floor_number"], errors="coerce").fillna(1)
    fallback = pd.Series(np.nan, index=b.index)
    fallback[kind.isin(["SFH", "TH"])] = 1
    for name, label in (("MFH", "MFH"), ("AB", "AB")):
        mask = kind.eq(name)
        fallback[mask] = np.maximum(PYLOVO_MINIMUM_HOUSEHOLDS[label], np.rint(res.fillna(gross)[mask] / PYLOVO_AREA_PER_HOUSEHOLD_M2[label]))
    untyped = kind.isin(["Residential", "Mixed"]) & res.fillna(0).gt(0)
    fallback[untyped] = np.maximum(PYLOVO_MINIMUM_HOUSEHOLDS["untyped"], np.rint(res[untyped] / PYLOVO_AREA_PER_HOUSEHOLD_M2["untyped"]))
    fallback[kind.isin(["Commercial", "Public", "Unknown"])] = 1
    b["households"] = pd.to_numeric(b["households"], errors="coerce").where(b["households"].notna(), fallback)
    use = b["nonresidential_use"].where(b["nonresidential_use"].isin(list(PYLOVO_PEAK_W_PER_M2)))
    use = use.where(use.notna(), kind.where(kind.isin(list(PYLOVO_PEAK_W_PER_M2))))
    use = use.where(use.notna() | ~b["building_use"].eq("Residential"), "Commercial")
    b["nonresidential_use"] = use.where(nonres.gt(0))
    b["residential_peak_load_in_kw"] = np.where(res.fillna(0).gt(0), b["households"] * PYLOVO_PEAK_LOAD_HOUSEHOLD_KW, 0.0)
    b["nonresidential_peak_load_in_kw"] = np.where(
        nonres.gt(0), nonres * b["nonresidential_use"].map(PYLOVO_PEAK_W_PER_M2).fillna(0.0) / 1000.0, 0.0
    )
    direct = pd.Series(b["nonresidential_peak_load_in_kw"] > PYLOVO_MV_DIRECT_THRESHOLD_KW, index=b.index)
    b["nonresidential_mv_direct"] = [bool(value) if area > 0 else None for value, area in zip(direct, nonres)]
    b["peak_load_in_kw"] = b["residential_peak_load_in_kw"] + np.where(direct, 0.0, b["nonresidential_peak_load_in_kw"])
    for column in ("mix_score", "mix_confidence"):
        b[column] = np.nan
    return impute_missing_occupants(b)


def mv_direct_buildings(derived: pd.DataFrame) -> pd.Series:
    """Buildings whose whole load goes to MV: an MV-direct non-residential share and no residential floor area.

    pylovo takes such a building out of its LV grids (``building_exclusion`` reason ``mv_direct``); a building with
    a residential share keeps that share on LV.
    """
    residential = pd.to_numeric(derived["residential_floor_area"], errors="coerce").fillna(0.0).gt(0)
    return derived["nonresidential_mv_direct"].eq(True) & ~residential


def building_attributes(engine, objectids, *, schema: str = "basedata") -> pd.DataFrame:
    """One row per building: InfDB attributes with pylovo's derivations (read-only)."""
    ids = sorted({str(value) for value in objectids})
    frame = pylovo_equivalent_buildings(read_infdb_buildings(engine, ids, schema))
    for column in PYLOVO_ONLY_COLUMNS:
        frame[column] = pd.NA
    return frame.sort_values("objectid", kind="stable").reset_index(drop=True)
