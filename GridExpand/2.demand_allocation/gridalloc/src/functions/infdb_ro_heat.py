"""INFDB ``ro_heat`` space-heat loading and preliminary DHW generation."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import OpenDHW
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

from config import config


EXPECTED_HOURS = 8760
PRELIMINARY_SOURCE_FALLBACK = "duplicate_last_day"
RO_HEAT_SERIES_NAME = "ro_heat_heating_load"
RO_HEAT_UNIT = "W"


def _database_engine():
    """Create the configured INFDB engine without importing the heat pipeline."""
    gridexpand_dir = Path(__file__).resolve().parents[4]
    load_dotenv(gridexpand_dir / ".env", override=True)
    return create_engine(
        "postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}"
    )


def _building_id_column(buildings: pd.DataFrame) -> str:
    for column in ("building_objectid", "objectid"):
        if column in buildings:
            return column
    raise ValueError("INFDB ro_heat requires a building_objectid/objectid column.")


def _last_day_fallback(values: pd.Series, timestamps: pd.Series) -> tuple[pd.Series, bool]:
    """Return 8,760 values, using the explicitly temporary 8,736-hour fallback."""
    values = values.reset_index(drop=True)
    timestamps = timestamps.reset_index(drop=True)
    if len(values) == EXPECTED_HOURS:
        return values, False
    if len(values) != EXPECTED_HOURS - 24:
        raise ValueError(
            f"ro_heat series has {len(values)} hours; expected {EXPECTED_HOURS} "
            f"or exactly {EXPECTED_HOURS - 24} for the preliminary fallback."
        )

    start = timestamps.iloc[0]
    end = timestamps.iloc[-1]
    expected_start = pd.Timestamp(year=start.year, month=1, day=1, tz="UTC")
    expected_end = pd.Timestamp(
        year=start.year, month=12, day=30, hour=23, tz="UTC"
    )
    if start != expected_start or end != expected_end:
        raise ValueError(
            "The preliminary ro_heat fallback requires a contiguous Jan 1--Dec 30 "
            "calendar with the final day available for duplication."
        )
    return pd.concat([values, values.iloc[-24:]], ignore_index=True), True


def _validate_and_normalize_series(group: pd.DataFrame) -> tuple[pd.Series, bool, int]:
    timestamps = pd.to_datetime(group["time"], utc=True, errors="coerce")
    values = pd.to_numeric(group["value"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("ro_heat contains invalid timestamps.")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("ro_heat contains missing or non-finite values.")
    if (values < 0).any():
        raise ValueError("ro_heat contains negative space-heat demand.")
    if timestamps.duplicated().any():
        raise ValueError("ro_heat contains duplicate timestamps.")

    order = timestamps.argsort(kind="stable")
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    values = values.iloc[order].reset_index(drop=True)
    if len(timestamps) < 2:
        raise ValueError("ro_heat series is too short to determine its resolution.")
    interval = timestamps.diff().dropna()
    if not (interval == pd.Timedelta(hours=1)).all():
        raise ValueError("ro_heat must provide a complete hourly series without gaps.")
    normalized, used_fallback = _last_day_fallback(values, timestamps)
    return normalized / 1000.0, used_fallback, int(timestamps.iloc[0].year)


def _total_floor_area(row: pd.Series | dict[str, object]) -> float:
    area = pd.to_numeric(pd.Series([row.get("floor_area")]), errors="coerce").iloc[0]
    floors = pd.to_numeric(
        pd.Series([row.get("floor_number", 1.0)]), errors="coerce"
    ).iloc[0]
    if pd.isna(area) or float(area) <= 0.0:
        return math.nan
    if pd.isna(floors) or float(floors) <= 0.0:
        floors = 1.0
    return float(area) * float(floors)


def _nearest_heat_source(
    target: pd.Series,
    candidates: pd.DataFrame,
) -> tuple[pd.Series, str, float]:
    """Choose the closest available regional building, broadening if needed."""
    building_type = str(target.get("building_type", "")).strip().upper()
    same_type = candidates["building_type"].astype(str).str.upper().eq(building_type)
    typed = candidates.loc[same_type]
    if not typed.empty:
        candidates = typed
        scope = "same_building_type"
    else:
        scope = "regional"

    target_area = _total_floor_area(target)
    positive_area = candidates["total_floor_area"].gt(0.0)
    if np.isfinite(target_area) and target_area > 0.0 and positive_area.any():
        candidates = candidates.loc[positive_area]
        distance = (
            np.log(candidates["total_floor_area"].astype(float))
            - np.log(target_area)
        ).abs()
        source = candidates.loc[distance.idxmin()]
        scale = target_area / float(source["total_floor_area"])
    else:
        source = candidates.sort_values("objectid").iloc[0]
        scale = 1.0
        scope = f"{scope}_missing_target_area"
    return source, scope, scale


def load_space_heat(
    buildings: pd.DataFrame,
    *,
    engine=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one hourly kWh space-heat series per physical building.

    The temporary source fallback duplicates the last available 24-hour day
    only for the current 8,736-hour ro_heat export. Missing individual
    buildings borrow the nearest-area profile of the same building type; if
    necessary, matching broadens to the complete regional profile pool.
    """
    if buildings.empty:
        return pd.DataFrame(), {
            "space_heat_source": "infdb_ro_heat",
            "space_heat_source_fallback": None,
            "space_heat_source_buildings": 0,
        }
    id_column = _building_id_column(buildings)
    source_ids = buildings[id_column].astype(str)
    if source_ids.duplicated().any():
        raise ValueError("INFDB ro_heat requires unique physical building IDs.")

    metadata_query = text(
        """
        SELECT
            m.id,
            m.objectid,
            m.name,
            m.unit,
            m.type,
            m.source,
            m.changelog,
            b.floor_area,
            b.floor_number,
            b.building_type
        FROM ro_heat.entise_ts_metadata m
        LEFT JOIN ro_heat.buildings_refurbished_status b
          ON b.building_objectid = m.objectid
        WHERE m.name = :series_name
        ORDER BY m.objectid, m.id DESC
        """
    )
    data_query = text(
        """
        SELECT m.objectid, d.time, d.value
        FROM ro_heat.entise_ts_metadata m
        JOIN ro_heat.entise_ts_data d ON d.ts_metadata_id = m.id
        WHERE m.id IN :metadata_ids
        ORDER BY m.objectid, d.time
        """
    ).bindparams(bindparam("metadata_ids", expanding=True))
    db_engine = engine or _database_engine()
    with db_engine.connect() as connection:
        candidates = pd.read_sql_query(
            metadata_query,
            connection,
            params={"series_name": RO_HEAT_SERIES_NAME},
        )
        if candidates.empty:
            raise ValueError("ro_heat contains no heating-load series.")
        candidates["objectid"] = candidates["objectid"].astype(str)
        candidates = candidates.drop_duplicates("id", keep="first")
        if set(candidates["unit"].dropna().astype(str)) != {RO_HEAT_UNIT}:
            raise ValueError("ro_heat heating-load units are ambiguous or unsupported.")

        exact = candidates[candidates["objectid"].isin(source_ids)].copy()
        if exact["changelog"].nunique(dropna=False) > 1:
            raise ValueError(
                "Selected exact ro_heat profiles span multiple changelog/model runs."
            )
        if exact.empty:
            changelog = (
                candidates.groupby("changelog", dropna=False)
                .size()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            changelog = exact["changelog"].iloc[0]
        if pd.isna(changelog):
            candidates = candidates[candidates["changelog"].isna()].copy()
        else:
            candidates = candidates[candidates["changelog"].eq(changelog)].copy()
        candidates = candidates.sort_values(
            ["objectid", "id"], ascending=[True, False]
        ).drop_duplicates("objectid", keep="first")
        candidates["total_floor_area"] = (
            pd.to_numeric(candidates["floor_area"], errors="coerce")
            * pd.to_numeric(candidates["floor_number"], errors="coerce").fillna(1.0)
        )

        source_by_target: dict[str, str] = {}
        scale_by_target: dict[str, float] = {}
        scope_by_target: dict[str, str] = {}
        candidate_by_id = candidates.set_index("objectid", drop=False)
        buildings_by_id = buildings.assign(
            _building_objectid=source_ids.to_numpy()
        ).set_index("_building_objectid", drop=False)
        for building_id, building in buildings_by_id.iterrows():
            if building_id in candidate_by_id.index:
                source_by_target[str(building_id)] = str(building_id)
                scale_by_target[str(building_id)] = 1.0
                scope_by_target[str(building_id)] = "exact"
                continue
            source, scope, scale = _nearest_heat_source(building, candidates)
            source_by_target[str(building_id)] = str(source["objectid"])
            scale_by_target[str(building_id)] = float(scale)
            scope_by_target[str(building_id)] = scope

        metadata_ids = (
            candidate_by_id.loc[
                sorted(set(source_by_target.values())),
                "id",
            ]
            .astype(int)
            .tolist()
        )
        data = pd.read_sql_query(
            data_query,
            connection,
            params={"metadata_ids": metadata_ids},
        )

    normalized: dict[str, pd.Series] = {}
    used_day_fallback: dict[str, bool] = {}
    years: set[int] = set()
    for building_id, group in data.groupby("objectid", sort=True):
        series, used_fallback, source_year = _validate_and_normalize_series(group)
        normalized[str(building_id)] = series
        used_day_fallback[str(building_id)] = used_fallback
        years.add(source_year)
    missing_sources = sorted(set(source_by_target.values()) - set(normalized))
    if missing_sources:
        raise ValueError(
            "ro_heat has no complete data rows for selected fallback source(s), "
            f"examples={missing_sources[:5]}"
        )
    if len(years) != 1:
        raise ValueError(f"ro_heat source years are ambiguous: {sorted(years)}")

    by_bus: dict[int, list[pd.Series]] = {}
    for _, building in buildings.iterrows():
        target_id = str(building[id_column])
        source_id = source_by_target[target_id]
        scaled = normalized[source_id] * scale_by_target[target_id]
        by_bus.setdefault(int(building["bus"]), []).append(scaled)
    result = pd.DataFrame(
        {bus: sum(series_list) for bus, series_list in by_bus.items()}
    )
    result.columns = pd.MultiIndex.from_tuples(
        [(bus, "space_heat") for bus in result.columns]
    )

    similar_targets = [
        target for target, scope in scope_by_target.items() if scope != "exact"
    ]
    duplicated_targets = [
        target
        for target, source in source_by_target.items()
        if used_day_fallback[source]
    ]
    fallback_methods = []
    if duplicated_targets:
        fallback_methods.append(PRELIMINARY_SOURCE_FALLBACK)
    if similar_targets:
        fallback_methods.append("nearest_building_area_scaled")
    audit = {
        "space_heat_source": "infdb_ro_heat",
        "space_heat_source_schema": "ro_heat",
        "space_heat_source_series": RO_HEAT_SERIES_NAME,
        "space_heat_source_unit": RO_HEAT_UNIT,
        "space_heat_source_value_kind": "hourly_power_converted_to_kwh",
        "space_heat_source_changelog": str(changelog),
        "space_heat_source_year": min(years),
        "space_heat_source_hours": EXPECTED_HOURS,
        "space_heat_source_buildings": int(len(source_ids)),
        "space_heat_source_fallback": (
            "+".join(fallback_methods) if fallback_methods else None
        ),
        "space_heat_source_fallback_buildings": len(
            set(duplicated_targets) | set(similar_targets)
        ),
        "space_heat_last_day_fallback_buildings": len(duplicated_targets),
        "space_heat_similar_building_fallback_buildings": len(similar_targets),
        "space_heat_similar_building_fallback_scopes": {
            scope: list(scope_by_target.values()).count(scope)
            for scope in sorted(set(scope_by_target.values()) - {"exact"})
        },
    }
    return result, audit


def _temperature_difference(hours: int) -> np.ndarray:
    if hours != EXPECTED_HOURS:
        raise ValueError("OpenDHW generation requires 8,760 hourly values.")
    days = np.arange(365)
    mixed = 45 + 3 * np.cos(np.pi * (2 / 365 * days - 2 * 355 / 365))
    cold = 10 + 7 * np.cos(np.pi * (2 / 365 * days - 2 * 225 / 365))
    return np.repeat(mixed - cold, 24)


def generate_opendhw(buildings: pd.DataFrame, base_seed: int = 0) -> pd.DataFrame:
    """Generate the existing building-level OpenDHW demand without TEASER."""
    result: dict[int, np.ndarray] = {}
    temperature_difference = _temperature_difference(EXPECTED_HOURS)
    for _, building in buildings.iterrows():
        building_type = str(building["building_type"])
        if building_type not in {"SFH", "MFH", "TH", "AB"}:
            raise ValueError(f"Unsupported residential building type for OpenDHW: {building_type!r}")
        occupants = building.get("occ_list")
        if not isinstance(occupants, (list, tuple, np.ndarray)):
            raise ValueError("OpenDHW requires an occupant list for every building.")
        from common.reproducibility import (
            legacy_random_state,
            physical_building_id,
            stable_seed,
        )

        building_id = physical_building_id(building)
        total_w = np.zeros(EXPECTED_HOURS)
        for flat_index, value in enumerate(occupants):
            number_occupants = int(round(float(value)))
            if number_occupants < 0:
                raise ValueError("OpenDHW occupant counts cannot be negative.")
            dhw_seed = stable_seed(base_seed, building_id, "dhw", flat_index)
            with legacy_random_state(dhw_seed):
                profile = OpenDHW.generate_dhw_profile(
                    s_step=60,
                    categories=1,
                    occupancy=number_occupants,
                    building_type=building_type,
                    weekend_weekday_factor=1.2,
                    holidays=config.HOLIDAYS,
                    mean_drawoff_vol_per_day=40,
                )
            water = OpenDHW.resample_water_series(profile, 3600)
            heat = OpenDHW.compute_heat(
                timeseries_df=water,
                temp_dT=temperature_difference,
            )
            values = pd.to_numeric(heat["Heat_W"], errors="coerce").to_numpy(dtype=float)
            if len(values) != EXPECTED_HOURS or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("OpenDHW produced an invalid hourly water-heat series.")
            total_w += values
        result[int(building["bus"])] = result.get(int(building["bus"]), np.zeros(EXPECTED_HOURS)) + total_w / 1000.0
    output = pd.DataFrame(result)
    output.columns = pd.MultiIndex.from_tuples(
        [(bus, "water_heat") for bus in output.columns]
    )
    return output
