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


def load_space_heat(
    buildings: pd.DataFrame,
    *,
    engine=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one hourly kWh space-heat series per physical building.

    The temporary source fallback duplicates the last available 24-hour day
    only for the current 8,736-hour ro_heat export. It is recorded in the
    returned audit metadata and must not be treated as a publication artifact.
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
        SELECT id, objectid, name, unit, type, source, changelog
        FROM ro_heat.entise_ts_metadata
        WHERE name = :series_name
          AND objectid IN :building_ids
        """
    ).bindparams(bindparam("building_ids", expanding=True))
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
        metadata = pd.read_sql_query(
            metadata_query,
            connection,
            params={"series_name": RO_HEAT_SERIES_NAME, "building_ids": source_ids.tolist()},
        )
        if metadata.empty:
            raise ValueError("No ro_heat heating-load series match the selected buildings.")
        metadata["objectid"] = metadata["objectid"].astype(str)
        if metadata["objectid"].duplicated().any():
            raise ValueError("ro_heat has duplicate heating series for a building.")
        missing = sorted(set(source_ids) - set(metadata["objectid"]))
        if missing:
            raise ValueError(
                f"ro_heat is missing {len(missing)} selected building(s), examples={missing[:5]}"
            )
        if set(metadata["unit"].dropna().astype(str)) != {RO_HEAT_UNIT}:
            raise ValueError("ro_heat heating-load units are ambiguous or unsupported.")
        if metadata["changelog"].nunique(dropna=False) != 1:
            raise ValueError("ro_heat has multiple changelog/model runs without a selector.")
        data = pd.read_sql_query(
            data_query,
            connection,
            params={"metadata_ids": metadata["id"].astype(int).tolist()},
        )

    normalized: dict[str, pd.Series] = {}
    years: set[int] = set()
    fallback_buildings = 0
    for building_id, group in data.groupby("objectid", sort=True):
        series, used_fallback, source_year = _validate_and_normalize_series(group)
        normalized[str(building_id)] = series
        years.add(source_year)
        fallback_buildings += int(used_fallback)
    if set(source_ids) != set(normalized):
        missing_data = sorted(set(source_ids) - set(normalized))
        raise ValueError(
            f"ro_heat has no complete data rows for {len(missing_data)} building(s), "
            f"examples={missing_data[:5]}"
        )
    if len(years) != 1:
        raise ValueError(f"ro_heat source years are ambiguous: {sorted(years)}")

    by_bus: dict[int, list[pd.Series]] = {}
    for _, building in buildings.iterrows():
        bus = int(building["bus"])
        by_bus.setdefault(bus, []).append(normalized[str(building[id_column])])
    result = pd.DataFrame(
        {bus: sum(series_list) for bus, series_list in by_bus.items()}
    )
    result.columns = pd.MultiIndex.from_tuples(
        [(bus, "space_heat") for bus in result.columns]
    )
    audit = {
        "space_heat_source": "infdb_ro_heat",
        "space_heat_source_schema": "ro_heat",
        "space_heat_source_series": RO_HEAT_SERIES_NAME,
        "space_heat_source_unit": RO_HEAT_UNIT,
        "space_heat_source_value_kind": "hourly_power_converted_to_kwh",
        "space_heat_source_changelog": str(metadata["changelog"].iloc[0]),
        "space_heat_source_year": min(years),
        "space_heat_source_hours": EXPECTED_HOURS,
        "space_heat_source_buildings": int(len(normalized)),
        "space_heat_source_fallback": (
            PRELIMINARY_SOURCE_FALLBACK if fallback_buildings else None
        ),
        "space_heat_source_fallback_buildings": fallback_buildings,
    }
    return result, audit


def _temperature_difference(hours: int) -> np.ndarray:
    if hours != EXPECTED_HOURS:
        raise ValueError("OpenDHW generation requires 8,760 hourly values.")
    days = np.arange(365)
    mixed = 45 + 3 * np.cos(np.pi * (2 / 365 * days - 2 * 355 / 365))
    cold = 10 + 7 * np.cos(np.pi * (2 / 365 * days - 2 * 225 / 365))
    return np.repeat(mixed - cold, 24)


def generate_opendhw(buildings: pd.DataFrame) -> pd.DataFrame:
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
        total_w = np.zeros(EXPECTED_HOURS)
        for value in occupants:
            number_occupants = int(round(float(value)))
            if number_occupants < 0:
                raise ValueError("OpenDHW occupant counts cannot be negative.")
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
