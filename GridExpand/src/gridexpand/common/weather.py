"""PVGIS typical-meteorological-year weather shared by Step 1 and Step 2.

Every pipeline part that needs site weather (the Step 1 grid export, Step 2
demand allocation, the mobility profile pool and the paired PV/weather
libraries) downloads it through :func:`get_pvgis_tmy_sarah3_dataframe`. An
aligned paired run can take one real calendar year instead
(:func:`get_pvgis_year_sarah3_dataframe`).
"""

from __future__ import annotations

import time
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import requests

from gridexpand.common.timeframe import REFERENCE_YEAR

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/tmy"
PVGIS_SERIES_URL = "https://re.jrc.ec.europa.eu/api/seriescalc"
# The pipeline works in fixed UTC+1 (CET without daylight saving time).
UTC_OFFSET_HOURS = 1
REQUEST_TIMEOUT_S = 60
MAX_ATTEMPTS = 5

PVGIS_COLUMNS = {
    "G(h)": "ghi",  # W/m2
    "Gb(n)": "dni",  # W/m2
    "Gd(h)": "dhi",  # W/m2
    "T2m": "temp_air",  # degC
    "RH": "relative_humidity",  # %
    "SP": "pressure",  # Pa
    "WS10m": "wind_speed",  # m/s
    "WD10m": "wind_direction",  # degrees
}


def _request_pvgis(url: str, params: dict, output_key: str, *, timeout: float, max_attempts: int) -> dict:
    """Return the PVGIS JSON with ``outputs[output_key]``; retry transient failures, raise on the rest."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                data = response.json()
                if "outputs" not in data or output_key not in data["outputs"]:
                    raise RuntimeError(
                        f"Unexpected PVGIS response structure: {str(data)[:500]}"
                    )
                return data
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(
                    f"PVGIS rejected the request (HTTP {response.status_code}, "
                    f"params {params}): {response.text[:500]}"
                )
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        if attempt < max_attempts:
            sleep_seconds = min(60, 2**attempt)
            print(
                f"PVGIS request failed (attempt {attempt}/{max_attempts}); "
                f"retrying in {sleep_seconds}s. Details: {last_error}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"PVGIS request failed after {max_attempts} attempts. "
        f"Params: {params}. Last error: {last_error}"
    )


def _to_reference_year(df: pd.DataFrame, offset: float, reference_year: int) -> pd.DataFrame:
    """Move hourly ``time(UTC)`` rows to ``reference_year`` in fixed UTC+1.

    ``time(inst)`` adds the satellite's instantaneous measurement offset (hours).
    When that offset is non-negative, the last hour lies in the next year and is
    moved to the start of the series.
    """
    year = int(reference_year)
    df["time(UTC)"] = df["time(UTC)"].apply(lambda x: x.replace(year=year))
    df["time(UTC)"] = df["time(UTC)"].dt.tz_localize(timezone.utc)
    fixed_utc_plus_1 = timezone(timedelta(hours=UTC_OFFSET_HOURS))
    df["time(UTC+1)"] = df["time(UTC)"].dt.tz_convert(fixed_utc_plus_1)
    # Actual (instantaneous) satellite measurement time in UTC+1.
    df["time(inst)"] = df["time(UTC+1)"] + pd.DateOffset(hours=offset)
    if offset >= 0:
        # The last instantaneous measurement falls into the next year: move it
        # to the beginning of the reference year.
        last_row = df.iloc[-1].copy()
        df = df.iloc[:-1]
        last_row["time(UTC+1)"] = last_row["time(UTC+1)"] - pd.DateOffset(years=1)
        last_row["time(inst)"] = last_row["time(inst)"] - pd.DateOffset(years=1)
        df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
        df = df.sort_values("time(inst)").reset_index(drop=True)
    return df


def get_pvgis_tmy_sarah3_dataframe(
    latitude: float,
    longitude: float,
    *,
    reference_year: int = REFERENCE_YEAR,
    timeout: float = REQUEST_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[pd.DataFrame, float]:
    """Download the PVGIS SARAH3 typical meteorological year for one site.

    Timestamps are moved to ``reference_year`` and converted to fixed UTC+1
    (``time(UTC+1)``); ``time(inst)`` adds the satellite's instantaneous
    measurement offset. When that offset is non-negative, the last hour lies in
    the next year and is moved to the start of the series.

    Args:
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        reference_year: Calendar year assigned to every TMY timestamp.
        timeout: Seconds to wait for each HTTP request.
        max_attempts: Attempts for timeouts, connection errors, HTTP 429 and 5xx.

    Returns:
        The hourly weather (columns renamed, e.g. ``ghi``, ``temp_air``) and the
        site elevation in metres reported by PVGIS.

    Raises:
        RuntimeError: If PVGIS rejects the request, returns an unexpected
            payload, or keeps failing.
    """
    params = {
        "lat": latitude,
        "lon": longitude,
        "raddatabase": "PVGIS-SARAH3",
        "outputformat": "json",
        "usehorizon": 1,
        "database": "SARAH3",
    }
    print(f"Requesting TMY data from PVGIS (SARAH3) for coordinates ({latitude}, {longitude})...")
    tmy_data = _request_pvgis(PVGIS_URL, params, "tmy_hourly", timeout=timeout, max_attempts=max_attempts)
    offset = tmy_data["inputs"]["location"]["irradiance_time_offset"]
    altitude = tmy_data["inputs"]["location"]["elevation"]

    df = pd.DataFrame(tmy_data["outputs"]["tmy_hourly"])
    if "time(UTC)" not in df.columns:
        raise RuntimeError("PVGIS TMY response has no 'time(UTC)' column.")
    df["time(UTC)"] = pd.to_datetime(df["time(UTC)"], format="%Y%m%d:%H%M")
    df = _to_reference_year(df, offset, reference_year)

    existing_columns = set(df.columns).intersection(PVGIS_COLUMNS)
    df = df.rename(columns={column: PVGIS_COLUMNS[column] for column in existing_columns})
    return df, altitude


def get_pvgis_year_sarah3_dataframe(
    latitude: float,
    longitude: float,
    year: int,
    *,
    reference_year: int = REFERENCE_YEAR,
    timeout: float = REQUEST_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[pd.DataFrame, float]:
    """Download one real calendar year of PVGIS SARAH3 hourly weather for one site.

    The TMY's sources (SARAH3 irradiance, ERA5 meteorology) for a single year, from
    PVGIS ``seriescalc`` on the horizontal plane: ``ghi = Gb + Gd``, ``dhi = Gd``,
    ``dni = Gb / sin(sun height)``. The series has no humidity, pressure or wind
    direction. 29 February is dropped; the timestamps are then moved to
    ``reference_year`` in fixed UTC+1 as for the TMY. PVGIS stamps each hour with
    the instantaneous measurement time, whose minutes give ``time(inst)``.

    Returns:
        The hourly weather (``ghi``, ``dni``, ``dhi``, ``temp_air``,
        ``wind_speed`` and the time columns) and the site elevation in metres.
    """
    params = {
        "lat": latitude,
        "lon": longitude,
        "raddatabase": "PVGIS-SARAH3",
        "startyear": int(year),
        "endyear": int(year),
        "outputformat": "json",
        "usehorizon": 1,
        "angle": 0,
        "aspect": 0,
        "components": 1,
    }
    print(f"Requesting {year} hourly data from PVGIS (SARAH3) for coordinates ({latitude}, {longitude})...")
    data = _request_pvgis(PVGIS_SERIES_URL, params, "hourly", timeout=timeout, max_attempts=max_attempts)
    altitude = data["inputs"]["location"]["elevation"]

    raw = pd.DataFrame(data["outputs"]["hourly"])
    stamps = pd.to_datetime(raw["time"], format="%Y%m%d:%H%M")
    keep = ~((stamps.dt.month == 2) & (stamps.dt.day == 29))
    raw, stamps = raw[keep].reset_index(drop=True), stamps[keep].reset_index(drop=True)
    sin_height = np.sin(np.radians(raw["H_sun"].to_numpy(dtype=float)))
    beam = raw["Gb(i)"].to_numpy(dtype=float)
    df = pd.DataFrame({
        "time(UTC)": stamps.dt.floor("h"),
        "ghi": beam + raw["Gd(i)"].to_numpy(dtype=float),
        "dni": np.divide(beam, sin_height, out=np.zeros_like(beam), where=(beam > 0) & (sin_height > 0)),
        "dhi": raw["Gd(i)"].to_numpy(dtype=float),
        "temp_air": raw["T2m"].to_numpy(dtype=float),
        "wind_speed": raw["WS10m"].to_numpy(dtype=float),
    })
    offset = float(stamps.dt.minute.iloc[0]) / 60.0
    return _to_reference_year(df, offset, reference_year), altitude


def get_dew_point(temp_celsius, relative_humidity):
    """Return the dew point in degC (Magnus-Tetens) from temperature and RH in %."""
    a = 17.27
    b = 237.7  # degC
    gamma = (a * temp_celsius) / (b + temp_celsius) + np.log(relative_humidity / 100.0)
    return (b * gamma) / (a - gamma)


def read_weather_hdf(path) -> tuple[pd.DataFrame, float | None]:
    """Read a provider weather file (``raw_data/weather``, ``raw_data/region``).

    Returns the weather frame, with the dew point for the mobility model added when
    relative humidity is present, and the altitude of the weather location (or None).
    """
    weather = pd.read_hdf(path, key="raw_data/weather").copy()
    if "temp_air" not in weather.columns:
        raise ValueError(f"Weather file {path} has no temp_air column.")
    if "dew_point" not in weather.columns and "relative_humidity" in weather.columns:
        weather["dew_point"] = get_dew_point(weather["temp_air"], weather["relative_humidity"])
    try:
        region = pd.read_hdf(path, key="raw_data/region")
        altitude = float(region["altitude"].iloc[0]) if "altitude" in region else None
    except KeyError:
        altitude = None
    return weather, altitude
