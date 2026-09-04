"""Shared timeframe selection and metadata helpers for GridExpand pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_YEAR = 2009
FULL_YEAR_HOURS = 8760
TIMESLICE_HOURS = 168
TIMEFRAME_MODES = (
    "full_year",
    "min_temperature_week",
    "max_solar_radiation_week",
    "max_base_electricity_demand_week",
)
FULL_YEAR_START = f"{REFERENCE_YEAR}-01-01T00:00:00+00:00"
FULL_YEAR_END = f"{REFERENCE_YEAR}-12-31T23:00:00+00:00"
TIMEFRAME_HDF_KEY = "metadata/timeframe"
DST_TRANSITION_HOURS = (2090, 7130)


def build_full_year_metadata() -> dict[str, Any]:
    return {
        "timeframe_mode": "full_year",
        "horizon_hours": FULL_YEAR_HOURS,
        "timeframe_start": FULL_YEAR_START,
        "timeframe_end": FULL_YEAR_END,
        "source_year_or_reference_year": REFERENCE_YEAR,
        "timeframe_kind": "full_year",
        "methodological_note": (
            "Full 8760-hour reference-year run. Cost and investment outputs keep "
            "their existing annual interpretation."
        ),
        "cost_investment_interpretation": "annual_valid",
        "annual_valid": True,
    }


def build_initial_metadata(timeframe_mode: str) -> dict[str, Any]:
    if timeframe_mode == "full_year":
        return build_full_year_metadata()
    return {
        "timeframe_mode": timeframe_mode,
        "horizon_hours": TIMESLICE_HOURS,
        "timeframe_start": None,
        "timeframe_end": None,
        "source_year_or_reference_year": REFERENCE_YEAR,
        "timeframe_kind": "timeslice",
        "methodological_note": (
            "Selected one-week operational stress-screening run. This horizon is "
            "not a full annual scenario and should not be interpreted as an "
            "annual investment optimum."
        ),
        "cost_investment_interpretation": "operational_stress_screening_only",
        "annual_valid": False,
    }


def scenario_key_for_timeframe(timeframe_mode: str, base_key: str = "baseline_static") -> str:
    if timeframe_mode == "full_year":
        return base_key
    return f"{base_key}_{timeframe_mode}"


def scenario_output_directory(base_directory: str | Path, scenario_key: str) -> Path:
    """Return a validated scenario-specific artifact directory."""
    key = str(scenario_key).strip()
    if not key or key in {".", ".."} or Path(key).name != key:
        raise ValueError(f"scenario_key must be one directory-safe path component: {scenario_key!r}")
    return Path(base_directory) / key


def output_filename_for_timeframe(filename: str, timeframe_mode: str) -> str:
    if timeframe_mode == "full_year":
        return filename
    path = Path(filename)
    suffix = f"_{timeframe_mode}"
    if path.stem.endswith(suffix):
        return path.name
    return f"{path.stem}{suffix}{path.suffix}"


def timestamp_for_hour(hour: int) -> str:
    return (
        pd.Timestamp(f"{REFERENCE_YEAR}-01-01T00:00:00+00:00")
        + pd.Timedelta(hours=int(hour))
    ).isoformat()


def build_timeslice_metadata(
    timeframe_mode: str,
    start_hour: int,
    *,
    selection_metric: str,
    selection_value: float,
) -> dict[str, Any]:
    if timeframe_mode == "full_year":
        return build_full_year_metadata()
    end_hour = int(start_hour) + TIMESLICE_HOURS - 1
    return {
        **build_initial_metadata(timeframe_mode),
        "timeframe_start": timestamp_for_hour(start_hour),
        "timeframe_end": timestamp_for_hour(end_hour),
        "selected_start_hour": int(start_hour),
        "selected_end_hour": int(end_hour),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "mobility_profile_pool_assumption": (
            "Pregenerated full-year pool profiles are assigned first and then sliced "
            "to the selected week. Initial and final EV storage states remain the "
            "standard URBS storage treatment for this operational stress run."
        ),
    }


def ensure_supported_slice(start_hour: int, horizon_hours: int = TIMESLICE_HOURS) -> None:
    start_hour = int(start_hour)
    end_exclusive = start_hour + int(horizon_hours)
    for transition_hour in DST_TRANSITION_HOURS:
        if start_hour <= transition_hour < end_exclusive:
            raise ValueError(
                "Selected timeframe crosses a daylight-saving transition hour "
                f"({transition_hour}). DST-transition weeks are not supported yet."
            )


def _rolling_window_start(series: pd.Series, mode: str) -> tuple[int, float]:
    if len(series) < TIMESLICE_HOURS:
        raise ValueError(
            f"Need at least {TIMESLICE_HOURS} hourly rows to select a one-week timeframe."
        )
    rolling = series.reset_index(drop=True).rolling(TIMESLICE_HOURS).sum().dropna()
    if mode == "min":
        end_hour = int(rolling.idxmin())
        value = float(rolling.loc[end_hour])
    elif mode == "max":
        end_hour = int(rolling.idxmax())
        value = float(rolling.loc[end_hour])
    else:
        raise ValueError(f"Unknown rolling selection mode: {mode}")
    return end_hour - TIMESLICE_HOURS + 1, value


def select_timeframe_from_weather(df_weather: pd.DataFrame, timeframe_mode: str) -> dict[str, Any]:
    if timeframe_mode == "min_temperature_week":
        start_hour, value = _rolling_window_start(df_weather["temp_air"], "min")
        metric = "rolling_168h_temperature_sum"
    elif timeframe_mode == "max_solar_radiation_week":
        start_hour, value = _rolling_window_start(df_weather["ghi"], "max")
        metric = "rolling_168h_ghi_sum"
    else:
        raise ValueError(f"Weather cannot select timeframe mode: {timeframe_mode}")
    ensure_supported_slice(start_hour)
    return build_timeslice_metadata(
        timeframe_mode,
        start_hour,
        selection_metric=metric,
        selection_value=value,
    )


def select_timeframe_from_electricity(df_demand: pd.DataFrame, timeframe_mode: str) -> dict[str, Any]:
    if timeframe_mode != "max_base_electricity_demand_week":
        raise ValueError(f"Electricity cannot select timeframe mode: {timeframe_mode}")
    start_hour, value = _rolling_window_start(df_demand.sum(axis=1), "max")
    ensure_supported_slice(start_hour)
    return build_timeslice_metadata(
        timeframe_mode,
        start_hour,
        selection_metric="rolling_168h_base_electricity_demand_sum",
        selection_value=value,
    )


def serialize_metadata(metadata: dict[str, Any]) -> pd.Series:
    return pd.Series({key: json.dumps(value) for key, value in metadata.items()})


def parse_metadata(series: pd.Series) -> dict[str, Any]:
    parsed = {}
    for key, value in series.items():
        try:
            parsed[str(key)] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed[str(key)] = value
    return parsed


def write_hdf_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    serialize_metadata(metadata).to_hdf(path, key=TIMEFRAME_HDF_KEY, mode="a")


def read_hdf_metadata(path: str | Path) -> dict[str, Any]:
    try:
        series = pd.read_hdf(path, key=TIMEFRAME_HDF_KEY)
    except (FileNotFoundError, KeyError):
        return build_full_year_metadata()
    return parse_metadata(series)


def horizon_hours_from_hdf(path: str | Path) -> int:
    try:
        metadata = parse_metadata(pd.read_hdf(path, key=TIMEFRAME_HDF_KEY))
    except (FileNotFoundError, KeyError):
        metadata = {}
    horizon = metadata.get("horizon_hours")
    if horizon is not None:
        return int(horizon)
    return len(pd.read_hdf(path, key="urbs_in/demand"))
