"""Bulk writers for Step 2 and Step 4 results (PostgreSQL ``COPY``).

Every writer runs in one transaction and streams CSV through
``COPY ... FROM STDIN``. Values are stored as the former ``to_sql`` inserts
stored them: floats in shortest round-trip form, NaN/None as NULL, floats in
integer columns rounded half away from zero, ``-0.0`` as ``0.0``.

Time stamps follow one rule: ``ts = timeframe_start(run) + t_index hours``,
where ``timeframe_start`` comes from the run's assumptions (or its scenario's
when the run has none) and defaults to 2009-01-01 00:00 UTC.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

TIME_INDEX_START = "2009-01-01 00:00:00+00:00"
COPY_CHUNK_ROWS = 500_000
NULL = "\\N"
INTEGER_TYPES = frozenset({"smallint", "integer", "bigint"})
FLOAT_TYPES = frozenset({"double precision", "real"})
TEXT_TYPES = frozenset({"text", "character varying"})
BOUNDARY_SUMMARY_COLUMN_NAMES = (
    "boundary_first_24h_max_percent",
    "boundary_last_24h_max_percent",
    "boundary_outside_24h_max_percent",
    "boundary_24h_excess_percent",
    "boundary_first_168h_max_percent",
    "boundary_last_168h_max_percent",
    "boundary_outside_168h_max_percent",
    "boundary_168h_excess_percent",
    "boundary_overall_max_percent",
    "boundary_peak_t_index",
    "boundary_peak_in_first_24h",
    "boundary_peak_in_last_24h",
    "boundary_peak_in_first_168h",
    "boundary_peak_in_last_168h",
)

_column_types: dict[tuple[str, str], dict[str, str]] = {}


# COPY -------------------------------------------------------------------------


def _table_types(conn: Connection, table: str) -> dict[str, str]:
    key = (str(conn.engine.url), table)
    types = _column_types.get(key)
    if types is None:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'surrogrid' AND table_name = :table"
            ),
            {"table": table},
        )
        types = {str(name): str(data_type) for name, data_type in rows}
        if not types:
            raise ValueError(f"Unknown table surrogrid.{table}")
        _column_types[key] = types
    return types


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (
        isinstance(value, (float, np.floating)) and bool(np.isnan(value))
    )


def _round_half_away(value: float) -> int:
    # INSERT sent floats as numeric literals; numeric -> integer rounds half away from zero.
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


def _integer_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (float, np.floating)):
        return _round_half_away(float(value))
    return value


def _integer_column(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "iu" or isinstance(series.dtype, pd.Int64Dtype):
        return series
    return pd.Series([_integer_value(v) for v in series.tolist()], index=series.index, dtype=object)


def _float_column(series: pd.Series) -> pd.Series:
    if series.dtype.kind == "f":
        return series + 0.0  # -0.0 -> 0.0 (as the former inserts stored it)
    if series.dtype.kind in "iu":
        return series
    values = [None if _is_missing(v) else float(v) + 0.0 for v in series.tolist()]
    return pd.Series(values, index=series.index, dtype=object)


def _text_column(series: pd.Series) -> pd.Series:
    # PostgreSQL casts a boolean bound to a text column to 'true'/'false'.
    if series.dtype.kind == "b":
        return series.map({True: "true", False: "false"})
    if series.dtype == object:
        return series.map(lambda v: ("true" if v else "false") if isinstance(v, (bool, np.bool_)) else v)
    return series


def csv_frame(frame: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """Normalise ``frame`` so that ``to_csv`` yields the values INSERT stored.

    Args:
        frame: rows to write; every column must exist in the target table.
        types: ``information_schema`` data type per table column.
    """
    out = {}
    for column in frame.columns:
        data_type = types.get(column)
        if data_type is None:
            raise ValueError(f"Column {column!r} does not exist in the target table.")
        series = frame[column]
        if data_type in INTEGER_TYPES:
            series = _integer_column(series)
        elif data_type in FLOAT_TYPES:
            series = _float_column(series)
        elif data_type in TEXT_TYPES:
            series = _text_column(series)
        out[column] = series
    return pd.DataFrame(out, index=frame.index)


def copy_frame(conn: Connection, table: str, frame: pd.DataFrame) -> int:
    """Append ``frame`` to ``surrogrid.<table>`` with COPY; returns the row count."""
    if frame.empty:
        return 0
    prepared = csv_frame(frame, _table_types(conn, table))
    columns = ", ".join(f'"{column}"' for column in prepared.columns)
    statement = f"COPY surrogrid.{table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '{NULL}')"
    cursor = conn.connection.dbapi_connection.cursor()
    try:
        for start in range(0, len(prepared), COPY_CHUNK_ROWS):
            buffer = io.StringIO()
            prepared.iloc[start:start + COPY_CHUNK_ROWS].to_csv(buffer, index=False, header=False, na_rep=NULL)
            buffer.seek(0)
            cursor.copy_expert(statement, buffer)
    finally:
        cursor.close()
    return len(prepared)


# Time stamps --------------------------------------------------------------------


class RunTimestamps:
    """``timeframe_start`` per run (cached) and the derived hourly ``ts``."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._start: dict[tuple[str, int], str] = {}

    def start(self, run_table: str, run_id: int) -> str:
        """``timeframe_start`` of a ``demand_allocation_run`` / ``powerflow_run`` row."""
        key = (run_table, int(run_id))
        if key not in self._start:
            with self.engine.connect() as conn:
                assumptions = conn.execute(
                    text(
                        f"""
                        SELECT CASE WHEN r.assumptions <> '{{}}'::jsonb THEN r.assumptions
                                    ELSE sc.assumptions END
                        FROM surrogrid.{run_table} r
                        JOIN surrogrid.scenario sc ON sc.scenario_id = r.scenario_id
                        WHERE r.{run_table}_id = :run_id
                        """
                    ),
                    {"run_id": int(run_id)},
                ).scalar_one_or_none()
            if isinstance(assumptions, str):
                try:
                    assumptions = json.loads(assumptions)
                except json.JSONDecodeError:
                    assumptions = {}
            if not isinstance(assumptions, dict):
                assumptions = {}
            self._start[key] = assumptions.get("timeframe_start") or TIME_INDEX_START
        return self._start[key]

    def forget(self, run_table: str, run_id: int) -> None:
        """Drop the cached start of one run (its assumptions changed)."""
        self._start.pop((run_table, int(run_id)), None)

    def index(self, run_table: str, run_id: int, n_rows: int) -> pd.DatetimeIndex:
        """``ts`` of t_index 0 .. n_rows-1."""
        return pd.date_range(self.start(run_table, run_id), periods=n_rows, freq="h")

    def at(self, run_table: str, run_id: int, t_index: Any) -> pd.Timestamp | None:
        """``ts`` of one t_index (None for a missing index)."""
        if t_index is None or pd.isna(t_index):
            return None
        return pd.Timestamp(self.start(run_table, run_id)) + pd.Timedelta(hours=int(t_index))


def _column_values(df: pd.DataFrame, column: Any) -> np.ndarray:
    """Column values, or NULLs for a missing column."""
    if column in df.columns:
        return df[column].to_numpy()
    return np.full(len(df), None, dtype=object)


def _ts_text(ts: pd.DatetimeIndex, t_index: np.ndarray) -> np.ndarray:
    if not len(ts):
        return np.array([], dtype=object)
    return np.asarray(ts.astype(str), dtype=object)[t_index]


def long_frame(
    run_id: int,
    stage: str | None,
    ts: pd.DatetimeIndex,
    asset_column: str,
    assets: list[Any],
    values: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """Raw power-flow rows, asset-major then t_index (the former writers' row order).

    Args:
        run_id: ``powerflow_run_id`` of every row.
        stage: ``pre``/``post``, or None for tables without a stage column.
        ts: time stamps of t_index 0 .. n-1.
        asset_column: ``bus`` or ``line``.
        assets: asset ids in row order.
        values: per value column, one array of length n per asset.
    """
    n_rows, n_assets = len(ts), len(assets)
    t_index = np.tile(np.arange(n_rows), n_assets)
    data: dict[str, Any] = {"powerflow_run_id": np.full(n_rows * n_assets, int(run_id))}
    if stage is not None:
        data["stage"] = stage
    data["ts"] = _ts_text(ts, t_index)
    data["t_index"] = t_index
    data[asset_column] = np.repeat(np.asarray([int(a) for a in assets], dtype=np.int64), n_rows)
    for name, arrays in values.items():
        data[name] = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    return pd.DataFrame(data)


# Step 2 -----------------------------------------------------------------------------

ELECTRIFICATION_ASSIGNMENT_COLUMNS = (
    "building_objectid", "technology", "selection_scope_id",
    "adoption_mode", "configured_share", "eligible",
    "selection_score", "selection_rank", "selected",
    "exclusion_reason", "source_evidence", "profile_seed",
)
DEMAND_COMPONENT_AUDIT_COLUMNS = (
    "component_id", "objectid", "scenario_unit_id", "bus", "category",
    "commodity", "annual_energy_kwh", "max_profile_value", "profile_hash",
    "profile_method", "stable_seed", "source_asset_count",
    "matched_swf_asset_count", "included_in_lv", "suppression_reason",
    "pylovo_version_id", "mix_score", "mix_rule", "mix_confidence", "mv_direct",
)


def allocated_timeseries_frame(
    run_id: int, df: pd.DataFrame, ts: pd.DatetimeIndex, *, label_column: str
) -> pd.DataFrame:
    """Long rows of ``urbs_in/demand`` / ``urbs_in/eff_factor`` ((bus, label) columns)."""
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError(f"Expected MultiIndex (bus, {label_column}) columns.")
    out = df.copy().reset_index(drop=True)
    out.columns = pd.MultiIndex.from_tuples(
        [(int(col[0]), str(col[1])) for col in out.columns.to_flat_index()],
        names=["bus", label_column],
    )
    out.index = pd.RangeIndex(len(out), name="t_index")
    out = out.stack(["bus", label_column], future_stack=True).rename("value").reset_index()
    out.insert(1, "ts", ts.take(out["t_index"].to_numpy()))
    out.insert(0, "demand_allocation_run_id", int(run_id))
    return out.dropna(subset=["value"])


def write_allocated_timeseries(
    engine: Engine, timestamps: RunTimestamps, run_id: int, df: pd.DataFrame, *, label_column: str, table: str
) -> None:
    """Write one Step 2 hourly table (``allocated_demand`` / ``allocated_eff_factor``)."""
    if df.empty:
        return
    ts = timestamps.index("demand_allocation_run", run_id, len(df))
    frame = allocated_timeseries_frame(run_id, df, ts, label_column=label_column)
    with engine.begin() as conn:
        copy_frame(conn, table, frame)


def electrification_assignment_frame(run_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``electrification_assignment`` (validated columns)."""
    columns = list(ELECTRIFICATION_ASSIGNMENT_COLUMNS)
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Electrification assignment is missing columns: {missing}")
    out = df[columns].copy().astype(object)
    out["eligible"] = out["eligible"].map(bool)
    out["selected"] = out["selected"].map(bool)
    out["profile_seed"] = pd.to_numeric(out["profile_seed"], errors="raise").astype(int)
    out = out.where(pd.notna(out), None)
    out.insert(0, "demand_allocation_run_id", int(run_id))
    return out


def write_electrification_assignment(engine: Engine, run_id: int, df: pd.DataFrame) -> None:
    """Persist the auditable physical-building technology assignment."""
    if df.empty:
        return
    frame = electrification_assignment_frame(run_id, df)
    with engine.begin() as conn:
        copy_frame(conn, "electrification_assignment", frame)


def write_demand_component_audit(engine: Engine, run_id: int, df: pd.DataFrame) -> None:
    """Persist compact component profile evidence, never hourly profiles."""
    if df.empty:
        return
    columns = list(DEMAND_COMPONENT_AUDIT_COLUMNS)
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Demand component audit is missing columns: {missing}")
    out = df[columns].copy()
    out.insert(0, "demand_allocation_run_id", int(run_id))
    with engine.begin() as conn:
        copy_frame(conn, "demand_component_audit", out)


def allocated_vehicle_frame(run_id: int, df_buildings: pd.DataFrame, battery_dict: dict | None = None) -> pd.DataFrame:
    """One row per allocated vehicle (from the buildings' ``car_dict``)."""
    battery_dict = battery_dict or {}
    rows = []
    for car_dict in df_buildings.get("car_dict", []):
        if not isinstance(car_dict, dict):
            continue
        for (bus, vehicle_id), cfg in car_dict.items():
            rows.append(
                {
                    "demand_allocation_run_id": int(run_id),
                    "bus": int(bus),
                    "vehicle_id": int(vehicle_id),
                    "model": str(cfg["model"]),
                    "schedule": str(cfg["schedule"]),
                    "seed": int(cfg["seed"]),
                    "profile_id": cfg.get("profile_id"),
                    "battery_cap_kwh": cfg.get("battery_cap_kwh", battery_dict.get((int(bus), int(vehicle_id)))),
                }
            )
    return pd.DataFrame(rows)


def write_allocated_vehicles(engine: Engine, run_id: int, df_buildings: pd.DataFrame, battery_dict: dict | None = None) -> None:
    """Persist the vehicle (pool profile or emobpy vehicle) of every building bus."""
    frame = allocated_vehicle_frame(run_id, df_buildings, battery_dict)
    if not frame.empty:
        with engine.begin() as conn:
            copy_frame(conn, "allocated_vehicle", frame)


# Step 4 summaries --------------------------------------------------------------------

SYNTHETIC_GRID_SUMMARY_COLUMNS = (
    "transformer_s_rated_mva", "trafo_mean_s_mva", "trafo_max_s_mva", "trafo_max_p_mw",
    "trafo_max_q_mvar", "trafo_critical_t_index",
    "trafo_loading_p50_time_percent", "trafo_loading_p90_time_percent",
    "trafo_loading_p95_time_percent", "trafo_loading_p99_time_percent",
    "trafo_loading_max_time_percent", "trafo_loading_hours_above_100",
    "cable_loading_p95_asset_percent", "cable_hours_above_100_p95_asset",
    "voltage_p05_load_bus_hour_pu", "voltage_hours_below_0_90_p95_asset",
    "voltage_hours_above_1_03_p95_asset", "voltage_hours_above_1_10_p95_asset",
)
REAL_GRID_SUMMARY_COLUMNS = (
    "transformer_s_rated_mva",
    "trafo_loading_p50_time_percent", "trafo_loading_p90_time_percent",
    "trafo_loading_p95_time_percent", "trafo_loading_p99_time_percent",
    "trafo_loading_max_time_percent", "trafo_loading_hours_above_100",
    "cable_loading_p95_asset_percent", "cable_hours_above_100_p95_asset",
    "voltage_p05_load_bus_hour_pu", "voltage_hours_below_0_90_p95_asset",
)
REAL_CABLE_SUMMARY_COLUMNS = (
    "cable", "cable_loading_p50_time_percent", "cable_loading_p90_time_percent",
    "cable_loading_p95_time_percent", "cable_loading_p99_time_percent",
    "cable_loading_max_time_percent", "cable_loading_hours_above_100",
    "cable_max_i_ka", "cable_parallel", "cable_installed_capacity_ka",
)
REAL_BUS_VOLTAGE_SUMMARY_COLUMNS = (
    "bus", "voltage_p50_time_pu", "voltage_p10_time_pu", "voltage_p05_time_pu",
    "voltage_p01_time_pu", "voltage_min_time_pu", "voltage_hours_below_0_90",
)


def _summary_part(
    summary: dict[str, Any], key: str, run_column: str, run_id: int, stage: str, columns: tuple[str, ...] | None
) -> pd.DataFrame | None:
    frame = summary.get(key)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    out = frame.copy()
    if columns is not None:
        out = out[[column for column in columns if column in out.columns]]
    out.insert(0, "stage", stage)
    out.insert(0, run_column, int(run_id))
    return out


def summary_frames(
    timestamps: RunTimestamps, run_id: int, stage: str, summary: dict[str, Any], *, real: bool = False
) -> list[tuple[str, pd.DataFrame]]:
    """``(table, rows)`` of one stage's summary: grid, cable, bus, tail[, diagnostic].

    Synthetic runs (``real=False``) keep every column of the cable and bus
    frames and add ``trafo_critical_ts`` and the transformer diagnostic;
    real-grid runs keep the columns of their tables only.
    """
    prefix = "real_" if real else ""
    run_table = f"{prefix}powerflow_run"
    run_column = f"{run_table}_id"
    grid = summary.get("grid_summary", summary)
    row: dict[str, Any] = {
        run_column: int(run_id),
        "stage": stage,
        "n_timesteps": int(grid.get("n_timesteps", 0)),
        "n_converged_timesteps": grid.get("n_converged_timesteps"),
        "n_failed_timesteps": grid.get("n_failed_timesteps"),
        "n_voltage_buses": int(grid.get("n_voltage_buses", 0)),
        "n_cables": int(grid.get("n_cables", 0)),
    }
    for column in REAL_GRID_SUMMARY_COLUMNS if real else SYNTHETIC_GRID_SUMMARY_COLUMNS:
        row[column] = grid.get(column)
        if column == "trafo_critical_t_index":
            row["trafo_critical_ts"] = timestamps.at(run_table, run_id, grid.get(column))
    row.update({name: grid.get(name) for name in BOUNDARY_SUMMARY_COLUMN_NAMES})

    parts: list[tuple[str, pd.DataFrame]] = [(f"{prefix}powerflow_summary", pd.DataFrame([row]))]
    cable = _summary_part(summary, "cable_summary", run_column, run_id, stage, REAL_CABLE_SUMMARY_COLUMNS if real else None)
    if cable is not None:
        cable["cable"] = cable["cable"].astype(int)
        parts.append((f"{prefix}powerflow_cable_summary", cable))
    bus = _summary_part(
        summary, "bus_voltage_summary", run_column, run_id, stage, REAL_BUS_VOLTAGE_SUMMARY_COLUMNS if real else None
    )
    if bus is not None:
        bus["bus"] = bus["bus"].astype(int)
        parts.append((f"{prefix}powerflow_bus_voltage_summary", bus))
    tail = _summary_part(summary, "tail_summary", run_column, run_id, stage, None)
    if tail is not None:
        tail["asset_id"] = tail["asset_id"].astype(int)
        tail["t_index"] = tail["t_index"].astype(int)
        parts.append((f"{prefix}powerflow_tail_value", tail))
    if not real:
        diagnostic = _summary_part(summary, "transformer_diagnostic", run_column, run_id, stage, None)
        if diagnostic is not None:
            diagnostic["point_index"] = diagnostic["point_index"].astype(int)
            diagnostic["x_value"] = diagnostic["x_value"].astype(float)
            diagnostic["t_index"] = pd.to_numeric(diagnostic["t_index"], errors="coerce").astype("Int64")
            diagnostic["ts"] = [timestamps.at(run_table, run_id, value) for value in diagnostic["t_index"]]
            parts.append(("powerflow_transformer_diagnostic", diagnostic))
    return parts


def write_summary(
    engine: Engine, timestamps: RunTimestamps, run_id: int, stage: str, summary: dict[str, Any], *, real: bool = False
) -> None:
    """Write one stage's compact power-flow summary in one transaction."""
    parts = summary_frames(timestamps, run_id, stage, summary, real=real)
    with engine.begin() as conn:
        for table, frame in parts:
            copy_frame(conn, table, frame)


# Step 4 raw series ------------------------------------------------------------------


def powerflow_demand_frame(run_id: int, stage: str, df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-bus active/reactive demand ((bus, electricity[-reactive]) columns)."""
    buses = sorted({int(col[0]) for col in df.columns})
    return long_frame(
        run_id, stage, ts, "bus", buses,
        {
            "p_kw": [_column_values(df, (bus, "electricity")) for bus in buses],
            "q_kvar": [_column_values(df, (bus, "electricity-reactive")) for bus in buses],
        },
    )


def powerflow_import_frame(run_id: int, stage: str, df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """External-grid import per timestep (p_mw, q_mvar)."""
    return pd.DataFrame(
        {
            "powerflow_run_id": int(run_id),
            "stage": stage,
            "ts": _ts_text(ts, np.arange(len(df))),
            "t_index": np.arange(len(df)),
            "p_mw": _column_values(df, "p_mw"),
            "q_mvar": _column_values(df, "q_mvar"),
        }
    )


def powerflow_bus_voltage_frame(run_id: int, stage: str, df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Bus voltages (one column per bus, in the frame's column order)."""
    buses = [int(col) for col in df.columns]
    return long_frame(
        run_id, stage, ts, "bus", buses,
        {"vm_pu": [df.iloc[:, position].to_numpy() for position in range(len(buses))]},
    )


def powerflow_line_result_frame(run_id: int, stage: str, df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Line flows ((line, p_from_mw|q_from_mvar|i_from_ka) columns)."""
    lines = sorted({int(col[0]) for col in df.columns})
    return long_frame(
        run_id, stage, ts, "line", lines,
        {name: [_column_values(df, (line, name)) for line in lines] for name in ("p_from_mw", "q_from_mvar", "i_from_ka")},
    )


def powerflow_reactive_frame(run_id: int, df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Reactive power per (bus, component, source) column, in column order."""
    columns = list(df.columns)
    n_rows = len(df)
    t_index = np.tile(np.arange(n_rows), len(columns))
    return pd.DataFrame(
        {
            "powerflow_run_id": np.full(n_rows * len(columns), int(run_id)),
            "ts": _ts_text(ts, t_index),
            "t_index": t_index,
            "bus": np.repeat(np.asarray([int(bus) for bus, _, _ in columns], dtype=np.int64), n_rows),
            "component": np.repeat(np.asarray([str(c) for _, c, _ in columns], dtype=object), n_rows),
            "source": np.repeat(np.asarray([str(s) for _, _, s in columns], dtype=object), n_rows),
            "q_kvar": np.concatenate([df[column].to_numpy() for column in columns]) if columns else [],
        }
    )


RAW_FRAMES = {
    "powerflow_demand": powerflow_demand_frame,
    "powerflow_import": powerflow_import_frame,
    "powerflow_bus_voltage": powerflow_bus_voltage_frame,
    "powerflow_line_result": powerflow_line_result_frame,
}


def write_raw(
    engine: Engine, timestamps: RunTimestamps, table: str, run_id: int, stage: str | None, df: pd.DataFrame
) -> None:
    """Write one raw power-flow table of one run and stage in one transaction."""
    ts = timestamps.index("powerflow_run", run_id, len(df))
    if table == "powerflow_reactive_component":
        frame = powerflow_reactive_frame(run_id, df, ts)
    else:
        frame = RAW_FRAMES[table](run_id, stage, df, ts)
    with engine.begin() as conn:
        copy_frame(conn, table, frame)
