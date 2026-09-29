"""COPY writer frames equal the rows of the former to_sql writers (no database)."""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from gridexpand.db import writers

START = "2009-01-13T22:00:00+00:00"
TYPES = {
    "powerflow_run_id": "bigint", "stage": "text", "ts": "timestamp with time zone", "t_index": "integer",
    "bus": "integer", "line": "integer", "vm_pu": "double precision", "p_kw": "double precision",
    "q_kvar": "double precision", "p_from_mw": "double precision", "q_from_mvar": "double precision",
    "i_from_ka": "double precision", "component": "text", "source": "text", "p_mw": "double precision",
    "q_mvar": "double precision",
}


class FixedTimestamps(writers.RunTimestamps):
    def __init__(self) -> None:
        self._start = {}

    def start(self, run_table: str, run_id: int) -> str:
        return START


def _csv(frame: pd.DataFrame) -> list[list[str]]:
    buffer = io.StringIO()
    writers.csv_frame(frame, TYPES).to_csv(buffer, index=False, header=False, na_rep=writers.NULL)
    return sorted(line.split(",") for line in buffer.getvalue().splitlines())


def _old_ts(n: int) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=n, freq="h")


def _old_series(df: pd.DataFrame, column) -> pd.Series:
    if column in df.columns:
        return df[column].reset_index(drop=True)
    return pd.Series([None] * len(df))


def _old_bus_voltage(run_id: int, stage: str, df: pd.DataFrame) -> pd.DataFrame:
    """The former write_powerflow_bus_voltage (chunks of 25 buses, melt)."""
    columns = [int(col) for col in df.columns]
    parts = []
    for start in range(0, len(columns), 25):
        out = df.iloc[:, start:start + 25].copy()
        out.columns = columns[start:start + 25]
        out.insert(0, "t_index", range(len(out)))
        out.insert(0, "ts", _old_ts(len(df)))
        out.insert(0, "stage", stage)
        out.insert(0, "powerflow_run_id", run_id)
        parts.append(out.melt(id_vars=["powerflow_run_id", "stage", "ts", "t_index"], var_name="bus", value_name="vm_pu"))
    return pd.concat(parts, ignore_index=True)


def _old_line_result(run_id: int, stage: str, df: pd.DataFrame) -> pd.DataFrame:
    lines = sorted({int(col[0]) for col in df.columns})
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "powerflow_run_id": run_id, "stage": stage, "ts": _old_ts(len(df)), "t_index": range(len(df)),
                    "line": line,
                    "p_from_mw": _old_series(df, (line, "p_from_mw")),
                    "q_from_mvar": _old_series(df, (line, "q_from_mvar")),
                    "i_from_ka": _old_series(df, (line, "i_from_ka")),
                }
            )
            for line in lines
        ],
        ignore_index=True,
    )


def _old_reactive(run_id: int, df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "powerflow_run_id": run_id, "ts": _old_ts(len(df)), "t_index": range(len(df)), "bus": int(bus),
                    "component": str(component), "source": str(source),
                    "q_kvar": df[(bus, component, source)].to_numpy(),
                }
            )
            for bus, component, source in df.columns
        ],
        ignore_index=True,
    )


def _values(rng: np.random.Generator, n: int) -> np.ndarray:
    values = rng.normal(size=n) * 10.0 ** rng.integers(-8, 8, n)
    values[::17] = np.nan
    values[5] = -0.0
    return values


def test_bus_voltage_frame_equals_former_rows() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({bus: _values(rng, 30) for bus in [7, 3, 12] + list(range(20, 50))})
    new = writers.powerflow_bus_voltage_frame(4, "post", df, _old_ts(len(df)))
    old = _old_bus_voltage(4, "post", df)
    assert _csv(new) == _csv(old[new.columns.tolist()])


def test_line_result_frame_equals_former_rows_with_missing_columns() -> None:
    rng = np.random.default_rng(4)
    columns = pd.MultiIndex.from_tuples(
        [(line, name) for line in (5, 1, 9) for name in ("p_from_mw", "i_from_ka")] + [(1, "q_from_mvar")]
    )
    df = pd.DataFrame(np.column_stack([_values(rng, 24) for _ in columns]), columns=columns)
    new = writers.powerflow_line_result_frame(2, "pre", df, _old_ts(len(df)))
    old = _old_line_result(2, "pre", df)
    assert _csv(new) == _csv(old[new.columns.tolist()])


def test_reactive_frame_equals_former_rows() -> None:
    rng = np.random.default_rng(5)
    columns = pd.MultiIndex.from_tuples([(3, "heat", "hp"), (1, "mobility", "ev"), (3, "pv", "inv")])
    df = pd.DataFrame(np.column_stack([_values(rng, 10) for _ in columns]), columns=columns)
    new = writers.powerflow_reactive_frame(9, df, _old_ts(len(df)))
    old = _old_reactive(9, df)
    assert _csv(new) == _csv(old[new.columns.tolist()])


def test_import_frame() -> None:
    df = pd.DataFrame({"p_mw": [0.1, np.nan], "q_mvar": [-0.0, 2.5]})
    frame = writers.powerflow_import_frame(1, "pre", df, _old_ts(2))
    assert _csv(frame) == sorted(
        [
            ["1", "pre", "2009-01-13 22:00:00+00:00", "0", "0.1", "0.0"],
            ["1", "pre", "2009-01-13 23:00:00+00:00", "1", writers.NULL, "2.5"],
        ]
    )


def test_csv_frame_matches_insert_semantics() -> None:
    types = {"a": "integer", "b": "double precision", "c": "text", "d": "boolean", "e": "bigint"}
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.5, -2.5, np.nan],
            "b": [-0.0, np.inf, np.nan, 1 / 3],
            "c": ["x", True, None, "y,z"],
            "d": [True, False, None, np.bool_(True)],
            "e": pd.array([1, None, 3, 2**62], dtype="Int64"),
        }
    )
    buffer = io.StringIO()
    writers.csv_frame(frame, types).to_csv(buffer, index=False, header=False, na_rep=writers.NULL)
    assert buffer.getvalue().splitlines() == [
        "1,0.0,x,True,1",
        "3,inf,true,False,\\N",
        "-3,\\N,\\N,\\N,3",
        f"\\N,{1 / 3!r},\"y,z\",True,{2**62}",
    ]


def test_csv_frame_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        writers.csv_frame(pd.DataFrame({"nope": [1]}), {"a": "integer"})


def test_timestamps_follow_timeframe_start() -> None:
    stamps = FixedTimestamps()
    assert stamps.at("powerflow_run", 1, 25) == pd.Timestamp("2009-01-14T23:00:00+00:00")
    assert stamps.at("powerflow_run", 1, None) is None
    assert stamps.at("powerflow_run", 1, pd.NA) is None
    index = stamps.index("powerflow_run", 1, 30)
    assert all(index[i] == stamps.at("powerflow_run", 1, i) for i in range(30))


def test_summary_frames_synthetic_and_real() -> None:
    stamps = FixedTimestamps()
    summary = {
        "grid_summary": {"n_timesteps": 168, "n_voltage_buses": 3, "n_cables": 2, "trafo_critical_t_index": 5,
                         "lv_busbar_vm_pu": 0.96 / 0.975, "tap_steps": 1},
        "cable_summary": pd.DataFrame({"cable": [1.0, 2.0], "cable_max_i_ka": [0.1, 0.2], "extra": [1, 2]}),
        "transformer_diagnostic": pd.DataFrame(
            {"diagnostic": ["x", "x"], "point_index": [0, 1], "x_value": [1, 2], "t_index": [3, None]}
        ),
    }
    synthetic = dict(writers.summary_frames(stamps, 8, "post", summary))
    assert set(synthetic) == {"powerflow_summary", "powerflow_cable_summary", "powerflow_transformer_diagnostic"}
    assert synthetic["powerflow_summary"].loc[0, "trafo_critical_ts"] == pd.Timestamp("2009-01-14T03:00:00+00:00")
    assert "extra" in synthetic["powerflow_cable_summary"]
    assert pd.isna(list(synthetic["powerflow_transformer_diagnostic"]["ts"])[1])
    real = dict(writers.summary_frames(stamps, 8, "base_electricity", summary, real=True))
    assert set(real) == {"real_powerflow_summary", "real_powerflow_cable_summary"}
    assert "extra" not in real["real_powerflow_cable_summary"]
    assert "trafo_critical_ts" not in real["real_powerflow_summary"]
    assert real["real_powerflow_summary"].loc[0, "real_powerflow_run_id"] == 8
    for row in (synthetic["powerflow_summary"], real["real_powerflow_summary"]):
        assert row.loc[0, "tap_steps"] == 1 and row.loc[0, "lv_busbar_vm_pu"] == pytest.approx(0.96 / 0.975)


def test_allocated_timeseries_frame_drops_missing_values() -> None:
    df = pd.DataFrame(
        [[1.0, np.nan], [2.0, 3.0]], columns=pd.MultiIndex.from_tuples([(4, "electricity"), (5, "heat")])
    )
    frame = writers.allocated_timeseries_frame(2, df, _old_ts(2), label_column="commodity")
    assert frame[["t_index", "bus", "commodity", "value"]].values.tolist() == [
        [0, 4, "electricity", 1.0], [1, 4, "electricity", 2.0], [1, 5, "heat", 3.0],
    ]
    assert list(frame.columns) == ["demand_allocation_run_id", "t_index", "ts", "bus", "commodity", "value"]


def test_electrification_assignment_frame() -> None:
    df = pd.DataFrame(
        {column: ["v"] for column in writers.ELECTRIFICATION_ASSIGNMENT_COLUMNS}
        | {"eligible": [1], "selected": [0], "profile_seed": ["12"], "selection_rank": [math.nan]}
    )
    frame = writers.electrification_assignment_frame(3, df)
    assert isinstance(frame.loc[0, "eligible"], (bool, np.bool_))
    assert bool(frame.loc[0, "eligible"]) and not bool(frame.loc[0, "selected"])
    assert frame.loc[0, "profile_seed"] == 12 and frame.loc[0, "selection_rank"] is None
    with pytest.raises(ValueError, match="missing columns"):
        writers.electrification_assignment_frame(3, df.drop(columns="technology"))


def test_allocated_vehicle_frame() -> None:
    buildings = pd.DataFrame(
        {"car_dict": [{(3, 0): {"model": "m", "schedule": "s", "seed": 7}}, None]}
    )
    frame = writers.allocated_vehicle_frame(1, buildings, {(3, 0): 55.0})
    assert frame.to_dict("records") == [
        {"demand_allocation_run_id": 1, "bus": 3, "vehicle_id": 0, "model": "m", "schedule": "s",
         "seed": 7, "profile_id": None, "battery_cap_kwh": 55.0}
    ]
