"""DST helpers are byte-identical to the four former implementations (alloc D3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridexpand.allocation.functions.dst import dst_shift_input, dst_shift_output
from gridexpand.allocation.scenario_calibration.profiles.real_swf_electricity_profiles import (
    add_output_data_daylight_saving_shift,
)


# Verbatim copies of the former implementations (Grid static methods and the
# energy-preserving paired variant) as the reference.
def _old_input(df_ts):
    if len(df_ts) == 0:
        return df_ts.copy()
    ts_hour1, ts_hour2 = 2090, 7130
    df_ts = df_ts.copy()
    df_ts = df_ts.drop(index=ts_hour2).reset_index(drop=True)
    new_row = df_ts.iloc[ts_hour1 - 1].copy()
    new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
    return pd.concat([df_ts.iloc[:ts_hour1], new_row_df, df_ts.iloc[ts_hour1:]]).reset_index(drop=True)


def _old_output(df_ts, mobility_dmd=False):
    if len(df_ts) == 0:
        return df_ts.copy()
    ts_hour1, ts_hour2 = 2090, 7130
    df_ts = df_ts.copy()
    new_row = df_ts.iloc[ts_hour2].copy()
    new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
    df_ts = pd.concat([df_ts.iloc[:ts_hour2 + 1], new_row_df, df_ts.iloc[ts_hour2 + 1:]]).reset_index(drop=True)
    if mobility_dmd:
        df_ts.iloc[ts_hour2] = 0
    return df_ts.drop(index=ts_hour1).reset_index(drop=True)


def _old_paired_output(df_ts):
    if df_ts.empty:
        return df_ts.copy()
    ts_hour1, ts_hour2 = 2090, 7130
    df_ts = df_ts.copy()
    annual_energy = df_ts.sum(axis=0)
    new_row = df_ts.iloc[ts_hour2].copy()
    new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
    shifted = pd.concat([df_ts.iloc[: ts_hour2 + 1], new_row_df, df_ts.iloc[ts_hour2 + 1 :]]).reset_index(drop=True)
    shifted = shifted.drop(index=ts_hour1).reset_index(drop=True)
    shifted_energy = shifted.sum(axis=0)
    scale = annual_energy.divide(shifted_energy.where(shifted_energy.ne(0.0), 1.0))
    return shifted.mul(scale, axis=1)


def _profiles(seed=3, columns=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.random((8760, columns)))
    frame.columns = pd.MultiIndex.from_product([range(columns), ["electricity"]])
    frame.iloc[:, -1] = 0.0  # an all-zero column exercises the energy rescaling guard
    return frame


def _weather():
    rng = np.random.default_rng(5)
    index = pd.date_range("2009-01-01", periods=8760, freq="h", tz="UTC")
    return pd.DataFrame({"time(inst)": index, "temp_air": rng.normal(10, 5, 8760), "ghi": rng.random(8760)})


@pytest.mark.parametrize("frame", [_profiles(), _weather()])
def test_input_shift_matches_former_implementation(frame):
    pd.testing.assert_frame_equal(dst_shift_input(frame), _old_input(frame), check_exact=True)


@pytest.mark.parametrize("zero", [False, True])
def test_output_shift_matches_former_implementation(zero):
    frame = _profiles()
    pd.testing.assert_frame_equal(
        dst_shift_output(frame, zero_repeated_hour=zero), _old_output(frame, mobility_dmd=zero), check_exact=True
    )


def test_energy_preserving_shift_matches_paired_implementation():
    frame = _profiles(seed=11)
    expected = _old_paired_output(frame)
    pd.testing.assert_frame_equal(dst_shift_output(frame, preserve_energy=True), expected, check_exact=True)
    pd.testing.assert_frame_equal(add_output_data_daylight_saving_shift(frame), expected, check_exact=True)
    np.testing.assert_allclose(expected.sum().to_numpy(), frame.sum().to_numpy())


def test_empty_frames_pass_through():
    empty = pd.DataFrame()
    assert dst_shift_input(empty).empty and dst_shift_output(empty).empty
    assert add_output_data_daylight_saving_shift(empty).empty


def test_round_trip_keeps_length_and_moves_one_hour():
    frame = pd.DataFrame({"x": np.arange(8760, dtype=float)})
    shifted = dst_shift_output(dst_shift_input(frame))
    assert len(shifted) == 8760
    assert shifted["x"].iloc[2089] == 2089.0 and shifted["x"].iloc[2090] == 2090.0
