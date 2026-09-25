"""Daylight-saving-time alignment of UTC+1 profiles with civil-time activity.

Weather and all generated profiles use fixed UTC+1 (CET). Human activity
follows civil time, which runs one hour ahead between the two transitions of
the 2009 reference year. ``DST_TRANSITION_HOURS`` holds the zero-based hours
02:00-03:00 of 29 March (skipped) and 25 October (repeated).

Behaviour-driven generators (heat internal gains, occupancy, EV use) receive
inputs shifted with :func:`dst_shift_input`; their outputs, and profiles that
are sampled in civil time (base electricity), are mapped back to UTC+1 with
:func:`dst_shift_output`.
"""

from __future__ import annotations

import pandas as pd

from gridexpand.common.timeframe import DST_TRANSITION_HOURS


def dst_shift_input(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Shift an hourly UTC+1 input so that it lines up with civil time.

    Drops the repeated autumn hour and inserts a copy of the preceding hour at
    the skipped spring hour; :func:`dst_shift_output` later removes that dummy
    row again. The length is unchanged.
    """
    if len(df_ts) == 0:
        return df_ts.copy()
    spring, autumn = DST_TRANSITION_HOURS
    df_ts = df_ts.copy()
    df_ts = df_ts.drop(index=autumn).reset_index(drop=True)
    new_row = pd.DataFrame([df_ts.iloc[spring - 1].copy()], columns=df_ts.columns)
    return pd.concat([df_ts.iloc[:spring], new_row, df_ts.iloc[spring:]]).reset_index(drop=True)


def dst_shift_output(
    df_ts: pd.DataFrame,
    *,
    zero_repeated_hour: bool = False,
    preserve_energy: bool = False,
) -> pd.DataFrame:
    """Map an hourly civil-time profile back to UTC+1.

    Repeats the autumn hour (a copy is inserted after it) and removes the
    skipped spring hour. The length is unchanged.

    Args:
        df_ts: Hourly profile with one row per hour.
        zero_repeated_hour: Set the original autumn hour to zero instead of
            duplicating it (accumulated EV charging demand must not double).
        preserve_energy: Rescale every column to its original annual sum.

    Returns:
        The shifted copy.
    """
    if len(df_ts) == 0:
        return df_ts.copy()
    spring, autumn = DST_TRANSITION_HOURS
    df_ts = df_ts.copy()
    annual_energy = df_ts.sum(axis=0) if preserve_energy else None
    new_row = pd.DataFrame([df_ts.iloc[autumn].copy()], columns=df_ts.columns)
    df_ts = pd.concat(
        [df_ts.iloc[: autumn + 1], new_row, df_ts.iloc[autumn + 1 :]]
    ).reset_index(drop=True)
    if zero_repeated_hour:
        df_ts.iloc[autumn] = 0
    df_ts = df_ts.drop(index=spring).reset_index(drop=True)
    if preserve_energy:
        shifted_energy = df_ts.sum(axis=0)
        scale = annual_energy.divide(shifted_energy.where(shifted_energy.ne(0.0), 1.0))
        df_ts = df_ts.mul(scale, axis=1)
    return df_ts
