"""Shared EV charging-session contract for the full-year chronological reference.

One *session* is one modeled home stay of one physical vehicle. The controllers
(URBS/HEMS and the INFLEX heuristic) must serve the identical obligation:

    0 <= P_charge[s,t] <= P_max[s] * available_fraction[s,t]
    sum_t P_charge[s,t] * delta_t == energy_kwh[s]
    P_charge[s,t] == 0                       for t outside the session

This module is imported by Step 2 (Python 3.12), Step 3 (Python 3.10) and
Step 4 (Python 3.12); it therefore uses only ``from __future__ import
annotations``, pandas, numpy and the standard library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS_HDF_KEY = "urbs_in/ev_sessions"
SESSION_HOURS_HDF_KEY = "urbs_in/ev_session_hours"

# Model hour t maps to source timestamp start + (t - SESSION_HOUR_OFFSET) hours.
# t = 0 is the URBS storage initialization row and never carries EV service.
SESSION_HOUR_OFFSET = 1

ENERGY_TOL_KWH = 1e-6
POWER_TOL_KW = 1e-6
FRACTION_TOL = 1e-9

SESSION_COLUMNS = [
    "session_id",
    "site",
    "vehicle_index",
    "process",
    "profile_id",
    "charger_kw",
    "energy_kwh",
    "n_hours",
    "available_hours",
    "capacity_kwh",
    "wraps_year",
    "merged_from",
    "source_start_ts",
    "source_end_ts",
    "battery_cap_kwh",
    "energy_over_battery_ratio",
    "generation_version",
]

SESSION_HOUR_COLUMNS = ["session_id", "t", "order", "available_fraction"]

# Step 3 only: the HEMS charging limit per session hour (InFlex keeps available_fraction).
HEMS_FRACTION_COLUMN = "hems_available_fraction"


class SessionError(ValueError):
    """A session table violates the shared contract."""


class SessionInfeasible(SessionError):
    """A session cannot deliver its required energy inside its window."""

    def __init__(self, detail):
        self.detail = dict(detail)
        super().__init__(
            "EV charging session cannot deliver its required energy: "
            + ", ".join(f"{key}={value}" for key, value in sorted(self.detail.items()))
        )


class SessionDefinitionUnresolved(SessionError):
    """The source records do not support an unambiguous session definition."""


# ---------------------------------------------------------------------------
# Construction from raw source records
# ---------------------------------------------------------------------------


def _home_intervals(is_home: np.ndarray) -> list:
    """Maximal half-open runs [start, stop) of connected source steps."""
    intervals = []
    index = 0
    total = len(is_home)
    while index < total:
        if not is_home[index]:
            index += 1
            continue
        stop = index
        while stop < total and is_home[stop]:
            stop += 1
        intervals.append((index, stop))
        index = stop
    return intervals


def _hour_fractions(
    steps: np.ndarray,
    source_timestep_hours: float,
    delta_t: float,
) -> "dict":
    """Map source steps to model hours and the connected fraction of each hour."""
    fractions = {}
    steps_per_hour = delta_t / source_timestep_hours
    for step in steps:
        hour = int(np.floor(step * source_timestep_hours / delta_t))
        fractions[hour] = fractions.get(hour, 0.0) + 1.0 / steps_per_hour
    return {hour: min(value, 1.0) for hour, value in fractions.items()}


def _assert_no_shared_hours(records: list) -> None:
    """No two sessions of one vehicle may claim the same model hour.

    With a half-hourly source and an hourly model this cannot happen: an hour
    holds two source steps, and two separate home stays touching one hour would
    need home-away-home inside them, which needs three. The guard exists so a
    finer source or coarser model timestep fails loudly instead of silently
    changing the contract, since neither available answer is right -- merging the
    two sessions would let one stay's energy be delivered during the other, and
    keeping both would grant the vehicle two chargers in that hour.
    """
    owner = {}
    for position, record in enumerate(records):
        for hour in record["fractions"]:
            previous = owner.get(hour)
            if previous is not None and previous != position:
                raise SessionDefinitionUnresolved(
                    "Two charging sessions of one vehicle share model hour "
                    f"{hour + SESSION_HOUR_OFFSET} (source steps "
                    f"{records[previous]['start_step']}-{records[previous]['stop_step']} "
                    f"and {record['start_step']}-{record['stop_step']}). This "
                    "source/model timestep combination is unsupported."
                )
            owner[hour] = position


def build_sessions_from_source(
    records: pd.DataFrame,
    *,
    profile_id: str,
    charger_kw: float,
    battery_cap_kwh: float,
    source_timestep_hours: float,
    horizon_hours: int,
    site=0,
    vehicle_index: int = 0,
    delta_t: float = 1.0,
    generation_version: str = "unversioned",
    reference_start=None,
):
    """Build the session and session-hour tables for one vehicle profile.

    ``records`` must be indexed by source step 0..N-1 and contain
    ``charging_point`` and ``charge_grid`` (grid-side power in kW).
    """
    required = {"charging_point", "charge_grid"}
    missing = required - set(records.columns)
    if missing:
        raise SessionError(f"Source records are missing columns: {sorted(missing)}")

    charger_kw = float(charger_kw)
    if not np.isfinite(charger_kw) or charger_kw <= 0.0:
        raise SessionError(
            f"Charger power must be finite and positive, got {charger_kw!r}."
        )
    source_timestep_hours = float(source_timestep_hours)
    delta_t = float(delta_t)
    steps_per_hour = delta_t / source_timestep_hours
    if abs(steps_per_hour - round(steps_per_hour)) > 1e-9:
        raise SessionError(
            "The model timestep must be an integer multiple of the source "
            f"timestep, got delta_t={delta_t} and source={source_timestep_hours}."
        )
    expected_steps = int(round(horizon_hours * steps_per_hour))
    if len(records) != expected_steps:
        raise SessionError(
            f"Profile {profile_id} has {len(records)} source steps, expected "
            f"{expected_steps} for a {horizon_hours} h horizon."
        )

    power = pd.to_numeric(records["charge_grid"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(power).all():
        raise SessionError(f"Profile {profile_id} has non-finite grid charging power.")
    # emobpy writes tiny negative values (-0.0, -1e-13) for idle steps.
    if power.min() < -1e-9:
        raise SessionError(
            f"Profile {profile_id} has negative grid charging power "
            f"({power.min():.3e} kW)."
        )
    power = np.clip(power, 0.0, None)
    energy_per_step = power * source_timestep_hours

    is_home = records["charging_point"].astype(str).to_numpy() == "home"

    # Charging that happened away from home is en-route or destination charging
    # on a public charge point outside the low-voltage grid under study. It is
    # attributed to that network and never enters the household connection; the
    # profile ledger records the excluded amount explicitly. See
    # FULL_YEAR_REFERENCE_DESIGN.md section 2.7.
    home_energy_per_step = np.where(is_home, energy_per_step, 0.0)

    intervals = _home_intervals(is_home)
    if not intervals:
        return _empty_tables()

    if len(intervals) == 1 and intervals[0] == (0, len(is_home)):
        if float(home_energy_per_step.sum()) > ENERGY_TOL_KWH:
            raise SessionDefinitionUnresolved(
                f"Profile {profile_id} is connected for the whole year with "
                "positive energy; the source departure events do not support an "
                "unambiguous session definition."
            )
        return _empty_tables()

    wrap = (
        len(intervals) > 1
        and intervals[0][0] == 0
        and intervals[-1][1] == len(is_home)
    )

    raw = []
    for _position, (start, stop) in enumerate(intervals):
        raw.append(
            {
                "start_step": int(start),
                "stop_step": int(stop),
                "energy_kwh": float(home_energy_per_step[start:stop].sum()),
                "wraps_year": False,
                "merged_from": 1,
                "fractions": _hour_fractions(
                    np.arange(start, stop), source_timestep_hours, delta_t
                ),
            }
        )

    if wrap:
        trailing = raw[-1]
        leading = raw[0]
        fractions = dict(trailing["fractions"])
        for hour, value in leading["fractions"].items():
            fractions[hour] = min(fractions.get(hour, 0.0) + value, 1.0)
        merged = {
            "start_step": trailing["start_step"],
            "stop_step": leading["stop_step"],
            "energy_kwh": trailing["energy_kwh"] + leading["energy_kwh"],
            "wraps_year": True,
            "merged_from": 2,
            "fractions": fractions,
        }
        raw = [merged] + raw[1:-1]

    _assert_no_shared_hours(raw)
    raw.sort(key=lambda record: (not record["wraps_year"], record["start_step"]))

    reference_start = pd.Timestamp(reference_start) if reference_start is not None else None
    session_rows = []
    hour_rows = []
    for sequence, record in enumerate(raw):
        session_id = f"{site}:{vehicle_index}:{sequence}"
        start_hour = int(
            np.floor(record["start_step"] * source_timestep_hours / delta_t)
        )
        ordered = sorted(
            record["fractions"].items(),
            key=lambda item: (item[0] - start_hour) % horizon_hours,
        )
        available_hours = float(sum(value for _hour, value in ordered))
        capacity_kwh = charger_kw * delta_t * available_hours
        energy_kwh = float(record["energy_kwh"])
        if energy_kwh > capacity_kwh + ENERGY_TOL_KWH:
            raise SessionInfeasible(
                {
                    "session_id": session_id,
                    "site": site,
                    "profile_id": profile_id,
                    "first_hour": ordered[0][0] + SESSION_HOUR_OFFSET,
                    "last_hour": ordered[-1][0] + SESSION_HOUR_OFFSET,
                    "energy_kwh": round(energy_kwh, 9),
                    "capacity_kwh": round(capacity_kwh, 9),
                    "charger_kw": charger_kw,
                    "shortfall_kwh": round(energy_kwh - capacity_kwh, 9),
                }
            )
        for order, (hour, fraction) in enumerate(ordered):
            hour_rows.append(
                {
                    "session_id": session_id,
                    "t": int(hour) + SESSION_HOUR_OFFSET,
                    "order": int(order),
                    "available_fraction": float(fraction),
                }
            )
        session_rows.append(
            {
                "session_id": session_id,
                "site": site,
                "vehicle_index": int(vehicle_index),
                "process": f"charging_station{int(vehicle_index)}",
                "profile_id": str(profile_id),
                "charger_kw": charger_kw,
                "energy_kwh": energy_kwh,
                "n_hours": len(ordered),
                "available_hours": available_hours,
                "capacity_kwh": capacity_kwh,
                "wraps_year": bool(record["wraps_year"]),
                "merged_from": int(record["merged_from"]),
                "source_start_ts": _timestamp(
                    reference_start, record["start_step"], source_timestep_hours
                ),
                "source_end_ts": _timestamp(
                    reference_start, record["stop_step"], source_timestep_hours
                ),
                "battery_cap_kwh": float(battery_cap_kwh),
                "energy_over_battery_ratio": (
                    energy_kwh / float(battery_cap_kwh)
                    if float(battery_cap_kwh) > 0.0
                    else np.nan
                ),
                "generation_version": str(generation_version),
            }
        )

    sessions = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
    hours = pd.DataFrame(hour_rows, columns=SESSION_HOUR_COLUMNS)

    delivered = float(sessions["energy_kwh"].sum())
    expected = float(home_energy_per_step.sum())
    if abs(delivered - expected) > ENERGY_TOL_KWH:
        raise SessionError(
            f"Profile {profile_id} session energy {delivered:.9f} kWh does not "
            f"conserve the source home grid charging energy {expected:.9f} kWh."
        )
    return sessions, hours


def _timestamp(reference_start, step, source_timestep_hours):
    if reference_start is None:
        return ""
    return (
        reference_start + pd.Timedelta(hours=float(step) * source_timestep_hours)
    ).isoformat()


def _empty_tables():
    return (
        pd.DataFrame(columns=SESSION_COLUMNS),
        pd.DataFrame(columns=SESSION_HOUR_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_sessions(
    sessions: pd.DataFrame,
    hours: pd.DataFrame,
    *,
    horizon_hours: int,
    delta_t: float = 1.0,
    process_table: pd.DataFrame = None,
):
    """Validate the shared contract. Raises on any violation, returns a report."""
    if sessions is None or hours is None:
        raise SessionError("Session tables are missing.")
    missing = set(SESSION_COLUMNS) - set(sessions.columns)
    if missing:
        raise SessionError(f"Session table is missing columns: {sorted(missing)}")
    missing = set(SESSION_HOUR_COLUMNS) - set(hours.columns)
    if missing:
        raise SessionError(f"Session-hour table is missing columns: {sorted(missing)}")

    if sessions.empty:
        if not hours.empty:
            raise SessionError("Session-hour rows exist without session rows.")
        return {"sessions": 0, "session_hours": 0, "energy_kwh": 0.0, "max_shortfall_kwh": 0.0}

    if sessions["session_id"].duplicated().any():
        duplicates = sorted(
            sessions.loc[sessions["session_id"].duplicated(), "session_id"].unique()
        )
        raise SessionError(f"Duplicate session ids: {duplicates[:5]}")

    energy = pd.to_numeric(sessions["energy_kwh"], errors="coerce")
    power = pd.to_numeric(sessions["charger_kw"], errors="coerce")
    if not np.isfinite(energy).all() or (energy < 0.0).any():
        raise SessionError("Session energy must be finite and non-negative.")
    if not np.isfinite(power).all() or (power <= 0.0).any():
        raise SessionError("Session charger power must be finite and positive.")

    fraction = pd.to_numeric(hours["available_fraction"], errors="coerce")
    if not np.isfinite(fraction).all():
        raise SessionError("Session availability fractions must be finite.")
    if (fraction <= 0.0).any() or (fraction > 1.0 + FRACTION_TOL).any():
        raise SessionError(
            "Session availability fractions must lie in (0, 1]; found "
            f"[{fraction.min()}, {fraction.max()}]."
        )

    step = pd.to_numeric(hours["t"], errors="coerce")
    if not np.isfinite(step).all():
        raise SessionError("Session hour indices must be finite.")
    if (step < SESSION_HOUR_OFFSET).any() or (
        step > horizon_hours + SESSION_HOUR_OFFSET - 1
    ).any():
        raise SessionError(
            f"Session hour indices must lie in "
            f"[{SESSION_HOUR_OFFSET}, {horizon_hours + SESSION_HOUR_OFFSET - 1}]."
        )

    unknown = set(hours["session_id"]) - set(sessions["session_id"])
    if unknown:
        raise SessionError(f"Session-hour rows reference unknown sessions: {sorted(unknown)[:5]}")

    counted = hours.groupby("session_id").size()
    declared = sessions.set_index("session_id")["n_hours"].astype(int)
    mismatch = declared[declared != counted.reindex(declared.index).fillna(0).astype(int)]
    if not mismatch.empty:
        raise SessionError(
            f"n_hours disagrees with the session-hour rows for {list(mismatch.index[:5])}."
        )

    available = hours.groupby("session_id")["available_fraction"].sum()
    capacity = sessions.set_index("session_id")["charger_kw"].astype(float) * float(
        delta_t
    ) * available.reindex(sessions["session_id"]).to_numpy()
    shortfall = sessions.set_index("session_id")["energy_kwh"].astype(float) - capacity
    infeasible = shortfall[shortfall > ENERGY_TOL_KWH]
    if not infeasible.empty:
        session_id = infeasible.index[0]
        row = sessions.set_index("session_id").loc[session_id]
        session_hours = hours[hours["session_id"] == session_id]
        raise SessionInfeasible(
            {
                "session_id": session_id,
                "site": row["site"],
                "profile_id": row["profile_id"],
                "first_hour": int(session_hours["t"].min()),
                "last_hour": int(session_hours["t"].max()),
                "energy_kwh": round(float(row["energy_kwh"]), 9),
                "capacity_kwh": round(float(capacity.loc[session_id]), 9),
                "charger_kw": float(row["charger_kw"]),
                "shortfall_kwh": round(float(infeasible.iloc[0]), 9),
            }
        )

    # Hours of one vehicle must be disjoint across its sessions.
    keyed = hours.merge(
        sessions[["session_id", "site", "vehicle_index"]], on="session_id", how="left"
    )
    overlapping = keyed.duplicated(subset=["site", "vehicle_index", "t"], keep=False)
    if bool(overlapping.any()):
        sample = keyed.loc[overlapping, ["site", "vehicle_index", "t"]].head(3)
        raise SessionError(
            "Sessions of one vehicle share model hours, which would duplicate "
            f"charger capacity: {sample.to_dict('records')}"
        )

    if process_table is not None and not process_table.empty:
        table = process_table.reset_index()
        if {"Site", "Process", "inst-cap", "cap-up"}.issubset(table.columns):
            table = table.set_index(
                [table["Site"].astype(str), table["Process"].astype(str)]
            )
            for _, row in sessions.iterrows():
                key = (str(row["site"]), str(row["process"]))
                if key not in table.index:
                    raise SessionError(
                        f"Session {row['session_id']} references missing process {key}."
                    )
                process_row = table.loc[key]
                if isinstance(process_row, pd.DataFrame):
                    process_row = process_row.iloc[0]
                installed = float(process_row["inst-cap"])
                upper = float(process_row["cap-up"])
                if not np.isclose(installed, upper, rtol=0.0, atol=POWER_TOL_KW):
                    raise SessionError(
                        f"Charger {key} is not a fixed heuristic asset "
                        f"(inst-cap={installed}, cap-up={upper})."
                    )
                if not np.isclose(
                    installed, float(row["charger_kw"]), rtol=0.0, atol=POWER_TOL_KW
                ):
                    raise SessionError(
                        f"Session {row['session_id']} charger power "
                        f"{row['charger_kw']} kW disagrees with process capacity "
                        f"{installed} kW."
                    )

    return {
        "sessions": int(len(sessions)),
        "session_hours": int(len(hours)),
        "energy_kwh": float(sessions["energy_kwh"].sum()),
        "max_shortfall_kwh": float(max(shortfall.max(), 0.0)),
    }


# ---------------------------------------------------------------------------
# INFLEX schedule
# ---------------------------------------------------------------------------


def earliest_feasible_schedule(
    sessions: pd.DataFrame,
    hours: pd.DataFrame,
    *,
    horizon_hours: int,
    delta_t: float = 1.0,
):
    """Fill every session from its own arrival, earliest feasible hour first.

    Returns a ``(horizon_hours, n_charger)`` frame of charging power in kW,
    indexed by model hour ``t`` and columned by ``(site, process)``.
    """
    index = pd.RangeIndex(
        SESSION_HOUR_OFFSET, horizon_hours + SESSION_HOUR_OFFSET, name="t"
    )
    if sessions.empty:
        return pd.DataFrame(index=index)

    ordered = hours.sort_values(["session_id", "order"])
    grouped = {
        session_id: (
            group["t"].to_numpy(dtype=int),
            group["available_fraction"].to_numpy(dtype=float),
        )
        for session_id, group in ordered.groupby("session_id", sort=False)
    }

    columns = {}
    residuals = []
    for row in sessions.to_dict("records"):
        key = (row["site"], row["process"])
        if key not in columns:
            columns[key] = np.zeros(horizon_hours, dtype=float)
        target = columns[key]
        steps, fractions = grouped[row["session_id"]]
        remaining = float(row["energy_kwh"])
        charger_kw = float(row["charger_kw"])
        for position in range(len(steps)):
            if remaining <= ENERGY_TOL_KWH:
                break
            limit_kw = charger_kw * fractions[position]
            power_kw = min(limit_kw, remaining / delta_t)
            offset = steps[position] - SESSION_HOUR_OFFSET
            if target[offset] > POWER_TOL_KW:
                raise SessionError(
                    f"Session {row['session_id']} would charge in hour "
                    f"{steps[position]}, which another session of the same "
                    "vehicle already uses."
                )
            target[offset] = power_kw
            remaining -= power_kw * delta_t
        residuals.append(max(remaining, 0.0))
        if remaining > ENERGY_TOL_KWH:
            raise SessionInfeasible(
                {
                    "session_id": row["session_id"],
                    "site": row["site"],
                    "profile_id": row["profile_id"],
                    "first_hour": int(steps[0]),
                    "last_hour": int(steps[-1]),
                    "energy_kwh": round(float(row["energy_kwh"]), 9),
                    "capacity_kwh": round(
                        charger_kw * delta_t * float(fractions.sum()), 9
                    ),
                    "charger_kw": charger_kw,
                    "shortfall_kwh": round(float(remaining), 9),
                }
            )

    frame = pd.DataFrame(
        {key: values for key, values in columns.items()}, index=index
    )
    frame.columns = pd.MultiIndex.from_tuples(list(columns))
    return frame


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def hems_session_fractions(
    sessions: pd.DataFrame,
    hours: pd.DataFrame,
    *,
    factor: float,
    uncapped: pd.Series,
) -> pd.DataFrame:
    """Session hours with the HEMS charging limit in ``hems_available_fraction``.

    HEMS spreads each session over its stay: in every hour the charger may draw at
    most ``factor`` times the session's average required power (energy over
    connected hours), as a fraction of its rating, and never more than the
    connected fraction. Hours where ``uncapped`` (a boolean Series indexed by
    ``(site, t)``, e.g. local PV surplus) is True keep the connected fraction.
    ``factor >= 1`` keeps every session feasible: with ``c`` the capped fraction,
    ``min(f, c) >= c * f`` for ``f <= 1`` gives at least ``factor`` times the
    energy over the session.
    """
    if not factor >= 1.0:
        raise ValueError("The HEMS session power factor must be at least 1.")
    frame = hours.merge(
        sessions[["session_id", "site", "charger_kw", "energy_kwh"]],
        on="session_id", how="left", validate="many_to_one",
    )
    if frame["site"].isna().any():
        raise SessionError("Session hours reference unknown sessions.")
    available = pd.to_numeric(frame["available_fraction"], errors="raise").to_numpy(dtype=float)
    connected = frame.groupby("session_id")["available_fraction"].transform("sum").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(
            connected > 0,
            float(factor) * frame["energy_kwh"].to_numpy(dtype=float)
            / connected / frame["charger_kw"].to_numpy(dtype=float),
            0.0,
        )
    keys = pd.MultiIndex.from_arrays([frame["site"], frame["t"].astype(int)], names=["site", "t"])
    keep = uncapped.reindex(keys, fill_value=False).to_numpy(dtype=bool)
    result = hours.copy()
    result[HEMS_FRACTION_COLUMN] = np.where(keep, available, np.minimum(available, cap))
    return result


def _put_table(store, key, frame, columns):
    """Persist a table so that an *empty* one is still a readable key.

    HDF's ``table`` format silently writes nothing for an empty frame, which
    would make a valid zero-EV contract indistinguishable from a legacy input
    that never had one. The fixed format preserves the key and its columns.
    """
    frame = frame.reset_index(drop=True)
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
        store.put(key, frame, format="fixed")
        return
    store.put(key, frame, format="table")


def write_sessions(store, sessions: pd.DataFrame, hours: pd.DataFrame) -> None:
    """Write both tables into an open ``pd.HDFStore``.

    Writing an empty pair is meaningful: it declares that this input uses the
    dedicated session contract and contains no electric vehicles.
    """
    _put_table(store, SESSIONS_HDF_KEY, sessions, SESSION_COLUMNS)
    _put_table(store, SESSION_HOURS_HDF_KEY, hours, SESSION_HOUR_COLUMNS)


def read_sessions(hdf_path):
    """Read both tables from an HDF file; raises if they are absent."""
    with pd.HDFStore(hdf_path, mode="r") as store:
        keys = set(store.keys())
        if f"/{SESSIONS_HDF_KEY}" not in keys:
            raise SessionError(
                f"{hdf_path} does not contain '{SESSIONS_HDF_KEY}'. It predates the "
                "dedicated EV session contract and cannot be used for the "
                "full-year reference."
            )
        sessions = store[SESSIONS_HDF_KEY]
        hours = (
            store[SESSION_HOURS_HDF_KEY]
            if f"/{SESSION_HOURS_HDF_KEY}" in keys
            else pd.DataFrame(columns=SESSION_HOUR_COLUMNS)
        )
    if sessions.empty:
        sessions = pd.DataFrame(columns=SESSION_COLUMNS)
    if hours.empty:
        hours = pd.DataFrame(columns=SESSION_HOUR_COLUMNS)
    return sessions.reset_index(drop=True), hours.reset_index(drop=True)
