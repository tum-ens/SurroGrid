"""Residential profile selection by annual energy, GHD shapes and the DST shift of paired profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridexpand.allocation.functions.dst import dst_shift_output

MEASURED_PROFILE_SELECTION_CLOSEST = "closest"
MEASURED_PROFILE_SELECTION_RANDOM_BAND = "random_band"
MEASURED_PROFILE_SELECTION_CHOICES = (
    MEASURED_PROFILE_SELECTION_CLOSEST,
    MEASURED_PROFILE_SELECTION_RANDOM_BAND,
)
DEFAULT_MEASURED_PROFILE_BAND_PCT = 10.0
DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES = 10


def load_electricity_module():
    """Return Step 2's electricity helper module (paths in its config are absolute)."""
    import gridexpand.allocation.functions.electricity as electricity

    return electricity


def add_output_data_daylight_saving_shift(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Apply the civil-time DST exchange while preserving annual energy."""
    if df_ts.empty:
        return df_ts.copy()
    return dst_shift_output(df_ts, preserve_energy=True)


def select_residential_profile(
    lps_total_demands: pd.DataFrame,
    demand: float,
    rng: np.random.Generator,
    measured_profile_selection: str,
    measured_profile_band_pct: float,
    measured_profile_min_candidates: int,
) -> dict[str, Any]:
    profile_kwh = pd.to_numeric(lps_total_demands["kWh"], errors="coerce")
    distances = (profile_kwh - float(demand)).abs().dropna()
    if distances.empty:
        raise ValueError(
            "No residential electricity load profiles with valid annual kWh values are available."
        )

    if measured_profile_selection == MEASURED_PROFILE_SELECTION_CLOSEST:
        candidate_index = pd.Index([distances.idxmin()])
        method = "closest"
    elif measured_profile_selection == MEASURED_PROFILE_SELECTION_RANDOM_BAND:
        band_abs = abs(float(demand)) * float(measured_profile_band_pct) / 100.0
        candidate_index = distances[distances <= band_abs].index
        if len(candidate_index) < int(measured_profile_min_candidates):
            n_candidates = min(int(measured_profile_min_candidates), len(distances))
            candidate_index = distances.nsmallest(n_candidates).index
            method = "nearest_fallback"
        else:
            method = "band"
    else:
        raise ValueError(
            f"measured_profile_selection must be one of {MEASURED_PROFILE_SELECTION_CHOICES}, "
            f"got {measured_profile_selection!r}."
        )

    chosen_index = rng.choice(candidate_index.to_numpy())
    return {
        "chosen_index": chosen_index,
        "chosen_profile_device": lps_total_demands.loc[chosen_index, "devicenumber"],
        "chosen_profile_kwh": float(profile_kwh.loc[chosen_index]),
        "candidate_count": int(len(candidate_index)),
        "candidate_method": method,
        "candidate_min_kwh": float(profile_kwh.loc[candidate_index].min()),
        "candidate_max_kwh": float(profile_kwh.loc[candidate_index].max()),
    }


def ghd_shape_by_building_use(electricity_module) -> dict[str, pd.Series]:
    profiles = pd.read_csv(electricity_module.config.ELEC_GHD_PATH, skiprows=1)
    profiles = profiles.drop(columns=["Unnamed: 0"], errors="ignore")
    type_distribution = pd.read_csv(
        Path(electricity_module.config.ELEC_GHD_PATH).with_name(
            "nonresbuilding_usetype_distribution.csv"
        ),
        skiprows=1,
    )

    def weighted_shape(probability_column: str) -> pd.Series:
        weights = (
            type_distribution.set_index("type")[probability_column]
            .reindex(profiles.columns)
            .fillna(0.0)
        )
        if float(weights.sum()) <= 0.0:
            raise ValueError(f"No GHD type weights available for {probability_column}.")
        shape = profiles.mul(weights, axis=1).sum(axis=1)
        total = float(shape.sum())
        if total <= 0.0:
            raise ValueError(
                f"GHD shape for {probability_column} has no positive annual energy."
            )
        return shape / total

    commercial = weighted_shape("commercial_prob")
    return {
        "Public": weighted_shape("public_prob"),
        "Commercial": commercial,
        "Residential": commercial,
        "default": commercial,
    }
