"""Shared calibrated electricity profile generation for real SWF scenario plans."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import GRIDALLOC_DIR

PF_ELC = 0.959
MEASURED_PROFILE_SELECTION_CLOSEST = "closest"
MEASURED_PROFILE_SELECTION_RANDOM_BAND = "random_band"
MEASURED_PROFILE_SELECTION_CHOICES = (
    MEASURED_PROFILE_SELECTION_CLOSEST,
    MEASURED_PROFILE_SELECTION_RANDOM_BAND,
)
DEFAULT_MEASURED_PROFILE_BAND_PCT = 10.0
DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES = 10


def load_electricity_module():
    """Load gridalloc's electricity helper with its local config module."""
    demand_dir = GRIDALLOC_DIR.resolve()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(demand_dir))
    previous_config = sys.modules.pop("config", None)
    try:
        os.chdir(demand_dir)
        spec = importlib.util.spec_from_file_location(
            "gridalloc_electricity",
            demand_dir / "src" / "functions" / "electricity.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load gridalloc electricity module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _make_relevant_config_paths_absolute(module)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(demand_dir))
        except ValueError:
            pass
        if previous_config is not None:
            sys.modules["config"] = previous_config
    return module


def _make_relevant_config_paths_absolute(module) -> None:
    """Make paths used by the shared profile builder independent of caller cwd."""
    for attr in ("ELEC_LPS_PATH", "ELEC_GHD_PATH"):
        value = Path(getattr(module.config, attr))
        if not value.is_absolute():
            setattr(module.config, attr, str(GRIDALLOC_DIR / value))


def read_allocation_plan(
    path: Path, scope: str | None = "full_local_demand_recommended"
) -> pd.DataFrame:
    plan = pd.read_csv(path)
    required = {
        "lv_id",
        "allocation_bus",
        "residential_equivalent_hh_rows",
        "residential_equivalent_hh_annual_kwh",
        "calibrated_annual_ghd_kwh",
    }
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(
            f"Allocation plan {path} is missing required columns: {sorted(missing)}"
        )
    if scope and "scenario_scope" in plan.columns:
        plan = plan[plan["scenario_scope"].astype(str).eq(scope)].copy()
    if plan.empty:
        raise ValueError(
            f"Allocation plan {path} contains no rows for scope={scope!r}."
        )
    plan["lv_id"] = plan["lv_id"].astype(int)
    plan["allocation_bus"] = plan["allocation_bus"].astype(int)
    return plan


def add_output_data_daylight_saving_shift(df_ts: pd.DataFrame) -> pd.DataFrame:
    if df_ts.empty:
        return df_ts.copy()
    ts_hour1 = 2090
    ts_hour2 = 7130
    df_ts = df_ts.copy()
    new_row = df_ts.iloc[ts_hour2].copy()
    new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
    df_ts = pd.concat(
        [df_ts.iloc[: ts_hour2 + 1], new_row_df, df_ts.iloc[ts_hour2 + 1 :]]
    ).reset_index(drop=True)
    return df_ts.drop(index=ts_hour1).reset_index(drop=True)


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


def build_scenario_base_electric_demand(
    allocation: pd.DataFrame,
    *,
    seed: int,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    include_reactive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if measured_profile_selection not in MEASURED_PROFILE_SELECTION_CHOICES:
        raise ValueError(
            f"measured_profile_selection must be one of {MEASURED_PROFILE_SELECTION_CHOICES}, "
            f"got {measured_profile_selection!r}."
        )
    if measured_profile_band_pct <= 0:
        raise ValueError("measured_profile_band_pct must be greater than zero.")
    if measured_profile_min_candidates <= 0:
        raise ValueError("measured_profile_min_candidates must be greater than zero.")

    electricity_module = load_electricity_module()
    df_normalized_lps_res = pd.read_hdf(
        electricity_module.config.ELEC_LPS_PATH, key="df_normalized_scaled"
    )
    lps_total_demands = pd.read_hdf(
        electricity_module.config.ELEC_LPS_PATH, key="df_sums"
    )
    ghd_shapes = ghd_shape_by_building_use(electricity_module)
    rng = np.random.default_rng(int(seed) + 2_000_003)

    active_by_bus: dict[int, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []

    for row in allocation.to_dict("records"):
        bus = int(row["allocation_bus"])
        hh_rows = int(
            round(float(row.get("residential_equivalent_hh_rows", 0.0) or 0.0))
        )
        hh_annual_kwh = float(
            row.get("residential_equivalent_hh_annual_kwh", 0.0) or 0.0
        )
        if hh_rows > 0 and hh_annual_kwh > 0.0:
            per_hh_kwh = hh_annual_kwh / hh_rows
            for sequence in range(hh_rows):
                selected = select_residential_profile(
                    lps_total_demands,
                    per_hh_kwh,
                    rng,
                    measured_profile_selection,
                    measured_profile_band_pct,
                    measured_profile_min_candidates,
                )
                profile = (
                    df_normalized_lps_res[selected["chosen_profile_device"]]
                    * per_hh_kwh
                ).reset_index(drop=True)
                active_by_bus[bus] = active_by_bus.get(bus, 0.0) + profile
                audit_rows.append(
                    {
                        "allocation_bus": bus,
                        "building_match_id": row.get("building_match_id"),
                        "demand_class": "HH",
                        "profile_sequence": sequence,
                        "annual_demand_kwh": per_hh_kwh,
                        **{
                            key: value
                            for key, value in selected.items()
                            if key != "chosen_index"
                        },
                    }
                )

        ghd_annual_kwh = float(row.get("calibrated_annual_ghd_kwh", 0.0) or 0.0)
        if ghd_annual_kwh > 0.0:
            building_use = str(row.get("building_use") or "default")
            shape = ghd_shapes.get(building_use, ghd_shapes["default"])
            profile = (shape * ghd_annual_kwh).reset_index(drop=True)
            active_by_bus[bus] = active_by_bus.get(bus, 0.0) + profile
            audit_rows.append(
                {
                    "allocation_bus": bus,
                    "building_match_id": row.get("building_match_id"),
                    "demand_class": "GHD",
                    "profile_sequence": 0,
                    "annual_demand_kwh": ghd_annual_kwh,
                    "chosen_profile_device": f"weighted_{building_use if building_use in ghd_shapes else 'Commercial'}_ghd",
                    "chosen_profile_kwh": ghd_annual_kwh,
                    "candidate_count": np.nan,
                    "candidate_method": "weighted_ghd_shape",
                    "candidate_min_kwh": np.nan,
                    "candidate_max_kwh": np.nan,
                }
            )

    if not active_by_bus:
        raise ValueError("Allocation plan produced no active electricity demand.")

    df_elec = pd.DataFrame(active_by_bus).reset_index(drop=True)
    df_elec.columns = pd.MultiIndex.from_tuples(
        [(int(bus), "electricity") for bus in df_elec.columns],
        names=["bus", "component"],
    )
    df_elec = add_output_data_daylight_saving_shift(df_elec)
    if not include_reactive:
        return df_elec.sort_index(axis=1, level=[0, 1]), pd.DataFrame(audit_rows)

    q_factor = -math.tan(math.acos(PF_ELC))
    df_reactive = df_elec.copy() * q_factor
    df_reactive.columns = pd.MultiIndex.from_tuples(
        [(bus, "electricity-reactive") for bus, _ in df_reactive.columns],
        names=["bus", "component"],
    )
    demand = pd.concat([df_elec, df_reactive], axis=1).sort_index(axis=1, level=[0, 1])
    return demand, pd.DataFrame(audit_rows)


def profile_selection_summary(demand_audit: pd.DataFrame) -> dict[str, Any]:
    if "chosen_profile_device" not in demand_audit.columns:
        return {}
    chosen = demand_audit["chosen_profile_device"].dropna().astype(str)
    if chosen.empty:
        return {}
    counts = chosen.value_counts()
    method_counts = (
        demand_audit.get("candidate_method", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .value_counts()
    )
    return {
        "measured_profile_unique_devices": int(counts.size),
        "measured_profile_largest_reuse_count": int(counts.iloc[0]),
        "measured_profile_largest_reuse_share": float(counts.iloc[0] / len(chosen)),
        "measured_profile_top5_reuse_share": float(counts.head(5).sum() / len(chosen)),
        "measured_profile_band_candidate_rows": int(method_counts.get("band", 0)),
        "measured_profile_nearest_fallback_rows": int(
            method_counts.get("nearest_fallback", 0)
        ),
    }
