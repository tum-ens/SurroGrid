"""Shared invariants for reproducible scenario-profile allocation."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd

from common.reproducibility import stable_seed


def profile_key(row: pd.Series | dict, *parts: object) -> tuple[object, ...]:
    """Identify one physical allocation independently of its target network bus."""
    getter = row.get
    building = getter("building_objectid")
    if building is None or pd.isna(building):
        building = getter("building_match_id")
    source_grid = getter("source_lv_id", getter("lv_id"))
    return (source_grid, building, *parts)


def sum_series_by_key(
    entries: Iterable[tuple[Hashable, pd.Series]],
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Sum time series sharing one output key without overwriting entries."""
    totals: dict[Hashable, pd.Series] = {}
    for key, values in entries:
        series = (
            pd.to_numeric(values, errors="coerce").fillna(0.0).reset_index(drop=True)
        )
        totals[key] = (
            series if key not in totals else totals[key].add(series, fill_value=0.0)
        )
    if not totals:
        return pd.DataFrame(index=index)
    frame = pd.DataFrame(totals)
    if index is not None:
        if len(frame) != len(index):
            raise ValueError(
                f"Time-series length {len(frame)} does not match expected length {len(index)}."
            )
        frame.index = index
    return frame


def assert_unique_columns(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate columns after aggregation: {duplicates[:10]}"
        )


def assert_energy_conserved(
    frame: pd.DataFrame,
    expected_kwh: float,
    *,
    component: str = "electricity",
    label: str,
    atol_kwh: float = 1e-3,
) -> None:
    if getattr(frame.columns, "nlevels", 1) < 2:
        raise ValueError(f"{label} must use (bus, component) MultiIndex columns.")
    columns = frame.columns.get_level_values(1).astype(str) == component
    generated_kwh = float(frame.loc[:, columns].sum().sum())
    if not np.isclose(
        generated_kwh, float(expected_kwh), rtol=0.0, atol=float(atol_kwh)
    ):
        raise ValueError(
            f"{label} does not conserve annual energy: expected={expected_kwh:.6f} kWh, "
            f"generated={generated_kwh:.6f} kWh."
        )


def assert_paired_plan_equivalence(
    real_plan: pd.DataFrame,
    synthetic_plan: pd.DataFrame,
) -> None:
    """Require both target networks to carry the same physical scenario."""
    real_buildings = set(real_plan["building_objectid"].dropna().astype(str))
    synthetic_buildings = set(synthetic_plan["building_objectid"].dropna().astype(str))
    if real_buildings != synthetic_buildings:
        only_real = sorted(real_buildings - synthetic_buildings)[:10]
        only_synthetic = sorted(synthetic_buildings - real_buildings)[:10]
        raise ValueError(
            "Paired plans contain different physical buildings: "
            f"only_real={only_real}, only_synthetic={only_synthetic}."
        )

    demand_columns = (
        "residential_equivalent_hh_rows",
        "residential_equivalent_hh_annual_kwh",
        "calibrated_annual_ghd_kwh",
    )
    for column in demand_columns:
        real_value = float(
            pd.to_numeric(real_plan[column], errors="coerce").fillna(0.0).sum()
        )
        synthetic_value = float(
            pd.to_numeric(synthetic_plan[column], errors="coerce").fillna(0.0).sum()
        )
        if not np.isclose(real_value, synthetic_value, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Paired plans differ in {column}: "
                f"real={real_value:.6f}, synthetic={synthetic_value:.6f}."
            )

    if "pv_roof_capacity_kw" in real_plan and "pv_roof_capacity_kw" in synthetic_plan:
        real_capacity = float(
            pd.to_numeric(real_plan["pv_roof_capacity_kw"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        synthetic_capacity = float(
            pd.to_numeric(synthetic_plan["pv_roof_capacity_kw"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        if not np.isclose(real_capacity, synthetic_capacity, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Paired plans differ in LoD2 PV roof capacity: "
                f"real={real_capacity:.6f}, synthetic={synthetic_capacity:.6f}."
            )
        real_units = set(
            real_plan.loc[
                real_plan["pv_roof_eligible"].astype(bool), "scenario_unit_id"
            ].astype(int)
        )
        synthetic_units = set(
            synthetic_plan.loc[
                synthetic_plan["pv_roof_eligible"].astype(bool), "scenario_unit_id"
            ].astype(int)
        )
        if real_units != synthetic_units:
            raise ValueError("Paired plans select different PV scenario units.")
