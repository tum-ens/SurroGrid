"""Transparent building-level sizing rules for residential heat assets."""

from __future__ import annotations

import numpy as np
import pandas as pd

WATER_HEAT_CAPACITY_WH_PER_L_K = 1.163


def calculate_full_load_hours(
    ambient_temperature_c,
    *,
    indoor_design_temperature_c: float,
    heating_limit_temperature_c: float,
    norm_outside_temperature_c: float,
) -> float:
    """Calculate VDI-style 20/15 full-load hours from daily mean weather."""
    ambient = pd.Series(ambient_temperature_c, dtype=float).reset_index(drop=True)
    if ambient.empty or ambient.isna().any():
        raise ValueError("Full-load-hour calculation requires complete ambient temperatures.")
    if len(ambient) % 24:
        raise ValueError("Full-load-hour calculation requires complete 24-hour days.")
    if norm_outside_temperature_c >= indoor_design_temperature_c:
        raise ValueError("Norm outside temperature must be below indoor design temperature.")
    daily_mean = ambient.groupby(np.arange(len(ambient)) // 24).mean()
    heating_days = daily_mean[daily_mean < heating_limit_temperature_c]
    degree_days_kd = (indoor_design_temperature_c - heating_days).sum()
    result = 24.0 * degree_days_kd / (
        indoor_design_temperature_c - norm_outside_temperature_c
    )
    if result <= 0.0:
        raise ValueError("Calculated heat full-load hours must be positive.")
    return float(result)


def _column(frame: pd.DataFrame, site: int, commodity: str) -> pd.Series:
    key = (site, commodity)
    if key not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").fillna(0.0)


def build_heat_asset_plan(
    buildings: pd.DataFrame,
    space_heat: pd.DataFrame,
    water_heat: pd.DataFrame,
    heat_pump_cop: pd.DataFrame,
    ambient_temperature_c,
    *,
    sizing_method: str,
    norm_outside_temperature_c: float,
    indoor_design_temperature_c: float,
    heating_limit_temperature_c: float,
    heat_pump_design_share: float,
    buffer_volume_l_per_kw_th: float,
    buffer_usable_temperature_spread_k: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compile one central residential heat-system plan per physical building."""
    required = {"building_objectid", "Site"}
    missing = required.difference(buildings.columns)
    if missing:
        raise ValueError(f"Heat sizing buildings lack columns: {sorted(missing)}")
    if sizing_method not in {"full_load_hours_rule", "optimization"}:
        raise ValueError(f"Unknown heat sizing method {sizing_method!r}.")
    source = buildings.copy()
    source["building_objectid"] = source["building_objectid"].astype(str)
    source["Site"] = pd.to_numeric(source["Site"], errors="raise").astype(int)
    if source["building_objectid"].duplicated().any():
        raise ValueError("Heat sizing requires one row per physical building.")
    if source["Site"].duplicated().any():
        raise ValueError("Heat sizing requires one central heat site per physical building.")

    ambient = pd.Series(ambient_temperature_c, dtype=float).reset_index(drop=True)
    full_load_hours = calculate_full_load_hours(
        ambient,
        indoor_design_temperature_c=indoor_design_temperature_c,
        heating_limit_temperature_c=heating_limit_temperature_c,
        norm_outside_temperature_c=norm_outside_temperature_c,
    )
    design_index = int((ambient - norm_outside_temperature_c).abs().idxmin())
    fixed = sizing_method == "full_load_hours_rule"
    rows = []
    for building in source.to_dict("records"):
        site = int(building["Site"])
        space = _column(space_heat, site, "space_heat")
        water = _column(water_heat, site, "water_heat")
        cop = _column(heat_pump_cop, site, "heatpump_air").clip(lower=0.1)
        if len(space) != len(ambient) or len(water) != len(ambient) or len(cop) != len(ambient):
            raise ValueError(f"Heat sizing timeseries length mismatch at site {site}.")
        annual_space = float(space.sum())
        annual_water = float(water.sum())
        space_design_kw_th = annual_space / full_load_hours
        dhw_allowance_kw_th = annual_water / len(water)
        design_kw_th = space_design_kw_th + dhw_allowance_kw_th
        design_cop = float(cop.iloc[design_index])
        heuristic_hp_kw_th = heat_pump_design_share * design_kw_th
        maximum_hp_kw_th = design_kw_th
        heuristic_hp_kw_el = heuristic_hp_kw_th / design_cop
        maximum_hp_kw_el = maximum_hp_kw_th / design_cop
        residual = (space + water - cop * heuristic_hp_kw_el).clip(lower=0.0)
        heuristic_aux_kw_el = float(residual.max())
        maximum_aux_kw_el = float((space + water).max())
        reference_hp_kw_th = heuristic_hp_kw_th if fixed else maximum_hp_kw_th
        buffer_l = buffer_volume_l_per_kw_th * reference_hp_kw_th
        buffer_kwh = (
            buffer_l * WATER_HEAT_CAPACITY_WH_PER_L_K
            * buffer_usable_temperature_spread_k / 1000.0
        )
        households = pd.to_numeric(
            pd.Series([building.get("number_of_households")]), errors="coerce"
        ).iloc[0]
        households = float(households) if pd.notna(households) and households > 0.0 else np.nan
        planned_hp_kw_el = heuristic_hp_kw_el if fixed else maximum_hp_kw_el
        planned_aux_kw_el = heuristic_aux_kw_el if fixed else maximum_aux_kw_el
        peak_coverage_margin = cop * planned_hp_kw_el + planned_aux_kw_el - space - water
        hp_installed_kw_el = heuristic_hp_kw_el if fixed else 0.0
        hp_upper_kw_el = heuristic_hp_kw_el if fixed else maximum_hp_kw_el
        aux_installed_kw_el = heuristic_aux_kw_el if fixed else 0.0
        aux_upper_kw_el = heuristic_aux_kw_el if fixed else maximum_aux_kw_el
        floor_area = pd.to_numeric(pd.Series([building.get("floor_area")]), errors="coerce").iloc[0]
        rows.append({
            **building,
            "annual_space_heat_kwh": annual_space,
            "annual_water_heat_kwh": annual_water,
            "full_load_hours_h": full_load_hours,
            "norm_outside_temperature_c": float(norm_outside_temperature_c),
            "heating_limit_temperature_c": float(heating_limit_temperature_c),
            "indoor_design_temperature_c": float(indoor_design_temperature_c),
            "design_weather_index": design_index,
            "design_weather_temperature_c": float(ambient.iloc[design_index]),
            "design_cop": design_cop,
            "space_design_load_kw_th": space_design_kw_th,
            "dhw_design_allowance_kw_th": dhw_allowance_kw_th,
            "total_design_load_kw_th": design_kw_th,
            "specific_design_load_w_per_m2": (
                design_kw_th * 1000.0 / float(floor_area)
                if pd.notna(floor_area) and float(floor_area) > 0.0 else np.nan
            ),
            "heat_pump_design_share": float(heat_pump_design_share),
            "heat_pump_installed_kw_el": hp_installed_kw_el,
            "heat_pump_capacity_upper_kw_el": hp_upper_kw_el,
            "heat_pump_reference_kw_th": reference_hp_kw_th,
            "auxiliary_installed_kw_el": aux_installed_kw_el,
            "auxiliary_capacity_upper_kw_el": aux_upper_kw_el,
            "number_of_households": households,
            "heat_pump_reference_kw_th_per_household": (
                reference_hp_kw_th / households if pd.notna(households) else np.nan
            ),
            "auxiliary_reference_kw_el_per_household": (
                planned_aux_kw_el / households if pd.notna(households) else np.nan
            ),
            "buffer_volume_l": buffer_l,
            "buffer_volume_m3": buffer_l / 1000.0,
            "buffer_l_per_hp_reference_kw_th": (
                buffer_l / reference_hp_kw_th if reference_hp_kw_th > 0.0 else np.nan
            ),
            "buffer_kwh_per_hp_reference_kw_th": (
                buffer_kwh / reference_hp_kw_th if reference_hp_kw_th > 0.0 else np.nan
            ),
            "buffer_l_per_household": (
                buffer_l / households if pd.notna(households) else np.nan
            ),
            "heat_capacity_peak_coverage_margin_kw_th": float(peak_coverage_margin.min()),
            "heat_capacity_peak_coverage_valid": bool(peak_coverage_margin.min() >= -1e-9),
            "buffer_rule_valid": bool(
                abs(buffer_l - buffer_volume_l_per_kw_th * reference_hp_kw_th) <= 1e-9
            ),
            "buffer_installed_kwh_th": buffer_kwh if fixed else 0.0,
            "buffer_capacity_upper_kwh_th": buffer_kwh,
            "buffer_installed_power_kw_th": reference_hp_kw_th if fixed else 0.0,
            "buffer_power_upper_kw_th": reference_hp_kw_th,
            "heat_conversion_capacity_kw_th": max(
                maximum_aux_kw_el,
                maximum_hp_kw_el * float(cop.max()) + maximum_aux_kw_el,
            ),
            "heat_sizing_method": sizing_method,
            "dhw_representation": "opendhw_hourly_direct_demand",
            "buffer_representation": "space_heat_only",
        })
    climate = {
        "full_load_hours_h": full_load_hours,
        "norm_outside_temperature_c": float(norm_outside_temperature_c),
        "heating_limit_temperature_c": float(heating_limit_temperature_c),
        "indoor_design_temperature_c": float(indoor_design_temperature_c),
    }
    return pd.DataFrame(rows), climate
