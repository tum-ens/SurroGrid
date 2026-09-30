"""Heat sizing rules: degree-day base, buffer volume, COP-curve charge efficiency."""

import numpy as np
import pandas as pd
import pytest

from gridexpand.allocation.assets.heat.materialization import materialize_heat_urbs_inputs
from gridexpand.allocation.assets.heat.sizing import (
    WATER_HEAT_CAPACITY_WH_PER_L_K,
    buffer_charge_efficiency,
    build_heat_asset_plan,
    calculate_full_load_hours,
)
from gridexpand.allocation.config import config
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config

HOURS = 8760


def _ambient():
    # One cold week, then a mild year: daily means of -10 degC and 5 degC below the limit.
    days = np.r_[np.full(7, -10.0), np.full(358, 5.0)]
    return np.repeat(days, 24)


def _cop_for_lift(lift):
    return config.ASHP_COP(np.asarray(lift, dtype=float)).to_numpy().ravel()


def test_default_base_is_the_indoor_temperature():
    ambient = _ambient()
    kwargs = dict(indoor_design_temperature_c=20.0, heating_limit_temperature_c=15.0, norm_outside_temperature_c=-12.0)
    default = calculate_full_load_hours(ambient, **kwargs)
    explicit = calculate_full_load_hours(ambient, degree_day_base_temperature_c=20.0, **kwargs)
    assert default == pytest.approx(explicit)
    assert default == pytest.approx(24.0 * (7 * 30.0 + 358 * 15.0) / 32.0)


def test_heating_limit_base_gives_heizgradtage_g15():
    ambient = _ambient()
    hours = calculate_full_load_hours(
        ambient, indoor_design_temperature_c=20.0, heating_limit_temperature_c=15.0,
        norm_outside_temperature_c=-12.0, degree_day_base_temperature_c=15.0,
    )
    assert hours == pytest.approx(24.0 * (7 * 25.0 + 358 * 10.0) / 32.0)


@pytest.mark.parametrize("base", [14.0, 21.0])
def test_degree_day_base_must_lie_between_limit_and_indoor(base):
    with pytest.raises(ValueError, match="Degree-day base"):
        calculate_full_load_hours(
            _ambient(), indoor_design_temperature_c=20.0, heating_limit_temperature_c=15.0,
            norm_outside_temperature_c=-12.0, degree_day_base_temperature_c=base,
        )


def test_charge_efficiency_is_the_cop_ratio_on_the_curve():
    lift = np.full(24, 40.0)
    cop = pd.Series(_cop_for_lift(lift))
    space = pd.Series(np.ones(24))
    expected = _cop_for_lift([50.0])[0] / _cop_for_lift([40.0])[0]
    assert buffer_charge_efficiency(cop, space, 10.0) == pytest.approx(expected)
    assert 0.8 < expected < 0.9


def test_charge_efficiency_is_weighted_by_space_heat_and_one_without_it():
    cop = pd.Series(_cop_for_lift([30.0, 60.0]))
    space = pd.Series([0.0, 2.0])
    expected = _cop_for_lift([70.0])[0] / _cop_for_lift([60.0])[0]
    assert buffer_charge_efficiency(cop, space, 10.0) == pytest.approx(expected)
    assert buffer_charge_efficiency(cop, pd.Series([0.0, 0.0]), 10.0) == 1.0


def test_charge_efficiency_stays_on_the_decreasing_branch():
    # A COP below the curve minimum maps to 90 K; the raised lift is capped there too.
    cop = pd.Series([1.5, 1.5])
    assert buffer_charge_efficiency(cop, pd.Series([1.0, 1.0]), 10.0) == pytest.approx(1.0)


def _plan(**overrides):
    index = pd.RangeIndex(HOURS)
    ambient = pd.Series(np.tile(np.r_[np.full(12, -12.0), np.full(12, 4.0)], HOURS // 24))
    space = pd.DataFrame({(1, "space_heat"): np.where(ambient < 0, 6.0, 2.0)}, index=index)
    water = pd.DataFrame({(1, "water_heat"): np.full(HOURS, 0.2)}, index=index)
    lift = np.where(ambient < 0, 40 - 2 * ambient, 40 - 2 * ambient).clip(15)
    cop = pd.DataFrame({(1, "heatpump_air"): _cop_for_lift(lift)}, index=index)
    buildings = pd.DataFrame({"building_objectid": ["b1"], "Site": [1], "floor_area": [150.0]})
    kwargs = dict(
        sizing_method="full_load_hours_rule", norm_outside_temperature_c=-12.0,
        indoor_design_temperature_c=20.0, heating_limit_temperature_c=15.0,
        heat_pump_design_share=0.65, buffer_volume_l_per_kw_th=86.0,
        buffer_usable_temperature_spread_k=10.0,
    )
    kwargs.update(overrides)
    plan, climate = build_heat_asset_plan(buildings, space, water, cop, ambient, **kwargs)
    return plan.iloc[0], climate


def test_plan_records_the_degree_day_base_and_a_smaller_flh_with_g15():
    default, _ = _plan()
    g15, climate = _plan(degree_day_base_temperature_c=15.0)
    assert default["degree_day_base_temperature_c"] == 20.0
    assert g15["degree_day_base_temperature_c"] == climate["degree_day_base_temperature_c"] == 15.0
    assert g15["full_load_hours_h"] < default["full_load_hours_h"]
    assert g15["heat_pump_installed_kw_el"] > default["heat_pump_installed_kw_el"]


def test_one_hour_buffer_volume_stores_one_hour_of_thermal_output():
    one_hour = 1000.0 / (WATER_HEAT_CAPACITY_WH_PER_L_K * 10.0)
    row, _ = _plan(buffer_volume_l_per_kw_th=one_hour)
    assert row["buffer_bridging_hours"] == pytest.approx(1.0)
    assert row["buffer_installed_kwh_th"] == pytest.approx(row["heat_pump_reference_kw_th"])


def test_charge_efficiency_method_sets_the_storage_row():
    scenario, _ = load_scenario_config(SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml")
    technologies = scenario.technologies
    technology, _ = _plan()
    curve, _ = _plan(buffer_charge_efficiency_method="cop_curve")
    assert np.isnan(technology["buffer_charge_efficiency"])
    assert 0.8 < curve["buffer_charge_efficiency"] < 0.95

    def storage_row(row):
        inputs = materialize_heat_urbs_inputs(
            pd.DataFrame([row]), sizing_method="full_load_hours_rule",
            process_parameters=technologies.processes,
            storage_parameters=technologies.storages["thermal_storage"],
        )
        return inputs.storage.iloc[0]

    thermal = technologies.storages["thermal_storage"]
    assert storage_row(technology)["eff-in"] == thermal["charge_efficiency"]
    assert storage_row(curve)["eff-in"] == pytest.approx(curve["buffer_charge_efficiency"])
    assert storage_row(curve)["discharge"] == thermal["self_discharge_per_timestep"] == 0.005


def test_unknown_charge_efficiency_method_is_rejected():
    with pytest.raises(ValueError, match="charge-efficiency method"):
        _plan(buffer_charge_efficiency_method="constant")
