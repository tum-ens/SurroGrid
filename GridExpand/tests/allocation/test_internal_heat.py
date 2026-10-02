"""SI units, once-only thermal area, physical identity and independent dispatch."""

import numpy as np
import pandas as pd
import pytest
from gridexpand.allocation.assets.heat.internal import (
    normalized_parameters,
    service_cops,
)
from gridexpand.scenario.scenario_config import InternalHeatConfig, HeatSizingConfig
from gridexpand.powerflow.demands import _inflex_internal_heat


def settings(uplift=2):
    return HeatSizingConfig(
        "internal",
        20,
        15,
        0.65,
        20,
        5,
        internal=InternalHeatConfig(hems_preheat_uplift_k=uplift),
    )


def rc_fixture():
    return pd.DataFrame(
        [
            dict(
                building_objectid="A",
                resistance=0.005,
                capacitance=72e6,
                source_footprint_m2=100,
                source_floor_number=2,
                source_window_area_m2=30,
            )
        ]
    )


def test_heated_area_and_si_units_are_applied_once():
    buildings = pd.DataFrame(
        [dict(building_objectid="A", Site=1, residential_effective_floor_area_m2=200)]
    )
    params = normalized_parameters(rc_fixture(), buildings, settings()).iloc[0]
    assert params.thermal_area_m2 == 160
    assert params.transmission_conductance_kw_per_k == 0.2
    assert params.capacitance_kwh_per_k == 20
    assert params.ventilation_conductance_kw_per_k == pytest.approx(
        1.2 * 1000 * 0.5 * 3.125 * 160 / 3600 / 1000
    )
    buildings["residential_effective_floor_area_m2"] = 120
    mixed = normalized_parameters(rc_fixture(), buildings, settings()).iloc[0]
    assert mixed.thermal_area_m2 == 96
    assert mixed.capacitance_kwh_per_k == 12
    assert mixed.transmission_conductance_kw_per_k == 0.12


def test_parameter_identity_omits_target_topology():
    b = pd.DataFrame(
        [
            dict(
                building_objectid="A",
                Site=1,
                residential_effective_floor_area_m2=200,
                target_network="real",
                allocation_bus=5,
            )
        ]
    )
    real = normalized_parameters(rc_fixture(), b, settings())
    b["target_network"] = "synthetic"
    b["allocation_bus"] = 999
    b["Site"] = 2
    synthetic = normalized_parameters(rc_fixture(), b, settings())
    pd.testing.assert_frame_equal(
        real.drop(columns="Site"), synthetic.drop(columns="Site")
    )


def test_buffer_cop_penalty_does_not_change_direct_services():
    space, water, buffer = service_cops("radiator", [-10, 0, 10], 10.0)
    assert space.shape == (3,)
    assert (buffer < space).all()
    assert (water > 0).all()


def test_buffer_cop_uplift_follows_the_usable_spread():
    space, _, small = service_cops("radiator", [-10, 0, 10], 5.0)
    _, _, large = service_cops("radiator", [-10, 0, 10], 10.0)
    assert (large < small).all() and (small < space).all()


def test_inflex_uses_reference_not_hems_dispatch():
    reference = pd.DataFrame(
        {
            ("A", "space_heat_kw"): [4.0, 4.0],
            ("A", "water_heat_kw"): [3.0, 3.0],
            ("A", "space_cop"): [4.0, 4.0],
            ("A", "water_cop"): [2.0, 2.0],
        }
    )
    inputs = {
        "thermal_parameters": pd.DataFrame([dict(Site=1, building_objectid="A")]),
        "internal_heat_reference": reference,
        "process": pd.DataFrame(
            [
                dict(Site=1, Process="heatpump_air", **{"inst-cap": 1.5}),
                dict(Site=1, Process="heatpump_booster", **{"inst-cap": 2}),
            ]
        ),
        "hems_temperatures": [22, 22],
    }
    total, hp, rod = _inflex_internal_heat(inputs, pd.RangeIndex(2))
    np.testing.assert_allclose(hp, 1.5)
    np.testing.assert_allclose(rod, 2)
    inputs["hems_temperatures"] = [20, 20]
    pd.testing.assert_frame_equal(
        total, _inflex_internal_heat(inputs, pd.RangeIndex(2))[0]
    )


def test_internal_heat_rejects_the_cop_curve_charge_efficiency():
    raw = {
        "space_heat_source": "internal",
        "indoor_design_temperature_c": 20.0,
        "heating_limit_temperature_c": 15.0,
        "heat_pump_design_share": 0.65,
        "buffer_volume_l_per_kw_th": 86.0,
        "buffer_usable_temperature_spread_k": 10.0,
        "buffer_charge_efficiency_method": "cop_curve",
    }
    with pytest.raises(ValueError, match="own COP route"):
        HeatSizingConfig.from_dict(raw)
    config = HeatSizingConfig.from_dict({**raw, "buffer_charge_efficiency_method": "technology"})
    assert config.space_heat_source == "internal"


def test_internal_tank_has_no_second_charging_penalty():
    from gridexpand.allocation.assets.heat.internal import materialize_internal_assets
    from gridexpand.paths import SCENARIO_CONFIG_DIR
    from gridexpand.scenario.config_loader import load_scenario_config

    scenario, _ = load_scenario_config(SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml")
    plan = pd.DataFrame([dict(
        building_objectid="A", Site=1,
        heat_pump_installed_kw_el=1.0, heat_pump_capacity_upper_kw_el=1.0,
        auxiliary_installed_kw_el=3.0, auxiliary_capacity_upper_kw_el=3.0,
        heat_conversion_capacity_kw_th=10.0,
        buffer_installed_kwh_th=3.0, buffer_capacity_upper_kwh_th=3.0,
        buffer_installed_power_kw_th=3.0, buffer_power_upper_kw_th=3.0,
    )])
    *_, storage = materialize_internal_assets(plan, scenario.technologies, "full_load_hours_rule")
    thermal = scenario.technologies.storages["thermal_storage"]
    assert thermal["charge_efficiency"] < 1.0
    assert storage["eff-in"].tolist() == [1.0]
    assert storage["discharge"].tolist() == [thermal["self_discharge_per_timestep"]]
