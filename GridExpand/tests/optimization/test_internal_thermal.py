"""Physical 1R1C checks and independent Pyomo/Linopy equivalence."""

import copy
import numpy as np
import pandas as pd
import pytest
from lp_fixture import write_input
from gridexpand.common.thermal import thermostat_reference, dispatch_heat_services


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_implicit_euler_and_gains(dt):
    # A periodic thermostat at 20 C requires H*(20-0)-gains = 1 kW.
    heat, temp, initial = thermostat_reference(
        np.zeros(24),
        np.ones(24),
        conductance_kw_per_k=0.1,
        capacitance_kwh_per_k=2,
        delta_t_hours=dt,
    )
    np.testing.assert_allclose(heat, 1)
    np.testing.assert_allclose(temp, 20)
    assert initial == 20
    heat, temp, initial = thermostat_reference(
        np.full(24, 30),
        np.zeros(24),
        conductance_kw_per_k=0.1,
        capacitance_kwh_per_k=2,
        delta_t_hours=dt,
    )
    assert heat.sum() == 0
    previous = np.r_[initial, temp[:-1]]
    np.testing.assert_allclose(
        temp, (2 / dt * previous + 0.1 * 30) / (2 / dt + 0.1), atol=1e-9
    )


def test_service_capacity_is_shared():
    hp, rod = dispatch_heat_services([[4, 3]], [[4, 2]], 1.5)
    np.testing.assert_allclose(hp, [[1, 0.5]])
    np.testing.assert_allclose(rod, [[0, 2]])
    assert hp.sum() == 1.5


def thermal_fixture(path, uplift=2, passive=False):
    from gridexpand.optimization.urbs.input import read_input_h5
    from gridexpand.optimization.urbs.scenarios import insert_scenario
    from gridexpand.optimization.urbs.identify import identify_mode

    write_input(path)
    with pd.HDFStore(path, "a") as store:
        pro = store["urbs_in/process"]
        pro = pro[~pro.Process.str.startswith("Rooftop")].copy()
        pro["inst-cap"] = pro["cap-up"]
        pro["inv-cost-fix"] = 0
        pro["inv-cost"] = 0
        store["urbs_in/process"] = pro
        store["urbs_in/storage"] = store["urbs_in/storage"].iloc[:0]
        store["urbs_in/supim"] = store["urbs_in/supim"].iloc[:, :0]
        pars = []
        series = {}
        for site, bid in [(1, "A"), (2, "B")]:
            outside = np.full(6, 0.0)
            gains = np.full(6, 3.0) if passive else np.zeros(6)
            heat, ref, initial = thermostat_reference(
                outside, gains, conductance_kw_per_k=0.1, capacitance_kwh_per_k=1
            )
            pars.append(
                dict(
                    Site=site,
                    building_objectid=bid,
                    heat_commodity="space_heat",
                    conductance_kw_per_k=0.1,
                    capacitance_kwh_per_k=1,
                    initial_temperature_c=initial,
                    terminal_temperature_c=initial,
                    room_heat_upper_kw=50,
                )
            )
            for field, values in dict(
                outside_temperature_c=outside,
                internal_gains_kw=gains,
                solar_gains_kw=np.zeros(6),
                minimum_temperature_c=np.full(6, 20),
                upper_temperature_c=np.maximum(20 + uplift, ref),
            ).items():
                series[bid, field] = values
        store["urbs_in/building_thermal_parameters"] = pd.DataFrame(pars)
        store["urbs_in/building_thermal_timeseries"] = pd.DataFrame(series)
        # Fixed demand is deliberately nonzero: thermal mode must replace it.
        demand = store["urbs_in/demand"]
        demand.loc[:, pd.IndexSlice[:, "space_heat"]] = 999
        store["urbs_in/demand"] = demand
    data = read_input_h5(path)
    settings = {"tsam": False, "hoursPerPeriod": 168, "dt": 1, "timesteps": range(7)}
    data = insert_scenario(data, settings)
    return data, identify_mode(data), settings


@pytest.mark.parametrize("uplift,passive", [(0, False), (2, False), (2, True)])
def test_backends_agree(tmp_path, uplift, passive):
    from pyomo.environ import SolverFactory
    from pyomo.opt import check_optimal_termination
    from gridexpand.optimization.urbs.model import create_model
    from gridexpand.optimization.urbs.saveload import create_result_cache
    from gridexpand.optimization.pypsa_model.solve import solve_group

    data, mode, settings = thermal_fixture(tmp_path / "input.h5", uplift, passive)
    model = create_model(copy.deepcopy(data), settings)
    assert check_optimal_termination(SolverFactory("appsi_highs").solve(model))
    urbs = create_result_cache(model)
    pypsa = solve_group(
        copy.deepcopy(data),
        mode,
        solver_name="appsi_highs",
        options={"output_flag": False},
    )
    assert pypsa["audit"]["objective"] == pytest.approx(
        model.objective_function(), rel=1e-8
    )
    for key in ["building_temperature", "building_heat"]:
        actual = pypsa["results"][key].sort_index()
        expected = urbs[key].sort_index()
        assert actual.index.names == expected.index.names
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    if passive:
        assert urbs["building_heat"].sum() == pytest.approx(0, abs=1e-6)
        assert urbs["building_temperature"].min() > 22


def test_preheating_has_flat_price_benefit(tmp_path):
    from gridexpand.optimization.pypsa_model.solve import solve_group

    objectives = []
    for uplift in [0, 2]:
        data, mode, _ = thermal_fixture(tmp_path / f"input{uplift}.h5", uplift)
        # Warm/COP-favourable early hours: preheating shifts useful room heat.
        for col in data["eff_factor"].columns:
            data["eff_factor"].loc[:, col] = [0, 5, 5, 5, 2, 2, 2]
        result = solve_group(
            data, mode, solver_name="appsi_highs", options={"output_flag": False}
        )
        objectives.append(result["audit"]["objective"])
    assert objectives[1] < objectives[0] - 1


def test_thermal_cluster_filter_and_tsam_rejection(tmp_path):
    from gridexpand.optimization.urbs.input import get_cluster_data
    from gridexpand.optimization.urbs.identify import identify_mode

    data, _, _ = thermal_fixture(tmp_path / "input.h5")
    cluster = get_cluster_data(data, [1])
    assert cluster["building_thermal_parameters"].building_objectid.tolist() == ["A"]
    assert set(cluster["building_thermal_timeseries"].columns.get_level_values(0)) == {
        "A"
    }
    data["global_prop"].loc[pd.IndexSlice[:, "tsam"], "value"] = True
    with pytest.raises(ValueError, match="TSAM"):
        identify_mode(data)


def test_optimized_internal_heat_has_one_capacity_and_equivalent_cost(tmp_path):
    from pyomo.environ import SolverFactory
    from gridexpand.optimization.urbs.model import create_model
    from gridexpand.optimization.pypsa_model.solve import solve_group

    data, mode, settings = thermal_fixture(tmp_path / "sized.h5", 2)
    frame = data["process"]
    selected = frame.index.get_level_values("Process") == "heatpump_air"
    frame.loc[selected, "inst-cap"] = 0
    frame.loc[selected, "inv-cost"] = 100
    frame.loc[selected, "inv-cost-fix"] = 0
    model = create_model(copy.deepcopy(data), settings)
    SolverFactory("appsi_highs").solve(model)
    result = solve_group(
        data, mode, solver_name="appsi_highs", options={"output_flag": False}
    )
    assert result["audit"]["objective"] == pytest.approx(
        model.objective_function(), rel=1e-8
    )
    assert len(result["results"]["cap_pro"].xs("heatpump_air", level="pro")) == 2
    assert result["results"]["building_energy_residual"].abs().max() < 1e-6


def test_preheating_needs_earlier_heating_headroom(tmp_path):
    from gridexpand.optimization.pypsa_model.solve import solve_group

    objectives = []
    for uplift in (0, 2):
        data, mode, _ = thermal_fixture(tmp_path / f"no_headroom_{uplift}.h5", uplift)
        for col in data["eff_factor"].columns:
            data["eff_factor"].loc[:, col] = [0, 5, 5, 5, 2, 2, 2]
        # The reference requires 2 kW every hour; an emitter capped at 2 kW
        # leaves no earlier heat-delivery headroom, despite a 2 K allowance.
        mask = data["process"].index.get_level_values("Process") == "Heat_dummy_space"
        data["process"].loc[mask, ["inst-cap", "cap-up"]] = 2.0
        result = solve_group(
            data, mode, solver_name="appsi_highs", options={"output_flag": False}
        )
        objectives.append(result["audit"]["objective"])
    assert objectives[0] == pytest.approx(objectives[1], rel=1e-9)


def test_shared_bus_keeps_two_temperatures_and_one_hp_investment(tmp_path):
    from gridexpand.scenario.config_loader import load_scenario_config
    from gridexpand.paths import SCENARIO_CONFIG_DIR
    from gridexpand.allocation.assets.heat.internal import materialize_internal_assets
    from gridexpand.optimization.urbs.input import read_input_h5
    from gridexpand.optimization.urbs.scenarios import insert_scenario
    from gridexpand.optimization.urbs.identify import identify_mode
    from gridexpand.optimization.urbs.model import create_model
    from gridexpand.optimization.pypsa_model.solve import solve_group
    from pyomo.environ import SolverFactory
    from pyomo.opt import check_optimal_termination

    path = tmp_path / "shared.h5"
    thermal_fixture(path, uplift=0)
    scenario, _ = load_scenario_config(
        SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml"
    )
    plan = pd.DataFrame(
        [
            dict(
                building_objectid=bid,
                Site=1,
                heat_pump_installed_kw_el=0.5,
                heat_pump_capacity_upper_kw_el=0.5,
                auxiliary_installed_kw_el=3.0,
                auxiliary_capacity_upper_kw_el=3.0,
                heat_conversion_capacity_kw_th=10.0,
                buffer_installed_kwh_th=2.0,
                buffer_capacity_upper_kwh_th=2.0,
                buffer_installed_power_kw_th=2.0,
                buffer_power_upper_kw_th=2.0,
            )
            for bid in ["A", "B"]
        ]
    )
    process, commodity, ratios, storage = materialize_internal_assets(
        plan, scenario.technologies, "full_load_hours_rule"
    )
    with pd.HDFStore(path, "a") as store:
        base = store["urbs_in/process"]
        base = base.loc[base.Site.eq(1) & base.Process.isin(["import", "feed_in"])]
        store["urbs_in/process"] = pd.concat([base, process], ignore_index=True)
        base = store["urbs_in/commodity"]
        base = base.loc[base.Site.eq(1) & base.Commodity.str.startswith("electricity")]
        store["urbs_in/commodity"] = pd.concat([base, commodity], ignore_index=True)
        base = store["urbs_in/process_commodity"]
        base = base.loc[base.Process.isin(["import", "feed_in"])]
        store["urbs_in/process_commodity"] = pd.concat(
            [base, ratios], ignore_index=True
        )
        store["urbs_in/storage"] = storage
        demand = pd.DataFrame(
            {(1, "electricity"): np.ones(6), (1, "water_heat"): np.ones(6)}
        )
        store["urbs_in/demand"] = demand
        cops = {}
        for bid in ["A", "B"]:
            for service, value in [("space", 3), ("water", 2), ("buffer", 2.5)]:
                cops[1, f"HP_{service}_{bid}"] = np.full(6, value)
        store["urbs_in/eff_factor"] = pd.DataFrame(cops)
        params = store["urbs_in/building_thermal_parameters"]
        params.Site = 1
        params.heat_commodity = "room_heat_" + params.building_objectid
        store["urbs_in/building_thermal_parameters"] = params
    data = insert_scenario(read_input_h5(path), {"tsam": False})
    mode = identify_mode(data)
    model = create_model(
        copy.deepcopy(data), {"dt": 1, "timesteps": range(7), "hoursPerPeriod": 168}
    )
    assert check_optimal_termination(SolverFactory("appsi_highs").solve(model))
    result = solve_group(
        data, mode, solver_name="appsi_highs", options={"output_flag": False}
    )
    assert result["audit"]["objective"] == pytest.approx(
        model.objective_function(), rel=1e-8
    )
    assert set(
        result["results"]["building_temperature"].index.get_level_values("building")
    ) == {"A", "B"}
    hp = result["results"]["tau_pro"].xs("heatpump_air", level="pro")
    assert hp.max() <= 1 + 1e-6
    assert len(result["results"]["cap_pro"].xs("heatpump_air", level="pro")) == 1
    assert result["results"]["building_energy_residual"].abs().max() < 1e-6
