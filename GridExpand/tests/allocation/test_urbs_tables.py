"""urbs table builders take the scenario parameter dicts (alloc D8).

The expected frames are built with the former alias-based code (the
config.<PREFIX>_<FIELD> attributes that Config.apply_scenario used to set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gridexpand.allocation.functions.electricity as electricity
import gridexpand.allocation.functions.mobility as mobility
from gridexpand.allocation.assets.urbs_rows import process_row, storage_parameter_fields
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config

PROCESS_FIELDS = (
    "installed_capacity_kw", "capacity_upper_kw", "fixed_investment_cost_eur",
    "investment_cost_eur_per_kw", "fixed_cost_eur_per_hour", "variable_cost_eur_per_kwh",
    "wacc", "depreciation_years", "minimum_power_factor",
)
STORAGE_FIELDS = (
    "installed_energy_kwh", "capacity_upper_kwh", "installed_power_kw", "power_upper_kw",
    "charge_efficiency", "discharge_efficiency", "self_discharge_per_timestep",
    "energy_to_power_hours", "investment_cost_eur_per_kw", "investment_cost_eur_per_kwh",
    "fixed_investment_cost_power_eur", "fixed_investment_cost_energy_eur",
    "variable_cost_eur_per_kwh", "wacc", "depreciation_years",
)


@pytest.fixture(scope="module", params=["forchheim_2045_synthetic.yaml", "schweinfurt_2045.yaml"])
def technologies(request):
    scenario, _ = load_scenario_config(SCENARIO_CONFIG_DIR / request.param)
    return scenario.technologies


def _values(parameters, fields):
    return tuple(parameters[field] for field in fields)


def test_grid_connection_processes(technologies):
    parameters = technologies.processes["grid_connection"]
    expected = pd.DataFrame([7, 9], columns=["Site"])
    expected[["Process", "inst-cap", "cap-up", "inv-cost-fix", "inv-cost", "fix-cost", "var-cost", "wacc", "depreciation", "pf-min"]] = (
        ("import",) + _values(parameters, PROCESS_FIELDS))
    feed = expected.copy()
    feed["Process"] = "feed_in"
    expected = pd.concat([expected, feed], axis=0).reset_index(drop=True)
    pd.testing.assert_frame_equal(electricity.create_pro_elec([7, 9], parameters), expected, check_exact=True)


@pytest.mark.parametrize("buses", [[], [3, 4]])
def test_generic_battery_storage(technologies, buses):
    parameters = technologies.storages["stationary_battery"]
    expected = pd.DataFrame(buses, columns=["Site"])
    expected[["Storage", "Commodity", "inst-cap-c", "cap-up-c", "inst-cap-p", "cap-up-p", "eff-in", "eff-out", "discharge", "ep-ratio",
              "inv-cost-p", "inv-cost-c", "fix-cost-p", "fix-cost-c", "var-cost-p", "wacc", "depreciation"]] = (
        ("battery_private", "electricity") + _values(parameters, STORAGE_FIELDS))
    pd.testing.assert_frame_equal(electricity.create_sto_elec(buses, parameters), expected, check_exact=True)


def test_mobility_processes_and_storages(technologies):
    battery_dict = {(5, 0): 58.0, (5, 1): 77.4, (8, 2): 40.0}
    charger = technologies.processes["home_charger"]
    expected = pd.DataFrame([(bus, f"charging_station{i}") for bus, i in battery_dict], columns=["Site", "Process"])
    expected[["inst-cap", "cap-up", "inv-cost-fix", "inv-cost", "fix-cost", "var-cost", "wacc", "depreciation", "pf-min"]] = (
        _values(charger, PROCESS_FIELDS))
    pd.testing.assert_frame_equal(mobility.create_pro_mob(battery_dict, charger), expected, check_exact=True)

    storage = technologies.storages["mobility_storage"]
    expected = pd.DataFrame(
        [(bus, f"mobility_storage{i}", f"mobility{i}", cap, cap, cap, cap) for (bus, i), cap in battery_dict.items()],
        columns=["Site", "Storage", "Commodity", "inst-cap-c", "cap-up-c", "inst-cap-p", "cap-up-p"])
    expected[["eff-in", "eff-out", "discharge", "ep-ratio", "inv-cost-p", "inv-cost-c", "fix-cost-p", "fix-cost-c", "var-cost-p", "wacc", "depreciation"]] = (
        _values(storage, STORAGE_FIELDS[4:]))
    pd.testing.assert_frame_equal(mobility.create_sto_mob(battery_dict, storage), expected, check_exact=True)
    assert mobility.create_pro_mob({}, charger).empty and mobility.create_sto_mob({}, storage).empty


@pytest.mark.parametrize("fixed", [True, False])
def test_row_builders_keep_column_order_and_values(technologies, fixed):
    parameters = technologies.processes["heatpump_air"]
    assert process_row(4, "heatpump_air", 1.5, 3.0, fixed=fixed, parameters=parameters) == {
        "Site": 4, "Process": "heatpump_air", "inst-cap": 1.5, "cap-up": 3.0,
        "inv-cost-fix": 0.0 if fixed else parameters["fixed_investment_cost_eur"],
        "inv-cost": 0.0 if fixed else parameters["investment_cost_eur_per_kw"],
        "fix-cost": parameters["fixed_cost_eur_per_hour"], "var-cost": parameters["variable_cost_eur_per_kwh"],
        "wacc": parameters["wacc"], "depreciation": parameters["depreciation_years"],
        "pf-min": parameters["minimum_power_factor"],
    }
    storage = technologies.storages["thermal_storage"]
    fields = storage_parameter_fields(storage, fixed=fixed, ep_ratio=0.25)
    assert list(fields) == ["eff-in", "eff-out", "discharge", "ep-ratio", "inv-cost-p", "inv-cost-c",
                            "fix-cost-p", "fix-cost-c", "var-cost-p", "wacc", "depreciation"]
    assert fields["ep-ratio"] == 0.25
    assert fields["inv-cost-c"] == (0.0 if fixed else storage["investment_cost_eur_per_kwh"])
    assert np.isclose(fields["eff-in"], storage["charge_efficiency"])


def test_paired_static_tables_and_prices(technologies):
    from gridexpand.allocation.scenario_calibration.pipeline.urbs_input_tables import (
        buy_sell_price,
        urbs_static_tables,
    )

    tables = urbs_static_tables([1, 2], technologies, include_generic_battery=False)
    pd.testing.assert_frame_equal(
        tables["process"], electricity.create_pro_elec([1, 2], technologies.processes["grid_connection"])
    )
    pd.testing.assert_frame_equal(
        tables["storage"], electricity.create_sto_elec([], technologies.storages["stationary_battery"])
    )
    assert len(urbs_static_tables([1, 2], technologies)["storage"]) == 2
    prices = buy_sell_price(3, import_price_eur_per_kwh=0.398, pv_feed_in_tariff_eur_per_kwh=0.0)
    assert prices.to_dict("list") == {"electricity_import": [0.398] * 3, "electricity_feed_in": [0.0] * 3}
    assert prices.index.name == "t"
