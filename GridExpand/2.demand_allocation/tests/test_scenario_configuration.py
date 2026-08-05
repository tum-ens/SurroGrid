from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from scenario_pipeline.configuration.loader import load_scenario_config
from scenario_pipeline.configuration.scenario_config import ScenarioConfig
from gridalloc.src.assets.battery.materialization import (
    materialize_battery_urbs_inputs,
)
from gridalloc.src.assets.battery.sizing import (
    build_battery_asset_plan, htw_usable_capacity_kwh,
)
from gridalloc.src.assets.pv.materialization import materialize_pv_urbs_inputs
from gridalloc.src.assets.pv.sizing import (
    build_pv_asset_plan,
    heuristic_pv_capacity,
)
from gridalloc.src.scenario_calibration.pipeline.urbs_input_tables import buy_sell_price


SCENARIO_PATH = (
    GRIDEXPAND_DIR
    / "scenario_pipeline"
    / "configurations"
    / "scenarios"
    / "forchheim_2045.yaml"
)


class ScenarioConfigurationTest(unittest.TestCase):
    def test_forchheim_2045_resolves_zero_feed_in_and_all_cases(self):
        scenario, fingerprint = load_scenario_config(SCENARIO_PATH)
        self.assertEqual(scenario.scenario_id, "forchheim_2045")
        self.assertEqual(scenario.economics.pv_feed_in_tariff_eur_per_kwh, 0.0)
        self.assertEqual(len(scenario.model_cases), 4)
        self.assertEqual(scenario.battery.energy_to_power_hours, 2.0)
        self.assertEqual(
            scenario.battery_sizing_method("post-inflex-heuristic"), "htw_2025_upper_bound")
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(
            scenario.pv_sizing_method("post-hems-optimized"), "optimization"
        )
        self.assertEqual(
            scenario.pv_sizing_method("post-inflex-heuristic"),
            "annual_electricity_rule",
        )

    def test_unknown_yaml_option_is_rejected(self):
        raw = {
            "scenario": {"id": "x", "milestone_year": 2045, "model_cases": ["pre"]},
            "economics": {"electricity": {
                "import_price_eur_per_kwh": 0.3,
                "pv_feed_in_tariff_eur_per_kwh": 0.0,
                "feed_in_tarif": 0.1,
            }},
            "asset_sizing": {
                "pv": {
                    "heuristic_method": "annual_electricity_rule",
                    "optimized_method": "optimization",
                    "location_mode": "all_buildings",
                    "demand_multiplier": 2.5,
                    "fallback_capacity_kwp": 14.5,
                    "maximum_fallback_share": 0.0,
                    "module_capacity_kw_per_m2": 0.202,
                    "flat_roof_utilization": 0.27,
                    "slanted_roof_utilization": 0.58,
                    "tilt_bin_degrees": 5.0,
                    "azimuth_bin_degrees": 15.0,
                },
                "battery": {
                    "heuristic_method": "htw_2025_upper_bound",
                    "optimized_method": "optimization",
                    "location_mode": "all_pv_buildings",
                    "predefined_locations_when_available": True,
                    "minimum_pv_kwp_per_annual_mwh": 0.5,
                    "maximum_usable_kwh_per_pv_kwp": 1.5,
                    "maximum_usable_kwh_per_annual_mwh": 1.5,
                    "energy_to_power_hours": 2.0,
                },
                "heat_pump": {"method": "temporary_placeholder"},
            },
            "time_aggregation": {
                "enabled": True,
                "number_of_typical_periods": 6,
                "hours_per_period": 168,
                "extreme_period_method": "replace_cluster_center",
                "clustering_method": "hierarchical",
                "cluster_representation": "medoid",
                "segmentation": False,
                "rescale_cluster_periods": False,
                "feature_weights": {"Tamb": 1.0},
                "extreme_features": ["minimum_mean_temperature"],
            },
        }
        with self.assertRaisesRegex(ValueError, "feed_in_tarif"):
            ScenarioConfig.from_dict(raw)


class PvSizingTest(unittest.TestCase):
    def setUp(self):
        self.roofs = pd.DataFrame({
            "building_objectid": ["a", "a"],
            "roof_surface_id": ["south", "north"],
            "profile_usable": [True, True],
            "profile_tilt_deg": [30.0, 30.0],
            "profile_azimuth_deg": [180.0, 0.0],
            "available_pv_kw": [6.0, 6.0],
            "quality_flag": ["lod2", "lod2"],
        })
        self.profiles = pd.DataFrame({
            "solar_30_180": [0.0, 1.0, 1.0],
            "solar_30_0": [0.0, 0.25, 0.25],
        })
        self.buildings = pd.DataFrame({
            "building_objectid": ["a"],
            "Site": [10],
            "annual_electricity_kwh": [3200.0],
            "pv_roof_eligible": [True],
        })

    def test_equation_and_physical_clipping(self):
        self.assertAlmostEqual(heuristic_pv_capacity(3200.0, 20.0), 8.0)
        self.assertAlmostEqual(heuristic_pv_capacity(10000.0, 12.0), 12.0)

    def test_best_yield_allocation_and_fixed_urbs_capacity(self):
        plan, selected = build_pv_asset_plan(
            self.buildings,
            self.roofs,
            self.profiles,
            sizing_method="annual_electricity_rule",
        )
        by_roof = selected.set_index("roof_surface_id")["selected_pv_kw"]
        self.assertAlmostEqual(by_roof["south"], 6.0)
        self.assertAlmostEqual(by_roof["north"], 2.0)
        urbs = materialize_pv_urbs_inputs(
            plan, selected, self.profiles,
            sizing_method="annual_electricity_rule",
        )
        row = urbs.process.iloc[0]
        self.assertAlmostEqual(row["inst-cap"], 8.0)
        self.assertAlmostEqual(row["cap-up"], 8.0)
        self.assertEqual(row["inv-cost"], 0.0)
        np.testing.assert_allclose(
            urbs.supim.iloc[:, 0],
            (self.profiles["solar_30_180"] * 6 + self.profiles["solar_30_0"] * 2) / 8,
        )

    def test_multiple_scenario_units_are_sized_once_per_building(self):
        buildings = pd.DataFrame({
            "building_objectid": ["a", "a"],
            "Site": [10, 11],
            "annual_electricity_kwh": [1200.0, 2000.0],
            "pv_roof_eligible": [True, True],
        })
        plan, selected = build_pv_asset_plan(
            buildings,
            self.roofs,
            self.profiles,
            sizing_method="annual_electricity_rule",
        )
        self.assertEqual(len(plan), 1)
        self.assertAlmostEqual(plan.iloc[0]["annual_electricity_kwh"], 3200.0)
        self.assertAlmostEqual(plan.iloc[0]["pv_max_kwp"], 12.0)
        self.assertAlmostEqual(plan.iloc[0]["pv_installed_kwp"], 8.0)
        self.assertAlmostEqual(selected["selected_pv_kw"].sum(), 8.0)

    def test_buy_sell_tariff_is_configurable(self):
        class Electricity:
            class config:
                BSP_IMPORT = 0.4
                BSP_FEED_IN = 0.2

        prices = buy_sell_price(
            3,
            Electricity,
            import_price_eur_per_kwh=0.31,
            pv_feed_in_tariff_eur_per_kwh=0.0,
        )
        self.assertTrue(prices["electricity_import"].eq(0.31).all())
        self.assertTrue(prices["electricity_feed_in"].eq(0.0).all())


class BatterySizingTest(unittest.TestCase):
    class TechnicalParameters:
        BS_EFF_IN = 0.961
        BS_EFF_OUT = 1.0
        BS_DISCHARGE = 0.0
        BS_INV_COST_P = 0.0
        BS_INV_COST_C = 976.0
        BS_FIX_COST_P = 0.0
        BS_FIX_COST_C = 0.0
        BS_VAR_COST_P = 0.001
        BS_WACC = 0.022
        BS_DEPRECIATION = 15

    def setUp(self):
        self.pv_plan = pd.DataFrame({
            "building_objectid": ["a"],
            "Site": [10],
            "annual_electricity_kwh": [4000.0],
            "pv_installed_kwp": [8.0],
            "pv_max_kwp": [12.0],
        })

    def test_htw_threshold_and_upper_bound(self):
        self.assertAlmostEqual(htw_usable_capacity_kwh(4000.0, 8.0), 6.0)
        self.assertAlmostEqual(htw_usable_capacity_kwh(4000.0, 2.0), 0.0)
        self.assertAlmostEqual(htw_usable_capacity_kwh(4000.0, 2.01), 3.015)

    def test_fixed_two_hour_battery(self):
        plan = build_battery_asset_plan(
            self.pv_plan,
            sizing_method="htw_2025_upper_bound",
            minimum_pv_kwp_per_annual_mwh=0.5,
            maximum_usable_kwh_per_pv_kwp=1.5,
            maximum_usable_kwh_per_annual_mwh=1.5,
        )
        urbs = materialize_battery_urbs_inputs(
            plan,
            sizing_method="htw_2025_upper_bound",
            energy_to_power_hours=2.0,
            technical_parameters=self.TechnicalParameters,
        )
        row = urbs.storage.iloc[0]
        self.assertAlmostEqual(row["inst-cap-c"], 6.0)
        self.assertAlmostEqual(row["cap-up-c"], 6.0)
        self.assertAlmostEqual(row["inst-cap-p"], 3.0)
        self.assertAlmostEqual(row["cap-up-p"], 3.0)
        self.assertEqual(row["ep-ratio"], 2.0)
        self.assertEqual(row["inv-cost-c"], 0.0)

    def test_optimized_battery_keeps_capacity_variable(self):
        plan = build_battery_asset_plan(
            self.pv_plan,
            sizing_method="optimization",
            minimum_pv_kwp_per_annual_mwh=0.5,
            maximum_usable_kwh_per_pv_kwp=1.5,
            maximum_usable_kwh_per_annual_mwh=1.5,
        )
        urbs = materialize_battery_urbs_inputs(
            plan,
            sizing_method="optimization",
            energy_to_power_hours=2.0,
            technical_parameters=self.TechnicalParameters,
        )
        row = urbs.storage.iloc[0]
        self.assertEqual(row["inst-cap-c"], 0.0)
        self.assertAlmostEqual(row["cap-up-c"], 6.0)
        self.assertEqual(row["inst-cap-p"], 0.0)
        self.assertAlmostEqual(row["cap-up-p"], 3.0)
        self.assertEqual(row["inv-cost-c"], 976.0)


if __name__ == "__main__":
    unittest.main()
