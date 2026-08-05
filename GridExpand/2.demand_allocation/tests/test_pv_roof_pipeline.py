from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gridalloc.src.scenario_calibration.allocation.paired_allocation import (
    _add_pv_roof_assignment,
    _pv_scenario_unit_assignments,
)
from gridalloc.src.scenario_calibration.allocation.pv_roof_potential import (
    FALLBACK_PV_CAPACITY_KW,
    add_missing_building_fallbacks,
    building_roof_capacity,
    normalize_lod2_roof_sections,
)
from gridalloc.src.scenario_calibration.profiles.paired_profiles import _build_paired_pv


class RoofCatalogTest(unittest.TestCase):
    def test_normalizes_angles_and_exact_surface_capacity(self):
        raw = pd.DataFrame(
            [
                {
                    "building_objectid": "flat",
                    "roof_surface_id": "1",
                    "dachneigung": 90.0,
                    "dachorientierung": -1.0,
                    "roof_area_m2": 100.0,
                },
                {
                    "building_objectid": "sloped",
                    "roof_surface_id": "2",
                    "dachneigung": 58.0,
                    "dachorientierung": 188.0,
                    "roof_area_m2": 100.0,
                },
            ]
        )
        catalog = normalize_lod2_roof_sections(raw)
        flat, sloped = catalog.iloc[0], catalog.iloc[1]
        self.assertEqual(flat["surface_tilt_deg"], 0.0)
        self.assertEqual(flat["surface_azimuth_deg"], 0.0)
        self.assertAlmostEqual(flat["available_pv_kw"], 100 * 0.27 * 0.202)
        self.assertEqual(sloped["surface_tilt_deg"], 32.0)
        self.assertEqual(sloped["profile_tilt_deg"], 30.0)
        self.assertEqual(sloped["profile_azimuth_deg"], 195.0)
        self.assertAlmostEqual(sloped["available_pv_kw"], 100 * 0.58 * 0.202)

    def test_fallback_only_when_building_has_no_usable_section(self):
        raw = pd.DataFrame(
            [
                {
                    "building_objectid": "bad",
                    "roof_surface_id": "1",
                    "dachneigung": 60.0,
                    "dachorientierung": -1.0,
                    "roof_area_m2": 100.0,
                }
            ]
        )
        catalog = add_missing_building_fallbacks(
            normalize_lod2_roof_sections(raw), ["bad", "missing"]
        )
        fallback = catalog[catalog["quality_flag"].eq("fallback_14_5_kw")]
        self.assertEqual(set(fallback["building_objectid"]), {"bad", "missing"})
        capacities = building_roof_capacity(catalog)
        self.assertEqual(capacities["bad"], FALLBACK_PV_CAPACITY_KW)
        self.assertEqual(capacities["missing"], FALLBACK_PV_CAPACITY_KW)


class PvEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.real = pd.DataFrame(
            [
                {
                    "building_objectid": "a",
                    "source_lv_id": 1,
                    "source_allocation_bus": 10,
                    "scenario_unit_id": 0,
                    "residential_equivalent_hh_annual_kwh": 1000.0,
                    "calibrated_annual_ghd_kwh": 0.0,
                },
                {
                    "building_objectid": "a",
                    "source_lv_id": 1,
                    "source_allocation_bus": 11,
                    "scenario_unit_id": 1,
                    "residential_equivalent_hh_annual_kwh": 2000.0,
                    "calibrated_annual_ghd_kwh": 0.0,
                },
                {
                    "building_objectid": "b",
                    "source_lv_id": 1,
                    "source_allocation_bus": 12,
                    "scenario_unit_id": 2,
                    "residential_equivalent_hh_annual_kwh": 500.0,
                    "calibrated_annual_ghd_kwh": 0.0,
                },
            ]
        )

    def test_swf_rows_are_cumulative_location_evidence_only(self):
        matches = pd.DataFrame(
            [
                {"building_objectid": "a", "lv_id": 1, "bus": 10, "matched": True, "asset_type": "Photovoltaik"},
                {"building_objectid": "a", "lv_id": 1, "bus": 10, "matched": True, "asset_type": "Photovoltaik"},
                {"building_objectid": "a", "lv_id": 1, "bus": 11, "matched": True, "asset_type": "Photovoltaik"},
            ]
        )
        assignments = _pv_scenario_unit_assignments(
            self.real, matches, location_mode="swf"
        )
        self.assertEqual(assignments["scenario_unit_id"].tolist(), [0])
        assigned = _add_pv_roof_assignment(
            self.real, assignments, pd.Series({"a": 20.0, "b": 30.0})
        )
        self.assertEqual(assigned["pv_roof_capacity_kw"].sum(), 20.0)

    def test_all_buildings_uses_primary_demand_connection(self):
        assignments = _pv_scenario_unit_assignments(
            self.real, pd.DataFrame(), location_mode="all_buildings"
        )
        selected = assignments.set_index("building_objectid")["scenario_unit_id"]
        self.assertEqual(selected["a"], 1)
        self.assertEqual(selected["b"], 2)


class PairedPvInputsTest(unittest.TestCase):
    def test_builds_one_process_per_site_and_angle_bin(self):
        allocation = pd.DataFrame(
            {
                "building_objectid": ["a"],
                "allocation_bus": [10],
                "_profile_site_id": [100],
                "pv_roof_eligible": [True],
            }
        )
        roofs = pd.DataFrame(
            {
                "building_objectid": ["a", "a"],
                "roof_surface_id": ["r1", "r2"],
                "profile_usable": [True, True],
                "profile_tilt_deg": [30.0, 30.0],
                "profile_azimuth_deg": [180.0, 180.0],
                "available_pv_kw": [5.0, 7.0],
                "quality_flag": ["lod2", "lod2"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "pv.h5"
            pd.DataFrame({"solar_30_180": [0.0, 0.5, 1.0]}).to_hdf(
                library, key="profiles"
            )
            result = _build_paired_pv(
                allocation,
                roof_catalog=roofs,
                hours=3,
                profile_library=library,
            )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.process), 1)
        self.assertAlmostEqual(result.process.iloc[0]["cap-up"], 12.0)
        np.testing.assert_allclose(result.supim.iloc[:, 0], [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
