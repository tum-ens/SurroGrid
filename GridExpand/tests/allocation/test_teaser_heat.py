"""TEASER heat path: corrected defaults, single fixes, and bit-exact legacy reproduction."""

import ast
import contextlib
import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridexpand.allocation.functions import heat
from gridexpand.scenario.scenario_config import HeatSizingConfig, TeaserHeatConfig

POSTCODE = "91341"
# Profiles of these four buildings simulated with SurroGrid dev 1c671ac (before the TEASER fixes) under
# PYTHONHASHSEED=0: (annual space heat kWh, peak kW, sha256 of the float64 profile, annual hot water kWh).
# The hash seed matters: envelope.py sums element areas over a set of strings, so without a fixed seed the
# summation order, and the last bits of every profile, change from process to process (old code included).
LEGACY_GOLDEN = {
    "B-SFH": (68206.85341206419, 27.874224620428485,
        "5d15f609b99c37c8ad16b63fc4737cb3591ea4640daa39baa15e80e6b4b78e8e", 1766.1063757756694),
    "B-TH": (39973.087851361975, 12.752821910502533,
        "38af1da19b6d6eb53d542ddabda04579dc51acd09b2dd7dbae7ec935a460c672", 1175.4015086102004),
    "B-MFH": (89828.75692578562, 30.6465506843839,
        "ba0a002193b55eca3aebd0beaaf7ea242d5a3cd38e827a7b266a9f120a2fc84d", 4708.343930603425),
    "B-AB": (338456.5746765582, 113.67686391014324,
        "eacbff928be255997fe6e021f47abe78932b7e2c5a5fa505624cbe7782e45bfe", 21202.629490605275),
}


def synthetic_weather():
    """Deterministic 8760 h weather with a winter, a summer, a daily cycle and plausible irradiance."""
    hours = np.arange(8760)
    day = hours // 24
    hour = hours % 24
    temp = 9.0 - 10.0 * np.cos(2 * np.pi * (day - 15) / 365) + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    declination = 23.45 * np.sin(2 * np.pi * (284 + day) / 365)
    hour_angle = 15.0 * (hour + 0.5 - 12.0)
    lat = np.radians(49.6)
    sin_elev = (np.sin(lat) * np.sin(np.radians(declination))
                + np.cos(lat) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle)))
    sin_elev = np.clip(sin_elev, 0.0, None)
    cloud = 0.55 + 0.35 * np.sin(2 * np.pi * day / 7.3)
    ghi = 1000.0 * sin_elev ** 1.15 * cloud
    dhi = ghi * (0.3 + 0.5 * (1 - cloud))
    dni = np.where(sin_elev > 0.087, (ghi - dhi) / np.maximum(sin_elev, 0.087), 0.0)
    return pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi, "temp_air": temp})


def sample_buildings(with_surfaces=False):
    buildings = pd.DataFrame({
        "objectid": ["B-SFH", "B-TH", "B-MFH", "B-AB"],
        "bus": [0, 1, 2, 3],
        "building_type": ["SFH", "TH", "MFH", "AB"],
        "construction_year": ["1949-1978", "-1919", "1979-1990", "1979-1990"],
        "floor_area": [110.0, 70.0, 220.0, 520.0],
        "floor_number": [2, 3, 3, 5],
        "households": [1, 1, 4, 18],
        "occ_list": [[3], [2], [2, 1, 3, 2], [2] * 18],
        "residential_effective_floor_area_m2": [220.0, 210.0, 660.0, 2600.0],
    })
    if with_surfaces:
        buildings["wall_surface_area_m2"] = [260.0, 230.0, 640.0, 1500.0]
        buildings["roof_surface_area_m2"] = [140.0, 90.0, 260.0, 540.0]
        buildings["shared_wall_area_m2"] = [0.0, 90.0, 0.0, 300.0]
    return buildings


def electricity(buildings):
    return pd.DataFrame({(bus, "electricity"): np.full(8760, 2.8 * hh / 8.76)
                         for bus, hh in zip(buildings["bus"], buildings["households"])})


def simulate(teaser, buildings=None):
    buildings = sample_buildings() if buildings is None else buildings
    weather = synthetic_weather()
    kwargs = {} if teaser == "old" else {"teaser": teaser}
    arrays = ([weather[c].to_numpy() for c in ("dni", "dhi", "temp_air")] if teaser == "old"
              else heat.teaser_weather_arrays(weather, teaser))
    with contextlib.redirect_stdout(io.StringIO()):
        space, water = heat.generate_heat_demands(buildings, electricity(buildings), arrays, POSTCODE, 0, **kwargs)
    result = {}
    for _, row in buildings.iterrows():
        q = space[(row.bus, "space_heat")].to_numpy(dtype=np.float64)
        result[row.objectid] = (float(q.sum()), float(q.max()), hashlib.sha256(q.tobytes()).hexdigest(),
                                float(water[(row.bus, "water_heat")].sum()))
    return result


def test_legacy_preset_reproduces_surrogrid_2_0_bit_for_bit():
    code = ("import sys; sys.path.insert(0, %r); import test_teaser_heat as t; "
            "from gridexpand.scenario.scenario_config import TeaserHeatConfig; "
            "print(repr(t.simulate(TeaserHeatConfig.legacy())))" % str(Path(__file__).parent))
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert ast.literal_eval(out.strip().splitlines()[-1]) == LEGACY_GOLDEN


def test_legacy_preset_matches_the_old_profiles_for_any_hash_seed():
    for key, (energy, peak, _, water) in simulate(TeaserHeatConfig.legacy()).items():
        assert energy == pytest.approx(LEGACY_GOLDEN[key][0], rel=1e-12)
        assert peak == pytest.approx(LEGACY_GOLDEN[key][1], rel=1e-12)
        assert water == LEGACY_GOLDEN[key][3]


def test_legacy_preset_from_yaml_equals_the_classmethod():
    assert TeaserHeatConfig.from_dict({"preset": "legacy"}) == TeaserHeatConfig.legacy()
    assert TeaserHeatConfig.from_dict(None) == TeaserHeatConfig()
    single = TeaserHeatConfig.from_dict({"preset": "legacy", "ground_floor_b": 0.5})
    assert single.ground_floor_b == 0.5 and single.heated_area_ratio == 1.0


def test_teaser_block_validation():
    with pytest.raises(ValueError, match="preset"):
        TeaserHeatConfig.from_dict({"preset": "new"})
    with pytest.raises(ValueError, match="heated_area_ratio"):
        TeaserHeatConfig.from_dict({"heated_area_ratio": 1.2})
    with pytest.raises(ValueError, match="Unknown"):
        TeaserHeatConfig.from_dict({"area": 0.5})
    base = {"indoor_design_temperature_c": 20.0, "heating_limit_temperature_c": 15.0,
            "heat_pump_design_share": 0.65, "buffer_volume_l_per_kw_th": 86.0,
            "buffer_usable_temperature_spread_k": 10.0}
    assert HeatSizingConfig.from_dict({**base, "space_heat_source": "teaser"}).teaser == TeaserHeatConfig()
    legacy = HeatSizingConfig.from_dict({**base, "space_heat_source": "teaser", "teaser": {"preset": "legacy"}})
    assert legacy.teaser == TeaserHeatConfig.legacy()
    with pytest.raises(ValueError, match="teaser heat block"):
        HeatSizingConfig.from_dict({**base, "space_heat_source": "infdb_ro_heat", "teaser": {}})


def test_weather_beam_is_horizontal_by_default_and_dni_for_legacy():
    weather = synthetic_weather()
    corrected = heat.teaser_weather_arrays(weather, TeaserHeatConfig())
    legacy = heat.teaser_weather_arrays(weather, TeaserHeatConfig.legacy())
    np.testing.assert_array_equal(corrected[0], (weather.ghi - weather.dhi).clip(lower=0).to_numpy())
    np.testing.assert_array_equal(legacy[0], weather.dni.to_numpy())


def test_heated_area_ratio_scales_heat_and_peak():
    legacy = simulate(TeaserHeatConfig.legacy())
    area = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "heated_area_ratio": 0.55}))
    for key in legacy:
        assert 0.45 < area[key][0] / legacy[key][0] < 0.6
        assert 0.45 < area[key][1] / legacy[key][1] < 0.6


def test_sfh_roof_fix_lowers_only_the_single_family_house():
    legacy = simulate(TeaserHeatConfig.legacy())
    fixed = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "sfh_roof_fix": True}))
    assert fixed["B-SFH"][0] < 0.9 * legacy["B-SFH"][0]
    for key in ("B-TH", "B-MFH", "B-AB"):  # flat roofs under orientation -1: identical
        assert fixed[key] == legacy[key]


def test_sfh_roof_fix_areas():
    from teaser.project import Project
    from gridexpand.allocation.external.districtgenerator.classes.envelope import Envelope
    from gridexpand.allocation.external.districtgenerator.classes.datahandler import Datahandler

    prj = Project(load_data=True)
    prj.add_residential(method="tabula_de", usage="single_family_house", name="x", year_of_construction=1975,
                        number_of_floors=1, height_of_floors=3.125, net_leased_area=100.0,
                        construction_type="tabula_standard")
    handler = Datahandler(pd.DataFrame({"id": [0]}))
    params = pd.Series({"id": 0, "year": 1975, "retrofit": 0, "building": "SFH"})
    args = (prj, params, "tabula_standard", handler.physics, handler.design_building_data, handler.dataPath)
    old, new = Envelope(*args), Envelope(*args, options=TeaserHeatConfig())
    roofs = sum(r.area for r in prj.buildings[0].thermal_zones[0].rooftops)
    assert new.A["opaque"]["roof"] == pytest.approx(roofs)
    assert new.A["opaque"]["wall"] == pytest.approx(old.A["opaque"]["wall"] - roofs)
    assert old.A["opaque"]["roof"] == pytest.approx(1.2 * old.A["opaque"]["floor"])


def test_ground_floor_factor_and_shading_lower_or_raise_heat_as_expected():
    legacy = simulate(TeaserHeatConfig.legacy())
    floor = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "ground_floor_b": 0.5}))
    assert all(floor[k][0] < legacy[k][0] for k in legacy)
    no_blinds = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "blind_threshold_w_per_m2": None}))
    assert all(no_blinds[k][0] < legacy[k][0] for k in legacy)  # more solar gain


def test_heating_limit_replaces_the_fixed_heating_break():
    legacy = simulate(TeaserHeatConfig.legacy())
    limit = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "fixed_heating_off_days": None,
                                                 "heating_limit_c": 12.0}))
    assert all(limit[k] != legacy[k] for k in legacy)


def test_stock_share_age_mapping():
    shares = heat.stock_class_shares("SFH", "1949-1978")
    assert set(shares) == {"D", "E", "F"} and sum(shares.values()) == pytest.approx(1.0)
    assert shares["E"] == pytest.approx(1509 / (859 + 1509 + 1507))
    assert set(heat.stock_class_shares("MFH", "1979-1990")) == {"G", "H"}
    assert heat.stock_class_shares("TH", "-1919") == pytest.approx({"B": 640 / (640 + 710 / 30), "C": (710 / 30) / (640 + 710 / 30)})
    assert heat.stock_class_shares("SFH", "2020-") == {"L": 1.0}
    row = pd.Series({"objectid": "B-SFH", "building_type": "SFH"})
    year = heat.teaser_year(row, "1949-1978", TeaserHeatConfig(), 0)
    assert year in (1953, 1963, 1973) and year == heat.teaser_year(row, "1949-1978", TeaserHeatConfig(), 0)
    assert heat.teaser_year(row, "1949-1978", TeaserHeatConfig.legacy(), 0) == 1978


def test_apartment_block_after_1978_uses_the_mfh_archetype_and_keeps_its_hot_water():
    legacy = simulate(TeaserHeatConfig.legacy())
    fixed = simulate(TeaserHeatConfig.from_dict({"preset": "legacy", "apartment_block_after_1978": "multi_family_house"}))
    assert fixed["B-AB"][0] != legacy["B-AB"][0]
    assert fixed["B-AB"][3] == legacy["B-AB"][3]
    assert all(fixed[k] == legacy[k] for k in ("B-SFH", "B-TH", "B-MFH"))


def test_lod2_envelope_uses_surfaces_and_falls_back_to_tabula():
    with_surfaces = sample_buildings(with_surfaces=True)
    lod2 = simulate(TeaserHeatConfig(), with_surfaces)
    tabula = simulate(TeaserHeatConfig.from_dict({"envelope_source": "tabula"}), with_surfaces)
    fallback = simulate(TeaserHeatConfig(), sample_buildings())
    assert all(lod2[k][0] != tabula[k][0] for k in lod2)
    assert fallback == tabula
    no_shared = simulate(TeaserHeatConfig.from_dict({"subtract_shared_walls": False}), with_surfaces)
    assert no_shared["B-TH"][0] > lod2["B-TH"][0] and no_shared["B-SFH"] == lod2["B-SFH"]


def test_options_s7_and_s8_are_off_by_default_and_work_when_set():
    config = TeaserHeatConfig()
    assert config.consumption_factor == "none" and config.retrofit_share_before_1979 == 0.0
    base = simulate(config)
    iwu = simulate(TeaserHeatConfig.from_dict({"consumption_factor": "iwu"}))
    for key in base:
        assert iwu[key][0] <= base[key][0] and iwu[key][1] == pytest.approx(base[key][1])
    retro = simulate(TeaserHeatConfig.from_dict({"retrofit_share_before_1979": 1.0}))
    assert retro["B-SFH"][0] < base["B-SFH"][0] and retro["B-MFH"] == base["B-MFH"]


def test_energy_only_scaling_keeps_the_peak():
    q = np.array([0.0, 1.0, 2.0, 5.0, 3.0, 0.5])
    scaled = heat.energy_only_scaling(q, 0.7)
    assert scaled.max() == pytest.approx(5.0) and scaled.sum() == pytest.approx(0.7 * q.sum(), rel=1e-9)
    assert scaled[0] == 0.0
    np.testing.assert_array_equal(heat.energy_only_scaling(q, 1.0), q)
