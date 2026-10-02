"""GHD rules (ghd: block): activity gating, single-volume storeys, OSM evidence (synthetic)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml
from shapely.geometry import Point, box

from gridexpand.common import ghd
from gridexpand.common.building_components import build_building_components
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.scenario_config import ScenarioConfig


def _row(objectid, building_use_id, *, use, res=0.0, nonres=0.0, floor_area=100.0, floors=2, direct=False):
    return {
        "objectid": objectid, "floor_area": floor_area, "floor_number": floors,
        "residential_floor_area": res, "nonresidential_floor_area": nonres, "nonresidential_use": use,
        "households": 2 if res > 0 else 1, "occupants": 4.0 if res > 0 else None,
        "building_use": "Mixed" if res > 0 and nonres > 0 else ("Residential" if res > 0 else "Commercial"),
        "building_use_id": building_use_id, "building_type": None, "mix_score": None,
        "mix_rule": "standard" if res > 0 and nonres > 0 else "full_nonresidential", "mix_confidence": None,
        "residential_peak_load_in_kw": 33.65 if res > 0 else 0.0,
        "nonresidential_peak_load_in_kw": nonres * 0.079 if nonres > 0 else 0.0,
        "nonresidential_mv_direct": direct if nonres > 0 else None, "peak_load_in_kw": 10.0, "bus": 1,
    }


def _physical():
    return pd.DataFrame([
        _row("GEN", "31001_2000", use="Commercial", nonres=200.0),
        _row("KITA", "31001_3065", use="Public", nonres=200.0),
        _row("CHURCH", "31001_3041", use="Public", nonres=1500.0, floor_area=300.0, floors=5),
        _row("SHOP", "31001_1000", use="Commercial", res=200.0, nonres=100.0, floors=3),
        _row("UNK", "31001_9998", use="Unknown", nonres=200.0),
        _row("BIG", "31001_2000", use="Commercial", nonres=2000.0, floor_area=1000.0, direct=True),
    ])


def _evidence(**rows):
    frame = ghd.empty_evidence(_physical()["objectid"])
    for objectid, values in rows.items():
        for key, value in values.items():
            frame.loc[objectid, key] = value
    return frame


def _areas(components):
    nonres = components[components["component_category"].isin(["Commercial", "Public"])]
    return dict(zip(nonres["component_id"], nonres["effective_floor_area_m2"]))


def test_rules_off_return_the_manifest_unchanged():
    physical = _physical()
    components = build_building_components(physical)
    result, audit = ghd.apply_ghd_policy(physical, components, ghd.GhdConfig(), None)
    pd.testing.assert_frame_equal(result, components)
    decisions = dict(zip(audit["objectid"], audit["decision"]))
    assert decisions == {"GEN": "active", "KITA": "active", "CHURCH": "active", "SHOP": "active",
                         "UNK": "non_demand_unknown", "BIG": "mv_direct"}


def test_gating_keeps_specific_functions_and_buildings_with_activity_evidence():
    physical = _physical()
    config = ghd.GhdConfig(activity_gating=True, osm=ghd.OsmEvidenceConfig(source="file", file="unused.csv"))
    result, audit = ghd.apply_ghd_policy(physical, build_building_components(physical), config, _evidence())
    assert set(_areas(result)) == {"KITA::public", "CHURCH::public", "BIG::commercial"}
    assert "SHOP::residential" in set(result["component_id"])  # the residential part stays
    decisions = dict(zip(audit["objectid"], audit["decision"]))
    assert decisions["GEN"] == decisions["SHOP"] == "inactive_no_activity_evidence"
    assert decisions["BIG"] == "mv_direct" and decisions["UNK"] == "non_demand_unknown"

    evidence = _evidence(GEN={"activity": True, "activity_tags": "shop=bakery", "activity_category": "Commercial"})
    result, _ = ghd.apply_ghd_policy(physical, build_building_components(physical), config, evidence)
    assert "GEN::commercial" in _areas(result)


def test_gating_turns_an_unknown_building_with_evidence_into_ghd():
    physical = _physical()
    evidence = _evidence(UNK={"activity": True, "activity_tags": "amenity=doctors", "activity_category": "Public"})
    gated = ghd.GhdConfig(activity_gating=True, osm=ghd.OsmEvidenceConfig(source="file", file="unused.csv"))
    result, audit = ghd.apply_ghd_policy(physical, build_building_components(physical), gated, evidence)
    added = result[result["objectid"].eq("UNK")]
    assert added["component_id"].tolist() == ["UNK::public"]
    assert added["effective_floor_area_m2"].tolist() == [200.0] and bool(added["included_in_lv"].iloc[0])
    assert audit.loc[audit["objectid"].eq("UNK"), "decision"].item() == "activated_unknown"
    # Without gating, Unknown stays non-demand whatever the evidence says.
    one_storey = ghd.GhdConfig(single_volume_one_storey=True)
    result, _ = ghd.apply_ghd_policy(physical, build_building_components(physical), one_storey, evidence)
    assert "UNK" not in set(result["objectid"])


def test_single_volume_buildings_count_one_storey():
    physical = _physical()
    config = ghd.GhdConfig(single_volume_one_storey=True)
    result, audit = ghd.apply_ghd_policy(physical, build_building_components(physical), config, None)
    areas = _areas(result)
    assert areas["CHURCH::public"] == 300.0  # ALKIS 3041: the footprint
    assert areas["GEN::commercial"] == 200.0 and areas["SHOP::commercial"] == 100.0  # unchanged
    assert audit.loc[audit["objectid"].eq("CHURCH"), "storey_rule"].item() == "single_volume_alkis"


def test_a_specific_osm_building_value_overrides_the_alkis_function():
    physical = _physical()
    config = ghd.GhdConfig(single_volume_one_storey=True)
    evidence = _evidence(CHURCH={"osm_building": "house"}, GEN={"osm_building": "church"}, KITA={"osm_building": "yes"})
    result, audit = ghd.apply_ghd_policy(physical, build_building_components(physical), config, evidence)
    areas = _areas(result)
    assert areas["CHURCH::public"] == 1500.0  # OSM says it is not single-volume
    assert areas["GEN::commercial"] == 100.0  # OSM church on a generic commercial building
    assert areas["KITA::public"] == 200.0  # building=yes: the ALKIS list decides (3065 is not listed)
    assert audit.loc[audit["objectid"].eq("GEN"), "storey_rule"].item() == "single_volume_osm"


def test_osm_levels_replace_floor_number_of_fully_non_residential_buildings():
    physical = _physical()
    config = ghd.GhdConfig(osm_levels=True, osm=ghd.OsmEvidenceConfig(source="file", file="unused.csv"))
    evidence = _evidence(GEN={"osm_levels": 1.0}, SHOP={"osm_levels": 1.0})
    result, _ = ghd.apply_ghd_policy(physical, build_building_components(physical), config, evidence)
    areas = _areas(result)
    assert areas["GEN::commercial"] == 100.0 and areas["SHOP::commercial"] == 100.0 and areas["KITA::public"] == 200.0


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("shop", "bakery", "Commercial"), ("shop", "vacant", None), ("craft", "carpenter", "Commercial"),
        ("office", "company", "Commercial"), ("office", "government", "Public"), ("healthcare", "doctor", "Public"),
        ("amenity", "restaurant", "Commercial"), ("amenity", "school", "Public"), ("amenity", "parking", None),
        ("amenity", "bench", None), ("tourism", "hotel", "Commercial"), ("tourism", "viewpoint", None),
        ("leisure", "fitness_centre", "Commercial"), ("leisure", "park", None), ("building", "retail", None),
    ],
)
def test_activity_tags(key, value, expected):
    assert ghd.activity_category(key, value) == expected


def test_osm_features_are_matched_to_footprints():
    footprints = pd.DataFrame({"objectid": ["A", "B"], "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)]})
    features = pd.DataFrame({
        "key": ["shop", "amenity", "shop", "amenity", "amenity"],
        "value": ["bakery", "cafe", "florist", "school", "parking"],
        "geometry": [Point(5, 5), Point(12, 5), Point(40, 40), box(19, -1, 31, 11), Point(25, 5)],
    })
    buildings = pd.DataFrame({"building": ["church"], "levels": [1], "geometry": [box(0, 0, 10, 10.5)]})
    evidence = ghd.match_osm_evidence(footprints, features, buildings, point_buffer_m=5.0)
    assert evidence.loc["A", "activity_tags"] == "amenity=cafe;shop=bakery"  # inside, and 2 m off the wall
    assert evidence.loc["A", "activity_category"] == "Commercial"
    assert evidence.loc["B", "activity_tags"] == "amenity=school"  # a school polygon covering B; parking ignored
    assert evidence.loc["B", "activity_category"] == "Public"
    assert evidence.loc["A", "osm_building"] == "church" and evidence.loc["A", "osm_levels"] == 1.0
    assert pd.isna(evidence.loc["B", "osm_building"])


def test_a_point_between_two_buildings_goes_to_the_nearest_within_the_buffer():
    footprints = pd.DataFrame({"objectid": ["A", "B"], "geometry": [box(0, 0, 10, 10), box(14, 0, 24, 10)]})
    features = pd.DataFrame({"key": ["shop"], "value": ["kiosk"], "geometry": [Point(13, 5)]})
    evidence = ghd.match_osm_evidence(footprints, features, point_buffer_m=5.0)
    assert evidence["activity"].to_dict() == {"A": False, "B": True}


def test_file_evidence_source(tmp_path):
    path = tmp_path / "evidence.csv"
    pd.DataFrame({"objectid": ["GEN"], "activity": [True], "activity_tags": ["shop=bakery"],
                  "activity_category": ["Commercial"], "osm_building": ["retail"], "osm_levels": [2]}).to_csv(path, index=False)
    source = ghd.evidence_source(ghd.OsmEvidenceConfig(source="file", file=str(path)))
    evidence = source.evidence(["GEN", "KITA"])
    assert evidence["activity"].to_dict() == {"GEN": True, "KITA": False}
    assert evidence.loc["GEN", "osm_levels"] == 2.0


def test_ghd_block_parsing():
    assert ScenarioConfig.from_dict(yaml.safe_load((SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml").read_text())).ghd == ghd.GhdConfig()
    raw = yaml.safe_load((SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml").read_text(encoding="utf-8"))
    raw["ghd"] = {
        "activity_gating": True, "single_volume_one_storey": True,
        "osm": {"source": "postgis", "activity_layers": [
            {"table": "opendata.osm_poi_point", "key_column": "osm_type", "value_column": "osm_subtype"},
            {"table": "opendata.osm_amenity_polygon", "key": "amenity", "value_column": "osm_type"},
        ], "building_layer": {"table": "opendata.osm_building_polygon"}},
    }
    config = ScenarioConfig.from_dict(copy.deepcopy(raw)).ghd
    assert config.activity_gating and config.single_volume_one_storey and not config.osm_levels
    assert [layer.key for layer in config.osm.activity_layers] == [None, "amenity"]
    layer = config.osm.building_layer  # pgosm-flex defaults
    assert (layer.building_column, layer.levels_column, layer.type_column, layer.type_value) == ("osm_subtype", "levels", "osm_type", "building")
    for broken, message in [
        ({"activity_gating": True}, "needs ghd.osm"),
        ({"osm_levels": True, "osm": {"source": "postgis", "activity_layers": []}}, "building_layer"),
        ({"osm": {"source": "postgis", "buildings_table": "basedata.buildings; drop"}}, "plain SQL"),
        ({"osm": {"source": "postgis", "activity_layers": [{"table": "t", "key": "shop", "key_column": "k"}]}}, "exactly one"),
        ({"single_volume_one_storey": "yes"}, "true or false"),
        ({"osm": {"source": "postgis", "building_layer": {"table": "t", "type_column": None}}}, "type_column and type_value"),
        ({"unknown_switch": True}, "Unknown ghd option"),
    ]:
        raw["ghd"] = broken
        with pytest.raises(ValueError, match=message):
            ScenarioConfig.from_dict(copy.deepcopy(raw))
