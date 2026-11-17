"""Rules of the connection-point matcher on small synthetic scenes (EPSG:25832 metres, 10 m square houses)."""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as rect

from gridexpand.grids.matching import (
    MatchingConfig,
    building_address,
    house_key,
    match_points,
    point_address,
    street_key,
)


def house(objectid, x, y, *, street="Hauptstraße", number="1", use="Residential", use_id="31001_1010",
          households=1, area=100.0, mv_direct=False, size=10.0):
    return {"objectid": objectid, "geometry": rect(x, y, x + size, y + size), "street": street,
            "house_number": number, "building_use": use, "building_use_id": use_id, "households": households,
            "residential_floor_area": area, "mv_direct": mv_direct}


def shed(objectid, x, y, **kwargs):
    return house(objectid, x, y, street=None, number=None, use="Commercial", use_id="31001_2000", households=0,
                 area=0.0, size=4.0, **kwargs)


def point(point_id, x, y, *, grid="g1", bus=1, address="", remark="", on_building=None, exclude=False):
    return {"point_id": point_id, "grid_id": grid, "bus": bus, "x": x, "y": y, "address": address,
            "remark": remark, "on_building": on_building, "exclude": exclude}


def match(points, buildings, grid_lines=None, **config):
    return match_points(pd.DataFrame(points), pd.DataFrame(buildings), MatchingConfig.from_dict(config),
                        grid_lines=grid_lines)


def decisions(result):
    return {row.point_id: (row.rule, row.building_objectid) for row in result.points.itertuples()}


def test_address_normalisation():
    assert street_key("Kiefernstr. 2") == street_key("Kiefernstraße") == "kiefernstrasse"
    assert house_key("30 Zählersäule") == "30" and house_key("55a") == "55a" and house_key("2 a") == "2"
    assert point_address("Hauptstr. 12a") == ("hauptstrasse", "12a")
    assert point_address("Hauptstraße") is None and building_address("Hauptstraße", None) is None
    assert building_address("Hauptstraße", "12A") == point_address("Hauptstr.12a")


def test_exact_address_and_covering_footprint():
    result = match([point("p", 5, 5, address="Hauptstraße 1")],
                   [house("a", 0, 0), house("b", 15, 0)])  # b: same address, does not cover the point
    assert decisions(result) == {"p": ("address_and_footprint", "a")}


def test_a_wall_box_counts_on_the_facade_a_pillar_does_not():
    scene = [house("a", 0, 0, number="3")]
    on_wall = match([point("p", -1, 5, address="Hauptstraße 3", on_building=True)], scene)
    pillar = match([point("p", -1, 5, address="Hauptstraße 3", on_building=False)], scene)
    assert decisions(on_wall) == {"p": ("address_and_footprint", "a")}
    assert decisions(pillar) == {"p": ("unique_address", "a")}


def test_unique_normalised_address_tells_2a_from_2():
    scene = [house("a", 20, 0, street="Kiefernstraße", number="2a"), house("b", 35, 0, street="Kiefernstraße",
                                                                            number="2")]
    assert decisions(match([point("p", 0, 5, address="Kiefernstr. 2a")], scene)) == {"p": ("unique_address", "a")}
    twins = scene + [house("c", 0, 20, street="Kiefernstraße", number="2a")]
    # Two buildings with the address: no unique match; the fallback takes the nearest free house.
    assert decisions(match([point("p", 0, 5, address="Kiefernstr. 2a")], twins))["p"][0] == "nearest_residential"


def test_covering_footprint_needs_a_house_or_a_wall_box():
    assert decisions(match([point("p", 5, 5)], [house("a", 0, 0)])) == {"p": ("covering_footprint", "a")}
    shop = [house("s", 0, 0, use="Commercial", households=0, area=0.0)]
    assert decisions(match([point("p", 5, 5, on_building=True)], shop)) == {"p": ("covering_footprint", "s")}
    assert decisions(match([point("p", 5, 5, on_building=False)], shop)) == {"p": ("no_residential_in_reach", None)}


def test_buildings_without_demand_and_mv_direct_buildings_carry_no_load():
    unknown_use = [house("u", 0, 0, use="Unknown")]
    kiosk = [house("k", 0, 0, use="Public", use_id="31001_2523", households=0, area=0.0)]
    mv = [house("m", 0, 0, mv_direct=True)]
    assert decisions(match([point("p", 5, 5)], unknown_use)) == {"p": ("building_without_demand", None)}
    assert decisions(match([point("p", 5, 5, address="Hauptstraße 1")], kiosk)) == {
        "p": ("building_without_demand", None)}
    result = match([point("p", 5, 5)], mv)
    assert decisions(result) == {"p": ("mv_direct_building", None)}
    assert result.points.at[0, "named_building_objectid"] == "m"
    assert decisions(match([point("p", 5, 5)], mv, exclude_mv_direct=False)) == {"p": ("covering_footprint", "m")}


def test_one_building_one_supply():
    scene = [house("a", 0, 0)]
    other_bus = match([point("p1", 5, 5, address="Hauptstraße 1", bus=1), point("p2", 6, 6, bus=2)], scene)
    assert decisions(other_bus) == {"p1": ("address_and_footprint", "a"), "p2": ("second_supply", None)}
    row = other_bus.points.set_index("point_id").loc["p2"]
    assert (row["named_building_objectid"], row["held_by_bus"]) == ("a", 1)
    assert other_bus.report["demoted_for_a_stronger_supply"] == {"covering_footprint": 1}
    same_bus = match([point("p1", 5, 5, address="Hauptstraße 1"), point("p2", 6, 6)], scene)
    assert decisions(same_bus) == {"p1": ("address_and_footprint", "a"), "p2": ("covering_footprint", "a")}
    assert same_bus.supply.to_dict("records") == [{"building_objectid": "a", "grid_id": "g1", "bus": 1,
                                                   "rule": "address_and_footprint", "points": 2,
                                                   "point_ids": "p1;p2"}]


def test_fallback_takes_the_nearest_free_house_never_a_shed():
    scene = [shed("garage", 2, 0), house("h1", 10, 0), house("h2", 30, 0)]
    result = match([point("p1", 0, 2, bus=1), point("p2", 0, 3, bus=2)], scene)
    # p1 is closer to a building (the garage), so it goes first and takes h1; h1 is then held by bus 1.
    assert decisions(result) == {"p1": ("nearest_residential", "h1"), "p2": ("no_residential_in_reach", None)}
    wider = match([point("p1", 0, 2, bus=1), point("p2", 0, 3, bus=2)], scene, nearest_residential_m=30.0)
    assert decisions(wider)["p2"] == ("nearest_residential", "h2")
    off = match([point("p1", 0, 2)], scene, nearest_residential_m=None)
    assert decisions(off) == {"p1": ("unmatched", None)}


def test_excluded_and_suspect_points():
    scene = [house("a", 0, 0)]
    assert decisions(match([point("p", 5, 5, address="Hauptstraße 1", exclude=True)], scene)) == {
        "p": ("excluded_point", None)}
    suspect = match([point("p", 5, 5, address="Hauptstraße 1", remark="Telekom Verteiler")], scene,
                    suspect_pattern="(?i)telekom")
    assert decisions(suspect) == {"p": ("unique_address", "a")}


def test_rules_can_be_switched_off():
    scene = [house("a", 0, 0)]
    result = match([point("p", 5, 5, address="Hauptstraße 1")], scene, address_and_footprint=False,
                   unique_address=False)
    assert decisions(result) == {"p": ("covering_footprint", "a")}
    nothing = match([point("p", 5, 5, address="Hauptstraße 1")], scene, address_and_footprint=False,
                    unique_address=False, covering_footprint=False, nearest_residential_m=None)
    assert decisions(nothing) == {"p": ("unmatched", None)}


def test_unconnected_buildings_near_the_lines():
    scene = [house("a", 0, 0), house("b", 20, 0), shed("s", 40, 0), house("m", 60, 0, mv_direct=True),
             house("far", 0, 100)]
    lines = {"g1": [LineString([(0, -5), (80, -5)])]}
    result = match([point("p", 5, 5, address="Hauptstraße 1")], scene, grid_lines=lines)
    assert result.unconnected[["building_objectid", "reason"]].to_dict("records") == [
        {"building_objectid": "b", "reason": "residential_without_a_point"},
        {"building_objectid": "m", "reason": "mv_direct_building"},
        {"building_objectid": "s", "reason": "outbuilding"},
    ]
    assert result.report["unconnected_buildings"] == {"residential_without_a_point": 1, "mv_direct_building": 1,
                                                      "outbuilding": 1}


def test_grid_quality_and_report():
    scene = [house("a", 0, 0), house("b", 40, 0, number="2")]
    result = match([point("p1", 5, 5, address="Hauptstraße 1"), point("p2", 45, 5, grid="g2"),
                    point("p3", 200, 200, grid="g2")], scene)
    quality = result.quality.set_index("grid_id")
    assert quality.loc["g1", ["points", "matched", "strong", "weak"]].tolist() == [1, 1, 1, 0]
    assert quality.loc["g2", ["points", "matched", "weak", "no_building"]].tolist() == [2, 1, 1, 1]
    assert quality.loc["g2", "excluded_share"] == 0.5
    assert result.report["rules"] == {"address_and_footprint": 1, "covering_footprint": 1,
                                      "no_residential_in_reach": 1}
    assert result.report["supplied_buildings"] == 2 and result.report["unconnected_buildings"] is None


def test_invalid_footprints_are_no_candidates():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    result = match([point("p", 5, 2)], [{**house("x", 0, 0), "geometry": bowtie}])
    assert decisions(result) == {"p": ("no_residential_in_reach", None)}
    assert result.report["buildings_without_valid_geometry"] == 1


@pytest.mark.parametrize("raw, message", [
    ({"radius": 10}, "Unknown matching keys"),
    ({"candidate_radius_m": -1}, "positive"),
    ({"nearest_residential_m": 40}, "must not exceed"),
    ({"unique_address": "yes"}, "true or false"),
    ({"suspect_pattern": "("}, "no regular expression"),
    ({"demand_uses": "Residential"}, "list of strings"),
])
def test_config_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        MatchingConfig.from_dict(raw)


def test_point_validation():
    with pytest.raises(ValueError, match="unique"):
        match([point("p", 5, 5), point("p", 6, 6)], [house("a", 0, 0)])
    with pytest.raises(ValueError, match="no coordinates"):
        match([point("p", None, 5)], [house("a", 0, 0)])
    with pytest.raises(ValueError, match="lack the columns"):
        match_points(pd.DataFrame([{"point_id": "p", "x": 1, "y": 1}]), pd.DataFrame([house("a", 0, 0)]))
