"""``gridexpand match`` end to end, with the database replaced by a building table."""

from __future__ import annotations

import json

import pandapower as pp
import pandas as pd
import pytest
from shapely.geometry import box

from gridexpand import cli
from gridexpand.grids import match


def grid(tmp_path, name, house_x):
    net = pp.create_empty_network()
    station = pp.create_bus(net, vn_kv=0.4, geodata=(house_x - 40.0, -5.0))
    house = pp.create_bus(net, vn_kv=0.4, geodata=(house_x + 5.0, 5.0))
    pp.create_line_from_parameters(net, station, house, length_km=0.05, r_ohm_per_km=0.2, x_ohm_per_km=0.08,
                                   c_nf_per_km=0.0, max_i_ka=0.2)
    pp.create_load(net, house, p_mw=0.01, name="house")
    pp.to_json(net, tmp_path / f"{name}.json")


def buildings(_box):
    frame = pd.DataFrame([
        {"objectid": "h1", "geometry": box(0, 0, 10, 10), "street": "Hauptstraße", "house_number": "1",
         "building_use": "Residential", "building_use_id": "31001_1010", "households": 1,
         "residential_floor_area": 90.0, "mv_direct": False},
        {"objectid": "h2", "geometry": box(100, 0, 110, 10), "street": "Hauptstraße", "house_number": "2",
         "building_use": "Residential", "building_use_id": "31001_1010", "households": 1,
         "residential_floor_area": 90.0, "mv_direct": False},
        {"objectid": "n1", "geometry": box(-20, -2, -10, 8), "street": None, "house_number": None,
         "building_use": "Residential", "building_use_id": "31001_1010", "households": 2,
         "residential_floor_area": 150.0, "mv_direct": False},
    ])
    return frame, {"pylovo_version_id": "1", "postcodes_without_pylovo_result": []}


def test_match_writes_the_tables_and_the_report(tmp_path):
    grids = tmp_path / "grids"
    grids.mkdir()
    grid(grids, "LV_1", 0.0)
    grid(grids, "LV_2", 100.0)
    out = tmp_path / "out"
    args = match.build_parser().parse_args([str(grids), "--output", str(out)])
    result = match.run(args, read_buildings=buildings)
    assert result.supply[["building_objectid", "grid_id"]].to_dict("records") == [
        {"building_objectid": "h1", "grid_id": "LV_1"}, {"building_objectid": "h2", "grid_id": "LV_2"}]
    assert sorted(path.name for path in out.iterdir()) == [
        "building_supply.csv", "grid_quality.csv", "matching_report.json", "point_matches.csv",
        "unconnected_buildings.csv"]
    report = json.loads((out / "matching_report.json").read_text(encoding="utf-8"))
    assert report["inputs"]["grids"] == ["LV_1", "LV_2"]
    assert report["rules"] == {"covering_footprint": 2}
    unconnected = pd.read_csv(out / "unconnected_buildings.csv")
    assert unconnected[["building_objectid", "reason"]].to_dict("records") == [
        {"building_objectid": "n1", "reason": "residential_without_a_point"}]


def test_points_csv_must_fit_the_grids(tmp_path):
    grids = tmp_path / "grids"
    grids.mkdir()
    grid(grids, "LV_1", 0.0)
    points = tmp_path / "points.csv"
    points.write_text("grid_id,bus,x,y\nLV_9,1,5,5\n", encoding="utf-8")
    args = match.build_parser().parse_args([str(grids), "--output", str(tmp_path / "o"), "--points", str(points)])
    with pytest.raises(ValueError, match="without a grid file"):
        match.run(args, read_buildings=buildings)
    points.write_text("grid_id,bus,x,y\nLV_1,7,5,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no buses"):
        match.run(args, read_buildings=buildings)
    points.write_text("grid_id,bus,x,y,address\nLV_1,1,15,5,Hauptstraße 1\n", encoding="utf-8")
    assert match.run(args, read_buildings=buildings).points.at[0, "rule"] == "unique_address"


def test_config_file_needs_a_matching_block(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("run:\n  id: x\nmatching:\n  nearest_residential_m: 10\n", encoding="utf-8")
    assert match.load_config(path).nearest_residential_m == 10.0
    path.write_text("run:\n  id: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no matching: block"):
        match.load_config(path)


def test_help_needs_no_database(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["match", "--help"])
    assert stop.value.code == 0
    assert "connection points" in capsys.readouterr().out
