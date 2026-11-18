"""Connection points and lines from grid files and CSVs, CRS transformation and the read-only building reader."""

from __future__ import annotations

import pandapower as pp
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from gridexpand.grids import sources


def small_net(*, line_geo=True):
    net = pp.create_empty_network()
    station = pp.create_bus(net, vn_kv=0.4, geodata=(100.0, 200.0))
    house = pp.create_bus(net, vn_kv=0.4, geodata=(150.0, 200.0))
    pp.create_bus(net, vn_kv=0.4)  # no coordinates, no load
    pp.create_line_from_parameters(net, station, house, length_km=0.05, r_ohm_per_km=0.2, x_ohm_per_km=0.08,
                                   c_nf_per_km=0.0, max_i_ka=0.2,
                                   geodata=[(100.0, 200.0), (125.0, 210.0), (150.0, 200.0)] if line_geo else None)
    pp.create_load(net, house, p_mw=0.01, name="Haus 1")
    pp.create_load(net, house, p_mw=0.01, name="Haus 2")
    net.load["address"] = ["Hauptstraße 1", None]
    return net


def test_points_and_lines_from_a_saved_grid(tmp_path):
    pp.to_json(small_net(), tmp_path / "LV_7.json")
    files = sources.grid_files([tmp_path])
    assert list(files) == ["LV_7"]
    net = pp.from_json(str(files["LV_7"]))
    points = sources.points_from_net(net, "LV_7")
    assert points[["point_id", "grid_id", "bus", "x", "y", "remark"]].to_dict("records") == [
        {"point_id": "LV_7:0", "grid_id": "LV_7", "bus": 1, "x": 150.0, "y": 200.0, "remark": "Haus 1"},
        {"point_id": "LV_7:1", "grid_id": "LV_7", "bus": 1, "x": 150.0, "y": 200.0, "remark": "Haus 2"},
    ]
    assert points["address"].tolist()[0] == "Hauptstraße 1"
    (line,) = sources.lines_from_net(net)
    assert list(line.coords) == [(100.0, 200.0), (125.0, 210.0), (150.0, 200.0)]
    (straight,) = sources.lines_from_net(small_net(line_geo=False))
    assert list(straight.coords) == [(100.0, 200.0), (150.0, 200.0)]


def test_loads_need_bus_coordinates():
    net = small_net()
    pp.create_load(net, 2, p_mw=0.01)
    with pytest.raises(ValueError, match="without coordinates"):
        sources.points_from_net(net, "g")


def test_grid_files_need_unique_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pp.to_json(small_net(), tmp_path / "a" / "LV_1.json")
    pp.to_json(small_net(), tmp_path / "b" / "LV_1.json")
    with pytest.raises(ValueError, match="Two grid files"):
        sources.grid_files([tmp_path / "a", tmp_path / "b"])
    with pytest.raises(ValueError, match="No grid files"):
        sources.grid_files([tmp_path / "missing.json"])


def test_points_from_csv(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("grid_id,bus,x,y,address,on_building\nLV_1,4,1.5,2.5,Hauptstraße 1,true\nLV_1,5,3,4,,\n",
                    encoding="utf-8")
    points = sources.points_from_csv(path)
    assert points["point_id"].tolist() == ["LV_1:0", "LV_1:1"]
    assert points["grid_id"].tolist() == ["LV_1", "LV_1"]
    path.write_text("grid,bus,x,y\n1,2,3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the columns"):
        sources.points_from_csv(path)


def test_points_and_lines_are_transformed_to_25832():
    points = pd.DataFrame({"x": [11.5], "y": [48.1]})
    lines = {"g": [LineString([(11.5, 48.1), (11.6, 48.2)])]}
    moved, moved_lines = sources.to_target_crs(points, lines, "EPSG:4326")
    assert moved.at[0, "x"] == pytest.approx(686_000, abs=2_000)
    assert moved.at[0, "y"] == pytest.approx(5_329_000, abs=2_000)
    assert moved_lines["g"][0].coords[0] == pytest.approx((moved.at[0, "x"], moved.at[0, "y"]))
    same, _ = sources.to_target_crs(points, lines, "EPSG:25832")
    assert same is points


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)

    def scalar(self):
        return self.rows[0][0] if self.rows else None


def _row(objectid, x, use, residential, nonresidential, households, street=None):
    return {"objectid": objectid, "postcode": 80331, "street": street, "house_number": "1" if street else None,
            "building_use": use, "building_use_id": "31001_1010", "building_type": None, "floor_area": 100.0,
            "floor_number": 2, "residential_floor_area": residential, "nonresidential_floor_area": nonresidential,
            "households": households, "occupants": None, "type": use,
            "nonresidential_use": "Commercial" if nonresidential else None, "srid": 25832,
            "wkb": box(x, 0, x + 10, 10).wkb}


class _Connection:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.log.append(sql.strip().split("\n")[0])
        if "basedata.buildings" in sql:
            return _Result([
                _row("a", 0, "Residential", 120.0, 0.0, 2, street="Hauptstraße"),
                _row("b", 20, "Commercial", 0.0, 2000.0, None),   # 158 kW non-residential, nothing else
                _row("c", 40, "Mixed", 300.0, 2000.0, 3),         # its flats stay on LV
            ])
        return _Result([])

    def rollback(self):
        self.log.append("ROLLBACK")


class _Engine:
    def __init__(self):
        self.log = []

    def connect(self):
        return _Connection(self.log)


def test_read_buildings_is_read_only_and_flags_mv_direct_from_infdb():
    engine = _Engine()
    frame, note = sources.read_buildings(engine, (0, 0, 60, 50))
    assert engine.log[0] == "SET TRANSACTION READ ONLY" and engine.log[-1] == "ROLLBACK"
    assert not any("pylovo" in line for line in engine.log)
    assert frame.set_index("objectid")["mv_direct"].to_dict() == {"a": False, "b": True, "c": False}
    assert frame["households"].isna().tolist() == [False, True, False]  # InfDB's own value, no fallback
    assert frame.at[0, "geometry"].equals(box(0, 0, 10, 10))
    assert "100 kW" in note["mv_direct_rule"]


def test_an_mv_external_grid_moves_to_the_lv_busbar(tmp_path):
    from gridexpand.powerflow.run_real_scenario_powerflow import load_real_net

    net = pp.create_empty_network()
    mv = pp.create_bus(net, vn_kv=20.0)
    lv = pp.create_bus(net, vn_kv=0.4, geodata=(0.0, 0.0))
    house = pp.create_bus(net, vn_kv=0.4, geodata=(50.0, 0.0))
    pp.create_ext_grid(net, mv)
    pp.create_transformer(net, mv, lv, std_type="0.4 MVA 20/0.4 kV")
    pp.create_line_from_parameters(net, lv, house, length_km=0.05, r_ohm_per_km=0.2, x_ohm_per_km=0.08,
                                   c_nf_per_km=0.0, max_i_ka=0.2)
    pp.create_load(net, house, p_mw=0.01)
    pp.to_json(net, tmp_path / "g.json")
    loaded, rating = load_real_net(tmp_path / "g.json")
    assert int(loaded.ext_grid.at[0, "bus"]) == lv and not loaded.bus.at[mv, "in_service"]
    assert not loaded.trafo["in_service"].any() and rating == pytest.approx(0.4)
