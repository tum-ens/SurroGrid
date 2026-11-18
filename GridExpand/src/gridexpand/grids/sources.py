"""Inputs of the matcher: connection points from grid files or a CSV, buildings from InfDB.

Grid files are pandapower JSON, one per LV grid; the file stem is the grid id. Every row of ``net.load`` is a
connection point at the coordinates of its bus (``net.bus.geo``); the optional load columns ``address``,
``on_building``, ``exclude`` and ``exclude_reason`` are taken over, the load ``name`` becomes the remark. The
points should sit where the service cable reaches the building; when the load buses of a grid are street nodes,
give the connection points as a CSV instead (:func:`points_from_csv`).

Buildings are read from InfDB ``basedata.buildings`` in the bounding box of the points; a building whose whole load
pylovo's rules put on MV (:func:`gridexpand.grids.attributes.mv_direct_buildings`) is flagged ``mv_direct``. Every
query runs in a read-only transaction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shapely
from shapely import from_wkb
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry

from gridexpand.grids import attributes

TARGET_CRS = "EPSG:25832"
OPTIONAL_LOAD_COLUMNS = ("address", "on_building", "exclude", "exclude_reason")

BUILDINGS_SQL = attributes.INFDB_SELECT.replace(
    "FROM {schema}.buildings", ", ST_SRID(geom) AS srid, ST_AsBinary(geom) AS wkb\nFROM {schema}.buildings"
) + "WHERE geom && ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 25832)\nORDER BY objectid\n"


def grid_files(paths: Iterable[Path]) -> dict[str, Path]:
    """Grid id (file stem) -> pandapower JSON file; directories contribute their ``*.json`` files."""
    result: dict[str, Path] = {}
    for path in paths:
        path = Path(path).expanduser()
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not files or not all(item.is_file() for item in files):
            raise ValueError(f"No grid files at {path}.")
        for item in files:
            if item.stem in result:
                raise ValueError(f"Two grid files with the grid id {item.stem!r}: {result[item.stem]} and {item}.")
            result[item.stem] = item
    return result


def _geojson(value: Any) -> dict[str, Any] | None:
    """A GeoJSON mapping from ``bus.geo``/``line.geo``; pandapower writes coordinate tuples as ``(x, y)``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = json.loads(value.replace("(", "[").replace(")", "]"))
            except json.JSONDecodeError:
                return None
    return value if isinstance(value, dict) else None


def bus_coordinates(net) -> dict[int, tuple[float, float]]:
    """Bus -> (x, y) from ``net.bus.geo`` (GeoJSON points); buses without coordinates are left out."""
    if "geo" not in net.bus.columns:
        return {}
    result = {}
    for bus, value in net.bus["geo"].items():
        geometry = _geojson(value)
        coordinates = geometry.get("coordinates") if geometry else None
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            x, y = float(coordinates[0]), float(coordinates[1])
            if math.isfinite(x) and math.isfinite(y):
                result[int(bus)] = (x, y)
    return result


def points_from_net(net, grid_id: str) -> pd.DataFrame:
    """One connection point per row of ``net.load``, at its bus coordinates."""
    coordinates = bus_coordinates(net)
    rows, missing = [], []
    for index, load in net.load.iterrows():
        bus = int(load["bus"])
        if bus not in coordinates:
            missing.append(bus)
            continue
        row = {"point_id": f"{grid_id}:{index}", "grid_id": str(grid_id), "bus": bus,
               "x": coordinates[bus][0], "y": coordinates[bus][1], "remark": load.get("name")}
        row.update({name: load.get(name) for name in OPTIONAL_LOAD_COLUMNS if name in net.load.columns})
        rows.append(row)
    if missing:
        raise ValueError(f"Grid {grid_id}: {len(missing)} loads sit on buses without coordinates, "
                         f"e.g. bus {missing[0]}.")
    return pd.DataFrame(rows)


def lines_from_net(net) -> list[BaseGeometry]:
    """Line geometries: ``net.line.geo`` where present, else the straight segment between the bus coordinates."""
    coordinates = bus_coordinates(net)
    result = []
    for _, line in net.line.iterrows():
        geometry = _geojson(line.get("geo")) if "geo" in net.line.columns else None
        if geometry:
            try:
                result.append(shape(geometry))
                continue
            except (ValueError, TypeError, AttributeError):
                pass
        ends = [coordinates.get(int(line["from_bus"])), coordinates.get(int(line["to_bus"]))]
        if all(ends) and ends[0] != ends[1]:
            result.append(LineString(ends))
    return result


def points_from_csv(path: Path) -> pd.DataFrame:
    """Connection points of a CSV: ``grid_id, bus, x, y`` and optionally ``point_id, address, remark,
    on_building, exclude, exclude_reason`` (``point_id`` defaults to ``<grid_id>:<row>``)."""
    frame = pd.read_csv(path, dtype={"point_id": str, "grid_id": str, "address": str, "remark": str,
                                     "exclude_reason": str})
    missing = [name for name in ("grid_id", "bus", "x", "y") if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}.")
    if "point_id" not in frame.columns:
        frame.insert(0, "point_id", [f"{grid}:{row}" for row, grid in enumerate(frame["grid_id"])])
    return frame


def to_target_crs(points: pd.DataFrame, lines: dict[str, list[BaseGeometry]], crs: str
                  ) -> tuple[pd.DataFrame, dict[str, list[BaseGeometry]]]:
    """Points and lines in EPSG:25832 (unchanged when ``crs`` is already that CRS)."""
    from pyproj import CRS, Transformer

    if CRS.from_user_input(crs) == CRS.from_user_input(TARGET_CRS):
        return points, lines
    transformer = Transformer.from_crs(crs, TARGET_CRS, always_xy=True)

    def move(xy: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    points = points.copy()
    moved = move(points[["x", "y"]].to_numpy(float))
    points["x"], points["y"] = moved[:, 0], moved[:, 1]
    lines = {grid_id: [shapely.transform(line, move) for line in items] for grid_id, items in lines.items()}
    return points, lines


def bounds(points: pd.DataFrame, margin: float) -> tuple[float, float, float, float]:
    """Bounding box of the points, widened by ``margin`` metres."""
    return (float(points["x"].min()) - margin, float(points["y"].min()) - margin,
            float(points["x"].max()) + margin, float(points["y"].max()) + margin)


def read_buildings(engine, box: tuple[float, float, float, float], *, schema: str = "basedata"
                   ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """InfDB buildings in ``box`` (EPSG:25832) with the MV-direct flag, and a note on the rule.

    ``households`` stays InfDB's own value (the matcher tells houses from outbuildings by it); pylovo's household
    fallback only enters the MV-direct derivation.
    """
    from sqlalchemy import text

    if not schema.isidentifier():
        raise ValueError(f"Invalid schema name {schema!r}.")
    xmin, ymin, xmax, ymax = box
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = conn.execute(text(BUILDINGS_SQL.format(schema=schema)),
                            {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}).mappings().all()
        conn.rollback()
    frame = pd.DataFrame([dict(row) for row in rows])
    note = {"mv_direct_rule": "InfDB non-residential share above "
                              f"{attributes.PYLOVO_MV_DIRECT_THRESHOLD_KW:g} kW and no residential floor area "
                              "(pylovo 8afedcc rules)"}
    if frame.empty:
        return frame, note
    if not frame["srid"].eq(25832).all():
        raise ValueError(f"{schema}.buildings must be in EPSG:25832.")
    frame["geometry"] = [from_wkb(bytes(value)) for value in frame.pop("wkb")]
    frame = frame.drop(columns="srid")
    frame["objectid"] = frame["objectid"].astype(str)
    derived = attributes.pylovo_equivalent_buildings(frame.drop(columns="geometry"))
    frame["mv_direct"] = attributes.mv_direct_buildings(derived).to_numpy()
    return frame, note
