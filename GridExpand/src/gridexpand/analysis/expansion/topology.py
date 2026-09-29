"""Radial topology of one LV grid for the staged expansion rules (synthetic and real grids alike).

The station root is the LV side of an in-service transformer fed by the external grid (pylovo nets),
else the external-grid bus (the real SWF and ÜZW grids carry the external grid on the LV busbar). The
station busbar is the root plus everything tied to it by zero-length lines or closed bus-bus switches.

- An **outlet** route leaves the station busbar.
- A **service line** is the route into a load bus with at most one line neighbour: the same terminal
  service connection that ``powerflow.network.comparison_backbone_scope`` removes.
- Bus coordinates are returned in metres (EPSG:25832). pylovo stores WGS84 longitude/latitude, the
  real grids already use EPSG:25832.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from gridexpand.powerflow import network

TARGET_EPSG = 25832
ZERO_LENGTH_KM = 1e-9


@dataclass
class GridTopology:
    """Tree of one grid rooted at the station busbar."""

    root: int
    station_buses: frozenset[int]
    parent: dict[int, int | None]
    distance_km: dict[int, float]
    load_buses: frozenset[int]
    line_neighbors: dict[int, set[int]]
    coordinates: dict[int, tuple[float, float]] = field(default_factory=dict)

    def child(self, bus_a: int, bus_b: int) -> int:
        """The downstream bus of an edge (the farther one if the edge is not a tree edge)."""
        if self.parent.get(bus_b) == bus_a:
            return bus_b
        if self.parent.get(bus_a) == bus_b:
            return bus_a
        return bus_b if self.distance_km.get(bus_b, math.inf) >= self.distance_km.get(bus_a, math.inf) else bus_a

    def is_outlet(self, bus_a: int, bus_b: int) -> bool:
        return (bus_a in self.station_buses) != (bus_b in self.station_buses)

    def is_service(self, bus_a: int, bus_b: int) -> bool:
        child = self.child(bus_a, bus_b)
        return child in self.load_buses and len(self.line_neighbors.get(child, ())) <= 1

    def coordinate_array(self) -> np.ndarray | None:
        if not self.coordinates:
            return None
        return np.array(list(self.coordinates.values()), dtype=float)


def station_root(net) -> int:
    """LV bus of the in-service transformer fed by the external grid, else the external-grid bus."""
    ext_bus = int(net.ext_grid["bus"].iloc[0]) if not net.ext_grid.empty else None
    trafo = getattr(net, "trafo", None)
    if trafo is not None and not trafo.empty:
        active = trafo[trafo["in_service"].fillna(True).astype(bool)] if "in_service" in trafo.columns else trafo
        for row in active.itertuples():
            if ext_bus is None or int(row.hv_bus) == ext_bus:
                return int(row.lv_bus)
    if ext_bus is not None:
        return ext_bus
    return int(net.bus.index[0])


@lru_cache(maxsize=4)
def _transformer(source_epsg: int):
    from pyproj import Transformer

    return Transformer.from_crs(source_epsg, TARGET_EPSG, always_xy=True)


def bus_coordinates(net) -> dict[int, tuple[float, float]]:
    """Bus coordinates in EPSG:25832 metres (GeoJSON points in ``bus.geo``; WGS84 is projected)."""
    if "geo" not in net.bus.columns:
        return {}
    points: dict[int, tuple[float, float]] = {}
    for bus, value in net.bus["geo"].items():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if not isinstance(value, dict):
            continue
        coordinates = value.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            continue
        try:
            x, y = float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points[int(bus)] = (x, y)
    if points and all(abs(x) <= 180.0 and abs(y) <= 90.0 for x, y in points.values()):
        transformer = _transformer(4326)
        xs, ys = transformer.transform([p[0] for p in points.values()], [p[1] for p in points.values()])
        points = {bus: (float(x), float(y)) for bus, x, y in zip(points, xs, ys)}
    return points


def build_topology(net) -> GridTopology:
    """Tree, station busbar, distances, load buses and coordinates of ``net``."""
    root = station_root(net)
    adjacency = network.grid_adjacency(net)
    parents = network.parent_tree_from_root(adjacency, root)
    lines = net.line.loc[network.active_line_index(net)]
    length_by_edge: dict[frozenset[int], float] = {}
    line_neighbors: dict[int, set[int]] = {}
    for line in lines.itertuples():
        a, b = int(line.from_bus), int(line.to_bus)
        edge = frozenset((a, b))
        length = float(line.length_km) if pd.notna(line.length_km) else 0.0
        zero = length <= ZERO_LENGTH_KM or (
            float(getattr(line, "r_ohm_per_km", 1.0) or 0.0) == 0.0
            and float(getattr(line, "x_ohm_per_km", 1.0) or 0.0) == 0.0
        )
        length_by_edge[edge] = min(length_by_edge.get(edge, math.inf), 0.0 if zero else length)
        line_neighbors.setdefault(a, set()).add(b)
        line_neighbors.setdefault(b, set()).add(a)
    distance = {root: 0.0}
    station = {root}
    # Breadth-first order of the parent tree: a parent always precedes its children.
    order = [root]
    children: dict[int, list[int]] = {}
    for bus, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, []).append(bus)
    head = 0
    while head < len(order):
        bus = order[head]
        head += 1
        for child in sorted(children.get(bus, [])):
            length = length_by_edge.get(frozenset((bus, child)), 0.0)  # switches: 0
            distance[child] = distance[bus] + length
            if bus in station and length <= ZERO_LENGTH_KM:
                station.add(child)
            order.append(child)
    loads = net.load
    if "in_service" in loads.columns:
        loads = loads[loads["in_service"].fillna(True).astype(bool)]
    load_buses = frozenset(int(bus) for bus in loads["bus"].dropna())
    return GridTopology(
        root=root,
        station_buses=frozenset(station),
        parent=parents,
        distance_km=distance,
        load_buses=load_buses,
        line_neighbors=line_neighbors,
        coordinates=bus_coordinates(net),
    )
