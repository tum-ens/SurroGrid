"""Match the connection points of given LV grids to InfDB buildings.

A given grid usually comes without its consumers: it holds connection points (the buses its loads hang on, often
with an address), not buildings. :func:`match_points` decides for every point which building it supplies and
records the rule that decided it. The rules are tried in the order of the strength of their evidence:

1. ``excluded_point``: the point is marked ``exclude`` (a street light, a telecom cabinet, a pump): no building.
2. ``address_and_footprint``: exactly one demand building has the point's exact address and covers the point, or,
   for a point mounted on a wall, lies within ``facade_offset_m`` of it.
3. ``unique_address``: exactly one building within ``candidate_radius_m`` has the point's normalised address
   ("str." written out, the first house number with a letter directly after it, so 2a is not 2).
4. ``covering_footprint``: exactly one footprint covers the point, and it is residential or the point is mounted on
   a wall.
5. ``building_without_demand``: the building named by rules 2-4 has no demand model (a use outside
   ``demand_uses`` or a use id in ``non_demand_use_ids``): the supply is real, but it is no building consumer.
6. ``mv_direct_building``: pylovo connects the building named by rules 2-4 straight to MV; no LV grid carries it.
7. ``second_supply``: the building named by rules 2-4 is already supplied from another bus by a stronger match.
   A building has one supply; several points on one bus may share it (the meters of an apartment house). Moving the
   point to the nearest free building instead would invent a consumer.
8. ``nearest_residential``: the remaining points, the closest first, take the nearest residential building within
   ``nearest_residential_m`` that no other bus supplies. Only a house: next to a house the nearest footprint is
   often a garage or a shed, which pylovo would load by its floor area.
9. ``no_residential_in_reach``, or ``unmatched`` with that fallback switched off: the point carries no load.

The defaults are the rules and distances of the real-vs-synthetic validation. Every rule can be switched off and
every distance changed (:class:`MatchingConfig`, the ``matching:`` block). Buildings near the grid lines that no
point supplies are listed with the reason, and every grid gets the shares of its points that rest on address
evidence, on geometry alone, or carry no load.

Coordinates are metres in EPSG:25832, the CRS of InfDB; :mod:`gridexpand.grids.sources` transforms other CRS.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import pandas as pd
from shapely import STRtree
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

#: Rules in the order of their strength; the description is the reason written for every decision.
RULES: dict[str, str] = {
    "excluded_point": "the point is marked as no building supply",
    "address_and_footprint": "exact address and a covering footprint, or the facade of a wall-mounted point",
    "unique_address": "exactly one building in reach has the normalised address",
    "covering_footprint": "exactly one footprint covers the point, and it is residential or the point is wall-mounted",
    "building_without_demand": "the named building has no demand model: a real supply, but no building consumer",
    "mv_direct_building": "pylovo connects the named building straight to MV, so no LV grid carries it",
    "second_supply": "the named building is already supplied from another bus by a stronger match",
    "nearest_residential": "the nearest residential building in reach that no other bus supplies",
    "no_residential_in_reach": "no residential building in reach that another bus does not supply",
    "unmatched": "no enabled rule names a building",
}
STRENGTH = {name: rank for rank, name in enumerate(RULES)}

#: Per-grid shares: address evidence, geometry alone, and the reasons a point carries no load.
QUALITY: dict[str, tuple[str, ...]] = {
    "strong": ("address_and_footprint", "unique_address"),
    "weak": ("covering_footprint", "nearest_residential"),
    "excluded_point": ("excluded_point",),
    "second_supply": ("second_supply",),
    "no_building": ("building_without_demand", "no_residential_in_reach", "unmatched"),
    "mv_direct": ("mv_direct_building",),
}

POINT_COLUMNS = ("point_id", "grid_id", "bus", "x", "y")
BUILDING_COLUMNS = ("objectid", "geometry")


@dataclass(frozen=True)
class MatchingConfig:
    """Rules and distances of the matcher (the ``matching:`` block; every key is optional).

    The three evidence rules can be switched off; ``nearest_residential_m: null`` switches the fallback off.
    ``suspect_pattern`` is a regular expression over a point's address and remark: a point it finds (a telecom
    cabinet named in the remark, say) never gets the exact-address rule, only the weaker ones.
    """

    candidate_radius_m: float = 30.0
    facade_offset_m: float = 2.0
    address_and_footprint: bool = True
    unique_address: bool = True
    covering_footprint: bool = True
    nearest_residential_m: float | None = 20.0
    exclude_mv_direct: bool = True
    demand_uses: tuple[str, ...] = ("Residential", "Commercial", "Public", "Mixed")
    non_demand_use_ids: tuple[str, ...] = ("31001_2523",)
    suspect_pattern: str | None = None
    unconnected_radius_m: float = 20.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "MatchingConfig":
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError("matching must be a mapping.")
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown matching keys {unknown}; allowed: {sorted(known)}.")
        values = {**asdict(cls()), **raw}
        for name in ("address_and_footprint", "unique_address", "covering_footprint", "exclude_mv_direct"):
            if not isinstance(values[name], bool):
                raise ValueError(f"matching.{name} must be true or false.")
        for name in ("candidate_radius_m", "unconnected_radius_m"):
            values[name] = _distance(values[name], f"matching.{name}")
        values["facade_offset_m"] = _distance(values["facade_offset_m"], "matching.facade_offset_m", allow_zero=True)
        if values["nearest_residential_m"] is not None:
            values["nearest_residential_m"] = _distance(values["nearest_residential_m"], "matching.nearest_residential_m")
        for name in ("facade_offset_m", "nearest_residential_m"):
            if values[name] is not None and values[name] > values["candidate_radius_m"]:
                raise ValueError(f"matching.{name} must not exceed candidate_radius_m.")
        for name in ("demand_uses", "non_demand_use_ids"):
            value = values[name]
            if isinstance(value, str) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"matching.{name} must be a list of strings.")
            values[name] = tuple(value)
        if values["suspect_pattern"] is not None:
            try:
                re.compile(values["suspect_pattern"])
            except (re.error, TypeError) as exc:
                raise ValueError(f"matching.suspect_pattern is no regular expression: {exc}") from exc
        return cls(**values)


@dataclass
class MatchResult:
    """Outputs of :func:`match_points` (one CSV each, plus the report as JSON)."""

    points: pd.DataFrame
    """One row per point: the deciding rule, the supplied building (or the named one of an exclusion)."""
    supply: pd.DataFrame
    """One row per supplied building: its grid and bus, the strongest rule, the points on it."""
    unconnected: pd.DataFrame
    """Demand buildings near the grid lines that no point supplies, with the reason."""
    quality: pd.DataFrame
    """Per grid: the points by evidence and the shares of weak and of excluded points."""
    report: dict[str, Any]


# --------------------------------------------------------------------------- addresses
def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def address_key(value: Any) -> str:
    """Key of the exact address comparison: case and punctuation ignored, ß as ss."""
    return re.sub(r"[^\w]", "", _text(value).casefold().replace("ß", "ss"))


def street_key(value: Any) -> str:
    """Street key: "Kiefernstr. 2" and "Kiefernstraße 2" name one street."""
    text = _text(value).casefold().replace("ß", "ss")
    text = re.sub(r"str\.?(?=\s|$|\d)", "strasse", text)
    return re.sub(r"[^a-zäöü]", "", text)


def house_key(value: Any) -> str:
    """The first house number, with a letter only when written directly after it ("30 Zählersäule" is 30)."""
    match = re.search(r"(\d+)([a-zA-Z])?", _text(value))
    return match.group(1) + (match.group(2) or "").casefold() if match else ""


def point_address(address: Any) -> tuple[str, str] | None:
    """(street, house) of a point address with street and number in one field, split at the first digit."""
    text = _text(address)
    match = re.search(r"\d", text)
    if not match:
        return None
    street, house = street_key(text[: match.start()]), house_key(text[match.start():])
    return (street, house) if street and house else None


def building_address(street: Any, house_number: Any) -> tuple[str, str] | None:
    """(street, house) of a building, or None without street or number."""
    if not _text(street) or not _text(house_number):
        return None
    key = street_key(street), house_key(house_number)
    return key if all(key) else None


# --------------------------------------------------------------------------- inputs
def _distance(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number of metres.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number of metres.") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'zero or ' if allow_zero else ''}positive.")
    return number


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _flag(value: Any) -> bool | None:
    """True, False or None (unknown) from booleans, numbers or strings such as "true", "no", "1"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _text(value).strip().casefold()
    if not text:
        return None
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    raise ValueError(f"{value!r} is no true/false value.")


def _point_records(points: pd.DataFrame) -> list[dict[str, Any]]:
    missing = [name for name in POINT_COLUMNS if name not in points.columns]
    if missing:
        raise ValueError(f"Connection points lack the columns {missing}.")
    records = []
    for row in points.to_dict("records"):
        try:
            x, y = float(row["x"]), float(row["y"])
        except (TypeError, ValueError):
            x = y = math.nan
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Connection point {row['point_id']} has no coordinates.")
        records.append({
            "point_id": _text(row["point_id"]),
            "grid_id": _text(row["grid_id"]),
            "bus": int(row["bus"]),
            "x": x,
            "y": y,
            "address": _text(row.get("address")),
            "remark": _text(row.get("remark")),
            "on_building": _flag(row.get("on_building")),
            "exclude": bool(_flag(row.get("exclude"))),
            "exclude_reason": _text(row.get("exclude_reason")),
        })
    ids = Counter(record["point_id"] for record in records)
    duplicated = sorted(point_id for point_id, count in ids.items() if count > 1)
    if duplicated or "" in ids:
        raise ValueError(f"Connection point ids must be unique and not empty: {duplicated[:5]}")
    return records


@dataclass(frozen=True)
class _Building:
    objectid: str
    geometry: BaseGeometry | None
    exact_address: str | None
    address: tuple[str, str] | None
    demand: bool
    residential: bool
    mv_direct: bool
    use: str


def _buildings(buildings: pd.DataFrame, config: MatchingConfig) -> dict[str, _Building]:
    missing = [name for name in BUILDING_COLUMNS if name not in buildings.columns]
    if missing:
        raise ValueError(f"Buildings lack the columns {missing}.")
    result: dict[str, _Building] = {}
    for row in buildings.to_dict("records"):
        objectid = _text(row["objectid"])
        if not objectid or objectid in result:
            raise ValueError(f"Building objectids must be unique and not empty: {objectid!r}")
        street, house = _text(row.get("street")), _text(row.get("house_number"))
        geometry = row["geometry"]
        valid = isinstance(geometry, BaseGeometry) and not geometry.is_empty and geometry.is_valid
        result[objectid] = _Building(
            objectid=objectid,
            geometry=geometry if valid else None,
            exact_address=address_key(f"{street} {house}") if street and house else None,
            address=building_address(street, house),
            demand=(_text(row.get("building_use")) in config.demand_uses
                    and _text(row.get("building_use_id")) not in config.non_demand_use_ids),
            residential=_number(row.get("households")) > 0 or _number(row.get("residential_floor_area")) > 0,
            mv_direct=bool(_flag(row.get("mv_direct"))),
            use=_text(row.get("building_use")),
        )
    return result


@dataclass(frozen=True)
class _Candidate:
    objectid: str
    distance_m: float
    covers: bool
    address_agrees: bool


def _candidates(point: dict[str, Any], tree: STRtree, located: list[_Building], radius: float) -> list[_Candidate]:
    """Every building within ``radius`` of the point, nearest first (distance rounded to 0.01 mm)."""
    spot = Point(point["x"], point["y"])
    wanted = address_key(point["address"])
    result = []
    for index in tree.query(box(spot.x - radius, spot.y - radius, spot.x + radius, spot.y + radius)):
        building = located[int(index)]
        distance = spot.distance(building.geometry)
        if distance > radius:
            continue
        result.append(_Candidate(
            objectid=building.objectid,
            distance_m=round(distance, 5),
            covers=bool(building.geometry.covers(spot)),
            address_agrees=building.exact_address is not None and building.exact_address == wanted,
        ))
    return sorted(result, key=lambda candidate: (candidate.distance_m, candidate.objectid))


# --------------------------------------------------------------------------- matching
def match_points(
    points: pd.DataFrame,
    buildings: pd.DataFrame,
    config: MatchingConfig | None = None,
    *,
    grid_lines: Mapping[str, Sequence[BaseGeometry]] | None = None,
) -> MatchResult:
    """Decide the building of every connection point.

    Args:
        points: ``point_id, grid_id, bus, x, y`` and optionally ``address`` (street and number in one field),
            ``remark``, ``on_building`` (true: mounted on a wall, false: free-standing, empty: unknown),
            ``exclude`` and ``exclude_reason``.
        buildings: every building around the points: ``objectid, geometry`` (shapely, EPSG:25832) and
            ``street, house_number, building_use, building_use_id, households, residential_floor_area,
            mv_direct``.
        config: rules and distances (default: :class:`MatchingConfig`).
        grid_lines: line geometries per grid id; buildings near them that no point supplies are listed in
            ``unconnected``.
    """
    config = config or MatchingConfig()
    records = _point_records(points)
    inventory = _buildings(buildings, config)
    located = [building for building in inventory.values() if building.geometry is not None]
    tree = STRtree([building.geometry for building in located])
    mv_direct = {key for key, building in inventory.items() if building.mv_direct} if config.exclude_mv_direct else set()
    suspect = re.compile(config.suspect_pattern) if config.suspect_pattern else None
    by_id = {record["point_id"]: record for record in records}
    pools = {record["point_id"]: _candidates(record, tree, located, config.candidate_radius_m) for record in records}

    # --- the rules that need nothing but the point's own evidence
    decided: dict[str, tuple[str, str | None, float | None]] = {}
    named: dict[str, str] = {}
    for record in records:
        point_id, pool = record["point_id"], pools[record["point_id"]]
        if record["exclude"]:
            decided[point_id] = ("excluded_point", None, None)
            continue
        flagged = bool(suspect and suspect.search(f"{record['address']} {record['remark']}"))
        if config.address_and_footprint and not flagged:
            matches = [
                candidate for candidate in pool
                if inventory[candidate.objectid].demand and candidate.address_agrees
                and (candidate.covers or (candidate.distance_m <= config.facade_offset_m
                                          and record["on_building"] is True))
            ]
            if len(matches) == 1:
                decided[point_id] = ("address_and_footprint", matches[0].objectid, matches[0].distance_m)
                continue
        if config.unique_address:
            wanted = point_address(record["address"])
            agree = [candidate for candidate in pool
                     if wanted and inventory[candidate.objectid].address == wanted]
            if len(agree) == 1:
                decided[point_id] = ("unique_address", agree[0].objectid, agree[0].distance_m)
                continue
        if config.covering_footprint:
            covering = [candidate for candidate in pool if candidate.covers]
            if len(covering) == 1 and (inventory[covering[0].objectid].residential or record["on_building"] is True):
                decided[point_id] = ("covering_footprint", covering[0].objectid, 0.0)

    # --- the named building must carry an LV load
    for point_id, (rule, objectid, _) in list(decided.items()):
        if objectid is None:
            continue
        if not inventory[objectid].demand:
            decided[point_id], named[point_id] = ("building_without_demand", None, None), objectid
        elif objectid in mv_direct:
            decided[point_id], named[point_id] = ("mv_direct_building", None, None), objectid

    # --- one building, one supply
    holder: dict[str, tuple[str, int]] = {}
    held_by: dict[str, tuple[str, int]] = {}

    def supply(point_id: str) -> tuple[str, int]:
        return by_id[point_id]["grid_id"], by_id[point_id]["bus"]

    def free(objectid: str, point_id: str) -> bool:
        return objectid not in holder or holder[objectid] == supply(point_id)

    demoted: Counter[str] = Counter()
    strong = sorted((point_id for point_id, decision in decided.items() if decision[1] is not None),
                    key=lambda point_id: (STRENGTH[decided[point_id][0]], decided[point_id][2] or 0.0, point_id))
    for point_id in strong:
        rule, objectid, _ = decided[point_id]
        if free(objectid, point_id):
            holder[objectid] = supply(point_id)
        else:
            demoted[rule] += 1
            named[point_id], held_by[point_id] = objectid, holder[objectid]
            decided[point_id] = ("second_supply", None, None)

    # --- the fallback, the closest points first
    pending = [record["point_id"] for record in records if record["point_id"] not in decided]
    radius = config.nearest_residential_m
    if radius is None:
        for point_id in pending:
            decided[point_id] = ("unmatched", None, None)
    else:
        def reach(point_id: str) -> float:
            return min((c.distance_m for c in pools[point_id] if c.distance_m <= radius), default=math.inf)

        for point_id in sorted(pending, key=lambda point_id: (reach(point_id), point_id)):
            homes = [
                candidate for candidate in pools[point_id]
                if candidate.distance_m <= radius and inventory[candidate.objectid].demand
                and candidate.objectid not in mv_direct and free(candidate.objectid, point_id)
                and inventory[candidate.objectid].residential
            ]
            if not homes:
                decided[point_id] = ("no_residential_in_reach", None, None)
                continue
            pick = homes[0]
            decided[point_id] = ("nearest_residential", pick.objectid, pick.distance_m)
            holder[pick.objectid] = supply(point_id)

    return _result(records, decided, named, held_by, holder, demoted, inventory, located, tree, mv_direct,
                   config, grid_lines)


def _result(records, decided, named, held_by, holder, demoted, inventory, located, tree, mv_direct, config,
            grid_lines) -> MatchResult:
    rows = []
    for record in records:
        rule, objectid, distance = decided[record["point_id"]]
        held = held_by.get(record["point_id"])
        rows.append({
            "point_id": record["point_id"],
            "grid_id": record["grid_id"],
            "bus": record["bus"],
            "rule": rule,
            "status": "matched" if objectid else "excluded",
            "building_objectid": objectid,
            "distance_m": distance,
            "named_building_objectid": named.get(record["point_id"]),
            "held_by_grid_id": held[0] if held else None,
            "held_by_bus": held[1] if held else None,
            "exclude_reason": record["exclude_reason"] or None,
        })
    points = pd.DataFrame(rows)

    matched = points[points["status"].eq("matched")]
    supply_rows = []
    for objectid, group in matched.groupby("building_objectid", sort=True):
        grid_id, bus = holder[objectid]
        if not (group["grid_id"].eq(grid_id).all() and group["bus"].eq(bus).all()):
            raise AssertionError(f"Building {objectid} is supplied from two buses.")
        supply_rows.append({
            "building_objectid": objectid,
            "grid_id": grid_id,
            "bus": bus,
            "rule": min(group["rule"], key=STRENGTH.__getitem__),
            "points": int(len(group)),
            "point_ids": ";".join(sorted(group["point_id"])),
        })
    supply = pd.DataFrame(supply_rows, columns=["building_objectid", "grid_id", "bus", "rule", "points", "point_ids"])

    quality_rows = []
    for grid_id, group in points.groupby("grid_id", sort=True):
        rules = Counter(group["rule"])
        total = int(len(group))
        counts = {key: sum(rules[name] for name in names) for key, names in QUALITY.items()}
        excluded = total - counts["strong"] - counts["weak"]
        quality_rows.append({"grid_id": grid_id, "points": total, "matched": counts["strong"] + counts["weak"],
                             **counts, "weak_share": round(counts["weak"] / total, 4),
                             "excluded_share": round(excluded / total, 4)})
    quality = pd.DataFrame(quality_rows)

    unconnected = _unconnected(grid_lines, inventory, located, tree, holder, mv_direct, config.unconnected_radius_m)
    report = {
        "config": {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(config).items()},
        "points": len(points),
        "grids": int(points["grid_id"].nunique()),
        "buildings_considered": len(inventory),
        "buildings_without_valid_geometry": len(inventory) - len(located),
        "rules": {name: int(count) for name, count in sorted(Counter(points["rule"]).items(),
                                                              key=lambda item: STRENGTH[item[0]])},
        "demoted_for_a_stronger_supply": dict(demoted),
        "matched_points": int(len(matched)),
        "supplied_buildings": len(holder),
        "mv_direct_buildings_considered": len(mv_direct),
        "unconnected_buildings": None if grid_lines is None else dict(Counter(unconnected["reason"])),
        "rule_reasons": RULES,
    }
    return MatchResult(points=points, supply=supply, unconnected=unconnected, quality=quality, report=report)


def _unconnected(grid_lines, inventory, located, tree, holder, mv_direct, radius) -> pd.DataFrame:
    """Demand buildings within ``radius`` of a grid line that no point supplies, with the reason."""
    columns = ["building_objectid", "grid_ids", "building_use", "reason"]
    if not grid_lines:
        return pd.DataFrame(columns=columns)
    near: dict[str, set[str]] = defaultdict(set)
    for grid_id, lines in grid_lines.items():
        for line in lines:
            if line is None or line.is_empty:
                continue
            xmin, ymin, xmax, ymax = line.bounds
            for index in tree.query(box(xmin - radius, ymin - radius, xmax + radius, ymax + radius)):
                building = located[int(index)]
                if building.demand and building.geometry.distance(line) <= radius:
                    near[building.objectid].add(str(grid_id))
    rows = []
    for objectid, grids in sorted(near.items()):
        if objectid in holder:
            continue
        building = inventory[objectid]
        if objectid in mv_direct:
            reason = "mv_direct_building"
        elif not building.residential:
            reason = "outbuilding"
        else:
            reason = "residential_without_a_point"
        rows.append({"building_objectid": objectid, "grid_ids": ";".join(sorted(grids)),
                     "building_use": building.use, "reason": reason})
    return pd.DataFrame(rows, columns=columns)
