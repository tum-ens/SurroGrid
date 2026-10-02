"""GHD (commercial/public) demand rules applied to the building component manifest.

GridExpand models the annual electricity of a Commercial/Public component as a synPRO
kWh/m² profile times its effective floor area (:mod:`gridexpand.allocation.functions.electricity`).
This module decides before that which non-residential components carry demand and with
which floor area. Every rule is off by default; the scenario YAML block ``ghd:`` switches
them on (:class:`GhdConfig`).

``single_volume_one_storey``
    A fully non-residential single-volume building (church, chapel, station, car park,
    castle) counts one storey: its GHD floor area is the footprint instead of
    footprint x ``floor_number``. The InfDB derives ``floor_number`` from the LoD2 height,
    so a 20 m nave counts five or six storeys. The single-volume status comes from the
    OSM ``building`` value of the footprint when the evidence source has a specific one
    (:data:`SINGLE_VOLUME_OSM_BUILDINGS`), otherwise from the ALKIS building function
    (:data:`SINGLE_VOLUME_ALKIS_FUNCTIONS`).
``osm_levels``
    For the other fully non-residential buildings, OSM ``building:levels`` replaces
    ``floor_number`` in the GHD floor area where the evidence source has a positive value.
``activity_gating``
    A GHD component is active only if its building has a specific non-residential ALKIS
    function or OSM activity evidence on or near its footprint (:func:`activity_category`).
    Generic commercial buildings (31001_2000) and the non-residential part of buildings
    with a residential function (31001_1xxx) need evidence. An Unknown building
    (31001_9998) is non-demand (:mod:`gridexpand.common.building_components`); with gating
    and activity evidence it gets a GHD component of the evidence's category.

Mixed buildings keep their non-residential floor area (the InfDB split already assigns it
to the ground floor); the storey rules apply to fully non-residential buildings only.
MV-direct components stay as pylovo classified them.

OSM evidence is pluggable (:func:`evidence_source`): a PostGIS source with configurable
tables and columns (:class:`OsmEvidenceConfig`), a CSV file of per-building evidence, or an
in-memory frame (:class:`FrameEvidence`). All return :data:`EVIDENCE_COLUMNS` per building.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridexpand.common.building_components import NON_DEMAND_NONRESIDENTIAL_USES, NONRESIDENTIAL_CATEGORIES

# ALKIS building functions (InfDB ``building_use_id``) of single-volume buildings. Names from
# the AdV function code list as commented in the InfDB ``classify_building_use``. Not included:
# 3044 Gemeindehaus and 3048 Kloster (storeyed buildings), 3092 Flughafengebäude (terminal).
SINGLE_VOLUME_ALKIS_FUNCTIONS: dict[str, str] = {
    "31001_3040": "Gebäude für religiöse Zwecke",
    "31001_3041": "Kirche",
    "31001_3042": "Synagoge",
    "31001_3043": "Kapelle",
    "31001_3045": "Gotteshaus",
    "31001_3046": "Moschee",
    "31001_3047": "Tempel",
    "31001_3031": "Schloss",
    "31001_3038": "Burg, Festung",
    "31001_3090": "Empfangsgebäude",
    "31001_3091": "Bahnhofsgebäude",
    "31001_3094": "Gebäude zum U-Bahnhof",
    "31001_3095": "Gebäude zum S-Bahnhof",
    "31001_3097": "Gebäude zum Busbahnhof",
    "31001_2460": "Gebäude zum Parken",
    "31001_2461": "Parkhaus",
    "31001_2462": "Parkdeck",
    "31001_2463": "Garage",
    "31001_2464": "Fahrzeughalle",
    "31001_2465": "Tiefgarage",
}

# OSM ``building`` values of single-volume buildings (same families as the ALKIS list).
SINGLE_VOLUME_OSM_BUILDINGS = frozenset({
    "church", "chapel", "cathedral", "mosque", "synagogue", "temple", "shrine", "religious",
    "train_station", "transportation",
    "castle",
    "parking", "garage", "garages", "carport",
})
# OSM ``building`` values that say nothing about the building type (the ALKIS list decides).
GENERIC_OSM_BUILDINGS = frozenset({"yes", "building", "construction", "no"})

# ALKIS functions that need OSM activity evidence under ``activity_gating``: the generic
# commercial function, besides every residential function (31001_1xxx) and 31001_9998.
GENERIC_NONRESIDENTIAL_FUNCTIONS = frozenset({"31001_2000"})
UNSPECIFIED_FUNCTION = "31001_9998"

# OSM tags that evidence an independent, electricity-consuming commercial or public
# activity inside a building (value -> category; see activity_category).
_COMMERCIAL_AMENITIES = frozenset({
    "restaurant", "cafe", "fast_food", "bar", "pub", "biergarten", "food_court", "ice_cream",
    "nightclub", "cinema", "casino", "bank", "bureau_de_change", "pharmacy", "veterinary",
    "car_wash", "car_rental", "fuel", "driving_school", "language_school", "music_school",
    "internet_cafe", "coworking_space", "events_venue", "conference_centre", "studio",
})
_PUBLIC_AMENITIES = frozenset({
    "school", "kindergarten", "childcare", "college", "university", "library", "townhall",
    "courthouse", "police", "fire_station", "post_office", "community_centre",
    "social_facility", "social_centre", "arts_centre", "theatre", "place_of_worship",
    "hospital", "clinic", "doctors", "dentist", "nursing_home", "prison", "public_building",
    "research_institute", "training",
})
_COMMERCIAL_TOURISM = frozenset({"hotel", "guest_house", "hostel", "motel", "apartment", "chalet"})
_PUBLIC_TOURISM = frozenset({"museum", "gallery"})
_LEISURE = frozenset({
    "fitness_centre", "sports_centre", "sports_hall", "bowling_alley", "dance", "sauna",
    "adult_gaming_centre", "amusement_arcade", "escape_game", "ice_rink",
})
_PUBLIC_OFFICES = frozenset({"government", "administrative", "diplomatic"})
_NOT_ACTIVE = frozenset({"no", "none", "vacant", "disused", "abandoned"})

EVIDENCE_COLUMNS = ("activity", "activity_tags", "activity_category", "osm_building", "osm_levels")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def activity_category(key: Any, value: Any) -> str | None:
    """Category of one OSM tag as activity evidence: ``Commercial``, ``Public`` or None.

    ``shop``, ``office`` and ``craft`` are businesses with their own supply (government
    offices count as Public); ``healthcare`` practices and the listed ``amenity``,
    ``tourism`` and indoor ``leisure`` values are building-bound services. Street furniture
    and outdoor amenities (parking, benches, waste, vending, post boxes, toilets, shelters,
    charging points, ATMs) and vacant or disused shops are not evidence.
    """
    if key is None or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    key, value = str(key).strip().lower(), str(value).strip().lower()
    if not value or value in _NOT_ACTIVE:
        return None
    if key in {"shop", "craft"}:
        return "Commercial"
    if key == "office":
        return "Public" if value in _PUBLIC_OFFICES else "Commercial"
    if key == "healthcare":
        return "Public"
    if key == "amenity":
        if value in _COMMERCIAL_AMENITIES:
            return "Commercial"
        if value in _PUBLIC_AMENITIES:
            return "Public"
        return None
    if key == "tourism":
        if value in _COMMERCIAL_TOURISM:
            return "Commercial"
        if value in _PUBLIC_TOURISM:
            return "Public"
        return None
    if key == "leisure":
        return "Commercial" if value in _LEISURE else None
    return None


def is_specific_nonresidential_function(building_use_id: Any) -> bool:
    """True for an ALKIS function that names a non-residential use by itself.

    False for the generic commercial function 31001_2000, for residential functions
    (31001_1xxx, the shop part of a mixed building) and for 31001_9998 (unspecified).
    """
    code = "" if building_use_id is None or pd.isna(building_use_id) else str(building_use_id).strip()
    if not code or code == UNSPECIFIED_FUNCTION or code in GENERIC_NONRESIDENTIAL_FUNCTIONS:
        return False
    return not code.startswith("31001_1")


def single_volume_source(osm_building: Any, building_use_id: Any) -> str | None:
    """``osm`` or ``alkis`` if the building is single-volume by that source, else None.

    A specific OSM ``building`` value decides; a missing or generic one (``yes``) falls back
    to the ALKIS function list.
    """
    tag = None if osm_building is None or pd.isna(osm_building) else str(osm_building).strip().lower()
    if tag and tag not in GENERIC_OSM_BUILDINGS:
        return "osm" if tag in SINGLE_VOLUME_OSM_BUILDINGS else None
    code = "" if building_use_id is None or pd.isna(building_use_id) else str(building_use_id).strip()
    return "alkis" if code in SINGLE_VOLUME_ALKIS_FUNCTIONS else None


# ---------------------------------------------------------------------------------------
# Configuration (scenario YAML block ``ghd:``)
# ---------------------------------------------------------------------------------------


def _only(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(mapping).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} option(s): {sorted(unknown)}")


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false.")
    return value


def _identifier(value: Any, label: str, *, qualified: bool = False) -> str:
    text = str(value)
    parts = text.split(".")
    if len(parts) > (2 if qualified else 1) or not all(_IDENTIFIER.match(part) for part in parts):
        kind = "[schema.]table" if qualified else "column"
        raise ValueError(f"{label} must be a plain SQL {kind} name, got {value!r}.")
    return text


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a non-negative number.")
    return float(value)


@dataclass(frozen=True)
class OsmActivityLayer:
    """One OSM feature table: tags as a key and a value column, or a fixed key.

    pgosm-flex ``poi_*`` tables hold the key in ``osm_type`` and the value in
    ``osm_subtype``; single-key tables (``amenity_*``) use ``key: amenity`` and
    ``value_column: osm_type``.
    """

    table: str
    geometry_column: str = "geom"
    key_column: str | None = None
    key: str | None = None
    value_column: str = "osm_subtype"

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> "OsmActivityLayer":
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a mapping.")
        _only(raw, {"table", "geometry_column", "key_column", "key", "value_column"}, label)
        if "table" not in raw:
            raise ValueError(f"{label}.table is required.")
        key_column = raw.get("key_column")
        key = raw.get("key")
        if (key_column is None) == (key is None):
            raise ValueError(f"{label} needs exactly one of key_column and key.")
        return cls(
            table=_identifier(raw["table"], f"{label}.table", qualified=True),
            geometry_column=_identifier(raw.get("geometry_column", "geom"), f"{label}.geometry_column"),
            key_column=None if key_column is None else _identifier(key_column, f"{label}.key_column"),
            key=None if key is None else str(key),
            value_column=_identifier(raw.get("value_column", "osm_subtype"), f"{label}.value_column"),
        )


@dataclass(frozen=True)
class OsmBuildingLayer:
    """OSM building polygons: the ``building`` value and ``building:levels``.

    Defaults follow the pgosm-flex building layer (``osm_building_polygon``): the
    ``building`` value in ``osm_subtype``, rows of ``osm_type = 'building'`` only (not
    ``building_part`` or ``address``).
    """

    table: str
    geometry_column: str = "geom"
    building_column: str = "osm_subtype"
    levels_column: str | None = "levels"
    type_column: str | None = "osm_type"
    type_value: str | None = "building"

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> "OsmBuildingLayer":
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a mapping.")
        _only(raw, {"table", "geometry_column", "building_column", "levels_column", "type_column", "type_value"}, label)
        if "table" not in raw:
            raise ValueError(f"{label}.table is required.")
        levels = raw.get("levels_column", "levels")
        type_column = raw.get("type_column", "osm_type")
        type_value = raw.get("type_value", "building")
        if (type_column is None) != (type_value is None):
            raise ValueError(f"{label} needs both type_column and type_value, or neither.")
        return cls(
            table=_identifier(raw["table"], f"{label}.table", qualified=True),
            geometry_column=_identifier(raw.get("geometry_column", "geom"), f"{label}.geometry_column"),
            building_column=_identifier(raw.get("building_column", "osm_subtype"), f"{label}.building_column"),
            levels_column=None if levels is None else _identifier(levels, f"{label}.levels_column"),
            type_column=None if type_column is None else _identifier(type_column, f"{label}.type_column"),
            type_value=None if type_value is None else str(type_value),
        )


@dataclass(frozen=True)
class OsmEvidenceConfig:
    """Where the per-building OSM evidence comes from (``ghd.osm``)."""

    source: str
    file: str | None = None
    buildings_table: str = "basedata.buildings"
    building_id_column: str = "objectid"
    building_geometry_column: str = "geom"
    point_buffer_m: float = 5.0
    polygon_min_overlap: float = 0.5
    activity_layers: tuple[OsmActivityLayer, ...] = ()
    building_layer: OsmBuildingLayer | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "OsmEvidenceConfig":
        if not isinstance(raw, dict):
            raise ValueError("ghd.osm must be a mapping.")
        _only(raw, {
            "source", "file", "buildings_table", "building_id_column", "building_geometry_column",
            "point_buffer_m", "polygon_min_overlap", "activity_layers", "building_layer",
        }, "ghd.osm")
        source = raw.get("source")
        if source not in {"postgis", "file"}:
            raise ValueError("ghd.osm.source must be 'postgis' or 'file'.")
        if source == "file" and not raw.get("file"):
            raise ValueError("ghd.osm.source 'file' needs ghd.osm.file.")
        overlap = _positive_number(raw.get("polygon_min_overlap", 0.5), "ghd.osm.polygon_min_overlap")
        if not 0.0 < overlap <= 1.0:
            raise ValueError("ghd.osm.polygon_min_overlap must be in (0, 1].")
        layers = raw.get("activity_layers") or []
        if not isinstance(layers, list):
            raise ValueError("ghd.osm.activity_layers must be a list.")
        building_layer = raw.get("building_layer")
        return cls(
            source=source,
            file=None if raw.get("file") is None else str(raw["file"]),
            buildings_table=_identifier(raw.get("buildings_table", "basedata.buildings"), "ghd.osm.buildings_table", qualified=True),
            building_id_column=_identifier(raw.get("building_id_column", "objectid"), "ghd.osm.building_id_column"),
            building_geometry_column=_identifier(raw.get("building_geometry_column", "geom"), "ghd.osm.building_geometry_column"),
            point_buffer_m=_positive_number(raw.get("point_buffer_m", 5.0), "ghd.osm.point_buffer_m"),
            polygon_min_overlap=overlap,
            activity_layers=tuple(
                OsmActivityLayer.from_dict(layer, f"ghd.osm.activity_layers[{i}]") for i, layer in enumerate(layers)
            ),
            building_layer=None if building_layer is None else OsmBuildingLayer.from_dict(building_layer, "ghd.osm.building_layer"),
        )


@dataclass(frozen=True)
class GhdConfig:
    """Scenario YAML block ``ghd:`` (optional; every rule off by default)."""

    activity_gating: bool = False
    single_volume_one_storey: bool = False
    osm_levels: bool = False
    osm: OsmEvidenceConfig | None = None

    @property
    def enabled(self) -> bool:
        return self.activity_gating or self.single_volume_one_storey or self.osm_levels

    @classmethod
    def from_dict(cls, raw: Any) -> "GhdConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("ghd must be a YAML mapping.")
        _only(raw, {"activity_gating", "single_volume_one_storey", "osm_levels", "osm"}, "ghd")
        osm = None if raw.get("osm") is None else OsmEvidenceConfig.from_dict(raw["osm"])
        config = cls(
            activity_gating=_bool(raw.get("activity_gating", False), "ghd.activity_gating"),
            single_volume_one_storey=_bool(raw.get("single_volume_one_storey", False), "ghd.single_volume_one_storey"),
            osm_levels=_bool(raw.get("osm_levels", False), "ghd.osm_levels"),
            osm=osm,
        )
        if config.activity_gating and (osm is None or (osm.source == "postgis" and not osm.activity_layers)):
            raise ValueError("ghd.activity_gating needs ghd.osm with activity_layers (or a file source).")
        if config.osm_levels and (osm is None or (osm.source == "postgis" and osm.building_layer is None)):
            raise ValueError("ghd.osm_levels needs ghd.osm with a building_layer (or a file source).")
        return config


# ---------------------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------------------


def empty_evidence(objectids: Sequence[str]) -> pd.DataFrame:
    """No evidence for any building (the frame every source returns for unknown ids)."""
    index = pd.Index([str(value) for value in objectids], name="objectid")
    return pd.DataFrame(
        {
            "activity": pd.Series(False, index=index, dtype=bool),
            "activity_tags": pd.Series("", index=index, dtype=object),
            "activity_category": pd.Series(None, index=index, dtype=object),
            "osm_building": pd.Series(None, index=index, dtype=object),
            "osm_levels": pd.Series(np.nan, index=index, dtype=float),
        }
    )


class FrameEvidence:
    """Evidence from a frame with ``objectid`` and any of :data:`EVIDENCE_COLUMNS`."""

    def __init__(self, frame: pd.DataFrame):
        frame = frame.copy()
        if "objectid" in frame.columns:
            frame = frame.set_index("objectid")
        frame.index = frame.index.astype(str)
        frame.index.name = "objectid"
        if frame.index.duplicated().any():
            raise ValueError("OSM evidence has more than one row per objectid.")
        unknown = set(frame.columns).difference(EVIDENCE_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown OSM evidence column(s): {sorted(unknown)}")
        self._frame = frame

    def evidence(self, objectids: Sequence[str]) -> pd.DataFrame:
        result = empty_evidence(objectids)
        known = result.index.intersection(self._frame.index)
        for column in EVIDENCE_COLUMNS:
            if column in self._frame.columns and len(known):
                result.loc[known, column] = self._frame.loc[known, column].to_numpy()
        result["activity"] = result["activity"].fillna(False).astype(bool)
        result["activity_tags"] = result["activity_tags"].fillna("").astype(str)
        result["osm_levels"] = pd.to_numeric(result["osm_levels"], errors="coerce")
        return result


def match_osm_evidence(
    footprints: pd.DataFrame,
    features: pd.DataFrame,
    buildings: pd.DataFrame | None = None,
    *,
    point_buffer_m: float = 5.0,
    polygon_min_overlap: float = 0.5,
) -> pd.DataFrame:
    """Assign OSM features and building polygons to building footprints.

    Args:
        footprints: ``objectid`` and a shapely ``geometry`` (one footprint per building).
        features: ``key``, ``value`` and ``geometry`` of OSM features (points, lines, polygons).
        buildings: optional ``building``, ``levels`` and ``geometry`` of OSM building polygons.
        point_buffer_m: a point (or the representative point of a line) outside every
            footprint belongs to the nearest footprint within this distance.
        polygon_min_overlap: a feature polygon belongs to every footprint it covers by at
            least this share of the footprint area; an OSM building polygon gives its tags to
            the footprint it covers most, if by at least this share.

    A point inside several footprints belongs to the smallest one; ties of the nearest
    footprint go to the smaller objectid. Coordinates must share one metric CRS.

    Returns:
        :data:`EVIDENCE_COLUMNS` indexed by ``objectid`` (every footprint).
    """
    from shapely import STRtree
    from shapely.geometry import Point

    objectids = footprints["objectid"].astype(str).tolist()
    result = empty_evidence(objectids)
    geoms = list(footprints["geometry"])
    if not geoms:
        return result
    tree = STRtree(geoms)
    areas = np.array([geom.area for geom in geoms])
    tags: dict[int, set[str]] = {}
    categories: dict[int, set[str]] = {}

    def assign(position: int, key: str, value: str, category: str) -> None:
        tags.setdefault(position, set()).add(f"{key}={value}")
        categories.setdefault(position, set()).add(category)

    def point_target(point) -> int | None:
        inside = [int(i) for i in tree.query(point, predicate="within")]
        if inside:
            return min(inside, key=lambda i: (areas[i], objectids[i]))
        nearest = tree.query_nearest(point, max_distance=point_buffer_m, all_matches=True)
        if len(nearest) == 0:
            return None
        return min((int(i) for i in nearest), key=lambda i: objectids[i])

    for key, value, geom in zip(features["key"], features["value"], features["geometry"]):
        category = activity_category(key, value)
        if category is None or geom is None or geom.is_empty:
            continue
        key_text, value_text = str(key).strip().lower(), str(value).strip().lower()
        targets = []
        if geom.area > 0:
            for i in tree.query(geom, predicate="intersects"):
                i = int(i)
                if areas[i] > 0 and geom.intersection(geoms[i]).area >= polygon_min_overlap * areas[i]:
                    targets.append(i)
        if not targets:
            point = geom if isinstance(geom, Point) else geom.representative_point()
            target = point_target(point)
            targets = [] if target is None else [target]
        for i in targets:
            assign(i, key_text, value_text, category)

    for i, found in tags.items():
        oid = objectids[i]
        result.loc[oid, "activity"] = True
        result.loc[oid, "activity_tags"] = ";".join(sorted(found))
        result.loc[oid, "activity_category"] = "Commercial" if "Commercial" in categories[i] else "Public"

    if buildings is not None and len(buildings):
        best: dict[int, tuple[float, int]] = {}
        building_geoms = list(buildings["geometry"])
        for row, geom in enumerate(building_geoms):
            if geom is None or geom.is_empty or geom.area <= 0:
                continue
            for i in tree.query(geom, predicate="intersects"):
                i = int(i)
                if areas[i] <= 0:
                    continue
                share = geom.intersection(geoms[i]).area / areas[i]
                if share >= polygon_min_overlap and share > best.get(i, (-1.0, -1))[0]:
                    best[i] = (share, row)
        levels = pd.to_numeric(buildings["levels"], errors="coerce") if "levels" in buildings else None
        for i, (_, row) in best.items():
            oid = objectids[i]
            value = buildings["building"].iloc[row]
            result.loc[oid, "osm_building"] = None if pd.isna(value) else str(value).strip().lower()
            if levels is not None and pd.notna(levels.iloc[row]) and levels.iloc[row] > 0:
                result.loc[oid, "osm_levels"] = float(levels.iloc[row])
    return result


class PostgisOsmEvidence:
    """Evidence from OSM tables next to the building footprints in one PostGIS database.

    Reads the footprints of the requested buildings and the OSM features inside their
    bounding box (expanded by the point buffer), transformed to the footprint CRS, then
    matches them with :func:`match_osm_evidence`. Read-only.
    """

    def __init__(self, engine, config: OsmEvidenceConfig):
        self.engine = engine
        self.config = config

    def _read(self, sql: str, **params) -> pd.DataFrame:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)

    def evidence(self, objectids: Sequence[str]) -> pd.DataFrame:
        from shapely import wkb

        ids = sorted({str(value) for value in objectids})
        if not ids:
            return empty_evidence([])
        cfg = self.config
        fp = self._read(
            f"SELECT {cfg.building_id_column}::text AS objectid, "
            f"ST_AsBinary({cfg.building_geometry_column}) AS wkb, ST_SRID({cfg.building_geometry_column}) AS srid "
            f"FROM {cfg.buildings_table} WHERE {cfg.building_id_column}::text = ANY(:ids)",
            ids=ids,
        )
        if fp.empty:
            return empty_evidence(ids)
        srids = fp["srid"].dropna().astype(int).unique()
        if len(srids) != 1:
            raise ValueError(f"Building footprints in {cfg.buildings_table} use several SRIDs: {sorted(srids)}")
        srid = int(srids[0])
        footprints = pd.DataFrame({"objectid": fp["objectid"], "geometry": [wkb.loads(bytes(v)) for v in fp["wkb"]]})
        xmin, ymin, xmax, ymax = np.array([geom.bounds for geom in footprints["geometry"]]).T
        box = dict(
            xmin=float(xmin.min() - cfg.point_buffer_m), ymin=float(ymin.min() - cfg.point_buffer_m),
            xmax=float(xmax.max() + cfg.point_buffer_m), ymax=float(ymax.max() + cfg.point_buffer_m), srid=srid,
        )
        frames = []
        for layer in cfg.activity_layers:
            key = f"{layer.key_column}::text" if layer.key_column else "CAST(:fixed_key AS text)"
            params = dict(box, **({} if layer.key_column else {"fixed_key": layer.key}))
            frames.append(self._read(
                f"SELECT {key} AS key, {layer.value_column}::text AS value, "
                f"ST_AsBinary(ST_Transform({layer.geometry_column}, :srid)) AS wkb FROM {layer.table} "
                f"WHERE {layer.geometry_column} && ST_Transform(ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, :srid), "
                f"ST_SRID({layer.geometry_column}))",
                **params,
            ))
        features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["key", "value", "wkb"])
        features["geometry"] = [wkb.loads(bytes(v)) for v in features["wkb"]]
        buildings = None
        if cfg.building_layer is not None:
            layer = cfg.building_layer
            levels = f"{layer.levels_column}" if layer.levels_column else "NULL"
            kind = f" AND {layer.type_column}::text = :type_value" if layer.type_column else ""
            raw = self._read(
                f"SELECT {layer.building_column}::text AS building, {levels} AS levels, "
                f"ST_AsBinary(ST_Transform({layer.geometry_column}, :srid)) AS wkb FROM {layer.table} "
                f"WHERE {layer.geometry_column} && ST_Transform(ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, :srid), "
                f"ST_SRID({layer.geometry_column})){kind}",
                **box, **({"type_value": layer.type_value} if layer.type_column else {}),
            )
            raw["geometry"] = [wkb.loads(bytes(v)) for v in raw["wkb"]]
            buildings = raw
        matched = match_osm_evidence(
            footprints, features, buildings,
            point_buffer_m=cfg.point_buffer_m, polygon_min_overlap=cfg.polygon_min_overlap,
        )
        return FrameEvidence(matched.reset_index()).evidence(ids)


def evidence_source(config: OsmEvidenceConfig, engine=None):
    """The evidence source named by ``ghd.osm`` (a PostGIS source needs ``engine``)."""
    if config.source == "file":
        path = Path(config.file)
        frame = pd.read_csv(path, dtype={"objectid": str})
        return FrameEvidence(frame)
    if engine is None:
        from gridexpand.db.engine import get_engine

        engine = get_engine()
    return PostgisOsmEvidence(engine, config)


def load_evidence(config: GhdConfig, objectids: Sequence[str], engine=None) -> pd.DataFrame | None:
    """Evidence for ``objectids`` when a rule needs it and ``ghd.osm`` is set, else None."""
    if not config.enabled or config.osm is None:
        return None
    return evidence_source(config.osm, engine).evidence(objectids)


# ---------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------

AUDIT_COLUMNS = (
    "objectid", "building_use_id", "source_nonresidential_use", "component_id", "component_category",
    "decision", "storey_rule", "source_area_m2", "ghd_area_m2", "activity_tags", "osm_building", "osm_levels",
)


def _float(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def apply_ghd_policy(
    physical: pd.DataFrame,
    components: pd.DataFrame,
    config: GhdConfig,
    evidence: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the ``ghd:`` rules to a component manifest.

    Args:
        physical: one row per building (the ``validate_physical_buildings`` columns).
        components: the manifest of :func:`build_building_components` for these buildings.
        config: the scenario's ``ghd`` block.
        evidence: :data:`EVIDENCE_COLUMNS` by ``objectid`` (None: no OSM evidence).

    Returns:
        ``(components, audit)``: the manifest without inactive GHD components, with the
        storey rules applied to ``effective_floor_area_m2`` and with activated Unknown
        parts; and one audit row per building with a non-residential part. ``decision`` is
        ``active``, ``inactive_no_activity_evidence``, ``mv_direct``, ``non_demand_unknown``
        or ``activated_unknown``; ``storey_rule`` is ``floor_number``,
        ``single_volume_osm``, ``single_volume_alkis`` or ``osm_levels``. With every rule
        off the manifest is returned unchanged.
    """
    buildings = physical.copy()
    buildings["objectid"] = buildings["objectid"].astype(str)
    buildings = buildings.set_index("objectid")
    comps = components.copy()
    comps["objectid"] = comps["objectid"].astype(str)
    evidence = empty_evidence(buildings.index) if evidence is None else FrameEvidence(evidence).evidence(buildings.index)

    nonres_use = buildings["nonresidential_use"].astype("string").str.strip()
    nonres_area = pd.to_numeric(buildings["nonresidential_floor_area"], errors="coerce").fillna(0.0)
    res_area = pd.to_numeric(buildings["residential_floor_area"], errors="coerce").fillna(0.0)
    floor_area = pd.to_numeric(buildings["floor_area"], errors="coerce")

    def ghd_area(oid: str, area: float) -> tuple[float, str]:
        if res_area.loc[oid] > 0 or not (config.single_volume_one_storey or config.osm_levels):
            return area, "floor_number"
        if config.single_volume_one_storey:
            source = single_volume_source(evidence.at[oid, "osm_building"], buildings.at[oid, "building_use_id"])
            if source is not None:
                return float(floor_area.loc[oid]), f"single_volume_{source}"
        levels = evidence.at[oid, "osm_levels"]
        if config.osm_levels and pd.notna(levels) and levels > 0:
            return float(floor_area.loc[oid]) * float(levels), "osm_levels"
        return area, "floor_number"

    audit_rows: list[dict[str, Any]] = []
    keep = pd.Series(True, index=comps.index)
    nonres_rows = comps.index[comps["component_category"].isin(NONRESIDENTIAL_CATEGORIES)]
    for position in nonres_rows:
        row = comps.loc[position]
        oid = str(row["objectid"])
        source_area = float(row["effective_floor_area_m2"])
        area, rule = ghd_area(oid, source_area)
        mv_direct = not bool(row["included_in_lv"])
        if mv_direct:
            decision = "mv_direct"
        elif config.activity_gating and not (
            is_specific_nonresidential_function(buildings.at[oid, "building_use_id"]) or bool(evidence.at[oid, "activity"])
        ):
            decision = "inactive_no_activity_evidence"
            keep.loc[position] = False
        else:
            decision = "active"
        comps.loc[position, "effective_floor_area_m2"] = area
        audit_rows.append({
            "objectid": oid, "building_use_id": buildings.at[oid, "building_use_id"],
            "source_nonresidential_use": nonres_use.loc[oid], "component_id": row["component_id"],
            "component_category": row["component_category"], "decision": decision, "storey_rule": rule,
            "source_area_m2": source_area, "ghd_area_m2": area if decision == "active" else 0.0,
            "activity_tags": evidence.at[oid, "activity_tags"], "osm_building": evidence.at[oid, "osm_building"],
            "osm_levels": evidence.at[oid, "osm_levels"],
        })

    added = []
    unknown = buildings.index[nonres_area.gt(0) & nonres_use.isin(NON_DEMAND_NONRESIDENTIAL_USES).fillna(False)]
    for oid in unknown:
        source_area = float(nonres_area.loc[oid])
        activated = config.activity_gating and bool(evidence.at[oid, "activity"])
        area, rule = ghd_area(oid, source_area)
        category = evidence.at[oid, "activity_category"] if activated else None
        category = category if category in NONRESIDENTIAL_CATEGORIES else "Public"
        component_id = f"{oid}::{category.lower()}" if activated else None
        audit_rows.append({
            "objectid": oid, "building_use_id": buildings.at[oid, "building_use_id"],
            "source_nonresidential_use": nonres_use.loc[oid], "component_id": component_id,
            "component_category": category if activated else None,
            "decision": "activated_unknown" if activated else "non_demand_unknown",
            "storey_rule": rule if activated else "floor_number", "source_area_m2": source_area,
            "ghd_area_m2": area if activated else 0.0, "activity_tags": evidence.at[oid, "activity_tags"],
            "osm_building": evidence.at[oid, "osm_building"], "osm_levels": evidence.at[oid, "osm_levels"],
        })
        if not activated:
            continue
        building = buildings.loc[oid]
        direct_value = building.get("nonresidential_mv_direct")
        direct = False if direct_value is None or pd.isna(direct_value) else bool(direct_value)
        added.append({
            "component_id": component_id,
            "grid_case_id": building.get("grid_case_id", pd.NA),
            "objectid": oid,
            "pylovo_grid_result_id": building.get("pylovo_grid_result_id"),
            "pylovo_version_id": building.get("pylovo_version_id"),
            "component_category": category,
            "effective_floor_area_m2": area,
            "gross_floor_area_m2": _float(building["floor_area"]) * _float(building["floor_number"]),
            "households": pd.NA,
            "occupants": pd.NA,
            "installed_peak_kw": _float(building.get("nonresidential_peak_load_in_kw")),
            "load_units": 1.0,
            "consumer_vertex": building.get("consumer_vertex", building.get("vertice_id")),
            "bus": building["bus"],
            "included_in_lv": not direct,
            "mv_direct": direct,
            "mix_score": building.get("mix_score"),
            "mix_rule": building.get("mix_rule"),
            "mix_confidence": building.get("mix_confidence"),
            "source_building_use": building.get("building_use"),
            "source_building_use_id": building.get("building_use_id"),
            "source_building_type": building.get("building_type"),
        })

    audit = pd.DataFrame(audit_rows, columns=list(AUDIT_COLUMNS))
    if not config.enabled:
        return components.copy(), audit
    result = comps.loc[keep]
    if added:
        result = pd.concat([result, pd.DataFrame(added).reindex(columns=comps.columns)], ignore_index=True)
    # Stable: the builder's order (residential first) is kept within a building.
    result = result.sort_values("objectid", kind="stable").reset_index(drop=True)
    return result, audit


def summarize_ghd_audit(audit: pd.DataFrame) -> dict[str, Any]:
    """Counts and floor areas per decision and storey rule (for run assumptions)."""
    if audit.empty:
        return {"buildings": 0}
    return {
        "buildings": int(audit["objectid"].nunique()),
        "decisions": {str(k): int(v) for k, v in audit["decision"].value_counts().items()},
        "storey_rules": {str(k): int(v) for k, v in audit["storey_rule"].value_counts().items()},
        "source_area_m2": float(pd.to_numeric(audit["source_area_m2"], errors="coerce").sum()),
        "ghd_area_m2": float(pd.to_numeric(audit["ghd_area_m2"], errors="coerce").sum()),
    }
