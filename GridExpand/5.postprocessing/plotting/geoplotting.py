"""Geospatial diagnostic plots for synthetic and real grid comparison."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy.spatial import ConvexHull
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402

COST_CMAP = LinearSegmentedColormap.from_list(
    "cost_green_red",
    ["#1a9850", "#fee08b", "#d73027"],
)


def _cost_colormap(cmap: str | LinearSegmentedColormap):
    if isinstance(cmap, str) and cmap == "cost_green_red":
        return COST_CMAP
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _display_value_scale(value_column: str) -> tuple[float, str]:
    if value_column.endswith("_eur"):
        return 1000.0, "k€"
    return 1.0, ""


def _display_label(value_column: str, unit: str) -> str:
    label = value_column.replace("_eur", "").replace("_", " ")
    if unit:
        return f"{label} [{unit}]"
    return label


def _add_osm_basemap(
    ax: plt.Axes,
    *,
    target_epsg: int,
    source: object | None = None,
    zoom: str | int = "auto",
    alpha: float = 0.72,
) -> None:
    try:
        import contextily as ctx
    except ImportError as exc:
        raise ImportError(
            "OSM basemap support requires contextily. Install/sync the "
            "GridExpand/5.postprocessing uv environment first."
        ) from exc

    if source is None:
        source = ctx.providers.OpenStreetMap.Mapnik
    ctx.add_basemap(
        ax,
        crs=f"EPSG:{int(target_epsg)}",
        source=source,
        zoom=zoom,
        alpha=alpha,
        attribution_size=6,
        reset_extent=True,
    )


@dataclass(frozen=True)
class GridEnvelope:
    """Point cloud and convex envelope for one transformer-supplied grid."""

    source: str
    grid_id: str
    points: np.ndarray
    envelope: np.ndarray | None
    n_points: int


def _parse_geo(value: object) -> dict | None:
    if pd.isna(value):
        return None

    obj = value
    for _ in range(3):
        if isinstance(obj, dict):
            return obj
        if not isinstance(obj, str):
            return None
        text_value = obj.strip()
        if not text_value or text_value.lower() == "nan":
            return None
        try:
            obj = json.loads(text_value)
        except json.JSONDecodeError:
            return None

    return obj if isinstance(obj, dict) else None


def _point_from_geo(value: object) -> tuple[float, float] | None:
    geom = _parse_geo(value)
    if not geom or geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        return None
    return float(coords[0]), float(coords[1])


def _convex_envelope(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None

    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 3:
        return None

    try:
        hull = ConvexHull(unique_points)
    except Exception:
        return None

    envelope = unique_points[hull.vertices]
    return np.vstack([envelope, envelope[0]])


def _make_envelopes(df: pd.DataFrame, source: str, id_column: str) -> list[GridEnvelope]:
    envelopes: list[GridEnvelope] = []
    for grid_id, group in df.groupby(id_column, sort=True):
        points = group[["x", "y"]].dropna().to_numpy(dtype=float)
        if len(points) == 0:
            continue
        envelopes.append(
            GridEnvelope(
                source=source,
                grid_id=str(grid_id),
                points=points,
                envelope=_convex_envelope(points),
                n_points=len(points),
            )
        )
    return envelopes


def load_synthetic_grid_points(
    *,
    plz: int = 91301,
    synthetic_run_name: str | None = "baseline_synthetic_hh_only",
    pylovo_version_id: str | None = None,
    building_use: str | Sequence[str] | None = "Residential",
    target_epsg: int = 25832,
) -> pd.DataFrame:
    """Load synthetic building/bus points from the SurroGrid database.

    Synthetic centroids are stored in EPSG:4326. They are transformed in PostGIS
    to ``target_epsg`` so that they can be plotted together with SWF Excel
    coordinates, which are stored in projected metre coordinates.
    """

    db = SurroGridDatabase()
    filters = ["gbb.plz = :plz", "gbb.centroid IS NOT NULL"]
    params: dict[str, object] = {"plz": int(plz), "target_epsg": int(target_epsg)}

    if pylovo_version_id is not None:
        filters.append("gbb.pylovo_version_id = :pylovo_version_id")
        params["pylovo_version_id"] = str(pylovo_version_id)

    run_join = ""
    if synthetic_run_name is not None:
        run_join = """
            JOIN (
                SELECT DISTINCT grid_case_id
                FROM surrogrid.powerflow_run
                WHERE run_name = :synthetic_run_name
            ) pr ON pr.grid_case_id = gbb.grid_case_id
        """
        params["synthetic_run_name"] = synthetic_run_name

    query = text(
        f"""
        SELECT gbb.grid_case_id,
               gbb.pylovo_grid_result_id,
               gbb.kcid,
               gbb.bcid,
               gbb.building_use,
               ST_X(ST_Transform(gbb.centroid, :target_epsg)) AS x,
               ST_Y(ST_Transform(gbb.centroid, :target_epsg)) AS y
        FROM surrogrid.grid_building_bus gbb
        {run_join}
        WHERE {' AND '.join(filters)}
        """
    )

    with db.engine.connect() as conn:
        points = pd.read_sql_query(query, conn, params=params)

    if building_use is not None:
        if isinstance(building_use, str):
            selected_uses = {building_use}
        else:
            selected_uses = set(building_use)
        points = points[points["building_use"].isin(selected_uses)].copy()

    return points


def load_real_grid_points(
    *,
    real_grid_data_path: str | Path = "/home/breveron/data/swf_split_hybrid",
    real_load_type: str | Sequence[str] | None = "HH",
    variant: str = "radialized",
    category: str = "regular",
    load_status: str = "lvload",
) -> pd.DataFrame:
    """Load real SWF load-bus points from radialized split-grid Excel files."""

    root = Path(real_grid_data_path)
    legacy_manifest = root / "split_manifest.csv"
    station_manifest = root / "station_split_manifest.csv"
    radialization_manifest = root / "station_radialization_manifest.csv"
    if legacy_manifest.exists():
        manifest = pd.read_csv(legacy_manifest)
        selected = manifest[
            (manifest["variant"] == variant)
            & (manifest["category"] == category)
            & (manifest["load_status"] == load_status)
            & (manifest["status"] == "exported")
        ].copy()
    elif station_manifest.exists() and radialization_manifest.exists():
        if variant != "radialized":
            raise ValueError(
                "Station-based splitter output currently supports only "
                "variant='radialized' in this plot."
            )
        stations = pd.read_csv(station_manifest)
        stations = stations[
            (stations["category"] == category)
            & (stations["load_status"] == load_status)
            & (stations["status"] == "ready")
        ].copy()
        radialized = pd.read_csv(radialization_manifest)
        radialized = radialized[radialized["status"] == "ok"].copy()
        radialized_files = radialized[["grid", "file"]].rename(
            columns={"file": "radialized_file"}
        )
        selected = stations.merge(
            radialized_files,
            left_on="station_id",
            right_on="grid",
            how="inner",
            validate="one_to_one",
        )
        selected["file"] = selected.pop("radialized_file")
        selected["lv_id"] = (
            selected["station_id"].astype(str).str.removeprefix("LV_")
        )
    else:
        raise FileNotFoundError(
            "Could not find a supported real-grid manifest in "
            f"{root}. Expected split_manifest.csv or the station-based "
            "split/radialization manifests."
        )

    frames: list[pd.DataFrame] = []
    for row in selected.to_dict("records"):
        source_file = Path(str(row["file"]))
        if not source_file.is_absolute():
            source_file = root / source_file
        if not source_file.exists():
            continue

        bus = pd.read_excel(source_file, sheet_name="bus")
        load = pd.read_excel(source_file, sheet_name="load")
        if "in_service" in load.columns:
            load = load[load["in_service"] == True]  # noqa: E712
        if real_load_type is not None and "type" in load.columns:
            if isinstance(real_load_type, str):
                selected_types = {real_load_type}
            else:
                selected_types = set(real_load_type)
            load = load[load["type"].astype(str).isin(selected_types)]

        load_buses = set(load["bus"].dropna().astype(int).tolist())
        if not load_buses:
            continue

        bus_id_col = "Unnamed: 0"
        bus = bus.copy()
        bus[bus_id_col] = pd.to_numeric(bus[bus_id_col], errors="coerce")
        bus = bus.dropna(subset=[bus_id_col])
        bus = bus[bus[bus_id_col].astype(int).isin(load_buses)].copy()
        points = []
        for _, bus_row in bus.iterrows():
            point = _point_from_geo(bus_row.get("geo"))
            if point is None:
                continue
            points.append(
                {
                    "lv_id": str(row["lv_id"]),
                    "bus": int(bus_row[bus_id_col]),
                    "x": point[0],
                    "y": point[1],
                    "source_file": str(source_file),
                }
            )
        if points:
            frames.append(pd.DataFrame(points))

    if not frames:
        return pd.DataFrame(columns=["lv_id", "bus", "x", "y", "source_file"])

    return pd.concat(frames, ignore_index=True)


def _add_envelopes(
    ax: plt.Axes,
    envelopes: Iterable[GridEnvelope],
    *,
    facecolor: str,
    edgecolor: str,
    point_color: str,
    alpha: float,
    linewidth: float,
    show_points: bool,
) -> None:
    polygons = []
    lines = []
    point_clouds = []

    for envelope in envelopes:
        if envelope.envelope is not None:
            polygons.append(envelope.envelope)
        elif envelope.n_points == 2:
            lines.append(envelope.points)
        else:
            point_clouds.append(envelope.points)

        if show_points:
            ax.scatter(
                envelope.points[:, 0],
                envelope.points[:, 1],
                s=3,
                color=point_color,
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )

    if polygons:
        collection = PolyCollection(
            polygons,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=2,
        )
        ax.add_collection(collection)

    if lines:
        ax.add_collection(
            LineCollection(lines, colors=edgecolor, linewidths=linewidth, alpha=0.8, zorder=3)
        )

    for points in point_clouds:
        ax.scatter(points[:, 0], points[:, 1], s=10, color=edgecolor, alpha=0.8, zorder=4)


def plot_grid_area_envelope_comparison(
    *,
    plz: int = 91301,
    synthetic_run_name: str | None = "baseline_synthetic_hh_only",
    pylovo_version_id: str | None = None,
    synthetic_building_use: str | Sequence[str] | None = "Residential",
    real_grid_data_path: str | Path = "/home/breveron/data/swf_split_hybrid",
    real_load_type: str | Sequence[str] | None = "HH",
    target_epsg: int = 25832,
    show_points: bool = False,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 7.0),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, pd.DataFrame]]:
    """Plot synthetic and real one-transformer supplied-area envelopes side by side.

    The envelopes are convex hulls around building/load-bus points. They are a
    compact diagnostic abstraction of supplied areas, not exact service-area
    polygons.
    """

    synthetic_points = load_synthetic_grid_points(
        plz=plz,
        synthetic_run_name=synthetic_run_name,
        pylovo_version_id=pylovo_version_id,
        building_use=synthetic_building_use,
        target_epsg=target_epsg,
    )
    real_points = load_real_grid_points(
        real_grid_data_path=real_grid_data_path,
        real_load_type=real_load_type,
    )

    synthetic_envelopes = _make_envelopes(synthetic_points, "Synthetic", "grid_case_id")
    real_envelopes = _make_envelopes(real_points, "Real", "lv_id")

    if not synthetic_envelopes:
        raise ValueError("No synthetic points found for the selected filters.")
    if not real_envelopes:
        raise ValueError("No real points found for the selected filters.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_syn, ax_real = axes

    _add_envelopes(
        ax_syn,
        synthetic_envelopes,
        facecolor="#4C78A8",
        edgecolor="#1F4E79",
        point_color="#1F4E79",
        alpha=0.20,
        linewidth=0.9,
        show_points=show_points,
    )
    _add_envelopes(
        ax_real,
        real_envelopes,
        facecolor="#F58518",
        edgecolor="#B75D00",
        point_color="#B75D00",
        alpha=0.20,
        linewidth=0.9,
        show_points=show_points,
    )

    all_points = pd.concat(
        [
            synthetic_points[["x", "y"]],
            real_points[["x", "y"]],
        ],
        ignore_index=True,
    ).dropna()
    x_min, x_max = all_points["x"].min(), all_points["x"].max()
    y_min, y_max = all_points["y"].min(), all_points["y"].max()
    x_pad = max((x_max - x_min) * 0.04, 50.0)
    y_pad = max((y_max - y_min) * 0.04, 50.0)

    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.25)
        ax.set_xlabel(f"x [EPSG:{target_epsg}]")
        ax.set_ylabel(f"y [EPSG:{target_epsg}]")

    ax_syn.set_title(
        f"Synthetic envelopes ({len(synthetic_envelopes)} grids, {len(synthetic_points)} points)"
    )
    ax_real.set_title(f"Real envelopes ({len(real_envelopes)} grids, {len(real_points)} points)")
    fig.suptitle(
        f"Transformer-supplied area envelopes in PLZ {plz}\\n"
        "Convex hulls around building/load-bus points",
        fontsize=13,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)

    return fig, (ax_syn, ax_real), {
        "synthetic_points": synthetic_points,
        "real_points": real_points,
        "synthetic_envelopes": pd.DataFrame(
            {
                "grid_id": [item.grid_id for item in synthetic_envelopes],
                "n_points": [item.n_points for item in synthetic_envelopes],
                "has_polygon": [item.envelope is not None for item in synthetic_envelopes],
            }
        ),
        "real_envelopes": pd.DataFrame(
            {
                "grid_id": [item.grid_id for item in real_envelopes],
                "n_points": [item.n_points for item in real_envelopes],
                "has_polygon": [item.envelope is not None for item in real_envelopes],
            }
        ),
    }


def _latest_expansion_analysis_key(db: SurroGridDatabase) -> str:
    query = text(
        """
        SELECT analysis_key
        FROM surrogrid.expansion_analysis_run
        ORDER BY created_at DESC, expansion_analysis_run_id DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        key = conn.execute(query).scalar_one_or_none()
    if key is None:
        raise ValueError("No expansion analysis run found.")
    return str(key)


def load_synthetic_expansion_envelope_points(
    *,
    analysis_key: str | None = None,
    building_use: str | Sequence[str] | None = None,
    target_epsg: int = 25832,
) -> pd.DataFrame:
    """Load synthetic building points with grid-level expansion costs."""

    db = SurroGridDatabase()
    analysis_key = analysis_key or _latest_expansion_analysis_key(db)
    params: dict[str, object] = {"analysis_key": analysis_key, "target_epsg": int(target_epsg)}
    filters = ["gbb.centroid IS NOT NULL"]
    if building_use is not None:
        if isinstance(building_use, str):
            selected_uses = [building_use]
        else:
            selected_uses = list(building_use)
        filters.append("gbb.building_use = ANY(:building_use)")
        params["building_use"] = selected_uses

    query = text(
        f"""
        WITH ar AS (
            SELECT expansion_analysis_run_id, analysis_key
            FROM surrogrid.expansion_analysis_run
            WHERE analysis_key = :analysis_key
        ), line_cost AS (
            SELECT
                grid_case_id,
                COUNT(*) AS cable_segments,
                COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur,
                MAX(loading_percent) AS max_cable_loading_percent
            FROM surrogrid.expansion_line_result
            WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            GROUP BY grid_case_id
        ), transformer_cost AS (
            SELECT
                grid_case_id,
                BOOL_OR(requires_expansion) AS transformer_requires_expansion,
                COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur,
                MAX(loading_percent) AS transformer_loading_percent,
                MAX(additional_transformer_kva) AS additional_transformer_kva
            FROM surrogrid.expansion_transformer_result
            WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            GROUP BY grid_case_id
        ), grid_cost AS (
            SELECT
                COALESCE(line_cost.grid_case_id, transformer_cost.grid_case_id) AS grid_case_id,
                COALESCE(cable_segments, 0) AS cable_segments,
                COALESCE(cable_segments_requiring_expansion, 0) AS cable_segments_requiring_expansion,
                COALESCE(cable_cost_eur, 0.0) AS cable_cost_eur,
                max_cable_loading_percent,
                COALESCE(transformer_requires_expansion, FALSE) AS transformer_requires_expansion,
                COALESCE(transformer_cost_eur, 0.0) AS transformer_cost_eur,
                transformer_loading_percent,
                COALESCE(additional_transformer_kva, 0.0) AS additional_transformer_kva,
                COALESCE(cable_cost_eur, 0.0) + COALESCE(transformer_cost_eur, 0.0) AS total_cost_eur
            FROM line_cost
            FULL OUTER JOIN transformer_cost USING (grid_case_id)
        )
        SELECT
            gbb.grid_case_id,
            gbb.kcid,
            gbb.bcid,
            gbb.building_use,
            ST_X(ST_Transform(gbb.centroid, :target_epsg)) AS x,
            ST_Y(ST_Transform(gbb.centroid, :target_epsg)) AS y,
            gc.cable_segments,
            gc.cable_segments_requiring_expansion,
            gc.cable_cost_eur,
            gc.max_cable_loading_percent,
            gc.transformer_requires_expansion,
            gc.transformer_cost_eur,
            gc.transformer_loading_percent,
            gc.additional_transformer_kva,
            gc.total_cost_eur
        FROM grid_cost gc
        JOIN surrogrid.grid_building_bus gbb USING (grid_case_id)
        WHERE {' AND '.join(filters)}
        """
    )
    with db.engine.connect() as conn:
        points = pd.read_sql_query(query, conn, params=params)
    points.attrs["analysis_key"] = analysis_key
    return points


def _grid_metrics_from_points(points: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "kcid",
        "bcid",
        "cable_segments",
        "cable_segments_requiring_expansion",
        "cable_cost_eur",
        "max_cable_loading_percent",
        "transformer_requires_expansion",
        "transformer_cost_eur",
        "transformer_loading_percent",
        "additional_transformer_kva",
        "total_cost_eur",
    ]
    available_cols = [col for col in metric_cols if col in points.columns]
    return points.groupby("grid_case_id", as_index=False)[available_cols].first()



def _synthetic_expansion_envelope_layers(
    points: pd.DataFrame,
    *,
    value_column: str,
    value_scale: float,
) -> dict[str, object]:
    envelopes = _make_envelopes(points, "Synthetic", "grid_case_id")
    metrics = _grid_metrics_from_points(points)
    values_by_grid = metrics.set_index("grid_case_id")[value_column].astype(float).to_dict()

    polygons = []
    polygon_values = []
    line_segments = []
    line_values = []
    point_clouds = []
    point_values = []
    for envelope in envelopes:
        value = float(values_by_grid.get(int(envelope.grid_id), np.nan))
        display_value = value / value_scale
        if envelope.envelope is not None:
            polygons.append(envelope.envelope)
            polygon_values.append(display_value)
        elif envelope.n_points == 2:
            line_segments.append(envelope.points)
            line_values.append(display_value)
        else:
            point_clouds.append(envelope.points)
            point_values.append(display_value)

    display_values = pd.Series(
        polygon_values + line_values + point_values,
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "points": points,
        "grid_metrics": metrics,
        "polygons": polygons,
        "polygon_values": polygon_values,
        "line_segments": line_segments,
        "line_values": line_values,
        "point_clouds": point_clouds,
        "point_values": point_values,
        "display_values": display_values,
    }


def _draw_synthetic_expansion_envelope_layers(
    ax: plt.Axes,
    layers: dict[str, object],
    *,
    cmap_obj,
    norm,
    target_epsg: int,
    show_points: bool,
    show_buildings: bool,
    building_point_size: float,
    building_alpha: float,
    envelope_alpha: float,
    add_osm_layer: bool,
    osm_source: object | None,
    osm_zoom: str | int,
    osm_alpha: float,
) -> object:
    points = layers["points"]
    polygons = layers["polygons"]
    polygon_values = layers["polygon_values"]
    line_segments = layers["line_segments"]
    line_values = layers["line_values"]
    point_clouds = layers["point_clouds"]
    point_values = layers["point_values"]

    if polygons:
        collection = PolyCollection(
            polygons,
            array=np.asarray(polygon_values, dtype=float),
            cmap=cmap_obj,
            norm=norm,
            edgecolors="#263238",
            linewidths=0.65,
            alpha=envelope_alpha,
            zorder=2,
        )
        ax.add_collection(collection)
        color_source = collection
    else:
        color_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    if line_segments:
        colors = cmap_obj(norm(np.asarray(line_values, dtype=float)))
        ax.add_collection(
            LineCollection(
                line_segments,
                colors=colors,
                linewidths=1.6,
                alpha=min(0.9, envelope_alpha + 0.18),
                zorder=3,
            )
        )
    for cloud, value in zip(point_clouds, point_values):
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=20,
            color=cmap_obj(norm(value)),
            edgecolor="#263238",
            linewidth=0.4,
            alpha=min(0.95, envelope_alpha + 0.25),
            zorder=4,
        )

    if show_buildings:
        ax.scatter(
            points["x"],
            points["y"],
            s=building_point_size,
            color="#202020",
            alpha=building_alpha,
            linewidths=0,
            zorder=5,
        )

    if show_points:
        ax.scatter(points["x"], points["y"], s=2, color="#111111", alpha=0.12, linewidths=0, zorder=6)

    ax.set_aspect("equal", adjustable="box")
    if add_osm_layer:
        _add_osm_basemap(
            ax,
            target_epsg=target_epsg,
            source=osm_source,
            zoom=osm_zoom,
            alpha=osm_alpha,
        )
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_xlabel(f"x [EPSG:{target_epsg}]")
    ax.set_ylabel(f"y [EPSG:{target_epsg}]")
    return color_source


def _ordered_analysis_items(
    analysis_keys: Mapping[str, str] | Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    if isinstance(analysis_keys, Mapping):
        items = list(analysis_keys.items())
    else:
        items = list(analysis_keys)
    if not items:
        raise ValueError("analysis_keys must contain at least one labeled analysis key.")
    return [(str(label), str(analysis_key)) for label, analysis_key in items]


def plot_synthetic_expansion_envelope_panels(
    *,
    analysis_keys: Mapping[str, str] | Sequence[tuple[str, str]],
    value_column: str = "total_cost_eur",
    building_use: str | Sequence[str] | None = None,
    target_epsg: int = 25832,
    clip_quantile: float | None = 0.95,
    log_scale: bool = False,
    cmap: str | LinearSegmentedColormap = "cost_green_red",
    show_points: bool = False,
    show_buildings: bool = True,
    building_point_size: float = 1.2,
    building_alpha: float = 0.06,
    envelope_alpha: float = 0.46,
    show_axis_ticks: bool = True,
    add_osm_layer: bool = True,
    osm_source: object | None = None,
    osm_zoom: str | int = "auto",
    osm_alpha: float = 0.72,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ncols: int | None = None,
    real_grid_data_path: str | Path | None = None,
    real_load_type: str | Sequence[str] | None = "HH",
) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, pd.DataFrame]]]:
    """Plot synthetic expansion envelopes, optionally with a real-reference row."""

    analysis_items = _ordered_analysis_items(analysis_keys)
    value_scale, value_unit = _display_value_scale(value_column)
    datasets = []
    for label, analysis_key in analysis_items:
        points = load_synthetic_expansion_envelope_points(
            analysis_key=analysis_key,
            building_use=building_use,
            target_epsg=target_epsg,
        )
        if points.empty:
            raise ValueError(f"No synthetic expansion envelope points found for analysis_key={analysis_key!r}.")
        if value_column not in points.columns:
            available = ", ".join(sorted(points.columns))
            raise ValueError(f"Unknown value_column {value_column!r}. Available columns: {available}.")
        layers = _synthetic_expansion_envelope_layers(points, value_column=value_column, value_scale=value_scale)
        datasets.append({"analysis_key": analysis_key, "label": label, "layers": layers})

    all_values = pd.concat([dataset["layers"]["display_values"] for dataset in datasets], ignore_index=True)
    all_values = all_values.replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        raise ValueError(f"No finite values available for {value_column!r}.")
    vmax = float(all_values.max())
    if clip_quantile is not None:
        q = float(clip_quantile)
        if q > 1:
            q = q / 100
        if q <= 0 or q > 1:
            raise ValueError("clip_quantile must satisfy 0 < value <= 1, or 0 < value <= 100.")
        vmax = float(all_values.quantile(q))
    vmax = max(vmax, 1e-9)
    if log_scale:
        positive = all_values[all_values > 0]
        if positive.empty:
            norm = Normalize(vmin=0.0, vmax=max(float(all_values.max()), 1e-9))
        else:
            norm = LogNorm(vmin=max(float(positive.min()), 1e-6), vmax=max(vmax, float(positive.min()) * 1.01))
    else:
        norm = Normalize(vmin=0.0, vmax=vmax)

    real_points = pd.DataFrame()
    real_envelopes = []
    if real_grid_data_path is not None:
        real_points = load_real_grid_points(
            real_grid_data_path=real_grid_data_path,
            real_load_type=real_load_type,
        )
        if not real_points.empty:
            real_envelopes = _make_envelopes(real_points, "Real SWF", "lv_id")

    # Keep both rows on the synthetic model scope. Real-grid coordinate
    # outliers must not change the comparison extent.
    point_frames = [dataset["layers"]["points"][["x", "y"]] for dataset in datasets]
    all_points = pd.concat(point_frames, ignore_index=True)
    x_min, x_max = float(all_points["x"].min()), float(all_points["x"].max())
    y_min, y_max = float(all_points["y"].min()), float(all_points["y"].max())
    pad_x = max((x_max - x_min) * 0.035, 25.0)
    pad_y = max((y_max - y_min) * 0.035, 25.0)
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    n_panels = len(datasets)
    has_real_row = bool(real_envelopes)
    if has_real_row:
        ncols = n_panels
        synthetic_rows = 1
    else:
        if ncols is None:
            ncols = n_panels
        ncols = max(1, min(int(ncols), n_panels))
        synthetic_rows = int(np.ceil(n_panels / ncols))
    nrows = synthetic_rows + (1 if has_real_row else 0)
    if figsize is None:
        figsize = (max(5.2 * ncols, 8.0), max(4.6 * nrows, 7.3))
    cmap_obj = _cost_colormap(cmap)
    fig, axes_raw = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    axes_grid = np.atleast_2d(axes_raw)
    axes = axes_grid.ravel()
    color_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    for ax, dataset in zip(axes, datasets):
        color_source = _draw_synthetic_expansion_envelope_layers(
            ax,
            dataset["layers"],
            cmap_obj=cmap_obj,
            norm=norm,
            target_epsg=target_epsg,
            show_points=show_points,
            show_buildings=show_buildings,
            building_point_size=building_point_size,
            building_alpha=building_alpha,
            envelope_alpha=envelope_alpha,
            add_osm_layer=False,
            osm_source=osm_source,
            osm_zoom=osm_zoom,
            osm_alpha=osm_alpha,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if add_osm_layer:
            _add_osm_basemap(
                ax,
                target_epsg=target_epsg,
                source=osm_source,
                zoom=osm_zoom,
                alpha=osm_alpha,
            )
        metric_count = len(dataset["layers"]["grid_metrics"])
        ax.set_title(f"{dataset['label']} ({metric_count} grids)")
        if not show_axis_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

    for ax in axes[len(datasets):synthetic_rows * ncols]:
        ax.set_axis_off()

    if has_real_row:
        real_axes = axes_grid[synthetic_rows, :]
        for col_idx, (ax, dataset) in enumerate(zip(real_axes, datasets)):
            _add_envelopes(
                ax,
                real_envelopes,
                facecolor="#F58518",
                edgecolor="#B75D00",
                point_color="#B75D00",
                alpha=0.18,
                linewidth=0.8,
                show_points=show_points,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"Real SWF reference - {dataset['label']} ({len(real_envelopes)} grids)")
            ax.set_aspect("equal", adjustable="box")
            if add_osm_layer:
                _add_osm_basemap(
                    ax,
                    target_epsg=target_epsg,
                    source=osm_source,
                    zoom=osm_zoom,
                    alpha=osm_alpha,
                )
            if not show_axis_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
    title_label = _display_label(value_column, value_unit)
    cbar = fig.colorbar(color_source, ax=axes.ravel().tolist(), shrink=0.78)
    cbar.set_label(title_label)
    fig.suptitle(f"{value_column} estimation aggregated per grid envelope", y=1.02)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240, bbox_inches="tight")

    return fig, axes, {
        **{
            dataset["label"]: {
                "points": dataset["layers"]["points"],
                "grid_metrics": dataset["layers"]["grid_metrics"].sort_values(value_column, ascending=False).reset_index(drop=True),
            }
            for dataset in datasets
        },
        **({
            "Real SWF": {
                "points": real_points,
                "grid_metrics": pd.DataFrame({
                    "grid_id": [item.grid_id for item in real_envelopes],
                    "n_points": [item.n_points for item in real_envelopes],
                    "has_polygon": [item.envelope is not None for item in real_envelopes],
                }),
            }
        } if has_real_row else {}),
    }


def plot_synthetic_expansion_envelope_comparison(
    *,
    pre_analysis_key: str,
    post_analysis_key: str,
    labels: tuple[str, str] = ("Pre", "Post-all"),
    value_column: str = "total_cost_eur",
    building_use: str | Sequence[str] | None = None,
    target_epsg: int = 25832,
    clip_quantile: float | None = 0.95,
    log_scale: bool = False,
    cmap: str | LinearSegmentedColormap = "cost_green_red",
    show_points: bool = False,
    show_buildings: bool = True,
    building_point_size: float = 1.2,
    building_alpha: float = 0.06,
    envelope_alpha: float = 0.46,
    show_axis_ticks: bool = True,
    add_osm_layer: bool = True,
    osm_source: object | None = None,
    osm_zoom: str | int = "auto",
    osm_alpha: float = 0.72,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (15.5, 7.3),
) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, pd.DataFrame]]]:
    """Plot pre/post expansion envelopes side by side with shared map extent and color scale."""

    return plot_synthetic_expansion_envelope_panels(
        analysis_keys={labels[0]: pre_analysis_key, labels[1]: post_analysis_key},
        value_column=value_column,
        building_use=building_use,
        target_epsg=target_epsg,
        clip_quantile=clip_quantile,
        log_scale=log_scale,
        cmap=cmap,
        show_points=show_points,
        show_buildings=show_buildings,
        building_point_size=building_point_size,
        building_alpha=building_alpha,
        envelope_alpha=envelope_alpha,
        show_axis_ticks=show_axis_ticks,
        add_osm_layer=add_osm_layer,
        osm_source=osm_source,
        osm_zoom=osm_zoom,
        osm_alpha=osm_alpha,
        output_path=output_path,
        figsize=figsize,
    )

def plot_synthetic_expansion_envelopes(
    *,
    analysis_key: str | None = None,
    value_column: str = "total_cost_eur",
    building_use: str | Sequence[str] | None = None,
    target_epsg: int = 25832,
    clip_quantile: float | None = 0.95,
    log_scale: bool = False,
    cmap: str | LinearSegmentedColormap = "cost_green_red",
    show_points: bool = False,
    show_buildings: bool = True,
    building_point_size: float = 1.2,
    building_alpha: float = 0.06,
    envelope_alpha: float = 0.58,
    add_osm_layer: bool = True,
    osm_source: object | None = None,
    osm_zoom: str | int = "auto",
    osm_alpha: float = 0.72,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (9.5, 8.0),
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot synthetic supplied-area envelopes colored by expansion severity.

    By default all available building points are used. ``value_column`` can be one
    of the grid-level metrics returned by
    :func:`load_synthetic_expansion_envelope_points`, for example
    ``total_cost_eur``, ``cable_cost_eur``, or ``transformer_cost_eur``. Cost
    columns ending in ``_eur`` are displayed in thousand euros on the colorbar.
    Set ``add_osm_layer=False`` to disable the OSM background. The building
    context layer is controlled through ``show_buildings``.
    """

    points = load_synthetic_expansion_envelope_points(
        analysis_key=analysis_key,
        building_use=building_use,
        target_epsg=target_epsg,
    )
    if points.empty:
        raise ValueError("No synthetic expansion envelope points found for the selected filters.")
    if value_column not in points.columns:
        available = ", ".join(sorted(points.columns))
        raise ValueError(f"Unknown value_column {value_column!r}. Available columns: {available}.")

    analysis_key = str(points.attrs.get("analysis_key", analysis_key or "latest"))
    envelopes = _make_envelopes(points, "Synthetic", "grid_case_id")
    metrics = _grid_metrics_from_points(points)
    values_by_grid = metrics.set_index("grid_case_id")[value_column].astype(float).to_dict()

    polygons = []
    polygon_values = []
    line_segments = []
    line_values = []
    point_clouds = []
    point_values = []
    for envelope in envelopes:
        value = float(values_by_grid.get(int(envelope.grid_id), np.nan))
        if envelope.envelope is not None:
            polygons.append(envelope.envelope)
            polygon_values.append(value)
        elif envelope.n_points == 2:
            line_segments.append(envelope.points)
            line_values.append(value)
        else:
            point_clouds.append(envelope.points)
            point_values.append(value)

    value_scale, value_unit = _display_value_scale(value_column)
    polygon_display_values = [value / value_scale for value in polygon_values]
    line_display_values = [value / value_scale for value in line_values]
    point_display_values = [value / value_scale for value in point_values]
    all_values = pd.Series(
        polygon_display_values + line_display_values + point_display_values,
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        raise ValueError(f"No finite values available for {value_column!r}.")

    vmax = float(all_values.max())
    if clip_quantile is not None:
        q = float(clip_quantile)
        if q > 1:
            q = q / 100
        if q <= 0 or q > 1:
            raise ValueError("clip_quantile must satisfy 0 < value <= 1, or 0 < value <= 100.")
        vmax = float(all_values.quantile(q))
    vmax = max(vmax, 1e-9)
    if log_scale:
        positive = all_values[all_values > 0]
        if positive.empty:
            norm = Normalize(vmin=0.0, vmax=max(float(all_values.max()), 1e-9))
        else:
            norm = LogNorm(vmin=max(float(positive.min()), 1e-6), vmax=max(vmax, float(positive.min()) * 1.01))
    else:
        norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    cmap_obj = _cost_colormap(cmap)

    if polygons:
        collection = PolyCollection(
            polygons,
            array=np.asarray(polygon_display_values, dtype=float),
            cmap=cmap_obj,
            norm=norm,
            edgecolors="#263238",
            linewidths=0.65,
            alpha=envelope_alpha,
            zorder=2,
        )
        ax.add_collection(collection)
        color_source = collection
    else:
        color_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    if line_segments:
        colors = cmap_obj(norm(np.asarray(line_display_values, dtype=float)))
        ax.add_collection(LineCollection(line_segments, colors=colors, linewidths=1.6, alpha=min(0.9, envelope_alpha + 0.18), zorder=3))
    for cloud, value in zip(point_clouds, point_display_values):
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=20,
            color=cmap_obj(norm(value)),
            edgecolor="#263238",
            linewidth=0.4,
            alpha=min(0.95, envelope_alpha + 0.25),
            zorder=4,
        )

    if show_buildings:
        ax.scatter(
            points["x"],
            points["y"],
            s=building_point_size,
            color="#202020",
            alpha=building_alpha,
            linewidths=0,
            zorder=5,
        )

    if show_points:
        ax.scatter(points["x"], points["y"], s=2, color="#111111", alpha=0.12, linewidths=0, zorder=6)

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    if add_osm_layer:
        _add_osm_basemap(
            ax,
            target_epsg=target_epsg,
            source=osm_source,
            zoom=osm_zoom,
            alpha=osm_alpha,
        )
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_xlabel(f"x [EPSG:{target_epsg}]")
    ax.set_ylabel(f"y [EPSG:{target_epsg}]")
    title_label = _display_label(value_column, value_unit)
    ax.set_title(f"{value_column} estimation aggregated per grid envelope")
    cbar = fig.colorbar(color_source, ax=ax, shrink=0.82)
    cbar.set_label(title_label)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240)

    return fig, ax, {
        "points": points,
        "grid_metrics": metrics.sort_values(value_column, ascending=False).reset_index(drop=True),
    }

def load_paired_synthetic_building_coverage(
    *,
    paired_plan_path: str | Path,
    target_epsg: int = 25832,
) -> pd.DataFrame:
    """Mark which existing synthetic buildings carry paired scenario demand."""
    plan_path = Path(paired_plan_path)
    plan = pd.read_csv(plan_path)
    required = {"synthetic_grid_case_id", "building_objectid"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(
            f"Paired allocation plan is missing columns: {sorted(missing)}"
        )

    grid_case_ids = sorted(
        int(value)
        for value in pd.to_numeric(
            plan["synthetic_grid_case_id"], errors="raise"
        ).unique()
    )
    paired_buildings = set(
        plan["building_objectid"].dropna().astype(str).unique()
    )
    if not grid_case_ids or not paired_buildings:
        raise ValueError("Paired allocation plan contains no synthetic buildings.")

    query = text(
        """
        SELECT
            gbb.grid_case_id,
            gbb.objectid AS building_objectid,
            gbb.building_use,
            gbb.street,
            gbb.house_number,
            ST_X(ST_Transform(gbb.centroid, :target_epsg)) AS x,
            ST_Y(ST_Transform(gbb.centroid, :target_epsg)) AS y
        FROM surrogrid.grid_building_bus gbb
        WHERE gbb.grid_case_id = ANY(:grid_case_ids)
          AND gbb.centroid IS NOT NULL
        ORDER BY gbb.grid_case_id, gbb.objectid
        """
    )
    db = SurroGridDatabase()
    with db.engine.connect() as conn:
        points = pd.read_sql_query(
            query,
            conn,
            params={
                "grid_case_ids": grid_case_ids,
                "target_epsg": int(target_epsg),
            },
        )

    points["paired_status"] = np.where(
        points["building_objectid"].astype(str).isin(paired_buildings),
        "Included in paired demand",
        "Not selected for paired demand",
    )
    points.attrs.update(
        {
            "paired_plan_path": str(plan_path.resolve()),
            "target_epsg": int(target_epsg),
            "paired_buildings": int(
                points["paired_status"].eq("Included in paired demand").sum()
            ),
            "unselected_buildings": int(
                points["paired_status"].eq("Not selected for paired demand").sum()
            ),
        }
    )
    return points


def plot_paired_synthetic_building_coverage(
    *,
    paired_plan_path: str | Path,
    target_epsg: int = 25832,
    show_grid_envelopes: bool = True,
    add_osm_layer: bool = True,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (11.0, 10.0),
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Map existing synthetic buildings with and without paired scenario demand."""
    points = load_paired_synthetic_building_coverage(
        paired_plan_path=paired_plan_path,
        target_epsg=target_epsg,
    )
    retained = points[points["paired_status"].eq("Included in paired demand")]
    omitted = points[points["paired_status"].eq("Not selected for paired demand")]
    if retained.empty or omitted.empty:
        raise ValueError("Coverage map requires both paired and omitted buildings.")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if show_grid_envelopes:
        for envelope in _make_envelopes(points, "Synthetic", "grid_case_id"):
            if envelope.envelope is not None:
                ax.plot(
                    envelope.envelope[:, 0],
                    envelope.envelope[:, 1],
                    color="#5f6368",
                    linewidth=0.45,
                    alpha=0.28,
                    zorder=2,
                )

    ax.scatter(
        retained["x"],
        retained["y"],
        s=5.0,
        color="#2166ac",
        alpha=0.32,
        edgecolors="none",
        label=f"Included in paired demand ({len(retained):,})",
        zorder=3,
    )
    ax.scatter(
        omitted["x"],
        omitted["y"],
        s=7.0,
        color="#d73027",
        alpha=0.72,
        edgecolors="none",
        label=f"Existing, not selected for paired demand ({len(omitted):,})",
        zorder=4,
    )

    x_span = float(points["x"].max() - points["x"].min())
    y_span = float(points["y"].max() - points["y"].min())
    ax.set_xlim(
        float(points["x"].min()) - max(0.025 * x_span, 50.0),
        float(points["x"].max()) + max(0.025 * x_span, 50.0),
    )
    ax.set_ylim(
        float(points["y"].min()) - max(0.025 * y_span, 50.0),
        float(points["y"].max()) + max(0.025 * y_span, 50.0),
    )
    ax.set_aspect("equal", adjustable="box")

    if add_osm_layer:
        _add_osm_basemap(
            ax,
            target_epsg=target_epsg,
            alpha=0.62,
        )

    omitted_share = 100.0 * len(omitted) / len(points)
    ax.set_title(
        "Paired demand-allocation scope\n"
        f"{len(omitted):,} of {len(points):,} existing synthetic buildings not selected "
        f"({omitted_share:.1f}%)",
        fontsize=15,
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=10,
        markerscale=2.2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")

    return fig, ax, points

