"""Geospatial diagnostic plots for synthetic and real grid comparison."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from scipy.spatial import ConvexHull
from sqlalchemy import text


GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


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
    manifest = pd.read_csv(root / "split_manifest.csv")
    selected = manifest[
        (manifest["variant"] == variant)
        & (manifest["category"] == category)
        & (manifest["load_status"] == load_status)
        & (manifest["status"] == "exported")
    ].copy()

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

