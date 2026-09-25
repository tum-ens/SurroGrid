"""pylovo grid identity: candidate lists, grid references, grid cases, grid reads.

A *grid reference* (``grid_ref``) is the dict that names one pylovo grid for
the pipeline: ``ags, candidate_index, cell_id, bridge_filename,
grid_result_id, version_id, plz, kcid, bcid``. Candidates of an AGS are the
pylovo grids of its postcodes with at least ``min_buildings`` buildings,
numbered by ``(plz, kcid, bcid)`` from 0 (``cell_id = <ags>-<index:02d>``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from gridexpand.common.building_components import (
    build_building_components,
    validate_component_bus_metadata,
    validate_physical_buildings,
)
from gridexpand.db.engine import load_env
from gridexpand.db.schema import ensure_schema

MEAN_HOUSEHOLD_SIZE = 2.03  # persons, mean of Step 2's hh_size_distribution.csv
GRID_REF_KEYS = (
    "ags", "candidate_index", "cell_id", "bridge_filename",
    "grid_result_id", "version_id", "plz", "kcid", "bcid",
)
_AGS_ID = re.compile(r"^0*(\d+)(?:-(\d+))?$")
# Numeric pylovo versions sort numerically ('10' after '9'); others after them.
_LATEST_VERSION_ORDER = (
    "CASE WHEN {v}::text ~ '^[0-9]+$' THEN {v}::text::numeric END DESC NULLS LAST, {v} DESC"
)

CANDIDATES_SQL = f"""
WITH ags_plz AS (
    SELECT DISTINCT plz
    FROM pylovo.municipal_register
    WHERE ags = :ags
),
grids AS (
    SELECT gr.grid_result_id, gr.version_id, gr.plz, gr.kcid, gr.bcid
    FROM pylovo.grid_result gr
    JOIN ags_plz ap ON ap.plz = gr.plz
    WHERE CAST(:pylovo_version_id AS TEXT) IS NULL
       OR gr.version_id::text = CAST(:pylovo_version_id AS TEXT)
),
building_counts AS (
    SELECT
        b.grid_result_id,
        b.version_id,
        COUNT(*) AS n_buildings,
        COUNT(*) FILTER (WHERE b.residential_floor_area > 0) AS n_residential_buildings
    FROM pylovo.buildings_result b
    JOIN grids g
      ON g.grid_result_id = b.grid_result_id
     AND g.version_id = b.version_id
    GROUP BY b.grid_result_id, b.version_id
),
latest AS (
    SELECT DISTINCT ON (g.plz, g.kcid, g.bcid)
        g.grid_result_id,
        g.version_id,
        g.plz,
        g.kcid,
        g.bcid,
        bc.n_buildings,
        bc.n_residential_buildings
    FROM grids g
    JOIN building_counts bc
      ON bc.grid_result_id = g.grid_result_id
     AND bc.version_id = g.version_id
    WHERE CASE
            WHEN :demand_scope = 'residential' THEN bc.n_residential_buildings
            ELSE bc.n_buildings
          END >= :min_buildings
    ORDER BY g.plz, g.kcid, g.bcid, {_LATEST_VERSION_ORDER.format(v="g.version_id")}
)
SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY plz, kcid, bcid) - 1 AS candidate_index
FROM latest
ORDER BY candidate_index
"""


def get_pylovo_version_id() -> str | None:
    """``PYLOVO_VERSION_ID`` from ``.env``/environment, or None."""
    load_env()
    value = os.getenv("PYLOVO_VERSION_ID")
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def normalize_ags(value: str | int) -> int:
    """Store AGS as an integer, without a leading zero."""
    return int(str(value).strip().lstrip("0") or "0")


def parse_grid_filename(filename: str) -> dict[str, Any]:
    """Parse ``<ags>-<index>_<plz>_<kcid>_<bcid>.h5`` into its parts."""
    stem = Path(filename).name.removesuffix(".h5")
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError("DB storage filenames must follow <cell_id>_<plz>_<kcid>_<bcid>.h5.")
    cell_id = parts[0]
    ags_match = _AGS_ID.match(cell_id)
    if ags_match is None:
        raise ValueError("DB storage filenames must begin with an AGS-based cell_id.")
    return {
        "ags": normalize_ags(ags_match.group(1)),
        "candidate_index": int(ags_match.group(2) or 0),
        "cell_id": cell_id,
        "plz": int(parts[1]),
        "kcid": int(parts[2]),
        "bcid": int(parts[3]),
    }


def format_grid_ref(
    *,
    ags: int,
    row: dict[str, Any],
    candidate_index: int | None = None,
    bridge_filename: str | None = None,
) -> dict[str, Any]:
    """Build a grid reference from a ``pylovo.grid_result`` row."""
    if candidate_index is None:
        candidate_index = int(row.get("candidate_index", 0))
    cell_id = f"{int(ags)}-{int(candidate_index):02d}"
    bridge_filename = bridge_filename or f"{cell_id}_{int(row['plz'])}_{int(row['kcid'])}_{int(row['bcid'])}.h5"
    return {
        "ags": int(ags),
        "candidate_index": int(candidate_index),
        "cell_id": cell_id,
        "bridge_filename": bridge_filename,
        "grid_result_id": int(row["grid_result_id"]),
        "version_id": str(row["version_id"]),
        "plz": int(row["plz"]),
        "kcid": int(row["kcid"]),
        "bcid": int(row["bcid"]),
    }


def list_grid_candidates(
    engine: Engine,
    ags: str | int,
    *,
    min_buildings: int = 5,
    demand_scope: str = "all",
    pylovo_version_id: str | None = None,
) -> list[dict[str, Any]]:
    """All candidate grids of an AGS, numbered by ``(plz, kcid, bcid)``.

    Without ``pylovo_version_id`` each grid is taken from its latest pylovo
    version that meets ``min_buildings``.

    Returns:
        Grid references plus ``n_buildings``, ``n_residential_buildings`` and
        ``n_selected_buildings`` (the count ``min_buildings`` applies to).
    """
    ags = normalize_ags(ags)
    count_column = "n_residential_buildings" if demand_scope == "residential" else "n_buildings"
    with engine.connect() as conn:
        rows = conn.execute(
            text(CANDIDATES_SQL),
            {
                "ags": ags,
                "min_buildings": int(min_buildings),
                "demand_scope": demand_scope,
                "pylovo_version_id": pylovo_version_id,
            },
        ).mappings().all()
    return [
        format_grid_ref(ags=ags, row=dict(row), candidate_index=int(row["candidate_index"]))
        | {
            "n_buildings": int(row["n_buildings"]),
            "n_selected_buildings": int(row[count_column]),
            "n_residential_buildings": int(row["n_residential_buildings"]),
        }
        for row in rows
    ]


def grid_ref_from_specs(
    engine: Engine,
    *,
    ags: int,
    plz: int,
    kcid: int,
    bcid: int,
    candidate_index: int,
    bridge_filename: str | None = None,
    pylovo_version_id: str | None = None,
) -> dict[str, Any]:
    """Grid reference of one ``(plz, kcid, bcid)`` (latest pylovo version if none given)."""
    query = text(
        f"""
        SELECT grid_result_id, version_id, plz, kcid, bcid
        FROM pylovo.grid_result
        WHERE plz = :plz AND kcid = :kcid AND bcid = :bcid
          AND (CAST(:pylovo_version_id AS TEXT) IS NULL OR version_id::text = CAST(:pylovo_version_id AS TEXT))
        ORDER BY {_LATEST_VERSION_ORDER.format(v="version_id")}
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            query,
            {"plz": int(plz), "kcid": int(kcid), "bcid": int(bcid), "pylovo_version_id": pylovo_version_id},
        ).mappings().first()
    if row is None:
        version_hint = f" and PYLOVO_VERSION_ID={pylovo_version_id}" if pylovo_version_id else ""
        raise ValueError(f"No pylovo grid found for PLZ={plz}, KCID={kcid}, BCID={bcid}{version_hint}.")
    return format_grid_ref(ags=ags, row=dict(row), candidate_index=candidate_index, bridge_filename=bridge_filename)


def resolve_grid_identifier(
    engine: Engine,
    input_id: str | int,
    *,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
    demand_scope: str = "all",
    pylovo_version_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a CLI identifier to one concrete pylovo grid.

    Accepts an AGS (``09278140`` / ``9278140``, optionally ``-<index>``) or a
    bridge filename like ``9278140-00_94342_1_-1.h5``.
    """
    input_id_str = str(input_id).strip()
    if input_id_str.endswith(".h5"):
        parsed = parse_grid_filename(input_id_str)
        return grid_ref_from_specs(
            engine,
            ags=parsed["ags"],
            plz=parsed["plz"],
            kcid=parsed["kcid"],
            bcid=parsed["bcid"],
            candidate_index=parsed["candidate_index"],
            bridge_filename=input_id_str,
            pylovo_version_id=pylovo_version_id,
        )
    match = _AGS_ID.match(input_id_str)
    if not match:
        raise ValueError(
            "DB storage expects inputfile_id as AGS, for example 09278140, "
            "or a DB-mode filename like 9278140-00_94342_1_-1.h5."
        )
    ags = normalize_ags(match.group(1))
    if match.group(2) is not None:
        candidate_index = int(match.group(2))
    if (plz is None) != (kcid is None) or (plz is None) != (bcid is None):
        raise ValueError("Provide --plz, --kcid, and --bcid together, or omit all three.")
    if plz is not None:
        return grid_ref_from_specs(
            engine, ags=ags, plz=int(plz), kcid=int(kcid), bcid=int(bcid),
            candidate_index=candidate_index, pylovo_version_id=pylovo_version_id,
        )
    for candidate in list_grid_candidates(
        engine, ags, min_buildings=min_buildings, demand_scope=demand_scope, pylovo_version_id=pylovo_version_id
    ):
        if candidate["candidate_index"] == int(candidate_index):
            return {key: candidate[key] for key in GRID_REF_KEYS}
    raise ValueError(f"No pylovo grid candidate found for AGS={ags}, candidate_index={candidate_index}.")


_GRID_CASE_UPSERT = text(
    """
    INSERT INTO surrogrid.grid_case (
        ags, plz, kcid, bcid, pylovo_grid_result_id, pylovo_version_id, cell_id
    )
    VALUES (:ags, :plz, :kcid, :bcid, :grid_result_id, :version_id, :cell_id)
    ON CONFLICT (ags, plz, kcid, bcid, pylovo_grid_result_id)
    DO UPDATE SET
        pylovo_version_id = EXCLUDED.pylovo_version_id,
        cell_id = EXCLUDED.cell_id
    RETURNING grid_case_id
    """
)


def get_or_create_grid_cases(engine: Engine, grid_refs: list[dict[str, Any]]) -> list[int]:
    """Upsert ``surrogrid.grid_case`` rows (one transaction); ids in input order."""
    ensure_schema(engine)
    with engine.begin() as conn:
        return [
            int(conn.execute(_GRID_CASE_UPSERT, {key: ref[key] for key in (
                "ags", "plz", "kcid", "bcid", "grid_result_id", "version_id", "cell_id"
            )}).scalar_one())
            for ref in grid_refs
        ]


def get_or_create_grid_case(engine: Engine, grid_ref: dict[str, Any]) -> int:
    """Upsert the ``surrogrid.grid_case`` row of one grid reference."""
    return get_or_create_grid_cases(engine, [grid_ref])[0]


def read_region(engine: Engine, grid_ref: dict[str, Any]) -> pd.DataFrame:
    """Municipal-register row of the grid's postcode (transformer coordinates)."""
    query = text(
        """
        WITH selected_grid AS (
            SELECT grid_result_id
            FROM pylovo.grid_result
            WHERE grid_result_id = :grid_result_id
        ),
        trafo AS (
            SELECT
                ST_Y(ST_Transform(tp.geom, 4326)) AS lat,
                ST_X(ST_Transform(tp.geom, 4326)) AS lon
            FROM pylovo.transformer_positions tp
            JOIN selected_grid sg ON sg.grid_result_id = tp.grid_result_id
            LIMIT 1
        )
        SELECT
            mr.plz,
            mr.pop,
            mr.area,
            mr.name_city,
            mr.pop_den,
            mr.regio7,
            COALESCE((SELECT lat FROM trafo), mr.lat) AS lat,
            COALESCE((SELECT lon FROM trafo), mr.lon) AS lon,
            :kcid AS kcid,
            :bcid AS bcid,
            :ags AS ags
        FROM pylovo.municipal_register mr
        WHERE mr.ags = :ags AND mr.plz = :plz
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        df_region = pd.read_sql_query(
            query,
            conn,
            params={key: grid_ref[key] for key in ("grid_result_id", "ags", "plz", "kcid", "bcid")},
        )
    if df_region.empty:
        raise ValueError(f"No municipal_register row found for AGS={grid_ref['ags']} and PLZ={grid_ref['plz']}.")
    return df_region


BUILDINGS_SQL = """
SELECT
    b.grid_result_id AS pylovo_grid_result_id,
    b.version_id AS pylovo_version_id,
    b.objectid,
    b.id,
    b.feature_id,
    b.vertice_id,
    b.height,
    b.floor_area,
    b.floor_number,
    b.residential_floor_area,
    b.nonresidential_floor_area,
    b.nonresidential_use,
    b.mix_score,
    b.mix_rule,
    b.mix_confidence,
    b.building_use,
    b.building_use_id,
    b.building_type,
    b.type,
    b.occupants,
    b.households,
    CAST(b.construction_year AS VARCHAR) AS construction_year,
    b.postcode,
    b.address_street_id,
    b.street,
    b.house_number,
    b.gemeindeschluessel,
    b.assigned_way_id,
    b.residential_peak_load_in_kw,
    b.nonresidential_peak_load_in_kw,
    b.nonresidential_mv_direct,
    b.peak_load_in_kw,
    b.connection_point,
    b.vertice_id AS consumer_vertex,
    ST_Y(ST_Transform(b.centroid, 4326)) AS lat,
    ST_X(ST_Transform(b.centroid, 4326)) AS lon
FROM pylovo.buildings_result b
WHERE b.grid_result_id = :grid_result_id
  AND b.version_id = :version_id
"""


def read_buildings(engine: Engine, grid_ref: dict[str, Any]) -> pd.DataFrame:
    """Physical buildings of the grid with their bus (pandas twin of grid_building_bus).

    Missing occupants of residential buildings with households are imputed as
    ``households * MEAN_HOUSEHOLD_SIZE`` and flagged in ``occupants_imputed``.
    """
    with engine.connect() as conn:
        df_buildings = pd.read_sql_query(
            text(BUILDINGS_SQL),
            conn,
            params={"grid_result_id": grid_ref["grid_result_id"], "version_id": grid_ref["version_id"]},
        )
        df_bus = pd.read_sql_query(
            text("SELECT pp_index AS bus, name FROM pylovo.pandapower_bus WHERE grid_result_id = :grid_result_id"),
            conn,
            params={"grid_result_id": grid_ref["grid_result_id"]},
        )
    if df_buildings.empty:
        raise ValueError(f"No buildings found for pylovo grid_result_id={grid_ref['grid_result_id']}.")

    df_buildings["grid_case_id"] = int(get_or_create_grid_case(engine, grid_ref))
    if not df_bus.empty:
        df_id = pd.DataFrame()
        df_id["vertice_id"] = df_bus["name"].astype(str).str.extract(r"^Consumer Nodebus (\d+)$")[0]
        df_id = df_id.dropna()
        df_id["vertice_id"] = df_id["vertice_id"].astype(int)
        df_id["bus"] = df_bus.loc[df_id.index, "bus"].astype(int)
        df_buildings = df_buildings.merge(df_id, on="vertice_id", how="left")
    else:
        df_buildings["bus"] = pd.NA
    if "connection_point" in df_buildings.columns:
        df_buildings["bus"] = df_buildings["bus"].fillna(df_buildings["connection_point"])

    # pylovo fills missing households but not occupants (58 ÜZW buildings in v1).
    missing_occupants = (
        pd.to_numeric(df_buildings["residential_floor_area"], errors="coerce").gt(0)
        & pd.to_numeric(df_buildings["households"], errors="coerce").gt(0)
        & df_buildings["occupants"].isna()
    )
    df_buildings["occupants_imputed"] = missing_occupants
    df_buildings.loc[missing_occupants, "occupants"] = (
        pd.to_numeric(df_buildings.loc[missing_occupants, "households"]) * MEAN_HOUSEHOLD_SIZE
    )
    validate_physical_buildings(df_buildings, require_grid_identity=True)

    cols = df_buildings.columns.tolist()
    cols.insert(0, cols.pop(cols.index("bus")))
    return df_buildings[cols].sort_values(by="bus").reset_index(drop=True)


def read_building_components(
    engine: Engine, grid_ref: dict[str, Any], df_buildings: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Component manifest of one grid, checked against ``surrogrid.grid_building_component``.

    Raises:
        ValueError: the SQL view and the pandas manifest differ.
    """
    physical = df_buildings if df_buildings is not None else read_buildings(engine, grid_ref)
    grid_case_id = int(get_or_create_grid_case(engine, grid_ref))
    components = build_building_components(physical, grid_case_id=grid_case_id)
    with engine.connect() as conn:
        sql_components = pd.read_sql_query(
            text(
                "SELECT * FROM surrogrid.grid_building_component "
                "WHERE grid_case_id = :grid_case_id ORDER BY component_id"
            ),
            conn,
            params={"grid_case_id": grid_case_id},
        )
        load_metadata = pd.read_sql_query(
            text("SELECT bus, category FROM pylovo.pandapower_load WHERE grid_result_id = :grid_result_id"),
            conn,
            params={"grid_result_id": grid_ref["grid_result_id"]},
        )
    if sql_components.empty:
        raise ValueError(f"No SQL component rows found for grid_case_id={grid_case_id}.")
    compare_columns = [
        "component_id", "objectid", "component_category", "effective_floor_area_m2",
        "installed_peak_kw", "bus", "included_in_lv", "mv_direct",
    ]
    expected = components[compare_columns].sort_values("component_id").reset_index(drop=True)
    actual = sql_components[compare_columns].sort_values("component_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-7)
    except AssertionError as exc:
        raise ValueError("SQL and normalized mixed-use component manifests differ.") from exc
    validate_component_bus_metadata(components, load_metadata)
    return components


def read_pandapower_grid(engine: Engine, grid_ref: dict[str, Any]):
    """The pylovo pandapower network of the grid."""
    import pandapower as pp

    with engine.connect() as conn:
        payload = conn.execute(
            text("SELECT grid FROM pylovo.grid_result WHERE grid_result_id = :grid_result_id"),
            {"grid_result_id": grid_ref["grid_result_id"]},
        ).scalar_one()
    grid_json = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
    return pp.from_json_string(grid_json)
