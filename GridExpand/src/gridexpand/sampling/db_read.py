"""pylovo readers of Step 1 (grid sampling and the HDF5 export).

``DataBase`` reads from the database of ``GridExpand/.env`` through
``gridexpand.db`` (shared engine, the SQL of ``gridexpand.db.grids``). Without
``PYLOVO_VERSION_ID`` the numerically latest pylovo version of a grid is used.
Nothing is written to the database.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from gridexpand.common.building_components import validate_physical_buildings
from gridexpand.db.engine import get_engine
from gridexpand.db.grids import (
    BUILDINGS_SQL,
    MEAN_HOUSEHOLD_SIZE,
    get_pylovo_version_id,
    grid_ref_from_specs,
    read_pandapower_grid,
)

# pylovo stores 25832 today; the census grid of the sampling notebooks is EPSG:3035.
CENSUS_SRID = 3035
_LATEST_VERSION = "CASE WHEN gr.version_id::text ~ '^[0-9]+$' THEN gr.version_id::text::numeric END DESC NULLS LAST, gr.version_id DESC"


def consumer_bus_frame(df_bus: pd.DataFrame) -> pd.DataFrame:
    """``vertice_id -> bus`` of the ``Consumer Nodebus <vertice_id>`` buses of a pandapower bus table."""
    vertice = df_bus["name"].astype(str).str.extract(r"^Consumer Nodebus (\d+)$")[0].dropna().astype(int)
    return pd.DataFrame({"vertice_id": vertice.to_numpy(), "bus": vertice.index.astype(int)})


def impute_missing_occupants(df_buildings: pd.DataFrame, mean_household_size: float = MEAN_HOUSEHOLD_SIZE) -> pd.DataFrame:
    """Occupants of residential buildings with households but none recorded (flag ``occupants_imputed``).

    Same rule as ``gridexpand.db.grids.read_buildings`` (pylovo fills households but
    not occupants, 58 ÜZW buildings in v1).
    """
    missing = (
        pd.to_numeric(df_buildings["residential_floor_area"], errors="coerce").gt(0)
        & pd.to_numeric(df_buildings["households"], errors="coerce").gt(0)
        & df_buildings["occupants"].isna()
    )
    df_buildings["occupants_imputed"] = missing
    df_buildings.loc[missing, "occupants"] = (
        pd.to_numeric(df_buildings.loc[missing, "households"]) * mean_household_size
    )
    return df_buildings


class DataBase:
    """Read-only pylovo access for the sampling notebooks and ``export_single_grid``."""

    def __init__(self) -> None:
        self.engine = get_engine()

    def show_contents(self) -> None:
        """Print the tables of the pylovo schema."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema_name ORDER BY table_name;"
                ),
                {"schema_name": "pylovo"},
            )
            tables = [row[0] for row in result]
        print("Available tables in schema 'pylovo':")
        print(tables)

    def read_grid_identifiers_from_positions(self, min_buildings: int = 5) -> pd.DataFrame:
        """Candidate grids ``(plz, kcid, bcid, loc)`` with at least ``min_buildings`` buildings.

        ``loc`` is the transformer position as WKT in EPSG:3035 (the census grid).
        """
        query = text(
            f"""
            WITH building_counts AS (
                SELECT
                    b.grid_result_id,
                    b.version_id,
                    COUNT(*) AS n_buildings
                FROM pylovo.buildings_result b
                GROUP BY b.grid_result_id, b.version_id
            )
            SELECT DISTINCT ON (gr.plz, gr.kcid, gr.bcid)
                gr.plz,
                gr.kcid,
                gr.bcid,
                ST_AsText(ST_Transform(tp.geom, {CENSUS_SRID})) AS loc
            FROM pylovo.grid_result gr
            JOIN pylovo.transformer_positions tp
              ON tp.grid_result_id = gr.grid_result_id
             AND tp.version_id = gr.version_id
            JOIN building_counts bc
              ON bc.grid_result_id = gr.grid_result_id
             AND bc.version_id = gr.version_id
            WHERE bc.n_buildings >= :min_buildings
              AND (CAST(:pylovo_version_id AS TEXT) IS NULL OR gr.version_id::text = CAST(:pylovo_version_id AS TEXT))
            ORDER BY gr.plz, gr.kcid, gr.bcid, {_LATEST_VERSION};
            """
        )
        df_generated_grids = pd.read_sql_query(
            query,
            self.engine,
            params={"min_buildings": int(min_buildings), "pylovo_version_id": get_pylovo_version_id()},
        )
        print(
            "Retrieved "
            f"{len(df_generated_grids)} generated grids from transformer_positions (global pool) "
            f"with >= {min_buildings} buildings!"
        )
        return df_generated_grids

    def _grid_ref(self, grid_specs: dict[str, Any]) -> dict[str, Any]:
        return grid_ref_from_specs(
            self.engine,
            ags=0,
            plz=int(grid_specs["plz"]),
            kcid=int(grid_specs["kcid"]),
            bcid=int(grid_specs["bcid"]),
            candidate_index=0,
            pylovo_version_id=get_pylovo_version_id(),
        )

    def read_single_ppgrid(self, grid_specs: dict[str, Any]):
        """pandapower network of one grid (``plz``, ``kcid``, ``bcid``)."""
        return read_pandapower_grid(self.engine, self._grid_ref(grid_specs))

    def read_trafo_pos(self, grid_specs: dict[str, Any]) -> dict[str, float]:
        """Transformer position ``{"lat", "lon"}`` (EPSG:4326) of one grid."""
        grid_ref = self._grid_ref(grid_specs)
        query = text(
            """
            SELECT ST_Y(ST_Transform(tp.geom, 4326)) AS lat, ST_X(ST_Transform(tp.geom, 4326)) AS lon
            FROM pylovo.transformer_positions tp
            WHERE tp.grid_result_id = :grid_result_id
              AND tp.version_id = :version_id
            LIMIT 1
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                query, {"grid_result_id": grid_ref["grid_result_id"], "version_id": grid_ref["version_id"]}
            ).mappings().first()
        if row is None or row["lat"] is None:
            raise ValueError(
                f"No transformer position found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, "
                f"BCID={grid_specs['bcid']}."
            )
        return {"lat": float(row["lat"]), "lon": float(row["lon"])}

    def read_regional_stats(self, plz: int) -> pd.DataFrame:
        """``pylovo.municipal_register`` rows of one postcode."""
        query = text(
            """
            SELECT plz, pop, area, name_city, pop_den, regio7
            FROM pylovo.municipal_register
            WHERE plz = :plz;
            """
        )
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"plz": int(plz)})

    def read_buildings(self, grid_specs: dict[str, Any], df_bus: pd.DataFrame) -> pd.DataFrame:
        """Physical buildings of one grid with their consumer bus (the DB-mode reader's columns).

        ``lat``/``lon`` come from the pylovo centroid (EPSG:4326; NULL centroids stay NaN),
        missing occupants are imputed as in DB mode.
        """
        grid_ref = self._grid_ref(grid_specs)
        with self.engine.connect() as conn:
            df_buildings = pd.read_sql_query(
                text(BUILDINGS_SQL),
                conn,
                params={"grid_result_id": grid_ref["grid_result_id"], "version_id": grid_ref["version_id"]},
            )
        if df_buildings.empty:
            raise ValueError(
                f"No buildings found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, BCID={grid_specs['bcid']}."
            )
        df_buildings = df_buildings.merge(consumer_bus_frame(df_bus), on="vertice_id", how="left")
        if "connection_point" in df_buildings.columns:
            df_buildings["bus"] = df_buildings["bus"].fillna(df_buildings["connection_point"])
        df_buildings = impute_missing_occupants(df_buildings)
        validate_physical_buildings(df_buildings)

        cols = df_buildings.columns.tolist()
        cols.insert(0, cols.pop(cols.index("bus")))
        return df_buildings[cols].sort_values(by="bus").reset_index(drop=True)
