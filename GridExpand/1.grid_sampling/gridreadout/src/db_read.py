"""Readout helpers for the pylovo grid database.

This module provides the `DataBase` class, a thin SQLAlchemy wrapper around a
PostgreSQL database that stores:

- pandapower grid JSON blobs
- transformer positions / grid identifiers (PLZ/KCID/BCID)
- building attributes used to map buildings to consumer buses

Connection credentials are taken from environment variables loaded in
`gridreadout/config.py`.
"""

from pathlib import Path
import sys

GRIDEXPAND_DIR = Path(__file__).resolve().parents[3]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
from common.building_components import validate_physical_buildings
from config import config

from sqlalchemy import create_engine, text
import pandapower as pp
from pyproj import Transformer

import pandas as pd
import json
import re

class DataBase:
    def __init__(self):
        self.connection_settings = {
            "host": config.DB_HOST,
            "port": config.DB_PORT,
            "name": config.DB_NAME,
            "user": config.DB_USER,
            "password": config.DB_PASSWORD
        }
        self.engine = self._get_engine()


    def _get_engine(self):
        """ Establish connection engine to database """
        # Access the database settings
        host = self.connection_settings["host"]
        port = self.connection_settings["port"]
        name = self.connection_settings["name"]
        user = self.connection_settings["user"]
        password = self.connection_settings["password"]

        # Create a database engine using SQLAlchemy.
        # This connection string uses psycopg2 as the driver.
        engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}')

        return engine


    def show_contents(self):
        """ Show all database sheets """
        schema_name = "pylovo"
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema_name ORDER BY table_name;"
                ),
                {"schema_name": schema_name},
            )
            tables = [row[0] for row in result]
            print(f"Available tables in schema '{schema_name}':")
            print(tables)


    def read_all_grid_identifiers(self):
        """Retrieve all available pylovo grid identifiers (PLZ, KCID, BCID)."""

        query = """
            SELECT DISTINCT plz, kcid, bcid
            FROM pylovo.grid_result;
        """
        # Execute the query with Pandas. This will only read data.
        df_generated_grids = pd.read_sql_query(query, self.engine)
        print(f"Retrieved {len(df_generated_grids)} generated grids!")


        return df_generated_grids

    def read_grid_identifiers_from_positions(self, min_buildings=5):
        """Retrieve candidate grids from pylovo transformer positions.

        Only grids with at least `min_buildings` associated buildings are
        returned.
        """

        query = text("""
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
                ST_AsText(tp.geom) AS loc,
                bc.n_buildings
            FROM pylovo.grid_result gr
            JOIN pylovo.transformer_positions tp
              ON tp.grid_result_id = gr.grid_result_id
             AND tp.version_id = gr.version_id
            JOIN building_counts bc
              ON bc.grid_result_id = gr.grid_result_id
             AND bc.version_id = gr.version_id
            WHERE bc.n_buildings >= :min_buildings
              AND (:pylovo_version_id IS NULL OR gr.version_id::text = :pylovo_version_id)
            ORDER BY gr.plz, gr.kcid, gr.bcid, gr.version_id DESC;
        """)
        df_generated_grids = pd.read_sql_query(
            query,
            self.engine,
            params={
                "min_buildings": int(min_buildings),
                "pylovo_version_id": config.PYLOVO_VERSION_ID,
            },
        )
        if "n_buildings" in df_generated_grids.columns:
            df_generated_grids = df_generated_grids.drop(columns=["n_buildings"])
        print(
            "Retrieved "
            f"{len(df_generated_grids)} generated grids from transformer_positions (global pool) "
            f"with >= {min_buildings} buildings!"
        )

        return df_generated_grids

    def read_single_ppgrid(self, grid_specs):
        """ Reads out single database grid based on plz, kcid, bcid
            
            Args:
                engine: SQLAlchemy engine to connect to database
                grid_specs: dict including plz, kcid, bcid

            Returns:
                net: pandapower grid topology (with lines, transformer, buses)
        """

        query = text("""
            SELECT grid
            FROM pylovo.grid_result
            WHERE (plz= :plz) AND (kcid= :kcid) AND (bcid= :bcid)
              AND (:pylovo_version_id IS NULL OR version_id::text = :pylovo_version_id)
            ORDER BY version_id DESC
            LIMIT 1;
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_grid = pd.read_sql(query, conn, params={
                "plz": int(grid_specs["plz"]),
                "kcid": int(grid_specs["kcid"]),
                "bcid": int(grid_specs["bcid"]),
                "pylovo_version_id": config.PYLOVO_VERSION_ID,
            })

        if df_grid.empty:
            raise ValueError(
                f"No grid found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, BCID={grid_specs['bcid']}."
            )

        grid_payload = df_grid.loc[0, "grid"]
        if isinstance(grid_payload, (dict, list)):
            grid_json = json.dumps(grid_payload)
        else:
            grid_json = str(grid_payload)

        # Transform to pandapower net
        net = pp.from_json_string(grid_json)

        return net


    def read_trafo_pos(self, grid_specs):
        """ Read out position of transformer position for given grid from database """

        query = text("""
            SELECT ST_AsText(tp.geom) as loc
            FROM pylovo.transformer_positions tp
            JOIN pylovo.grid_result gr
              ON gr.grid_result_id = tp.grid_result_id
             AND gr.version_id = tp.version_id
            WHERE (gr.plz= :plz) AND (gr.kcid= :kcid) AND (gr.bcid= :bcid)
              AND (:pylovo_version_id IS NULL OR gr.version_id::text = :pylovo_version_id)
            ORDER BY gr.version_id DESC
            LIMIT 1;
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_trafo = pd.read_sql(query, conn, params={
                "plz": int(grid_specs["plz"]),
                "kcid": int(grid_specs["kcid"]),
                "bcid": int(grid_specs["bcid"]),
                "pylovo_version_id": config.PYLOVO_VERSION_ID,
            })

        if df_trafo.empty:
            raise ValueError(
                f"No transformer position found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, BCID={grid_specs['bcid']}."
            )

        # Read out location string
        loc = df_trafo.loc[0, "loc"]
        match = re.match(r"POINT\(([-+]?[0-9]*\.?[0-9]+)\s*([-+]?[0-9]*\.?[0-9]+)\)", loc)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            # print(f"EPSG-3035:")
            # print(f"x: {x}, y: {y}")

        # Define the projections
        transformer = Transformer.from_crs(config.PYLOVO_COORD_FORMAT, config.TARGET_COORD_FORMAT, always_xy=True)
        # Convert from EPSG:3857 to EPSG:4326
        lon, lat = transformer.transform(x, y)
        # print(f"EPSG-4326:")
        # print(f"lat: {lat}, lon: {lon}")

        trafo_pos = {
            "lat": lat,
            "lon": lon
        }

        return trafo_pos
    
    def read_regional_stats(self, plz):
        query = text("""
            SELECT plz, pop, area, name_city, pop_den, regio7
            FROM pylovo.municipal_register
            WHERE plz=:plz;
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_region_specs = pd.read_sql(query, conn, params={"plz":int(plz)})

        return df_region_specs


    def read_buildings(self, grid_specs, df_bus):
        select_parts = [
            "b.grid_result_id AS pylovo_grid_result_id",
            "b.version_id AS pylovo_version_id",
            "b.objectid",
            "b.id",
            "b.feature_id",
            "b.vertice_id",
            "b.height",
            "b.floor_area",
            "b.floor_number",
            "b.residential_floor_area",
            "b.nonresidential_floor_area",
            "b.nonresidential_use",
            "b.mix_score",
            "b.mix_rule",
            "b.mix_confidence",
            "b.building_use",
            "b.building_use_id",
            "b.building_type",
            "b.type",
            "b.occupants",
            "b.households",
            "CAST(b.construction_year AS VARCHAR) AS construction_year",
            "b.postcode",
            "b.address_street_id",
            "b.street",
            "b.house_number",
            "b.gemeindeschluessel",
            "b.assigned_way_id",
            "b.residential_peak_load_in_kw",
            "b.nonresidential_peak_load_in_kw",
            "b.nonresidential_mv_direct",
            "b.peak_load_in_kw",
            "b.connection_point",
            "b.vertice_id AS consumer_vertex",
            "ST_AsText(b.centroid) AS centroid",
        ]

        query = text(
            """
            WITH selected_grid AS (
                SELECT grid_result_id, version_id
                FROM pylovo.grid_result
                WHERE plz = :plz AND kcid = :kcid AND bcid = :bcid
                  AND (:pylovo_version_id IS NULL OR version_id::text = :pylovo_version_id)
                ORDER BY version_id DESC
                LIMIT 1
            )
            SELECT
                """
            + ",\n                ".join(select_parts)
            + """
            FROM selected_grid sg
            JOIN pylovo.buildings_result b
              ON b.grid_result_id = sg.grid_result_id
             AND b.version_id = sg.version_id
            ;
            """
        )

        with self.engine.connect() as conn:
            df_buildings = pd.read_sql(
                query,
                conn,
                params={
                    "plz": int(grid_specs["plz"]),
                    "kcid": int(grid_specs["kcid"]),
                    "bcid": int(grid_specs["bcid"]),
                    "pylovo_version_id": config.PYLOVO_VERSION_ID,
                },
            )

        if df_buildings.empty:
            raise ValueError(
                f"No buildings found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, BCID={grid_specs['bcid']}."
            )

        ### Match bus to building
        df_id = pd.DataFrame()
        df_id["vertice_id"] = df_bus['name'].str.extract(r'^Consumer Nodebus (\d+)$')[0].dropna().astype(int)
        df_id = df_id.reset_index().rename(columns={"index":"bus"})
        df_buildings = df_buildings.merge(df_id, on='vertice_id', how="left")
        if "connection_point" in df_buildings.columns:
            df_buildings["bus"] = df_buildings["bus"].fillna(df_buildings["connection_point"])

        validate_physical_buildings(df_buildings)

        ### Take bus to front and order by it
        cols = df_buildings.columns.tolist()
        cols.insert(0, cols.pop(cols.index('bus')))
        df_buildings = df_buildings[cols]
        df_buildings = df_buildings.sort_values(by='bus').reset_index(drop=True)


        ### Read out location from string
        def _get_loc(loc_string):
            # Set dummy to Munich
            lat = 48.1351
            lon = 11.5820

            match = re.match(r"POINT\(([-+]?[0-9]*\.?[0-9]+)\s*([-+]?[0-9]*\.?[0-9]+)\)", str(loc_string))
            if match:
                x = float(match.group(1))
                y = float(match.group(2))

                # Define the projections
                transformer = Transformer.from_crs(config.PYLOVO_COORD_FORMAT, config.TARGET_COORD_FORMAT, always_xy=True)
                # Convert from EPSG:3857 to EPSG:4326
                lon, lat = transformer.transform(x, y)

            return lat, lon

        df_buildings[["lat", "lon"]] = df_buildings["centroid"].apply(_get_loc).apply(pd.Series)
        df_buildings.drop(columns=["centroid"], inplace=True)

        return df_buildings