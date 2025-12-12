"""Readout helpers for the pylovo grid database.

This module provides the `DataBase` class, a thin SQLAlchemy wrapper around a
PostgreSQL database that stores:

- pandapower grid JSON blobs
- transformer positions / grid identifiers (PLZ/KCID/BCID)
- building attributes used to map buildings to consumer buses

Connection credentials are taken from environment variables loaded in
`gridreadout/config.py`.
"""

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
        schema_name = "public"  # Change this if needed
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}';"))
            tables = [row[0] for row in result]
            print("Available tables:")
            print(tables)


    def read_all_grid_identifiers(self):
        """ Retrieve all grids' (PLZ, kcid, bcid, regiostar) that were generated and stored in the grids database """

        query = """
            SELECT plz, kcid, bcid
            FROM public.grids;
        """
        # Execute the query with Pandas. This will only read data.
        df_generated_grids = pd.read_sql_query(query, self.engine)
        print(f"Retrieved {len(df_generated_grids)} generated grids!")


        return df_generated_grids

    def read_nonfiller_grid_identifiers(self):
        """ Retrieve all grids' (PLZ, kcid, bcid, location) that were generated and stored in the grids database """

        query = """
            SELECT plz, kcid, bcid, ST_AsText(geom) as loc
            FROM public.transformer_classified
            WHERE kmeans_clusters!=0;
        """
        # Execute the query with Pandas. This will only read data.
        df_generated_grids = pd.read_sql_query(query, self.engine)
        print(f"Retrieved {len(df_generated_grids)} generated grids!")


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
            FROM public.grids
            WHERE (plz= :plz) AND (kcid= :kcid) AND (bcid= :bcid);
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_grid = pd.read_sql(query, conn, params={"plz":int(grid_specs["plz"]), "kcid":int(grid_specs["kcid"]), "bcid":int(grid_specs["bcid"])})

        # Transform to pandapower net
        net = pp.from_json_string(json.dumps(df_grid.loc[0, "grid"]))

        return net


    def read_trafo_pos(self, grid_specs):
        """ Read out position of transformer position for given grid from database """

        query = text("""
            SELECT ST_AsText(geom) as loc
            FROM public.transformer_positions
            WHERE (plz= :plz) AND (kcid= :kcid) AND (bcid= :bcid);
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_trafo = pd.read_sql(query, conn, params={"plz":int(grid_specs["plz"]), "kcid":int(grid_specs["kcid"]), "bcid":int(grid_specs["bcid"])})

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
            FROM public.municipal_register
            WHERE plz=:plz;
        """)

        # Execute the query with Pandas. This will only read data.
        with self.engine.connect() as conn:
            df_region_specs = pd.read_sql(query, conn, params={"plz":int(plz)})

        return df_region_specs


    def read_buildings(self, grid_specs, df_bus): #r.occupants, r.constructi, r.refurb_wal, r.refurb_roo, r.refurb_bas, r.refurb_win
        # query = text("""
        #     SELECT b.osm_id, b.vertice_id, b.type, r.use, b.houses_per_building, r.occupants, r.free_walls, b.floors, r.constructi, b.area, ST_AsText(b.center) as center, r.refurb_wal, r.refurb_roo, r.refurb_bas, r.refurb_win
        #     FROM public.buildings_result b
        #     RIGHT OUTER JOIN public.res r 
        #     ON b.osm_id = r.osm_id
        #     WHERE (b.plz= :plz) AND (b.kcid= :kcid) AND (b.bcid= :bcid);
        # """)

        query = text(''' 
            SELECT 
                b.osm_id, b.vertice_id, b.type,
                COALESCE(r.use, o.use) AS use,
                b.houses_per_building, r.occupants,
                COALESCE(r.free_walls, o.free_walls) AS free_walls,
                b.floors, r.constructi, b.area, ST_AsText(b.center) AS center
            FROM public.buildings_result b
            LEFT JOIN public.res r 
                ON b.osm_id = r.osm_id
            LEFT JOIN public.oth o 
                ON b.osm_id = o.osm_id
            WHERE 
                b.plz = :plz
                AND b.kcid = :kcid
                AND b.bcid = :bcid;
        ''')

        with self.engine.connect() as conn:
            df_buildings = pd.read_sql(query, conn, params={"plz":int(grid_specs["plz"]), "kcid":int(grid_specs["kcid"]), "bcid":int(grid_specs["bcid"])})

        ### Match bus to building
        df_id = pd.DataFrame()
        df_id["vertice_id"] = df_bus['name'].str.extract(r'^Consumer Nodebus (\d+)$')[0].dropna().astype(int)
        df_id = df_id.reset_index().rename(columns={"index":"bus"})
        df_buildings = df_buildings.merge(df_id, on='vertice_id', how="left")

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

            match = re.match(r"POINT\(([-+]?[0-9]*\.?[0-9]+)\s*([-+]?[0-9]*\.?[0-9]+)\)", loc_string)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))

                # Define the projections
                transformer = Transformer.from_crs(config.PYLOVO_COORD_FORMAT, config.TARGET_COORD_FORMAT, always_xy=True)
                # Convert from EPSG:3857 to EPSG:4326
                lon, lat = transformer.transform(x, y)

            return lat, lon

        df_buildings[["lat", "lon"]] = df_buildings["center"].apply(_get_loc).apply(pd.Series)
        df_buildings.drop(columns=["center"], inplace=True)

        return df_buildings