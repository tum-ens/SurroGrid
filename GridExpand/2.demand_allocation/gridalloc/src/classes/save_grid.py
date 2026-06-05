from config import config

import os
import pandas as pd
import shutil
import sys
from pathlib import Path

import warnings
from pandas.errors import PerformanceWarning

GRIDEXPAND_DIR = Path(__file__).resolve().parents[4]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


class SaveFile:
    def __init__(self, filename, storage="h5", grid_ref=None, allocation_settings=None):
        # Copy input file to destination directory
        self.storage = storage
        allocation_settings = allocation_settings or {}
        self.timeseries_storage = allocation_settings.get("timeseries_storage", "db")
        self.persist_allocated_timeseries = self.timeseries_storage in {"db", "both"}
        self.db = SurroGridDatabase() if self.storage == "db" else None
        self.grid_ref = grid_ref
        self.demand_allocation_run_id = None
        if self.storage == "db":
            if self.grid_ref is None:
                self.grid_ref = self.db.resolve_grid_identifier(filename)
            self.db.get_or_create_grid_case(self.grid_ref)
            filename = self.grid_ref["bridge_filename"]
            self.demand_allocation_run_id = self.db.create_demand_allocation_run(
                self.grid_ref,
                bridge_filename=filename,
                profiles=allocation_settings.get("profiles", "all"),
                mobility_source=allocation_settings.get("mobility_source", "emobpy"),
            )
        self.filename = filename
        self.input_path = self._get_readpath()
        self.output_path = self._generate_savepath()

        # Dirs from which to extract data within .h5 file
        self.building_dir = "raw_data/buildings"
        self.region_dir = "raw_data/region"
        self.weather_dir = "raw_data/weather"

    def _get_readpath(self):
        if self.storage == "db":
            return None
        directory = config.DATA_GRID_DIR
        return os.path.join(directory, self.filename)

    def _generate_savepath(self):
        directory = config.STORAGE_DIR
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self.filename)

    def get_input_data(self):
        if self.storage == "db":
            return self.db.read_step2_input_data(self.grid_ref)
        df_buildings = pd.read_hdf(self.input_path, key=self.building_dir)
        df_region = pd.read_hdf(self.input_path, key=self.region_dir)
        try: df_weather = pd.read_hdf(self.input_path, key=self.weather_dir)
        except: df_weather = None
        return df_buildings, df_region, df_weather

    def copy_save_file(self):
        if self.storage == "db":
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            return
        shutil.copy2(self.input_path, self.output_path)

    def save_df(self, df, dir):
        if self.storage == "db" and dir.startswith("raw_data/"):
            return
        if self.storage == "db" and self.persist_allocated_timeseries:
            clean_dir = dir.strip("/")
            if clean_dir == "urbs_in/demand":
                self.db.write_allocated_demand(self.demand_allocation_run_id, df)
            elif clean_dir == "urbs_in/eff_factor":
                self.db.write_allocated_eff_factor(self.demand_allocation_run_id, df)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
                store.put(dir, df)

    def save_allocated_vehicles(self, df_buildings, battery_dict):
        if self.storage != "db":
            return
        self.db.write_allocated_vehicles(self.demand_allocation_run_id, df_buildings, battery_dict)
