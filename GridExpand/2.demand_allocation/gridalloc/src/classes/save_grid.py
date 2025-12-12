from config import config

import os
import pandas as pd
import shutil

import warnings
from pandas.errors import PerformanceWarning

class SaveFile:
    def __init__(self, filename):
        # Copy input file to destination directory
        self.filename = filename
        self.input_path = self._get_readpath()
        self.output_path = self._generate_savepath()

        # Dirs from which to extract data within .h5 file
        self.building_dir = "raw_data/buildings"
        self.region_dir = "raw_data/region"
        self.weather_dir = "raw_data/weather"
        
    def _get_readpath(self):
        directory = config.DATA_GRID_DIR
        return os.path.join(directory, self.filename)

    def _generate_savepath(self):
        directory = config.STORAGE_DIR
        os.makedirs(directory, exist_ok=True)  
        return os.path.join(directory, self.filename)

    def get_input_data(self):
        df_buildings = pd.read_hdf(self.input_path, key=self.building_dir)
        df_region = pd.read_hdf(self.input_path, key=self.region_dir)
        try: df_weather = pd.read_hdf(self.input_path, key=self.weather_dir)
        except: df_weather = None
        return df_buildings, df_region, df_weather

    def copy_save_file(self):
        shutil.copy2(self.input_path, self.output_path)
        
    def save_df(self, df, dir):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
                store.put(dir, df)