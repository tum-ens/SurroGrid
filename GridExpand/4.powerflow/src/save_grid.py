"""I/O helper for scenario HDF5 files used in the powerflow step.

`SaveFile` is responsible for:

- locating the input `.h5` file in `Input/` (see `config.DATA_DIR`)
- copying it to `Output/` (see `config.STORAGE_DIR`) so results never modify
    the original input
- reading required inputs:
    - pandapower network from `raw_data/net` (JSON string)
    - demand tables from `urbs_in/demand` and `urbs_out/MILP/tau_pro`
- writing pandas DataFrames back into the output file via `pandas.HDFStore`

All writes append to the copied output file.
"""

from config import config

import os
import pandas as pd
import shutil
import h5py
import pandapower as pp


class SaveFile:
    def __init__(self, filename):
        # Copy input file to destination directory
        self.filename = filename
        self.input_path = self._get_readpath()
        print(self.input_path)
        self.output_path = self._generate_savepath()
        shutil.copy2(self.input_path, self.output_path)

        # Dirs from which to extract data within .h5 file
        self.grid_dir = "raw_data/net"
        self.raw_demand_dir = "urbs_in/demand"
        self.net_demand_dir = "urbs_out/MILP/tau_pro"
        
    def _get_readpath(self):
        directory = config.DATA_DIR
        return os.path.join(directory, self.filename)

    def _generate_savepath(self):
        directory = config.STORAGE_DIR
        os.makedirs(directory, exist_ok=True)  
        return os.path.join(directory, self.filename)

    def get_input_grid(self):
        with h5py.File(self.input_path, 'r') as f:
            json_data = f['raw_data/net'][()]
            grid = pp.from_json_string(json_data)
        return grid

    def get_input_demands(self):
        df_raw_demand = pd.read_hdf(self.input_path, key=self.raw_demand_dir)
        df_net_demand = pd.read_hdf(self.input_path, key=self.net_demand_dir)
        return df_raw_demand, df_net_demand

    def save_df(self, df, dir):
        with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
            store.put(dir, df)
