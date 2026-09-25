"""HDF5 output writer for sampled grids.

This module writes one `.h5` file per grid. It stores:

- A pandapower network (`net`) serialized via `pandapower.to_json(net)`.
- Tabular data (region metadata, consumer bus mapping, buildings, weather, …)
    stored via `pandas.HDFStore`.

The notebooks in `gridreadout/` use `SaveFile` to create the output that is
consumed by downstream pipeline steps.
"""

from config import config
from src.grid_topol import get_consumers

import pandapower as pp

import os
import h5py
import pandas as pd
import warnings



class SaveFile:
    def __init__(self, grid_specs):
        self.path = self._generate_savepath(grid_specs)
        self._create_empty_savefile()
        

    def _generate_savepath(self, grid_specs):
        directory = config.STORAGE_DIR
        os.makedirs(directory, exist_ok=True)  
        filename = f"{grid_specs['cell_id']}_{grid_specs['plz']}_{grid_specs['kcid']}_{grid_specs['bcid']}.h5"
        return os.path.join(directory, filename)


    def _create_empty_savefile(self):
        if os.path.exists(self.path):
            print(f"Warning: The grid file already existed and will be overwritten.")
        with h5py.File(self.path, 'w') as f:
            print(f"File {self.path} created!")


    def save_topology(self, net, dir):
        consumer_buses = get_consumers(net)
        json_net = pp.to_json(net)

        with h5py.File(self.path, 'a') as f:
            # Save the JSON string as an attribute or dataset
            f.create_dataset(f'{dir}/net', data=json_net)
        
        with pd.HDFStore(self.path, mode="a", complib='blosc', complevel=9) as store:
            store.put(f"{dir}/consumers", consumer_buses)

    def save_df(self, df, dir):
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        with pd.HDFStore(self.path, mode="a", complib='blosc', complevel=9) as store:
            store.put(dir, df)