"""I/O helper for scenario HDF5 files used in the powerflow step."""

from config import config

import os
import pandas as pd
import shutil
import h5py
import pandapower as pp
import sys
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


class SaveFile:
    def __init__(self, filename, storage="h5", pre_only=False):
        # Copy input file to destination directory
        self.filename = filename
        self.storage = storage
        self.db = SurroGridDatabase() if self.storage == "db" else None
        self.grid_ref = None
        self.powerflow_run_id = None
        self.input_path = self._get_readpath()
        print(self.input_path)
        self.output_path = self._generate_savepath()
        if self.storage == "h5":
            shutil.copy2(self.input_path, self.output_path)
        else:
            self.grid_ref = self.db.resolve_grid_identifier(filename)
            self.powerflow_run_id = self.db.create_powerflow_run(
                self.grid_ref,
                urbs_input_file=filename,
                pre_only=pre_only,
            )

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
        if self.storage == "db":
            return self.db.read_pandapower_grid(self.grid_ref)
        with h5py.File(self.input_path, 'r') as f:
            json_data = f['raw_data/net'][()]
            grid = pp.from_json_string(json_data)
        return grid

    def get_input_demands(self):
        df_raw_demand = pd.read_hdf(self.input_path, key=self.raw_demand_dir)
        df_net_demand = pd.read_hdf(self.input_path, key=self.net_demand_dir)
        return df_raw_demand, df_net_demand

    def save_df(self, df, dir):
        if self.storage == "db":
            self._save_df_to_db(df, dir)
            return
        with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
            store.put(dir, df)

    def _save_df_to_db(self, df, dir):
        clean_dir = dir.strip("/")
        print(f"Saving {clean_dir} to DB for powerflow_run_id={self.powerflow_run_id} shape={df.shape}", flush=True)
        if clean_dir == "pwrflw/input/demand_pre":
            self.db.write_powerflow_demand(self.powerflow_run_id, "pre", df)
        elif clean_dir == "pwrflw/input/demand_post":
            self.db.write_powerflow_demand(self.powerflow_run_id, "post", df)
        elif clean_dir == "pwrflw/output/pre/demand_import":
            self.db.write_powerflow_import(self.powerflow_run_id, "pre", df)
        elif clean_dir == "pwrflw/output/post/demand_import":
            self.db.write_powerflow_import(self.powerflow_run_id, "post", df)
        elif clean_dir == "pwrflw/output/pre/vm":
            self.db.write_powerflow_bus_voltage(self.powerflow_run_id, "pre", df)
        elif clean_dir == "pwrflw/output/post/vm":
            self.db.write_powerflow_bus_voltage(self.powerflow_run_id, "post", df)
        elif clean_dir == "pwrflw/output/pre/line_loads":
            self.db.write_powerflow_line_result(self.powerflow_run_id, "pre", df)
        elif clean_dir == "pwrflw/output/post/line_loads":
            self.db.write_powerflow_line_result(self.powerflow_run_id, "post", df)
        elif clean_dir == "pwrflw/urbs_out/MILP/reactive":
            self.db.write_powerflow_reactive(self.powerflow_run_id, df)
        else:
            raise ValueError(f"No DB writer is defined for HDF5 key '{dir}'.")
        print(f"Finished saving {clean_dir} to DB for powerflow_run_id={self.powerflow_run_id}", flush=True)
