"""I/O helper for scenario HDF5 files used in the powerflow step."""

from config import config

import os
import pandas as pd
import shutil
import h5py
import pandapower as pp
import sys
from sqlalchemy import text
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase
from common.timeframe import read_hdf_metadata, scenario_key_for_timeframe


class SaveFile:
    def __init__(self, filename, storage="h5", pre_only=False, run_name=None, assumptions_extra=None, grid_case_id=None):
        # Copy input file to destination directory
        self.filename = filename
        self.storage = storage
        self.db = SurroGridDatabase() if self.storage == "db" else None
        self.grid_ref = None
        self.powerflow_run_id = None
        self.input_path = self._get_readpath()
        print(self.input_path)
        self.output_path = self._generate_savepath()
        self.timeframe_metadata = read_hdf_metadata(self.input_path)
        if self.storage == "h5":
            shutil.copy2(self.input_path, self.output_path)
        else:
            self.grid_ref = (
                self._grid_ref_from_case_id(grid_case_id)
                if grid_case_id is not None
                else self.db.resolve_grid_identifier(filename)
            )
            assumptions = dict(self.timeframe_metadata)
            if assumptions_extra:
                assumptions.update(assumptions_extra)
            scenario_key = self.timeframe_metadata.get("scenario_key") or scenario_key_for_timeframe(
                self.timeframe_metadata.get("timeframe_mode", "full_year")
            )
            self.powerflow_run_id = self.db.create_powerflow_run(
                self.grid_ref,
                urbs_input_file=filename,
                pre_only=pre_only,
                scenario_key=scenario_key,
                run_name=run_name,
                assumptions=assumptions,
            )

        # Dirs from which to extract data within .h5 file
        self.grid_dir = "raw_data/net"
        self.raw_demand_dir = "urbs_in/demand"
        self.reduced_demand_dir = "urbs_out/reduced_data/demand"
        self.net_demand_dir = "urbs_out/MILP/tau_pro"
        self.cap_pro_dir = "urbs_out/MILP/cap_pro"
        self.raw_eff_factor_dir = "urbs_in/eff_factor"
        self.reduced_eff_factor_dir = "urbs_out/reduced_data/eff_factor"
        self.raw_supim_dir = "urbs_in/supim"
        self.reduced_supim_dir = "urbs_out/reduced_data/supim"
        self.raw_process_dir = "urbs_in/process"
        self.reduced_process_dir = "urbs_out/reduced_data/process"
        self.raw_storage_dir = "urbs_in/storage"
        self.reduced_storage_dir = "urbs_out/reduced_data/storage"

    def _grid_ref_from_case_id(self, grid_case_id):
        query = text(
            """
            SELECT
                ags, plz, kcid, bcid,
                pylovo_grid_result_id AS grid_result_id,
                pylovo_version_id AS version_id,
                cell_id
            FROM surrogrid.grid_case
            WHERE grid_case_id = :grid_case_id
            """
        )
        with self.db.engine.connect() as conn:
            row = conn.execute(
                query, {"grid_case_id": int(grid_case_id)}
            ).mappings().one_or_none()
        if row is None:
            raise ValueError(f"Unknown synthetic grid_case_id={grid_case_id}.")
        return {**dict(row), "grid_case_id": int(grid_case_id)}

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
            if 'raw_data/net' in f:
                json_data = f['raw_data/net'][()]
                return pp.from_json_string(json_data)
        db = SurroGridDatabase()
        return db.read_pandapower_grid(db.resolve_grid_identifier(self.filename))

    def _hdf_key_exists(self, key):
        clean_key = key.strip("/")
        with h5py.File(self.input_path, "r") as hdf_file:
            return clean_key in hdf_file

    def uses_reduced_demand(self):
        return self._hdf_key_exists(self.reduced_demand_dir)

    def get_pre_demand(self):
        demand_key = self.reduced_demand_dir if self.uses_reduced_demand() else self.raw_demand_dir
        return pd.read_hdf(self.input_path, key=demand_key)

    def get_allocation_plan(self):
        return self._read_required_hdf("raw_data/allocation_plan")

    def _read_preferred_hdf(self, reduced_key, raw_key):
        key = reduced_key if self._hdf_key_exists(reduced_key) else raw_key
        return pd.read_hdf(self.input_path, key=key)

    def _read_required_hdf(self, key):
        if not self._hdf_key_exists(key):
            raise KeyError(f"Required HDF5 key '{key}' is missing in {self.filename}.")
        return pd.read_hdf(self.input_path, key=key)

    def has_urbs_results(self):
        return self._hdf_key_exists(self.net_demand_dir)

    def has_reduced_no_flex_inputs(self):
        return all(
            self._hdf_key_exists(key)
            for key in (
                self.reduced_demand_dir,
                self.reduced_eff_factor_dir,
                self.reduced_supim_dir,
                self.reduced_process_dir,
            )
        )

    def get_input_demands(self):
        df_raw_demand = self.get_pre_demand()
        df_net_demand = pd.read_hdf(self.input_path, key=self.net_demand_dir)
        return df_raw_demand, df_net_demand

    def get_no_flex_inputs(self):
        """Read no-flex inputs from a post-flex Step 3 result file.

        No-flex demand reconstruction intentionally depends on the optimized
        post-flex capacities. Heat demand is dispatched without URBS temporal
        flexibility, but uses ``cap_pro`` to split heat-pump and auxiliary
        electric heating in the same technology sizing context.
        """
        if not self.has_urbs_results():
            raise KeyError(
                "No-flex post demand requires post-flex URBS results in "
                f"'{self.net_demand_dir}' so timestep alignment and optimized capacities are available."
            )
        if not self._hdf_key_exists(self.cap_pro_dir):
            raise KeyError(
                "No-flex post demand requires optimized post-flex capacities in "
                f"'{self.cap_pro_dir}'. Run Step 3 optimization before Step 4 no-flex power flow."
            )

        return {
            "source": "post-flex",
            "demand": self.get_pre_demand(),
            "eff_factor": self._read_preferred_hdf(self.reduced_eff_factor_dir, self.raw_eff_factor_dir),
            "supim": self._read_preferred_hdf(self.reduced_supim_dir, self.raw_supim_dir),
            "process": self._read_preferred_hdf(self.reduced_process_dir, self.raw_process_dir),
            "storage": self._read_preferred_hdf(
                self.reduced_storage_dir, self.raw_storage_dir
            ),
            "tsam_hours_per_period": (
                int(
                    self._read_required_hdf("urbs_out/tsam/hoursPerPeriod")
                    .to_numpy()
                    .reshape(-1)[0]
                )
                if self._hdf_key_exists("urbs_out/tsam/hoursPerPeriod")
                else None
            ),
            "cap_pro": self._read_required_hdf(self.cap_pro_dir),
            "reference": pd.read_hdf(self.input_path, key=self.net_demand_dir),
            "drop_initial_timestep": False,
        }

    def save_df(self, df, dir):
        if self.storage == "db":
            self._save_df_to_db(df, dir)
            return
        with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
            store.put(dir, df)

    def save_summary(self, summary, stage):
        if self.storage != "db":
            raise ValueError("Summary-only powerflow currently supports --storage db only.")
        grid_summary = summary.get("grid_summary", summary)
        cable_rows = len(summary.get("cable_summary", [])) if isinstance(summary, dict) else 0
        bus_rows = len(summary.get("bus_voltage_summary", [])) if isinstance(summary, dict) else 0
        tail_rows = len(summary.get("tail_summary", [])) if isinstance(summary, dict) else 0
        print(
            f"Saving pwrflw/summary/{stage} to DB for powerflow_run_id={self.powerflow_run_id} ",
            f"metrics={grid_summary} cable_rows={cable_rows} bus_rows={bus_rows} tail_rows={tail_rows}",
            flush=True,
        )
        self.db.write_powerflow_summary(self.powerflow_run_id, stage, summary)
        print(f"Finished saving pwrflw/summary/{stage} to DB for powerflow_run_id={self.powerflow_run_id}", flush=True)

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
