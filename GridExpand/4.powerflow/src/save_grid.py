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
from common.ev_sessions import (
    SESSION_HOURS_HDF_KEY,
    SESSIONS_HDF_KEY,
    SESSION_HOUR_COLUMNS,
    SESSION_COLUMNS,
    SessionError,
)
from common.timeframe import read_hdf_metadata, scenario_key_for_timeframe


def hdf_key_exists(path, key):
    clean_key = str(key).strip("/")
    with h5py.File(path, "r") as hdf_file:
        return clean_key in hdf_file


def read_temporal_method(path):
    """Return the Step-3 temporal-method audit, or None for older results."""
    if not hdf_key_exists(path, "urbs_out/temporal_method"):
        return None
    return pd.read_hdf(path, key="urbs_out/temporal_method").to_dict()


def require_temporal_method(path, expected):
    """Reject a result whose temporal method is not the requested one.

    A full-year request must never silently consume a stale representative-period
    result. The file name and the presence of the 'reduced_data' group prove
    nothing: Step 3 writes that group in both modes.
    """
    audit = read_temporal_method(path)
    name = Path(path).name
    if audit is None:
        raise KeyError(
            f"{name} has no 'urbs_out/temporal_method' record. It predates "
            f"temporal provenance and cannot be accepted for a {expected!r} run."
        )
    found = str(audit.get("temporal_method"))
    if found != str(expected):
        raise ValueError(
            f"{name} was produced with temporal_method={found!r}, but "
            f"{expected!r} was requested."
        )
    return audit


# HDF keys shared by every Step-4 adapter. Both the synthetic SaveFile and the
# real-grid adapter must consume exactly one inflex input contract; a second,
# independently maintained interpretation is how the real adapter silently lost
# the EV session tables.
HDF_KEYS = {
    "raw_demand": "urbs_in/demand",
    "reduced_demand": "urbs_out/reduced_data/demand",
    "net_demand": "urbs_out/MILP/tau_pro",
    "cap_pro": "urbs_out/MILP/cap_pro",
    "raw_eff_factor": "urbs_in/eff_factor",
    "reduced_eff_factor": "urbs_out/reduced_data/eff_factor",
    "raw_supim": "urbs_in/supim",
    "reduced_supim": "urbs_out/reduced_data/supim",
    "raw_process": "urbs_in/process",
    "reduced_process": "urbs_out/reduced_data/process",
    "raw_storage": "urbs_in/storage",
    "reduced_storage": "urbs_out/reduced_data/storage",
    "tsam_hours_per_period": "urbs_out/tsam/hoursPerPeriod",
}


def _preferred(path, reduced_key, raw_key):
    key = reduced_key if hdf_key_exists(path, reduced_key) else raw_key
    return pd.read_hdf(path, key=key)


def _required(path, key):
    if not hdf_key_exists(path, key):
        raise KeyError(
            f"Required HDF5 key {key!r} is missing in {Path(path).name}."
        )
    return pd.read_hdf(path, key=key)


def read_pre_demand(path):
    return _preferred(path, HDF_KEYS["reduced_demand"], HDF_KEYS["raw_demand"])


def read_ev_sessions(path):
    """Read the dedicated EV charging-session contract from a result file.

    An empty contract is valid: a building population can legitimately contain no
    electric vehicles. It is distinguished from a legacy input by the *presence*
    of the key, never by its row count.
    """
    name = Path(path).name
    if not hdf_key_exists(path, SESSIONS_HDF_KEY):
        raise SessionError(
            f"{name} does not contain '{SESSIONS_HDF_KEY}'. It predates the "
            "dedicated EV session contract, so its EV service cannot be "
            "reconstructed without falling back to the old energy-clipping "
            "heuristic."
        )
    sessions = pd.read_hdf(path, key=SESSIONS_HDF_KEY)
    if hdf_key_exists(path, SESSION_HOURS_HDF_KEY):
        hours = pd.read_hdf(path, key=SESSION_HOURS_HDF_KEY)
    else:
        hours = pd.DataFrame(columns=SESSION_HOUR_COLUMNS)
    if sessions.empty:
        sessions = pd.DataFrame(columns=SESSION_COLUMNS)
    if hours.empty:
        hours = pd.DataFrame(columns=SESSION_HOUR_COLUMNS)
    return sessions.reset_index(drop=True), hours.reset_index(drop=True)


def read_inflex_inputs(path):
    """The single inflex input contract shared by every Step-4 adapter.

    Heat demand is dispatched without temporal flexibility using the fixed
    heat-pump and auxiliary capacities in the scenario input process table.
    """
    name = Path(path).name
    if not hdf_key_exists(path, HDF_KEYS["net_demand"]):
        raise KeyError(
            "INFLEX post demand requires post-flex URBS results in "
            f"{HDF_KEYS['net_demand']!r} in {name} so timestep alignment and "
            "optimized capacities are available."
        )
    if not hdf_key_exists(path, HDF_KEYS["cap_pro"]):
        raise KeyError(
            "INFLEX post demand requires optimized post-flex capacities in "
            f"{HDF_KEYS['cap_pro']!r}. Run Step 3 optimization before Step 4 "
            "inflex power flow."
        )
    sessions, session_hours = read_ev_sessions(path)
    temporal = read_temporal_method(path)
    return {
        "source": "post-flex",
        "demand": read_pre_demand(path),
        "ev_sessions": sessions,
        "ev_session_hours": session_hours,
        "eff_factor": _preferred(
            path, HDF_KEYS["reduced_eff_factor"], HDF_KEYS["raw_eff_factor"]
        ),
        "supim": _preferred(path, HDF_KEYS["reduced_supim"], HDF_KEYS["raw_supim"]),
        "process": _preferred(
            path, HDF_KEYS["reduced_process"], HDF_KEYS["raw_process"]
        ),
        "storage": _preferred(
            path, HDF_KEYS["reduced_storage"], HDF_KEYS["raw_storage"]
        ),
        "tsam_hours_per_period": (
            int(
                _required(path, HDF_KEYS["tsam_hours_per_period"])
                .to_numpy()
                .reshape(-1)[0]
            )
            if hdf_key_exists(path, HDF_KEYS["tsam_hours_per_period"])
            else None
        ),
        "cap_pro": _required(path, HDF_KEYS["cap_pro"]),
        "reference": pd.read_hdf(path, key=HDF_KEYS["net_demand"]),
        "drop_initial_timestep": False,
        # Reconstruction is hourly end to end; the recorded duration is carried
        # so an incompatible one is rejected at entry rather than assumed.
        "delta_t_hours": (temporal or {}).get("delta_t_hours"),
    }


def component_audit_path(output_path, run_name=None):
    """Location of the compact component-audit sidecar for a run."""
    base = Path(output_path)
    suffix = f".{run_name}" if run_name else ""
    return base.with_name(f"{base.stem}{suffix}.component_audit.h5")


def write_component_audit(path, df, name):
    """Append one compact audit table to the sidecar, creating it if needed.

    Returns the path so the caller can report where the audit landed, or None
    when there is nothing to record.
    """
    if df is None or getattr(df, "empty", True):
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(path, mode="a", complib="blosc", complevel=9) as store:
        store.put(f"component_audit/{name}", df.reset_index(drop=True), format="table")
    return str(path)


class SaveFile:
    def __init__(self, filename, storage="h5", pre_only=False, run_name=None, assumptions_extra=None, grid_case_id=None, pylovo_version_id=None):
        # Copy input file to destination directory
        self.filename = filename
        self.storage = storage
        self.run_name = run_name
        self.db = SurroGridDatabase() if self.storage == "db" else None
        if self.db is not None and pylovo_version_id is not None:
            self.db.pylovo_version_id = str(pylovo_version_id)
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

    def has_reduced_inflex_inputs(self):
        return all(
            self._hdf_key_exists(key)
            for key in (
                self.reduced_demand_dir,
                self.reduced_eff_factor_dir,
                self.reduced_supim_dir,
                self.reduced_process_dir,
            )
        )

    def get_temporal_method(self):
        """Return the Step-3 temporal-method audit, or None for older results."""
        return read_temporal_method(self.input_path)

    def require_temporal_method(self, expected):
        """Reject a result whose temporal method is not the requested one."""
        return require_temporal_method(self.input_path, expected)

    def get_ev_sessions(self):
        """Delegate to the single shared session reader."""
        return read_ev_sessions(self.input_path)

    def get_input_demands(self):
        df_raw_demand = self.get_pre_demand()
        df_net_demand = pd.read_hdf(self.input_path, key=self.net_demand_dir)
        return df_raw_demand, df_net_demand

    def get_inflex_inputs(self):
        """Delegate to the single shared inflex input contract."""
        return read_inflex_inputs(self.input_path)

    def save_df(self, df, dir):
        if self.storage == "db":
            self._save_df_to_db(df, dir)
            return
        with pd.HDFStore(self.output_path, mode="a", complib='blosc', complevel=9) as store:
            store.put(dir, df)

    def audit_path(self):
        """Sidecar file holding compact component audits for this run."""
        return component_audit_path(self.output_path, self.run_name)

    def save_component_audit(self, df, name):
        """Persist a compact component audit, whatever the storage mode.

        Component audits must survive summary-only and database runs, where the
        reactive time-series tables are deliberately not written and the database
        has no matching table. They therefore go to a small sidecar HDF that is
        identified from the run rather than through save_df.
        """
        return write_component_audit(self.audit_path(), df, name)

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
