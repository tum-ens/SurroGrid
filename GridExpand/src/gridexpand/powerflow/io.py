"""Step 4 inputs (Step 2/3 HDF5 files, grids) and outputs (DB runs, HDF5, audits)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from sqlalchemy import text

from gridexpand.common.ev_sessions import read_sessions
from gridexpand.common.timeframe import read_hdf_metadata

# HDF keys shared by every Step-4 reader (synthetic and real grids).
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
    "temporal_method": "urbs_out/temporal_method",
    "solver_audit": "urbs_out/solver_audit",
    "allocation_plan": "raw_data/allocation_plan",
    "net": "raw_data/net",
}


def hdf_keys(path) -> set[str]:
    """All group and dataset paths of an HDF5 file (one open)."""
    names: list[str] = []
    with h5py.File(path, "r") as hdf_file:
        hdf_file.visit(names.append)
    return set(names)


def key_exists(path, key) -> bool:
    """True if ``key`` (with or without leading slash) exists in the HDF5 file."""
    with h5py.File(path, "r") as hdf_file:
        return str(key).strip("/") in hdf_file


def read_temporal_method(path):
    """Return the Step-3 temporal-method audit, or None for older results."""
    if not key_exists(path, HDF_KEYS["temporal_method"]):
        return None
    return pd.read_hdf(path, key=HDF_KEYS["temporal_method"]).to_dict()


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


def read_solver_audit(path):
    """The Step-3 ``urbs_out/solver_audit`` table, or None for older results."""
    if not key_exists(path, HDF_KEYS["solver_audit"]):
        return None
    return pd.read_hdf(path, key=HDF_KEYS["solver_audit"])


def temporal_assumptions(audit) -> dict[str, Any]:
    """Run-assumption fields taken from a temporal-method audit."""
    if not audit:
        return {}
    return {
        "temporal_method": str(audit.get("temporal_method")),
        "operating_hours": audit.get("operating_hours"),
        "storage_boundary_policy": audit.get("storage_boundary_policy"),
        "ev_boundary_policy": audit.get("ev_boundary_policy"),
    }


class ScenarioResultReader:
    """Read-only access to one Step 2 input or Step 3 result HDF5 file.

    Also usable as the ``SF`` argument of ``demands.obtain_demand`` (``save_df``
    does nothing; component audits go to a sidecar next to ``output_path``).
    """

    def __init__(self, path, *, output_path=None, run_name=None):
        self.path = Path(path)
        self.input_path = str(self.path)
        self.filename = self.path.name
        self.output_path = str(output_path or self.path)
        self.run_name = run_name
        self._keys = None
        self._metadata = None

    def has(self, key) -> bool:
        if self._keys is None:
            self._keys = hdf_keys(self.path)
        return str(key).strip("/") in self._keys

    def read(self, key):
        return pd.read_hdf(self.path, key=key)

    def _preferred(self, reduced_key, raw_key):
        return self.read(reduced_key if self.has(reduced_key) else raw_key)

    def _required(self, key):
        if not self.has(key):
            raise KeyError(f"Required HDF5 key {key!r} is missing in {self.filename}.")
        return self.read(key)

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = read_hdf_metadata(self.path)
        return self._metadata

    @property
    def timeframe_metadata(self) -> dict[str, Any]:
        return self.metadata

    def uses_reduced_demand(self) -> bool:
        return self.has(HDF_KEYS["reduced_demand"])

    def get_pre_demand(self):
        return self._preferred(HDF_KEYS["reduced_demand"], HDF_KEYS["raw_demand"])

    def get_input_demands(self):
        return self.get_pre_demand(), self.read(HDF_KEYS["net_demand"])

    def get_allocation_plan(self):
        return self._required(HDF_KEYS["allocation_plan"])

    def temporal_method(self):
        return read_temporal_method(self.path) if self.has(HDF_KEYS["temporal_method"]) else None

    def solver_audit(self):
        return self.read(HDF_KEYS["solver_audit"]) if self.has(HDF_KEYS["solver_audit"]) else None

    def get_inflex_inputs(self):
        """The single inflex input contract shared by every Step-4 front end.

        Heat demand is dispatched without temporal flexibility using the fixed
        heat-pump and auxiliary capacities in the scenario input process table.
        """
        if not self.has(HDF_KEYS["net_demand"]):
            raise KeyError(
                "INFLEX post demand requires post-flex URBS results in "
                f"{HDF_KEYS['net_demand']!r} in {self.filename} so timestep alignment and "
                "optimized capacities are available."
            )
        if not self.has(HDF_KEYS["cap_pro"]):
            raise KeyError(
                "INFLEX post demand requires optimized post-flex capacities in "
                f"{HDF_KEYS['cap_pro']!r}. Run Step 3 optimization before Step 4 "
                "inflex power flow."
            )
        sessions, session_hours = read_sessions(self.path)
        temporal = self.temporal_method()
        return {
            "source": "post-flex",
            "demand": self.get_pre_demand(),
            "ev_sessions": sessions,
            "ev_session_hours": session_hours,
            "eff_factor": self._preferred(HDF_KEYS["reduced_eff_factor"], HDF_KEYS["raw_eff_factor"]),
            "supim": self._preferred(HDF_KEYS["reduced_supim"], HDF_KEYS["raw_supim"]),
            "process": self._preferred(HDF_KEYS["reduced_process"], HDF_KEYS["raw_process"]),
            "storage": self._preferred(HDF_KEYS["reduced_storage"], HDF_KEYS["raw_storage"]),
            "tsam_hours_per_period": (
                int(self.read(HDF_KEYS["tsam_hours_per_period"]).to_numpy().reshape(-1)[0])
                if self.has(HDF_KEYS["tsam_hours_per_period"])
                else None
            ),
            "cap_pro": self._required(HDF_KEYS["cap_pro"]),
            "reference": self.read(HDF_KEYS["net_demand"]),
            "drop_initial_timestep": False,
            # Reconstruction is hourly end to end; the recorded duration is carried
            # so an incompatible one is rejected at entry rather than assumed.
            "delta_t_hours": (temporal or {}).get("delta_t_hours"),
        }

    def read_net(self):
        """The pandapower net stored in ``raw_data/net``, or None."""
        import pandapower as pp

        if not self.has(HDF_KEYS["net"]):
            return None
        with h5py.File(self.path, "r") as hdf_file:
            return pp.from_json_string(hdf_file[HDF_KEYS["net"]][()])

    # SF protocol of demands.obtain_demand ---------------------------------
    def save_df(self, df, key):
        return None

    def audit_path(self):
        return component_audit_path(self.output_path, self.run_name)

    def save_component_audit(self, df, name):
        return write_component_audit(self.audit_path(), df, name)


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


# Grids ----------------------------------------------------------------------

def grid_ref_from_case_id(db, grid_case_id):
    """Grid reference of a synthetic ``surrogrid.grid_case`` row."""
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
    with db.engine.connect() as conn:
        row = conn.execute(query, {"grid_case_id": int(grid_case_id)}).mappings().one_or_none()
    if row is None:
        raise ValueError(f"Unknown synthetic grid_case_id={grid_case_id}.")
    return {**dict(row), "grid_case_id": int(grid_case_id)}


def residential_buses(db, grid_ref) -> set[int]:
    """Pandapower buses of the included residential components of a grid case."""
    grid_case_id = db.get_or_create_grid_case(grid_ref)
    query = text(
        """
        SELECT DISTINCT gbc.bus
        FROM surrogrid.grid_building_component gbc
        WHERE gbc.grid_case_id = :grid_case_id
          AND gbc.bus IS NOT NULL
          AND gbc.included_in_lv
          AND gbc.component_category = 'Residential'
        ORDER BY gbc.bus
        """
    )
    with db.engine.connect() as conn:
        buses = [int(bus) for bus in conn.execute(query, {"grid_case_id": grid_case_id}).scalars()]
    if not buses:
        raise ValueError("--hh-only found no residential buses in surrogrid.grid_building_bus for this grid case.")
    return set(buses)


# Outputs --------------------------------------------------------------------

_DB_WRITERS = {
    "pwrflw/input/demand_pre": ("write_powerflow_demand", "pre"),
    "pwrflw/input/demand_post": ("write_powerflow_demand", "post"),
    "pwrflw/output/pre/demand_import": ("write_powerflow_import", "pre"),
    "pwrflw/output/post/demand_import": ("write_powerflow_import", "post"),
    "pwrflw/output/pre/vm": ("write_powerflow_bus_voltage", "pre"),
    "pwrflw/output/post/vm": ("write_powerflow_bus_voltage", "post"),
    "pwrflw/output/pre/line_loads": ("write_powerflow_line_result", "pre"),
    "pwrflw/output/post/line_loads": ("write_powerflow_line_result", "post"),
    "pwrflw/urbs_out/MILP/reactive": ("write_powerflow_reactive", None),
}


class DbRunSink:
    """One ``surrogrid.powerflow_run`` row and its tables (registered on creation)."""

    def __init__(self, db, grid_ref, *, urbs_input_file, pre_only, scenario_key, run_name, assumptions):
        self.db = db
        self.run_name = run_name
        self.powerflow_run_id = db.create_powerflow_run(
            grid_ref,
            urbs_input_file=urbs_input_file,
            pre_only=pre_only,
            scenario_key=scenario_key,
            run_name=run_name,
            assumptions=assumptions,
        )

    def save_df(self, df, key):
        clean = str(key).strip("/")
        if clean not in _DB_WRITERS:
            raise ValueError(f"No DB writer is defined for HDF5 key '{key}'.")
        method, stage = _DB_WRITERS[clean]
        print(f"Saving {clean} to DB for powerflow_run_id={self.powerflow_run_id} shape={df.shape}", flush=True)
        writer = getattr(self.db, method)
        if stage is None:
            writer(self.powerflow_run_id, df)
        else:
            writer(self.powerflow_run_id, stage, df)
        print(f"Finished saving {clean} to DB for powerflow_run_id={self.powerflow_run_id}", flush=True)

    def save_summary(self, summary, stage):
        grid_summary = summary.get("grid_summary", summary)
        print(
            f"Saving pwrflw/summary/{stage} to DB for powerflow_run_id={self.powerflow_run_id} ",
            f"metrics={grid_summary} cable_rows={len(summary.get('cable_summary', []))} "
            f"bus_rows={len(summary.get('bus_voltage_summary', []))} "
            f"tail_rows={len(summary.get('tail_summary', []))}",
            flush=True,
        )
        self.db.write_powerflow_summary(self.powerflow_run_id, stage, summary)
        print(f"Finished saving pwrflw/summary/{stage} to DB for powerflow_run_id={self.powerflow_run_id}", flush=True)


class HdfSink:
    """Copy of the input file with the ``pwrflw/*`` tables appended."""

    def __init__(self, input_path, output_path):
        self.output_path = str(output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, self.output_path)

    def save_df(self, df, key):
        with pd.HDFStore(self.output_path, mode="a", complib="blosc", complevel=9) as store:
            store.put(key, df)

    def save_summary(self, summary, stage):
        raise ValueError("Power-flow summaries are stored in the database only (--storage db).")
