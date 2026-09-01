"""Run compact power-flow summaries for real SWF scenario allocation plans.

This runner consumes the bus-level scenario calibration output from Step 2,
currently `swf_2045_full_local_demand_bus_allocation_plan.csv`, and maps the
calibrated annual demand back to the real SWF pandapower buses.

Implemented scope in this first runner:

- residential-equivalent HH electricity from calibrated annual HH rows
- calibrated GHD electricity from SWF annual GHD demand
- compact p99/p01-style power-flow summaries in the existing real summary DB tables

Sector-coupling assets are kept in the allocation plan and stored in the run
assumptions, but heat, EV, PV and battery time-series are intentionally not yet
translated here. That keeps this first executable step auditable before the post
flex/no-flex sector-coupling layer is added.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pandapower as pp
from dotenv import load_dotenv

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
STEP4_DIR = Path(__file__).resolve().parent
STEP2_DIR = GRIDEXPAND_DIR / "2.demand_allocation"
DEMAND_DIR = STEP2_DIR / "gridalloc"
ENV_PATH = GRIDEXPAND_DIR / ".env"
DEFAULT_ALLOCATION_PLAN = (
    GRIDEXPAND_DIR
    / "2.demand_allocation"
    / "gridalloc"
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_building_match_91301"
    / "swf_2045_full_local_demand_bus_allocation_plan.csv"
)

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))
from common.database import SurroGridDatabase  # noqa: E402
from common.timeframe import (  # noqa: E402
    build_full_year_metadata,
    read_hdf_metadata as read_timeframe_metadata,
)
import src.powerflow as pwrflw  # noqa: E402
import src.demands as dmnds  # noqa: E402
from config import config as pf_config  # noqa: E402


from gridalloc.src.scenario_calibration.profiles import (  # noqa: E402
    real_swf_electricity_profiles as _electricity_profiles,
)

DEFAULT_MEASURED_PROFILE_BAND_PCT = (
    _electricity_profiles.DEFAULT_MEASURED_PROFILE_BAND_PCT
)
DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES = (
    _electricity_profiles.DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES
)
MEASURED_PROFILE_SELECTION_CHOICES = (
    _electricity_profiles.MEASURED_PROFILE_SELECTION_CHOICES
)
MEASURED_PROFILE_SELECTION_RANDOM_BAND = (
    _electricity_profiles.MEASURED_PROFILE_SELECTION_RANDOM_BAND
)
build_scenario_base_electric_demand = (
    _electricity_profiles.build_scenario_base_electric_demand
)
profile_selection_summary = _electricity_profiles.profile_selection_summary
read_allocation_plan = _electricity_profiles.read_allocation_plan

from run_real_swf_powerflow import (  # noqa: E402
    _grid_ref,
    _prepare_real_grid,
    _select_manifest_rows,
)

DEFAULT_RUN_NAME = "real_swf_2045_full_local_base_electricity"
DEFAULT_SCENARIO_KEY = "real_swf_2045_full_local_base_electricity"
DEFAULT_SCENARIO_LABEL = "Real SWF 2045 full-local calibrated base electricity"
ASSUMPTION_TEXT = (
    "Real SWF scenario power flow based on the Step-2 SWF 2045 bus allocation plan. "
    "Residential-equivalent HH rows use measured annual kWh from the calibration plan and residential "
    "profile shapes from the synthetic demand-allocation library. Calibrated GHD rows use SWF annual "
    "GHD kWh and normalized Step-2 GHD profile shapes. Heat, EV, PV and batteries are audited in the "
    "plan but not yet translated into this first base-electricity runner."
)

URBS_ASSUMPTION_TEXT = (
    "Real SWF sector-coupling power flow based on a Step-3 URBS result HDF generated from "
    "the calibrated real SWF scenario allocation plan. Demand reconstruction uses the shared "
    "GridExpand Step-4 demand logic also used for synthetic grids; only the pandapower network "
    "and bus allocation differ."
)


class RealUrbsResultAdapter:
    """Small SaveFile-compatible adapter for real-grid URBS result HDFs."""

    def __init__(self, hdf_path: Path):
        self.input_path = str(hdf_path)
        self.output_path = str(hdf_path)
        self.filename = hdf_path.name
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

    def _hdf_key_exists(self, key: str) -> bool:
        import h5py

        with h5py.File(self.input_path, "r") as hdf_file:
            return key.strip("/") in hdf_file

    def _read_preferred_hdf(self, reduced_key: str, raw_key: str) -> pd.DataFrame:
        key = reduced_key if self._hdf_key_exists(reduced_key) else raw_key
        return pd.read_hdf(self.input_path, key=key)

    def _read_required_hdf(self, key: str) -> pd.DataFrame:
        if not self._hdf_key_exists(key):
            raise KeyError(f"Required HDF5 key {key!r} is missing in {self.filename}.")
        return pd.read_hdf(self.input_path, key=key)

    def uses_reduced_demand(self) -> bool:
        return self._hdf_key_exists(self.reduced_demand_dir)

    def get_pre_demand(self) -> pd.DataFrame:
        return self._read_preferred_hdf(self.reduced_demand_dir, self.raw_demand_dir)

    def get_input_demands(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.get_pre_demand(), pd.read_hdf(
            self.input_path, key=self.net_demand_dir
        )

    def has_urbs_results(self) -> bool:
        return self._hdf_key_exists(self.net_demand_dir)

    def get_no_flex_inputs(self) -> dict[str, Any]:
        if not self.has_urbs_results():
            raise KeyError(
                f"No-flex post demand requires {self.net_demand_dir!r} in {self.filename}."
            )
        return {
            "source": "post-flex",
            "demand": self.get_pre_demand(),
            "eff_factor": self._read_preferred_hdf(
                self.reduced_eff_factor_dir, self.raw_eff_factor_dir
            ),
            "supim": self._read_preferred_hdf(
                self.reduced_supim_dir, self.raw_supim_dir
            ),
            "process": self._read_preferred_hdf(
                self.reduced_process_dir, self.raw_process_dir
            ),
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

    def save_df(self, df: pd.DataFrame, dir: str) -> None:
        # Real-grid compact summary runs do not persist intermediate reactive tables.
        return None


def _read_hdf_metadata(hdf_path: Path) -> dict[str, Any]:
    return read_timeframe_metadata(hdf_path)


def _allocation_from_hdf(hdf_path: Path) -> pd.DataFrame:
    try:
        allocation = pd.read_hdf(hdf_path, key="raw_data/allocation_plan")
    except (KeyError, FileNotFoundError) as exc:
        raise ValueError(
            f"URBS result HDF {hdf_path} has no raw_data/allocation_plan table."
        ) from exc
    if "allocation_bus" not in allocation.columns:
        raise ValueError(
            f"URBS result HDF {hdf_path} allocation plan has no allocation_bus column."
        )
    return allocation


def _lv_id_from_hdf(hdf_path: Path) -> int | None:
    allocation = _allocation_from_hdf(hdf_path)
    if "lv_id" not in allocation.columns:
        return None
    values = (
        pd.to_numeric(allocation["lv_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(values) == 1:
        return int(values[0])
    return None


def _prepare_real_grid_for_allocation(
    net: pp.pandapowerNet,
    allocation_buses: list[int],
    summary_grid_scope: str = "full",
):
    grid, transformer_s_rated_mva, cable_max_i_ka, _, _, _, load_scope = (
        _prepare_real_grid(net)
    )
    allocation_buses = sorted({int(bus) for bus in allocation_buses})
    missing_buses = sorted(
        set(allocation_buses).difference(set(map(int, grid.bus.index)))
    )
    if missing_buses:
        raise ValueError(
            f"Allocation plan references buses missing from the real grid: {missing_buses[:10]}"
        )

    load_buses = pd.Index(allocation_buses, dtype=int).drop_duplicates().tolist()
    existing_load = grid.load.copy() if hasattr(grid, "load") else pd.DataFrame()
    template_columns = (
        list(existing_load.columns)
        if not existing_load.empty
        else ["bus", "p_mw", "q_mvar", "name"]
    )
    rows = []
    for bus in load_buses:
        rows.append(
            {
                "bus": int(bus),
                "p_mw": 0.0,
                "q_mvar": 0.0,
                "name": f"Scenario_Profile_{bus}",
            }
        )
    grid.load = (
        pd.DataFrame(rows).reindex(columns=template_columns).reset_index(drop=True)
    )
    grid.load["bus"] = grid.load["bus"].astype(int)
    grid.load["p_mw"] = 0.0
    grid.load["q_mvar"] = 0.0
    grid.load["max_p_mw"] = 1000.0
    for column, value in {
        "const_z_percent": 0.0,
        "const_i_percent": 0.0,
        "const_z_p_percent": 0.0,
        "const_z_q_percent": 0.0,
        "const_i_p_percent": 0.0,
        "const_i_q_percent": 0.0,
        "scaling": 1.0,
        "in_service": True,
    }.items():
        grid.load[column] = value

    summary_cable_ids, voltage_buses = pwrflw.comparison_evaluation_scope(
        grid, load_buses, scope=summary_grid_scope
    )
    if not voltage_buses:
        voltage_buses = load_buses
    load_scope.update(
        {
            "scenario_allocation_buses": int(len(load_buses)),
            "summary_grid_scope": summary_grid_scope,
            "scenario_summary_voltage_buses": int(len(voltage_buses)),
            "scenario_summary_cables": int(len(summary_cable_ids)),
        }
    )
    return (
        grid,
        transformer_s_rated_mva,
        cable_max_i_ka,
        voltage_buses,
        summary_cable_ids,
        load_scope,
    )


def _allocation_totals(
    allocation: pd.DataFrame, demand_audit: pd.DataFrame
) -> dict[str, Any]:
    selection_summary = profile_selection_summary(demand_audit)
    return {
        "allocation_plan_rows": int(len(allocation)),
        "allocation_plan_buses": int(allocation["allocation_bus"].nunique()),
        "allocation_plan_buildings": int(allocation["building_match_id"].nunique()),
        "allocation_hh_rows": int(
            pd.to_numeric(allocation["residential_equivalent_hh_rows"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "allocation_hh_annual_kwh": float(
            pd.to_numeric(
                allocation["residential_equivalent_hh_annual_kwh"], errors="coerce"
            )
            .fillna(0.0)
            .sum()
        ),
        "allocation_ghd_annual_kwh": float(
            pd.to_numeric(allocation["calibrated_annual_ghd_kwh"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "generated_profile_energy_kwh": float(
            pd.to_numeric(demand_audit["annual_demand_kwh"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "residential_ev_charger_kw_audited_not_simulated": float(
            pd.to_numeric(allocation.get("residential_ev_charger_kw"), errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "ghd_ev_charger_kw_audited_not_simulated": float(
            pd.to_numeric(allocation.get("ghd_ev_charger_kw"), errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "residential_pv_kw_audited_not_simulated": float(
            pd.to_numeric(allocation.get("residential_pv_kw"), errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "ghd_pv_kw_audited_not_simulated": float(
            pd.to_numeric(allocation.get("ghd_pv_kw"), errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        **selection_summary,
    }


def run_one(
    row: dict[str, Any],
    allocation_plan_path: str,
    run_name: str,
    scenario_key: str,
    scenario_label: str,
    seed: int,
    measured_profile_selection: str = MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    measured_profile_band_pct: float = DEFAULT_MEASURED_PROFILE_BAND_PCT,
    measured_profile_min_candidates: int = DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    max_timesteps: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    allocation_all = read_allocation_plan(Path(allocation_plan_path), scope=None)
    lv_id = int(str(row["lv_id"]).removeprefix("LV_"))
    allocation = allocation_all[allocation_all["lv_id"].astype(int).eq(lv_id)].copy()
    if allocation.empty:
        raise ValueError(f"Allocation plan contains no rows for LV {lv_id}.")

    source_file = Path(row["source_file"])
    net = pp.from_excel(source_file)
    (
        grid,
        transformer_s_rated_mva,
        cable_max_i_ka,
        voltage_buses,
        summary_cable_ids,
        load_scope,
    ) = _prepare_real_grid_for_allocation(
        net,
        allocation["allocation_bus"].astype(int).tolist(),
    )
    df_demand, demand_audit = build_scenario_base_electric_demand(
        allocation,
        seed=seed,
        measured_profile_selection=measured_profile_selection,
        measured_profile_band_pct=measured_profile_band_pct,
        measured_profile_min_candidates=measured_profile_min_candidates,
    )
    if max_timesteps is not None:
        df_demand = df_demand.iloc[: int(max_timesteps)].copy()
    summary = pwrflw.pf_summary(
        grid,
        df_demand,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_max_i_ka=cable_max_i_ka,
        voltage_buses=voltage_buses,
        algorithm=["nr", "iwamoto_nr"],
        cable_ids=summary_cable_ids,
        on_nonconvergence="nan",
        protect_grid_state=True,
    )

    db = SurroGridDatabase()
    db.ensure_schema()
    assumptions = {
        **build_full_year_metadata(),
        "demand_allocation": ASSUMPTION_TEXT,
        "allocation_plan_path": str(allocation_plan_path),
        "profile_seed": int(seed),
        "sector_assets_simulated": False,
        "stage_label": "base_electricity",
        "max_timesteps": None if max_timesteps is None else int(max_timesteps),
        **_allocation_totals(allocation, demand_audit),
        **load_scope,
        "nonconverged_timesteps": int(
            summary["grid_summary"].get("n_failed_timesteps", 0)
        ),
    }
    run_id = db.create_real_powerflow_run(
        _grid_ref(row),
        run_name=run_name,
        scenario_key=scenario_key,
        scenario_label=scenario_label,
        assumptions=assumptions,
    )
    db.write_real_powerflow_summary(run_id, "base_electricity", summary)
    elapsed = time.perf_counter() - start
    grid_summary = summary["grid_summary"]
    return {
        "lv_id": lv_id,
        "run_id": run_id,
        "source_file": str(source_file),
        "elapsed_s": elapsed,
        "n_timesteps": grid_summary["n_timesteps"],
        "n_converged_timesteps": grid_summary.get("n_converged_timesteps"),
        "n_failed_timesteps": grid_summary.get("n_failed_timesteps"),
        "n_voltage_buses": grid_summary["n_voltage_buses"],
        "n_cables": grid_summary["n_cables"],
        "tail_rows": len(summary.get("tail_summary", [])),
        **_allocation_totals(allocation, demand_audit),
        **load_scope,
    }


def _selected_rows_for_plan(
    root: Path, plz: int, limit: int | None, lv_id: str | None, plan: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = _select_manifest_rows(root, plz, limit=None, lv_id=lv_id)
    plan_lv_ids = set(plan["lv_id"].astype(int))
    rows = [
        row for row in rows if int(str(row["lv_id"]).removeprefix("LV_")) in plan_lv_ids
    ]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def run_one_urbs_result(
    row: dict[str, Any],
    urbs_result_hdf: str,
    run_name: str,
    scenario_key: str,
    scenario_label: str,
    post_demand_mode: str,
    max_timesteps: int | None = None,
    no_flex_ev_charger_kw: float | None = None,
    summary_grid_scope: str = "full",
) -> dict[str, Any]:
    start = time.perf_counter()
    hdf_path = Path(urbs_result_hdf).resolve()
    allocation = _allocation_from_hdf(hdf_path)
    lv_id = int(str(row["lv_id"]).removeprefix("LV_"))
    if "lv_id" in allocation.columns:
        allocation = allocation[
            pd.to_numeric(allocation["lv_id"], errors="coerce")
            .astype("Int64")
            .eq(lv_id)
        ].copy()
    if allocation.empty:
        raise ValueError(
            f"URBS result allocation plan contains no rows for LV {lv_id}."
        )

    source_file = Path(row["source_file"])
    net = pp.from_excel(source_file)
    (
        grid,
        transformer_s_rated_mva,
        cable_max_i_ka,
        voltage_buses,
        summary_cable_ids,
        load_scope,
    ) = _prepare_real_grid_for_allocation(
        net,
        allocation["allocation_bus"].astype(int).tolist(),
        summary_grid_scope,
    )

    adapter = RealUrbsResultAdapter(hdf_path)
    metadata = _read_hdf_metadata(hdf_path)
    if post_demand_mode == "pre-only":
        df_pre_demand = dmnds.obtain_pre_demand(adapter)
        df_post_demand = None
    else:
        charger_kw = (
            pf_config.EV_HOME_CHARGER_KW
            if no_flex_ev_charger_kw is None
            else float(no_flex_ev_charger_kw)
        )
        df_pre_demand, df_post_demand = dmnds.obtain_demand(
            adapter,
            save_reactive=False,
            post_demand_mode=post_demand_mode,
            ev_charger_kw=charger_kw,
        )

    if metadata.get("optimization_space") == "scenario_unit":
        df_pre_demand = dmnds.project_scenario_units_to_buses(df_pre_demand, allocation)
        if df_post_demand is not None:
            df_post_demand = dmnds.project_scenario_units_to_buses(
                df_post_demand, allocation
            )

    if max_timesteps is not None:
        df_pre_demand = df_pre_demand.iloc[: int(max_timesteps)].copy()
        if df_post_demand is not None:
            df_post_demand = df_post_demand.iloc[: int(max_timesteps)].copy()

    summaries: dict[str, dict[str, Any]] = {}
    summaries["pre"] = pwrflw.pf_summary(
        grid,
        df_pre_demand,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_max_i_ka=cable_max_i_ka,
        voltage_buses=voltage_buses,
        algorithm=["nr", "iwamoto_nr"],
        cable_ids=summary_cable_ids,
        on_nonconvergence="nan",
        protect_grid_state=True,
    )
    if df_post_demand is not None:
        summaries["post"] = pwrflw.pf_summary(
            grid,
            df_post_demand,
            transformer_s_rated_mva=transformer_s_rated_mva,
            cable_max_i_ka=cable_max_i_ka,
            voltage_buses=voltage_buses,
            algorithm=["nr", "iwamoto_nr"],
            cable_ids=summary_cable_ids,
            on_nonconvergence="nan",
            protect_grid_state=True,
        )

    assumptions = {
        **metadata,
        "demand_allocation": URBS_ASSUMPTION_TEXT,
        "urbs_result_hdf": str(hdf_path),
        "post_demand_mode": post_demand_mode,
        "stage_label": "sector_coupling",
        "max_timesteps": None if max_timesteps is None else int(max_timesteps),
        "allocation_plan_rows": int(len(allocation)),
        "allocation_plan_buses": int(allocation["allocation_bus"].nunique()),
        "no_flex_ev_charger_kw": None
        if no_flex_ev_charger_kw is None
        else float(no_flex_ev_charger_kw),
        **load_scope,
    }
    if post_demand_mode == "no-flex":
        assumptions["no_flex_battery_control"] = (
            "fixed SWF battery inventory; causal local PV self-consumption; "
            "no grid charging or battery export; cyclic state per TSAM period"
        )
    for stage, summary in summaries.items():
        assumptions[f"{stage}_nonconverged_timesteps"] = int(
            summary["grid_summary"].get("n_failed_timesteps", 0)
        )

    db = SurroGridDatabase()
    db.ensure_schema()
    run_id = db.create_real_powerflow_run(
        _grid_ref(row),
        run_name=run_name,
        scenario_key=scenario_key,
        scenario_label=scenario_label,
        assumptions=assumptions,
    )
    for stage, summary in summaries.items():
        db.write_real_powerflow_summary(run_id, stage, summary)

    elapsed = time.perf_counter() - start
    return {
        "lv_id": lv_id,
        "run_id": run_id,
        "source_file": str(source_file),
        "urbs_result_hdf": str(hdf_path),
        "elapsed_s": elapsed,
        "post_demand_mode": post_demand_mode,
        "stages": ",".join(summaries),
        "pre_failed_timesteps": summaries["pre"]["grid_summary"].get(
            "n_failed_timesteps"
        ),
        "post_failed_timesteps": summaries.get("post", {})
        .get("grid_summary", {})
        .get("n_failed_timesteps"),
        "n_voltage_buses": summaries["pre"]["grid_summary"]["n_voltage_buses"],
        "n_cables": summaries["pre"]["grid_summary"]["n_cables"],
        **load_scope,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real SWF scenario-plan base-electricity power-flow summaries."
    )
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--allocation-plan", type=Path, default=DEFAULT_ALLOCATION_PLAN)
    parser.add_argument(
        "--timeframe-mode",
        choices=["full_year"],
        default="full_year",
        help="Only full_year is supported until real-grid URBS/TSAM input generation is implemented.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lv-id", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--scenario-key", default=DEFAULT_SCENARIO_KEY)
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=481527,
        help="Arbitrary fixed seed for reproducible stochastic input profiles.",
    )
    parser.add_argument(
        "--measured-profile-selection",
        choices=MEASURED_PROFILE_SELECTION_CHOICES,
        default=MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    )
    parser.add_argument(
        "--measured-profile-band-pct",
        type=float,
        default=DEFAULT_MEASURED_PROFILE_BAND_PCT,
    )
    parser.add_argument(
        "--measured-profile-min-candidates",
        type=int,
        default=DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the full horizon.",
    )
    parser.add_argument(
        "--urbs-result-hdf",
        type=Path,
        default=None,
        help="Optional Step-3 URBS result HDF. If set, run real-grid pre/post sector power-flow from this HDF instead of base-electricity allocation.",
    )
    parser.add_argument(
        "--post-demand-mode",
        choices=["flexible", "no-flex", "pre-only"],
        default="flexible",
        help="Demand reconstruction for --urbs-result-hdf. 'pre-only' runs only urbs_in/reduced pre demand.",
    )
    parser.add_argument(
        "--no-flex-ev-charger-kw",
        type=float,
        default=None,
        help="Optional EV charger cap for --post-demand-mode no-flex; defaults to Step-4 config.",
    )
    parser.add_argument(
        "--scenario-label",
        default=None,
        help="Optional scenario label stored in the DB. Defaults depend on base-electricity vs URBS-result mode.",
    )
    parser.add_argument(
        "--summary-grid-scope",
        choices=["full", "backbone"],
        default="full",
        help="Include service lines/terminal buses (full) or evaluate the upstream backbone only.",
    )
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=True)
    root = args.grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    if args.urbs_result_hdf is not None:
        inferred_lv_id = _lv_id_from_hdf(args.urbs_result_hdf)
        lv_id = args.lv_id or (None if inferred_lv_id is None else str(inferred_lv_id))
        if lv_id is None:
            raise ValueError(
                "--urbs-result-hdf requires --lv-id when the LV id cannot be inferred from raw_data/allocation_plan."
            )
        rows = _select_manifest_rows(root, args.plz, limit=1, lv_id=lv_id)
        if not rows:
            raise ValueError(f"No SWF real-grid manifest row matched LV {lv_id}.")
        scenario_label = (
            args.scenario_label
            or f"Real SWF 2045 sector-coupling {args.post_demand_mode}"
        )
        print(
            f"Running real SWF URBS-result powerflow for LV {rows[0]['lv_id']} from {args.urbs_result_hdf} "
            f"with post_demand_mode={args.post_demand_mode}.",
            flush=True,
        )
        result = run_one_urbs_result(
            rows[0],
            str(args.urbs_result_hdf),
            args.run_name,
            args.scenario_key,
            scenario_label,
            args.post_demand_mode,
            args.max_timesteps,
            args.no_flex_ev_charger_kw,
            args.summary_grid_scope,
        )
        result["status"] = "ok"
        result["error"] = ""
        print(pd.DataFrame([result]).to_string(index=False), flush=True)
        return

    plan = read_allocation_plan(args.allocation_plan, scope=None)
    rows = _selected_rows_for_plan(root, args.plz, args.limit, args.lv_id, plan)
    if not rows:
        raise ValueError(
            "No SWF real-grid rows matched the allocation plan and requested filters."
        )

    print(
        f"Running {len(rows)} SWF scenario-plan base-electricity job(s) from {root} "
        f"using {args.allocation_plan} with {args.workers} worker(s).",
        flush=True,
    )
    scenario_label = args.scenario_label or DEFAULT_SCENARIO_LABEL
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for i, row in enumerate(rows):
            print(
                f"[{i + 1}/{len(rows)}] LV {row['lv_id']} {row['source_file']}",
                flush=True,
            )
            try:
                result = run_one(
                    row,
                    str(args.allocation_plan),
                    args.run_name,
                    args.scenario_key,
                    scenario_label,
                    args.profile_seed + i,
                    args.measured_profile_selection,
                    args.measured_profile_band_pct,
                    args.measured_profile_min_candidates,
                    args.max_timesteps,
                )
            except Exception as exc:
                result = {
                    "lv_id": row["lv_id"],
                    "run_id": None,
                    "status": "failed",
                    "error": str(exc),
                }
                print(f"  -> FAILED: {exc}", flush=True)
            else:
                result["status"] = "ok"
                result["error"] = ""
                print(
                    f"  -> run_id={result['run_id']} elapsed={result['elapsed_s']:.1f}s tail_rows={result['tail_rows']}",
                    flush=True,
                )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_one,
                    row,
                    str(args.allocation_plan),
                    args.run_name,
                    args.scenario_key,
                    scenario_label,
                    args.profile_seed + i,
                    args.measured_profile_selection,
                    args.measured_profile_band_pct,
                    args.measured_profile_min_candidates,
                    args.max_timesteps,
                ): row
                for i, row in enumerate(rows)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "lv_id": row["lv_id"],
                        "run_id": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                    print(
                        f"[{done}/{len(rows)}] LV {row['lv_id']} FAILED: {exc}",
                        flush=True,
                    )
                else:
                    result["status"] = "ok"
                    result["error"] = ""
                    print(
                        f"[{done}/{len(rows)}] LV {result['lv_id']} -> run_id={result['run_id']} "
                        f"elapsed={result['elapsed_s']:.1f}s tail_rows={result['tail_rows']}",
                        flush=True,
                    )
                results.append(result)

    elapsed = time.perf_counter() - start
    result_df = pd.DataFrame(results).sort_values("lv_id")
    print(result_df.to_string(index=False), flush=True)
    ok_count = (
        int((result_df.get("status", "ok") == "ok").sum())
        if "status" in result_df
        else len(result_df)
    )
    print(
        f"Finished {ok_count}/{len(results)} real scenario-plan job(s) successfully in {elapsed:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
