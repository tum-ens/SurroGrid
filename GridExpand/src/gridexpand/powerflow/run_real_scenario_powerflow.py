"""Step 4 of one real grid: power-flow summaries of a Step-3 result on the real network.

The paired runner calls this module once per real grid and model case. The real grid is a file named by the
paired dataset (``--grid-file``); the Step-3 result HDF carries the scenario units and their allocation buses
(``raw_data/allocation_plan``). Demand reconstruction is the shared Step-4 logic of the synthetic grids; only the
pandapower network and the bus allocation differ. Summaries go to the ``real_powerflow_*`` tables.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import pandas as pd
import pandapower as pp
from dotenv import load_dotenv

from gridexpand.db.database import SurroGridDatabase
from gridexpand.common.timeframe import read_hdf_metadata as read_timeframe_metadata
import gridexpand.powerflow.powerflow as pwrflw
import gridexpand.powerflow.demands as dmnds
from gridexpand.powerflow import network, station_voltage
from gridexpand.powerflow.io import (
    ScenarioResultReader,
    read_temporal_method,
    require_temporal_method,
    temporal_assumptions,
)
from gridexpand.paths import ENV_FILE

ENV_PATH = ENV_FILE
GRID_ASSUMPTION_TEXT = "Real grid from the file named by the paired dataset; scenario loads replace its loads."
URBS_ASSUMPTION_TEXT = (
    "Real-grid power flow of a Step-3 result HDF. Demand reconstruction uses the shared GridExpand Step-4 "
    "demand logic also used for synthetic grids; only the pandapower network and bus allocation differ."
)


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


def _grid_ref(row: dict[str, Any]) -> dict[str, Any]:
    """``real_grid_case`` identity of one real grid file."""
    return {
        "source": str(row["source"]),
        "plz": int(row["plz"]),
        "lv_id": str(row["lv_id"]),
        "variant": row.get("variant"),
        "category": row.get("category"),
        "load_status": row.get("load_status"),
        "status": row.get("status"),
        "source_file": str(row["source_file"]),
        "bus_count": None,
        "line_count": None,
        "load_count": None,
        "assumptions": {"demand_allocation": GRID_ASSUMPTION_TEXT},
    }


def load_real_net(source_file: Path) -> tuple[pp.pandapowerNet, float]:
    """Load one real LV grid and its station rating in MVA.

    DSO A workbooks keep an out-of-service trafo row for the rating. DSO B nets
    have no trafo; their bank rating is the root row of ``dso_b_roots``.
    """
    if source_file.suffix == ".json":
        net = pp.from_json(source_file)
        roots = net["dso_b_roots"]
        root = roots[roots["disposition"].astype(str).eq("root")]
        if len(root) != 1:
            raise ValueError(f"{source_file} must have exactly one dso_b_roots root row.")
        rating_kva = pd.to_numeric(root["rating_kva_total"], errors="coerce").iloc[0]
        return net, float(rating_kva) / 1000.0
    net = pp.from_excel(source_file)
    return net, network.transformer_rating_mva(net)


def _prepare_real_grid_for_allocation(
    net: pp.pandapowerNet,
    allocation_buses: list[int],
    summary_grid_scope: str = "full",
    transformer_s_rated_mva: float = float("nan"),
):
    """Prepare a real grid whose DSO loads are replaced by scenario loads.

    The ext_grid sits on the station LV busbar in both DSO models, so the
    transformer is represented by its rating only.
    """
    grid = deepcopy(net)
    pp.replace_zero_branches_with_switches(
        grid,
        elements=("line", "impedance"),
        zero_length=True,
        zero_impedance=True,
        in_service_only=True,
        drop_affected=False,
    )
    cable_max_i_ka = pd.to_numeric(grid.line["max_i_ka"], errors="coerce")
    if "in_service" in grid.line.columns:
        cable_max_i_ka = cable_max_i_ka.where(
            grid.line["in_service"].fillna(True).astype(bool)
        )
    if not grid.line.empty:
        grid.line["max_i_ka"] = 1000.0
    network.disable_static_injections(grid, fill_missing=True)
    if not grid.bus.empty:
        grid.bus[["min_vm_pu", "max_vm_pu"]] = (0.0, 10.0)
    load_scope: dict[str, Any] = {}
    load_buses = sorted({int(bus) for bus in allocation_buses})
    grid = network.set_scenario_load_buses(grid, load_buses)

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


def run_one_urbs_result(
    row: dict[str, Any],
    urbs_result_hdf: str,
    run_name: str,
    scenario_key: str,
    scenario_label: str,
    post_demand_mode: str,
    max_timesteps: int | None = None,
    inflex_ev_charger_kw: float | None = None,
    summary_grid_scope: str = "full",
    expect_temporal_method: str | None = None,
    post_only: bool = False,
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
    net, rating_mva = load_real_net(source_file)
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
        rating_mva,
    )

    # Compact real-grid summaries write no reactive tables; component audits go
    # to a sidecar next to the result file.
    adapter = ScenarioResultReader(hdf_path, output_path=hdf_path, run_name=run_name)
    metadata = _read_hdf_metadata(hdf_path)
    # Identity, not file name, decides whether this result may be consumed.
    if expect_temporal_method is not None:
        temporal_audit = require_temporal_method(str(hdf_path), expect_temporal_method)
    else:
        temporal_audit = read_temporal_method(str(hdf_path))
    metadata = {**metadata, **temporal_assumptions(temporal_audit)}
    if post_demand_mode == "pre-only":
        df_pre_demand = dmnds.obtain_pre_demand(adapter)
        df_post_demand = None
    else:
        # Per-vehicle charger ratings come from the EV session table; an
        # explicit value is only a cross-check.
        df_pre_demand, df_post_demand = dmnds.obtain_demand(
            adapter,
            save_reactive=False,
            post_demand_mode=post_demand_mode,
            ev_charger_kw=(
                None if inflex_ev_charger_kw is None else float(inflex_ev_charger_kw)
            ),
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
    if not post_only:
        summaries["pre"] = pwrflw.pf_summary(
            grid,
            df_pre_demand,
            transformer_s_rated_mva=transformer_s_rated_mva,
            cable_max_i_ka=cable_max_i_ka,
            voltage_buses=voltage_buses,
            algorithm=["nr", "iwamoto_nr"],
            cable_ids=summary_cable_ids,
            on_nonconvergence="nan",
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
        )

    assumptions = {
        **metadata,
        "demand_allocation": URBS_ASSUMPTION_TEXT,
        "urbs_result_hdf": str(hdf_path),
        "station_voltage": station_voltage.assumptions(),
        "post_demand_mode": post_demand_mode,
        **({"post_only": True} if post_only else {}),
        "stage_label": "sector_coupling",
        "max_timesteps": None if max_timesteps is None else int(max_timesteps),
        "allocation_plan_rows": int(len(allocation)),
        "allocation_plan_buses": int(allocation["allocation_bus"].nunique()),
        "inflex_ev_charger_kw": None
        if inflex_ev_charger_kw is None
        else float(inflex_ev_charger_kw),
        **load_scope,
    }
    if post_demand_mode == "inflex":
        assumptions["inflex_battery_control"] = (
            "fixed battery of the asset plan; causal local PV self-consumption; "
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
        "pre_failed_timesteps": summaries.get("pre", {})
        .get("grid_summary", {})
        .get("n_failed_timesteps"),
        "post_failed_timesteps": summaries.get("post", {})
        .get("grid_summary", {})
        .get("n_failed_timesteps"),
        "n_voltage_buses": next(iter(summaries.values()))["grid_summary"]["n_voltage_buses"],
        "n_cables": next(iter(summaries.values()))["grid_summary"]["n_cables"],
        **load_scope,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 of one real grid: pre/post power-flow summaries of a Step-3 result HDF."
    )
    parser.add_argument("--plz", type=int, required=True, help="Postcode stored with the real grid case.")
    parser.add_argument("--lv-id", default=None, help="Real grid id (default: from raw_data/allocation_plan).")
    parser.add_argument(
        "--provider",
        choices=["dso_a", "dso_b"],
        default="dso_a",
        help="Real-grid source recorded in real_grid_case.source.",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        required=True,
        help="The real grid file (DSO A .xlsx or DSO B .json) named by the paired dataset.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scenario-key", required=True)
    parser.add_argument(
        "--expect-temporal-method",
        choices=("full_year_no_tsam", "shared_weather_tsam"),
        default=None,
        help=(
            "Reject the Step-3 result unless it records this temporal method. "
            "Neither the file name nor the presence of the reduced_data group "
            "proves how a result was produced."
        ),
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
        required=True,
        help="Step-3 result HDF (Step-2 input HDF for --post-demand-mode pre-only).",
    )
    parser.add_argument(
        "--post-demand-mode",
        choices=["flexible", "inflex", "pre-only"],
        default="flexible",
        help="Demand reconstruction. 'pre-only' runs only urbs_in/reduced pre demand.",
    )
    parser.add_argument(
        "--inflex-ev-charger-kw",
        type=float,
        default=None,
        help="Optional EV charger cap for --post-demand-mode inflex; defaults to Step-4 config.",
    )
    parser.add_argument("--scenario-label", default=None, help="Scenario label stored in the DB.")
    parser.add_argument(
        "--summary-grid-scope",
        choices=["full", "backbone"],
        default="full",
        help="Include service lines/terminal buses (full) or evaluate the upstream backbone only.",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Run only the post stage; the pre stage comes from a separate pre-only run.",
    )
    args = parser.parse_args(argv)
    if args.post_only and args.post_demand_mode == "pre-only":
        parser.error("--post-only requires a post --post-demand-mode (flexible or inflex).")

    load_dotenv(ENV_PATH, override=True)
    inferred_lv_id = _lv_id_from_hdf(args.urbs_result_hdf)
    lv_id = args.lv_id or (None if inferred_lv_id is None else str(inferred_lv_id))
    if lv_id is None:
        raise ValueError(
            "--lv-id is required when the LV id cannot be inferred from raw_data/allocation_plan."
        )
    row = {
        "source": args.provider,
        "lv_id": str(int(lv_id)),
        "plz": int(args.plz),
        "source_file": str(args.grid_file.resolve()),
        "variant": "aligned",
        "status": "exported",
    }
    scenario_label = args.scenario_label or f"Real grid {args.post_demand_mode}"
    print(
        f"Running the real-grid power flow of LV {row['lv_id']} ({args.provider}) from {args.urbs_result_hdf} "
        f"with post_demand_mode={args.post_demand_mode}.",
        flush=True,
    )
    result = run_one_urbs_result(
        row,
        str(args.urbs_result_hdf),
        args.run_name,
        args.scenario_key,
        scenario_label,
        args.post_demand_mode,
        args.max_timesteps,
        args.inflex_ev_charger_kw,
        args.summary_grid_scope,
        args.expect_temporal_method,
        args.post_only,
    )
    result["status"] = "ok"
    result["error"] = ""
    print(pd.DataFrame([result]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
