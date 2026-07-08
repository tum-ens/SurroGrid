"""Demand-audit helpers for real and synthetic Forchheim comparison runs.

The real SWF grids contain annual HH demand metadata in load descriptions. The
synthetic grids do not have a one-to-one mapping to those real load rows, so this
module keeps two layers explicit:

- real-grid audit: parsed SWF annual kWh compared with generated HH profiles;
- synthetic-grid audit: allocated synthetic annual kWh and the aggregate scale
  factor implied by the real SWF total.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandapower as pp
from dotenv import load_dotenv
from sqlalchemy import text

POSTPROCESSING_DIR = Path(__file__).resolve().parent
GRIDEXPAND_DIR = POSTPROCESSING_DIR.parents[0]
REPO_ROOT = GRIDEXPAND_DIR.parents[0]
STEP4_DIR = GRIDEXPAND_DIR / "4.powerflow"
ENV_PATH = GRIDEXPAND_DIR / ".env"

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


def _load_real_runner():
    spec = importlib.util.spec_from_file_location(
        "run_real_swf_powerflow_audit",
        STEP4_DIR / "run_real_swf_powerflow.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load run_real_swf_powerflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_worst_real_voltage_grids(
    *,
    real_run_name: str,
    plz: int,
    stage: str = "pre",
    limit: int = 3,
) -> pd.DataFrame:
    """Return LV grids with the lowest retained-bus voltage minima."""
    db = SurroGridDatabase()
    query = text(
        """
        SELECT
            rgc.lv_id,
            MIN(rbv.voltage_min_time_pu) AS min_voltage_pu,
            COUNT(DISTINCT rbv.bus) AS voltage_buses
        FROM surrogrid.real_powerflow_bus_voltage_summary rbv
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        WHERE rpr.run_name = :real_run_name
          AND rbv.stage = :stage
          AND rgc.plz = :plz
        GROUP BY rgc.lv_id
        ORDER BY min_voltage_pu ASC NULLS LAST, rgc.lv_id
        LIMIT :limit
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={"real_run_name": real_run_name, "stage": stage, "plz": int(plz), "limit": int(limit)},
        )


def _percent_delta(value: float, reference: float) -> float:
    if reference == 0 or pd.isna(reference):
        return float("nan")
    return 100.0 * (value - reference) / reference


def _audit_real_row(row: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    real_runner = _load_real_runner()
    net = pp.from_excel(Path(row["source_file"]))
    (
        _grid,
        _transformer_s_rated_mva,
        _cable_max_i_ka,
        _voltage_buses,
        _backbone_cable_ids,
        selected_household_loads,
        load_scope,
    ) = real_runner._prepare_real_grid(net)
    _synthetic_demand, synthetic_audit = real_runner._build_real_electric_demand(
        net,
        seed=seed,
        load_rows=selected_household_loads,
        annual_demand_mode=real_runner.ANNUAL_DEMAND_MODE_SYNTHETIC,
        return_audit=True,
    )
    _measured_demand, measured_audit = real_runner._build_real_electric_demand(
        net,
        seed=seed,
        load_rows=selected_household_loads,
        annual_demand_mode=real_runner.ANNUAL_DEMAND_MODE_MEASURED,
        return_audit=True,
    )

    swf_total = float(pd.to_numeric(synthetic_audit["swf_annual_demand_kwh"], errors="coerce").sum())
    synthetic_total = float(pd.to_numeric(synthetic_audit["generated_profile_energy_kwh"], errors="coerce").sum())
    measured_total = float(pd.to_numeric(measured_audit["generated_profile_energy_kwh"], errors="coerce").sum())
    annual_rows = int(pd.to_numeric(synthetic_audit["swf_annual_demand_kwh"], errors="coerce").notna().sum())

    summary = {
        "lv_id": row["lv_id"],
        "source_file": row["source_file"],
        **load_scope,
        "selected_household_load_rows": int(len(selected_household_loads)),
        "selected_household_load_buses": int(selected_household_loads["bus"].dropna().astype(int).nunique()),
        "swf_annual_demand_rows": annual_rows,
        "swf_annual_demand_kwh": swf_total,
        "synthetic_profile_energy_kwh": synthetic_total,
        "measured_profile_energy_kwh": measured_total,
        "synthetic_vs_swf_percent": _percent_delta(synthetic_total, swf_total),
        "measured_vs_swf_percent": _percent_delta(measured_total, swf_total),
        "implied_synthetic_to_swf_scale": swf_total / synthetic_total if synthetic_total else float("nan"),
    }

    bus = (
        synthetic_audit.groupby("real_bus", as_index=False)
        .agg(
            load_rows=("profile_id", "count"),
            swf_annual_demand_kwh=("swf_annual_demand_kwh", "sum"),
            synthetic_profile_energy_kwh=("generated_profile_energy_kwh", "sum"),
        )
        .merge(
            measured_audit.groupby("real_bus", as_index=False).agg(
                measured_profile_energy_kwh=("generated_profile_energy_kwh", "sum")
            ),
            on="real_bus",
            how="left",
        )
    )
    bus.insert(0, "lv_id", row["lv_id"])
    return summary, bus


def audit_real_household_demand(
    *,
    lv_ids: list[str] | None,
    plz: int,
    seed: int = 91301,
    grid_data_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_runner = _load_real_runner()
    load_dotenv(ENV_PATH, override=True)
    root = grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    rows: list[dict[str, Any]] = []
    if lv_ids:
        for lv_id in lv_ids:
            rows.extend(real_runner._select_manifest_rows(root, plz, limit=None, lv_id=str(lv_id)))
    else:
        rows = real_runner._select_manifest_rows(root, plz, limit=None, lv_id=None)

    summaries = []
    buses = []
    for i, row in enumerate(rows):
        summary, bus = _audit_real_row(row, seed=seed + i)
        summaries.append(summary)
        buses.append(bus)
    return pd.DataFrame(summaries), pd.concat(buses, ignore_index=True) if buses else pd.DataFrame()


def _synthetic_powerflow_scope(*, synthetic_run_name: str, plz: int) -> pd.DataFrame:
    db = SurroGridDatabase()
    query = text(
        """
        SELECT DISTINCT
            pr.grid_case_id,
            gc.ags,
            gc.plz,
            gc.kcid,
            gc.bcid,
            gc.cell_id,
            pr.urbs_input_file
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :synthetic_run_name
          AND gc.plz = :plz
        ORDER BY gc.cell_id, gc.kcid, gc.bcid
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"synthetic_run_name": synthetic_run_name, "plz": int(plz)})


def _synthetic_residential_buses(grid_case_ids: list[int]) -> dict[int, set[int]]:
    if not grid_case_ids:
        return {}
    db = SurroGridDatabase()
    query = text(
        """
        SELECT DISTINCT grid_case_id, bus
        FROM surrogrid.grid_building_bus
        WHERE grid_case_id = ANY(:grid_case_ids)
          AND bus IS NOT NULL
          AND lower(COALESCE(building_use, '')) = 'residential'
        """
    )
    with db.engine.connect() as conn:
        rows = pd.read_sql_query(query, conn, params={"grid_case_ids": [int(value) for value in grid_case_ids]})
    grouped: dict[int, set[int]] = {}
    for row in rows.itertuples(index=False):
        grouped.setdefault(int(row.grid_case_id), set()).add(int(row.bus))
    return grouped


def _audit_synthetic_from_hdf(scope: pd.DataFrame, *, input_dir: Path) -> pd.DataFrame:
    residential_by_grid = _synthetic_residential_buses(scope["grid_case_id"].astype(int).tolist())
    rows: list[dict[str, Any]] = []
    for record in scope.to_dict("records"):
        grid_case_id = int(record["grid_case_id"])
        hdf_path = input_dir / str(record["urbs_input_file"])
        residential_buses = residential_by_grid.get(grid_case_id, set())
        if not hdf_path.exists() or not residential_buses:
            rows.append({**record, "synthetic_annual_demand_kwh": np.nan, "synthetic_demand_buses": 0, "timesteps": 0})
            continue
        demand = pd.read_hdf(hdf_path, "urbs_in/demand")
        if not isinstance(demand.columns, pd.MultiIndex):
            raise ValueError(f"Expected MultiIndex demand columns in {hdf_path}.")
        bus_level = demand.columns.get_level_values(0).astype(int)
        commodity_level = demand.columns.get_level_values(1).astype(str)
        mask = bus_level.isin(residential_buses) & (commodity_level == "electricity")
        selected = demand.loc[:, mask]
        rows.append(
            {
                **record,
                "demand_source": "hdf_urbs_in_demand_residential_buses",
                "synthetic_annual_demand_kwh": float(selected.sum().sum()),
                "synthetic_demand_buses": int(selected.columns.get_level_values(0).nunique()),
                "timesteps": int(len(selected)),
            }
        )
    return pd.DataFrame(rows)


def audit_synthetic_annual_demand(
    *,
    synthetic_run_name: str,
    plz: int,
    demand_profiles: str = "status_quo",
    input_dir: Path = GRIDEXPAND_DIR / "4.powerflow" / "Input",
) -> pd.DataFrame:
    """Return allocated synthetic residential electricity totals for grids in a powerflow run.

    Summary-only Forchheim runs often use temporary Step 2 storage, so the DB
    `allocated_demand` table may intentionally be empty. In that case, this
    falls back to the Step 4 HDF input files and filters to residential buses.
    """
    db = SurroGridDatabase()
    query = text(
        """
        WITH synthetic_scope AS (
            SELECT DISTINCT
                pr.grid_case_id,
                gc.ags,
                gc.plz,
                gc.kcid,
                gc.bcid,
                gc.cell_id,
                pr.urbs_input_file
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :synthetic_run_name
              AND gc.plz = :plz
        ), latest_demand AS (
            SELECT DISTINCT ON (dar.grid_case_id)
                dar.grid_case_id,
                dar.demand_allocation_run_id,
                dar.run_name AS demand_run_name,
                dar.created_at
            FROM surrogrid.demand_allocation_run dar
            JOIN synthetic_scope ss USING (grid_case_id)
            WHERE dar.profiles = :demand_profiles
            ORDER BY dar.grid_case_id, dar.created_at DESC, dar.demand_allocation_run_id DESC
        )
        SELECT
            ss.grid_case_id,
            ss.ags,
            ss.plz,
            ss.kcid,
            ss.bcid,
            ss.cell_id,
            ss.urbs_input_file,
            ld.demand_allocation_run_id,
            ld.demand_run_name,
            'db_allocated_demand' AS demand_source,
            SUM(ad.value) AS synthetic_annual_demand_kwh,
            COUNT(DISTINCT ad.bus) AS synthetic_demand_buses,
            COUNT(DISTINCT ad.t_index) AS timesteps
        FROM synthetic_scope ss
        JOIN latest_demand ld USING (grid_case_id)
        JOIN surrogrid.allocated_demand ad USING (demand_allocation_run_id)
        JOIN surrogrid.grid_building_bus gbb
          ON gbb.grid_case_id = ss.grid_case_id
         AND gbb.bus = ad.bus
         AND lower(COALESCE(gbb.building_use, '')) = 'residential'
        WHERE ad.commodity = 'electricity'
        GROUP BY
            ss.grid_case_id, ss.ags, ss.plz, ss.kcid, ss.bcid, ss.cell_id,
            ss.urbs_input_file, ld.demand_allocation_run_id, ld.demand_run_name
        ORDER BY ss.cell_id, ss.kcid, ss.bcid
        """
    )
    with db.engine.connect() as conn:
        result = pd.read_sql_query(
            query,
            conn,
            params={"synthetic_run_name": synthetic_run_name, "plz": int(plz), "demand_profiles": demand_profiles},
        )
    if not result.empty:
        return result
    scope = _synthetic_powerflow_scope(synthetic_run_name=synthetic_run_name, plz=plz)
    return _audit_synthetic_from_hdf(scope, input_dir=input_dir)


def build_comparison_table(real_summary: pd.DataFrame, synthetic_summary: pd.DataFrame) -> pd.DataFrame:
    real_total = float(real_summary["swf_annual_demand_kwh"].sum()) if not real_summary.empty else float("nan")
    synthetic_total = float(synthetic_summary["synthetic_annual_demand_kwh"].sum()) if not synthetic_summary.empty else float("nan")
    comparable_scope = int(len(real_summary)) == int(len(synthetic_summary)) and len(real_summary) > 0
    scale = real_total / synthetic_total if comparable_scope and synthetic_total else float("nan")
    return pd.DataFrame(
        [
            {
                "comparison_level": "real_selected_lv_grids",
                "grid_count": int(len(real_summary)),
                "load_or_bus_count": int(real_summary.get("selected_household_load_rows", pd.Series(dtype=int)).sum()),
                "annual_demand_kwh": real_total,
                "reference": "parsed SWF 2022 HH load descriptions",
                "synthetic_scale_to_match_swf": np.nan,
                "scope_note": "selected problematic real grids",
            },
            {
                "comparison_level": "synthetic_powerflow_scope",
                "grid_count": int(len(synthetic_summary)),
                "load_or_bus_count": int(synthetic_summary.get("synthetic_demand_buses", pd.Series(dtype=int)).sum()),
                "annual_demand_kwh": synthetic_total,
                "reference": "latest status_quo electricity demand or HDF fallback for synthetic run",
                "synthetic_scale_to_match_swf": scale,
                "scope_note": (
                    "scale only shown when real and synthetic grid counts match; "
                    "otherwise use this row as a separate scope diagnostic"
                ),
            },
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit real SWF and synthetic annual HH demand assumptions.")
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--real-run-name", default="real_hybrid")
    parser.add_argument("--synthetic-run-name", default="1_synthetic")
    parser.add_argument("--stage", default="pre")
    parser.add_argument("--limit", type=int, default=3, help="Number of lowest-voltage real LV grids to audit.")
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=POSTPROCESSING_DIR / "output" / "demand_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worst = get_worst_real_voltage_grids(
        real_run_name=args.real_run_name,
        plz=args.plz,
        stage=args.stage,
        limit=args.limit,
    )
    lv_ids = worst["lv_id"].astype(str).tolist()
    real_summary, real_bus = audit_real_household_demand(
        lv_ids=lv_ids,
        plz=args.plz,
        seed=args.seed,
        grid_data_path=args.grid_data_path,
    )
    if not worst.empty and not real_summary.empty:
        worst = worst.copy()
        worst["lv_id"] = worst["lv_id"].astype(str)
        real_summary = real_summary.copy()
        real_summary["lv_id"] = real_summary["lv_id"].astype(str)
        real_summary = real_summary.merge(worst, on="lv_id", how="left")
    synthetic_summary = audit_synthetic_annual_demand(
        synthetic_run_name=args.synthetic_run_name,
        plz=args.plz,
    )
    comparison = build_comparison_table(real_summary, synthetic_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worst.to_csv(args.output_dir / "worst_real_voltage_grids.csv", index=False)
    real_summary.to_csv(args.output_dir / "real_household_demand_summary.csv", index=False)
    real_bus.to_csv(args.output_dir / "real_household_demand_by_bus.csv", index=False)
    synthetic_summary.to_csv(args.output_dir / "synthetic_annual_demand_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "real_synthetic_annual_demand_comparison.csv", index=False)

    print("Worst real voltage grids:")
    print(worst.to_string(index=False))
    print("\nReal HH demand audit:")
    print(real_summary.to_string(index=False))
    print("\nReal/synthetic demand comparison:")
    print(comparison.to_string(index=False))
    print(f"\nWrote CSV tables to {args.output_dir}")


if __name__ == "__main__":
    main()
