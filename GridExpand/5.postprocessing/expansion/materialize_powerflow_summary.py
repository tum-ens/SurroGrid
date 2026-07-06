"""Materialize compact power-flow summaries from existing raw DB results.

This utility derives the same compact summary tables that ``run_pwrflw.py
--summary-only`` writes, but without running pandapower again. It reads the raw
``powerflow_import``, ``powerflow_line_result``, and ``powerflow_bus_voltage``
rows for an existing full power-flow run and writes the compact summary tables
used by the notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text


GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
POWERFLOW_DIR = GRIDEXPAND_DIR / "4.powerflow"
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
if str(POWERFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(POWERFLOW_DIR))

from database import SurroGridDatabase, normalize_ags
from src import powerflow as pwrflw


def _optional_ags(value: str | int | None) -> int | None:
    if value is None:
        return None
    return normalize_ags(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return int(value)


def _powerflow_assumptions(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _selected_runs(db: SurroGridDatabase, args: argparse.Namespace) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT
            pr.powerflow_run_id,
            pr.run_name,
            pr.scenario_id,
            pr.urbs_input_file,
            pr.assumptions,
            gc.grid_case_id,
            gc.ags,
            gc.plz,
            gc.kcid,
            gc.bcid,
            gc.pylovo_grid_result_id AS grid_result_id,
            gc.pylovo_version_id AS version_id,
            gc.cell_id
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:plz IS NULL OR gc.plz = :plz)
        ORDER BY gc.plz, gc.kcid, gc.bcid, pr.powerflow_run_id
        LIMIT COALESCE(:limit_rows, 2147483647)
        """
    )
    with db.engine.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                query,
                {
                    "run_name": args.run_name,
                    "scenario_id": args.scenario_id,
                    "ags": _optional_ags(args.ags),
                    "plz": args.plz,
                    "limit_rows": args.limit,
                },
            ).mappings()
        ]
    return rows


def _grid_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ags": int(row["ags"]),
        "candidate_index": 0,
        "cell_id": str(row.get("cell_id") or ""),
        "bridge_filename": str(row.get("urbs_input_file") or ""),
        "grid_result_id": int(row["grid_result_id"]),
        "version_id": str(row["version_id"]),
        "plz": int(row["plz"]),
        "kcid": int(row["kcid"]),
        "bcid": int(row["bcid"]),
    }


def _existing_summary_rows(db: SurroGridDatabase, run_id: int, stage: str) -> int:
    query = text(
        """
        SELECT
            (SELECT COUNT(*) FROM surrogrid.powerflow_summary WHERE powerflow_run_id = :run_id AND stage = :stage)
          + (SELECT COUNT(*) FROM surrogrid.powerflow_cable_summary WHERE powerflow_run_id = :run_id AND stage = :stage)
          + (SELECT COUNT(*) FROM surrogrid.powerflow_bus_voltage_summary WHERE powerflow_run_id = :run_id AND stage = :stage)
          + (SELECT COUNT(*) FROM surrogrid.powerflow_tail_value WHERE powerflow_run_id = :run_id AND stage = :stage)
        """
    )
    with db.engine.connect() as conn:
        return int(conn.execute(query, {"run_id": int(run_id), "stage": stage}).scalar_one())


def _delete_summary_rows(db: SurroGridDatabase, run_id: int, stage: str) -> None:
    with db.engine.begin() as conn:
        for table_name in (
            "powerflow_tail_value",
            "powerflow_cable_summary",
            "powerflow_bus_voltage_summary",
            "powerflow_summary",
        ):
            conn.execute(
                text(f"DELETE FROM surrogrid.{table_name} WHERE powerflow_run_id = :run_id AND stage = :stage"),
                {"run_id": int(run_id), "stage": stage},
            )


def _stage_timestep_count(db: SurroGridDatabase, run_id: int, stage: str, expected_horizon: int | None) -> int:
    query = text(
        """
        SELECT COALESCE(MAX(t_index), -1) + 1 AS max_index_count
        FROM (
            SELECT t_index FROM surrogrid.powerflow_import WHERE powerflow_run_id = :run_id AND stage = :stage
            UNION
            SELECT t_index FROM surrogrid.powerflow_line_result WHERE powerflow_run_id = :run_id AND stage = :stage
            UNION
            SELECT t_index FROM surrogrid.powerflow_bus_voltage WHERE powerflow_run_id = :run_id AND stage = :stage
        ) t
        """
    )
    with db.engine.connect() as conn:
        observed_count = int(conn.execute(query, {"run_id": int(run_id), "stage": stage}).scalar_one() or 0)
    if expected_horizon is not None and expected_horizon > 0:
        return max(int(expected_horizon), observed_count)
    return observed_count


def _pivot_matrix(
    df: pd.DataFrame,
    *,
    n_timesteps: int,
    columns: pd.Index,
    column_name: str,
    value_name: str,
) -> np.ndarray:
    if n_timesteps <= 0 or len(columns) == 0:
        return np.empty((max(n_timesteps, 0), len(columns)), dtype=float)
    if df.empty:
        return np.full((n_timesteps, len(columns)), np.nan, dtype=float)
    wide = (
        df.pivot_table(index="t_index", columns=column_name, values=value_name, aggfunc="max")
        .reindex(index=range(n_timesteps), columns=columns)
    )
    return wide.to_numpy(dtype=float)


def _raw_stage_tables(
    db: SurroGridDatabase,
    *,
    run_id: int,
    stage: str,
    cable_ids: pd.Index,
    voltage_buses: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with db.engine.connect() as conn:
        imports = pd.read_sql_query(
            text(
                """
                SELECT t_index, p_mw, q_mvar
                FROM surrogrid.powerflow_import
                WHERE powerflow_run_id = :run_id
                  AND stage = :stage
                ORDER BY t_index
                """
            ),
            conn,
            params={"run_id": int(run_id), "stage": stage},
        )
        line_results = pd.read_sql_query(
            text(
                """
                SELECT t_index, line, ABS(i_from_ka) AS i_from_ka
                FROM surrogrid.powerflow_line_result
                WHERE powerflow_run_id = :run_id
                  AND stage = :stage
                  AND line = ANY(:line_ids)
                ORDER BY t_index, line
                """
            ),
            conn,
            params={"run_id": int(run_id), "stage": stage, "line_ids": [int(line) for line in cable_ids]},
        )
        voltages = pd.read_sql_query(
            text(
                """
                SELECT t_index, bus, vm_pu
                FROM surrogrid.powerflow_bus_voltage
                WHERE powerflow_run_id = :run_id
                  AND stage = :stage
                  AND bus = ANY(:bus_ids)
                ORDER BY t_index, bus
                """
            ),
            conn,
            params={"run_id": int(run_id), "stage": stage, "bus_ids": [int(bus) for bus in voltage_buses]},
        )
    return imports, line_results, voltages


def _build_summary_from_raw(
    db: SurroGridDatabase,
    row: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    run_id = int(row["powerflow_run_id"])
    assumptions = _powerflow_assumptions(row.get("assumptions"))
    expected_horizon = _optional_int(assumptions.get("horizon_hours"))

    grid = db.read_pandapower_grid(_grid_ref(row))
    transformer_s_rated_mva = float(grid.trafo["sn_mva"].sum()) if "sn_mva" in grid.trafo.columns else float("nan")
    cable_max_i_ka = grid.line.get("max_i_ka")
    if cable_max_i_ka is None:
        cable_max_i_ka = grid.line.assign(max_i_ka=float("nan"))["max_i_ka"]
    load_buses = grid.load["bus"].dropna().astype(int).unique().tolist() if "bus" in grid.load.columns else []
    voltage_buses = load_buses or grid.bus.index.tolist()
    grid = pwrflw.prepare_grid(grid)
    backbone_cable_ids, voltage_buses = pwrflw.comparison_backbone_scope(grid, load_buses)
    if not voltage_buses:
        voltage_buses = load_buses or grid.bus.index.tolist()

    cable_ids = pd.Index([int(line) for line in backbone_cable_ids if int(line) in grid.line.index], name="cable")
    voltage_buses = pd.Index([int(bus) for bus in voltage_buses if int(bus) in grid.bus.index], name="bus")
    cable_max_i_ka = cable_max_i_ka.reindex(cable_ids).astype(float).replace(0.0, np.nan)
    if "parallel" in grid.line.columns:
        cable_parallel = grid.line["parallel"].reindex(cable_ids).fillna(1).astype(float)
    else:
        cable_parallel = pd.Series(1.0, index=cable_ids)
    cable_capacity = (cable_max_i_ka * cable_parallel).to_numpy(dtype=float)

    n_timesteps = _stage_timestep_count(db, run_id, stage, expected_horizon)
    if n_timesteps <= 0:
        raise ValueError(f"No raw power-flow rows found for powerflow_run_id={run_id}, stage={stage!r}.")

    imports, line_results, voltages = _raw_stage_tables(
        db,
        run_id=run_id,
        stage=stage,
        cable_ids=cable_ids,
        voltage_buses=voltage_buses,
    )
    if imports.empty:
        raise ValueError(f"No import rows found for powerflow_run_id={run_id}, stage={stage!r}.")

    p_mw = imports.set_index("t_index")["p_mw"].reindex(range(n_timesteps)).astype(float)
    q_mvar = imports.set_index("t_index")["q_mvar"].reindex(range(n_timesteps)).astype(float)
    if transformer_s_rated_mva > 0:
        transformer_loadings = (np.hypot(p_mw.to_numpy(), q_mvar.to_numpy()) / transformer_s_rated_mva) * 100.0
    else:
        transformer_loadings = np.full(n_timesteps, np.nan, dtype=float)

    line_matrix = _pivot_matrix(
        line_results,
        n_timesteps=n_timesteps,
        columns=cable_ids,
        column_name="line",
        value_name="i_from_ka",
    )
    if line_matrix.size:
        cable_loading_matrix = (line_matrix / cable_capacity[np.newaxis, :]) * 100.0
    else:
        cable_loading_matrix = np.empty((n_timesteps, 0), dtype=float)

    voltage_matrix = _pivot_matrix(
        voltages,
        n_timesteps=n_timesteps,
        columns=voltage_buses,
        column_name="bus",
        value_name="vm_pu",
    )

    cable_max_loading = pwrflw._safe_nanmax(cable_loading_matrix, axis=0) if len(cable_ids) else np.array([], dtype=float)
    cable_values = cable_max_loading[~np.isnan(cable_max_loading)]
    cable_hours_above_100 = np.nansum(cable_loading_matrix > 100.0, axis=0).astype(int) if len(cable_ids) else np.array([], dtype=int)
    voltage_hours_below_0_90 = np.nansum(voltage_matrix < 0.90, axis=0).astype(int) if len(voltage_buses) else np.array([], dtype=int)
    voltage_all = voltage_matrix[~np.isnan(voltage_matrix)]

    n_converged_timesteps = int(imports["t_index"].nunique())
    n_failed_timesteps = max(0, int(n_timesteps) - n_converged_timesteps)
    cable_summary = pd.DataFrame(
        {
            "cable": cable_ids,
            "cable_max_i_ka": cable_max_i_ka.to_numpy(dtype=float),
            "cable_parallel": cable_parallel.to_numpy(dtype=float),
            "cable_installed_capacity_ka": cable_capacity,
            "cable_loading_p50_time_percent": pwrflw._safe_nanpercentile(cable_loading_matrix, 50, axis=0),
            "cable_loading_p90_time_percent": pwrflw._safe_nanpercentile(cable_loading_matrix, 90, axis=0),
            "cable_loading_p95_time_percent": pwrflw._safe_nanpercentile(cable_loading_matrix, 95, axis=0),
            "cable_loading_p99_time_percent": pwrflw._safe_nanpercentile(cable_loading_matrix, 99, axis=0),
            "cable_loading_max_time_percent": cable_max_loading,
            "cable_loading_hours_above_100": cable_hours_above_100,
        }
    ).dropna(subset=["cable_loading_max_time_percent"])
    bus_voltage_summary = pd.DataFrame(
        {
            "bus": voltage_buses,
            "voltage_p50_time_pu": pwrflw._safe_nanpercentile(voltage_matrix, 50, axis=0),
            "voltage_p10_time_pu": pwrflw._safe_nanpercentile(voltage_matrix, 10, axis=0),
            "voltage_p05_time_pu": pwrflw._safe_nanpercentile(voltage_matrix, 5, axis=0),
            "voltage_p01_time_pu": pwrflw._safe_nanpercentile(voltage_matrix, 1, axis=0),
            "voltage_min_time_pu": pwrflw._safe_nanmax(-voltage_matrix, axis=0) * -1.0,
            "voltage_hours_below_0_90": voltage_hours_below_0_90,
        }
    ).dropna(subset=["voltage_p05_time_pu"])

    grid_summary = {
        "n_timesteps": int(n_timesteps),
        "n_converged_timesteps": n_converged_timesteps,
        "n_failed_timesteps": n_failed_timesteps,
        "n_voltage_buses": int(len(voltage_buses)),
        "n_cables": int(len(cable_values)),
        "transformer_s_rated_mva": float(transformer_s_rated_mva),
        "trafo_loading_p50_time_percent": float(pwrflw._safe_nanpercentile(transformer_loadings, 50)),
        "trafo_loading_p90_time_percent": float(pwrflw._safe_nanpercentile(transformer_loadings, 90)),
        "trafo_loading_p95_time_percent": float(pwrflw._safe_nanpercentile(transformer_loadings, 95)),
        "trafo_loading_p99_time_percent": float(pwrflw._safe_nanpercentile(transformer_loadings, 99)),
        "trafo_loading_max_time_percent": float(pwrflw._safe_nanmax(transformer_loadings)),
        "trafo_loading_hours_above_100": int(np.nansum(transformer_loadings > 100.0)) if transformer_loadings.size else 0,
        "cable_loading_p95_asset_percent": float(pwrflw._safe_nanpercentile(cable_values, 95)),
        "cable_hours_above_100_p95_asset": float(pwrflw._safe_nanpercentile(cable_hours_above_100, 95)) if cable_hours_above_100.size else np.nan,
        "voltage_p05_load_bus_hour_pu": float(pwrflw._safe_nanpercentile(voltage_all, 5)),
        "voltage_hours_below_0_90_p95_asset": float(pwrflw._safe_nanpercentile(voltage_hours_below_0_90, 95)) if voltage_hours_below_0_90.size else np.nan,
    }

    tail_frames = [
        pwrflw._tail_values_frame(
            transformer_loadings.reshape(-1, 1),
            [0],
            metric="Transformer",
            asset_type="transformer",
            tail="upper",
            threshold_percentile=99,
        ),
        pwrflw._tail_values_frame(
            cable_loading_matrix,
            cable_ids.to_numpy(dtype=int),
            metric="Cables",
            asset_type="cable",
            tail="upper",
            threshold_percentile=99,
        ),
        pwrflw._tail_values_frame(
            voltage_matrix,
            voltage_buses.to_numpy(dtype=int),
            metric="Voltage",
            asset_type="bus",
            tail="lower",
            threshold_percentile=1,
        ),
    ]
    tail_frames = [frame for frame in tail_frames if not frame.empty]
    tail_summary = pd.concat(tail_frames, ignore_index=True) if tail_frames else pd.DataFrame(
        columns=["metric", "asset_type", "asset_id", "tail", "threshold_value", "t_index", "value"]
    )

    return {
        "grid_summary": grid_summary,
        "cable_summary": cable_summary,
        "bus_voltage_summary": bus_voltage_summary,
        "tail_summary": tail_summary,
    }


def materialize_powerflow_summaries(args: argparse.Namespace) -> None:
    db = SurroGridDatabase()
    runs = _selected_runs(db, args)
    if not runs:
        raise ValueError("No matching powerflow_run rows found.")

    print(f"selected runs: {len(runs)}")
    written = 0
    skipped = 0
    for row in runs:
        run_id = int(row["powerflow_run_id"])
        for stage in args.stages:
            existing = _existing_summary_rows(db, run_id, stage)
            if existing and not args.replace:
                skipped += 1
                print(
                    f"skip powerflow_run_id={run_id} stage={stage}: "
                    f"{existing} summary rows already exist; use --replace to overwrite."
                )
                continue
            if args.dry_run:
                print(f"dry-run powerflow_run_id={run_id} stage={stage}: would materialize summary")
                continue
            if existing:
                _delete_summary_rows(db, run_id, stage)
            summary = _build_summary_from_raw(db, row, stage)
            db.write_powerflow_summary(run_id, stage, summary)
            written += 1
            grid_summary = summary["grid_summary"]
            print(
                f"wrote powerflow_run_id={run_id} stage={stage}: "
                f"timesteps={grid_summary['n_timesteps']} cables={grid_summary['n_cables']} "
                f"voltage_buses={grid_summary['n_voltage_buses']}"
            )
    print(f"summary stages written: {written}; skipped: {skipped}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize compact synthetic power-flow summaries from existing raw DB time series."
    )
    parser.add_argument("--run-name", required=True, help="Existing raw power-flow run name to summarize.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("pre", "post"),
        default=["post"],
        help="Power-flow stage(s) to summarize from raw tables.",
    )
    parser.add_argument("--scenario-id", type=int, help="Optional scenario_id filter.")
    parser.add_argument("--ags", help="Optional AGS filter.")
    parser.add_argument("--plz", type=int, help="Optional PLZ filter.")
    parser.add_argument("--limit", type=int, help="Optional number of runs to materialize for testing.")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing compact summaries for selected run/stage rows.")
    parser.add_argument("--dry-run", action="store_true", help="List selected summary work without writing rows.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    materialize_powerflow_summaries(args)


if __name__ == "__main__":
    main()
