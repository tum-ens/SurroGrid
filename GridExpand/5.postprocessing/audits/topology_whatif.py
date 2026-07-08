"""Run read-only topology what-if checks for critical real-grid voltage cases."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd
from sqlalchemy import text

POSTPROCESSING_DIR = Path(__file__).resolve().parents[1]
GRIDEXPAND_DIR = POSTPROCESSING_DIR.parents[0]
STEP4_DIR = GRIDEXPAND_DIR / "4.powerflow"

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))

from common.database import SurroGridDatabase

try:
    from .topology_bottleneck import critical_real_grids
except ImportError:
    from topology_bottleneck import critical_real_grids


def _load_real_runner():
    spec = importlib.util.spec_from_file_location(
        "run_real_swf_powerflow_topology_whatif",
        STEP4_DIR / "run_real_swf_powerflow.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load run_real_swf_powerflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assumptions_by_run(run_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not run_ids:
        return {}
    db = SurroGridDatabase()
    query = text(
        """
        SELECT real_powerflow_run_id, assumptions
        FROM surrogrid.real_powerflow_run
        WHERE real_powerflow_run_id = ANY(:run_ids)
        """
    )
    with db.engine.connect() as conn:
        rows = pd.read_sql_query(query, conn, params={"run_ids": [int(value) for value in run_ids]})
    return {int(row.real_powerflow_run_id): dict(row.assumptions or {}) for row in rows.itertuples(index=False)}


def _close_line_switches(grid, line_id: int) -> None:
    if hasattr(grid, "switch") and not grid.switch.empty and {"et", "element", "closed"}.issubset(grid.switch.columns):
        mask = (
            grid.switch["et"].astype(str).eq("l")
            & grid.switch["element"].dropna().astype(int).reindex(grid.switch.index, fill_value=-1).eq(int(line_id))
        )
        grid.switch.loc[mask, "closed"] = True


def _real_capacity_series(original_line_table: pd.DataFrame) -> pd.Series:
    if "max_i_ka" not in original_line_table.columns:
        return pd.Series(np.nan, index=original_line_table.index, name="max_i_ka")
    return pd.to_numeric(original_line_table["max_i_ka"], errors="coerce")


def _parse_line_ids(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    ids: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def _select_swap_open_line(best_alternative: pd.Series, path_lines: pd.DataFrame) -> int | None:
    bypassed_ids = _parse_line_ids(best_alternative.get("bypassed_critical_line_ids"))
    if not bypassed_ids or path_lines.empty:
        return None
    candidates = path_lines[path_lines["line_id"].astype(int).isin(bypassed_ids)].copy()
    if candidates.empty:
        return bypassed_ids[0]
    candidates = candidates.sort_values(
        ["cable_loading_max_time_percent", "path_r_ohm"],
        ascending=[False, False],
        na_position="last",
    )
    return int(candidates.iloc[0]["line_id"])


def _summary_record(
    *,
    lv_id: str,
    run_id: int,
    variant: str,
    candidate_line_id: int | None,
    opened_line_id: int | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    grid_summary = summary["grid_summary"]
    bus_summary = summary.get("bus_voltage_summary", pd.DataFrame())
    cable_summary = summary.get("cable_summary", pd.DataFrame())
    return {
        "lv_id": lv_id,
        "real_powerflow_run_id": int(run_id),
        "variant": variant,
        "candidate_line_id": candidate_line_id,
        "opened_line_id": opened_line_id,
        "n_timesteps": grid_summary.get("n_timesteps"),
        "n_failed_timesteps": grid_summary.get("n_failed_timesteps"),
        "voltage_min_time_pu": float(bus_summary["voltage_min_time_pu"].min()) if not bus_summary.empty else np.nan,
        "voltage_p05_load_bus_hour_pu": grid_summary.get("voltage_p05_load_bus_hour_pu"),
        "voltage_hours_below_0_90_p95_asset": grid_summary.get("voltage_hours_below_0_90_p95_asset"),
        "trafo_loading_max_time_percent": grid_summary.get("trafo_loading_max_time_percent"),
        "cable_loading_max_asset_percent": float(cable_summary["cable_loading_max_time_percent"].max()) if not cable_summary.empty else np.nan,
        "cable_loading_p95_asset_percent": grid_summary.get("cable_loading_p95_asset_percent"),
    }


def _select_peak_demand_timesteps(demand: pd.DataFrame, max_timesteps: int | None) -> pd.DataFrame:
    if max_timesteps is None or max_timesteps <= 0 or len(demand) <= max_timesteps:
        return demand
    electricity_columns = [column for column in demand.columns if column[1] == "electricity"]
    if not electricity_columns:
        return demand.head(max_timesteps)
    peak_index = demand[electricity_columns].sum(axis=1).nlargest(max_timesteps).index
    return demand.loc[peak_index].sort_index()


def _run_variant(
    *,
    row: dict[str, Any],
    real_runner,
    assumptions: dict[str, Any],
    candidate_line_id: int | None = None,
    opened_line_id: int | None = None,
    variant: str,
    max_timesteps: int | None = None,
) -> dict[str, Any]:
    source_file = Path(row["source_file"])
    net = pp.from_excel(source_file)
    original_line_table = net.line.copy()
    (
        grid,
        transformer_s_rated_mva,
        _cable_max_i_ka,
        _voltage_buses,
        _backbone_cable_ids,
        selected_household_loads,
        _load_scope,
    ) = real_runner._prepare_real_grid(net)

    if candidate_line_id is not None:
        grid.line.at[int(candidate_line_id), "in_service"] = True
        _close_line_switches(grid, int(candidate_line_id))
    if opened_line_id is not None:
        grid.line.at[int(opened_line_id), "in_service"] = False

    load_buses = selected_household_loads["bus"].dropna().astype(int).drop_duplicates().tolist()
    backbone_cable_ids, voltage_buses = real_runner.pwrflw.comparison_backbone_scope(grid, load_buses)
    if not voltage_buses:
        voltage_buses = load_buses

    annual_demand_mode = assumptions.get("annual_demand_mode", real_runner.ANNUAL_DEMAND_MODE_SYNTHETIC)
    measured_profile_selection = assumptions.get(
        "measured_profile_selection",
        real_runner.MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    )
    measured_profile_band_pct = float(
        assumptions.get("measured_profile_band_pct", real_runner.DEFAULT_MEASURED_PROFILE_BAND_PCT)
    )
    measured_profile_min_candidates = int(
        assumptions.get("measured_profile_min_candidates", real_runner.DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES)
    )
    seed = int(assumptions.get("profile_seed", 91301))
    demand = real_runner._build_real_electric_demand(
        net,
        seed=seed,
        load_rows=selected_household_loads,
        annual_demand_mode=annual_demand_mode,
        measured_profile_selection=measured_profile_selection,
        measured_profile_band_pct=measured_profile_band_pct,
        measured_profile_min_candidates=measured_profile_min_candidates,
        return_audit=False,
    )
    demand = _select_peak_demand_timesteps(demand, max_timesteps)
    summary = real_runner.pwrflw.pf_summary(
        grid,
        demand,
        transformer_s_rated_mva=transformer_s_rated_mva,
        cable_max_i_ka=_real_capacity_series(original_line_table),
        voltage_buses=voltage_buses,
        algorithm=["nr", "iwamoto_nr"],
        cable_ids=backbone_cable_ids,
        on_nonconvergence="nan",
        protect_grid_state=True,
    )
    return _summary_record(
        lv_id=str(row["lv_id"]),
        run_id=int(row["real_powerflow_run_id"]),
        variant=variant,
        candidate_line_id=candidate_line_id,
        opened_line_id=opened_line_id,
        summary=summary,
    )


def run_topology_whatifs(
    *,
    real_run_name: str,
    plz: int,
    stage: str,
    voltage_threshold: float,
    audit_dir: Path,
    output_file: Path,
    lv_ids: list[str] | None = None,
    max_timesteps: int | None = None,
) -> pd.DataFrame:
    critical = critical_real_grids(
        real_run_name=real_run_name,
        plz=plz,
        stage=stage,
        voltage_threshold=voltage_threshold,
    )
    if lv_ids:
        wanted = {str(value) for value in lv_ids}
        critical = critical[critical["lv_id"].astype(str).isin(wanted)]
    alternatives = pd.read_csv(audit_dir / "critical_path_alternative_lines.csv")
    path_lines = pd.read_csv(audit_dir / "critical_path_lines.csv")
    assumptions = _assumptions_by_run(critical["real_powerflow_run_id"].astype(int).tolist())
    real_runner = _load_real_runner()

    records: list[dict[str, Any]] = []
    for row in critical.to_dict("records"):
        lv_id = str(row["lv_id"])
        run_id = int(row["real_powerflow_run_id"])
        grid_alternatives = alternatives[alternatives["lv_id"].astype(str).eq(lv_id)].copy()
        if grid_alternatives.empty:
            continue
        grid_alternatives = grid_alternatives.sort_values(
            ["candidate_likely_relief", "bypassed_max_loading_percent", "candidate_vs_bypassed_z_ratio"],
            ascending=[False, False, True],
            na_position="last",
        )
        best = grid_alternatives.iloc[0]
        candidate_line_id = int(best["candidate_line_id"])
        grid_path_lines = path_lines[path_lines["lv_id"].astype(str).eq(lv_id)]
        swap_open_line_id = _select_swap_open_line(best, grid_path_lines)
        run_assumptions = assumptions.get(run_id, {})
        records.append(
            _run_variant(
                row=row,
                real_runner=real_runner,
                assumptions=run_assumptions,
                candidate_line_id=None,
                opened_line_id=None,
                variant="baseline_radialized",
                max_timesteps=max_timesteps,
            )
        )
        records.append(
            _run_variant(
                row=row,
                real_runner=real_runner,
                assumptions=run_assumptions,
                candidate_line_id=candidate_line_id,
                opened_line_id=None,
                variant="close_best_alternative_meshed",
                max_timesteps=max_timesteps,
            )
        )
        if swap_open_line_id is not None:
            records.append(
                _run_variant(
                    row=row,
                    real_runner=real_runner,
                    assumptions=run_assumptions,
                    candidate_line_id=candidate_line_id,
                    opened_line_id=swap_open_line_id,
                    variant="radial_swap_open_worst_bypassed",
                    max_timesteps=max_timesteps,
                )
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output_file, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local topology what-if power-flow checks for critical real grids.")
    parser.add_argument("--real-run-name", default="real_hybrid")
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--stage", default="pre")
    parser.add_argument("--voltage-threshold", type=float, default=0.90)
    parser.add_argument("--audit-dir", type=Path, default=POSTPROCESSING_DIR / "output" / "topology_bottleneck_audit")
    parser.add_argument("--output-file", type=Path, default=POSTPROCESSING_DIR / "output" / "topology_bottleneck_audit" / "critical_path_whatif.csv")
    parser.add_argument("--lv-id", action="append", dest="lv_ids")
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=168,
        help="Run only the highest aggregate-demand timesteps. Use 0 for the full year.",
    )
    args = parser.parse_args()
    result = run_topology_whatifs(
        real_run_name=args.real_run_name,
        plz=args.plz,
        stage=args.stage,
        voltage_threshold=args.voltage_threshold,
        audit_dir=args.audit_dir,
        output_file=args.output_file,
        lv_ids=args.lv_ids,
        max_timesteps=None if args.max_timesteps == 0 else args.max_timesteps,
    )
    print(f"Wrote {args.output_file} ({len(result)} rows)")
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
