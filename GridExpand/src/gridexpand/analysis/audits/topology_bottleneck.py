"""Audit topology bottlenecks for critical real-grid voltage cases.

This module is intentionally read-only with respect to the database. It combines
stored real power-flow summaries with the radialized pandapower grid files to
trace the physical path from the transformer/root to the bus with the lowest
voltage. The resulting CSVs help distinguish transformer-scale issues from weak
feeder paths, small attachment cables, and overloaded critical-path cables.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import networkx as nx
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

from common.database import SurroGridDatabase  # noqa: E402


def _load_real_runner():
    spec = importlib.util.spec_from_file_location(
        "run_real_swf_powerflow_topology_audit",
        STEP4_DIR / "run_real_swf_powerflow.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load run_real_swf_powerflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def critical_real_grids(*, real_run_name: str, plz: int, stage: str, voltage_threshold: float) -> pd.DataFrame:
    """Return real grids whose retained voltage buses fall below the threshold."""
    db = SurroGridDatabase()
    query = text(
        """
        WITH critical_bus AS (
            SELECT
                rbv.real_powerflow_run_id,
                rbv.bus AS critical_bus,
                rbv.voltage_min_time_pu,
                rbv.voltage_p01_time_pu,
                rbv.voltage_hours_below_0_90,
                ROW_NUMBER() OVER (
                    PARTITION BY rbv.real_powerflow_run_id
                    ORDER BY rbv.voltage_min_time_pu ASC NULLS LAST, rbv.bus
                ) AS rn
            FROM surrogrid.real_powerflow_bus_voltage_summary rbv
            WHERE rbv.stage = :stage
        )
        SELECT
            rpr.real_powerflow_run_id,
            rgc.real_grid_case_id,
            rgc.plz,
            rgc.lv_id,
            rgc.source_file,
            rps.stage,
            cb.critical_bus,
            cb.voltage_min_time_pu,
            cb.voltage_p01_time_pu,
            cb.voltage_hours_below_0_90,
            rps.n_timesteps,
            rps.n_converged_timesteps,
            rps.n_failed_timesteps,
            rps.transformer_s_rated_mva,
            rps.trafo_loading_p95_time_percent,
            rps.trafo_loading_max_time_percent,
            rps.trafo_loading_hours_above_100,
            rps.cable_loading_p95_asset_percent,
            rps.cable_hours_above_100_p95_asset,
            rps.voltage_p05_load_bus_hour_pu,
            rps.voltage_hours_below_0_90_p95_asset
        FROM critical_bus cb
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        JOIN surrogrid.real_powerflow_summary rps
          ON rps.real_powerflow_run_id = rpr.real_powerflow_run_id
         AND rps.stage = :stage
        WHERE cb.rn = 1
          AND rpr.run_name = :real_run_name
          AND rgc.plz = :plz
          AND cb.voltage_min_time_pu < :voltage_threshold
        ORDER BY cb.voltage_min_time_pu ASC NULLS LAST, rgc.lv_id
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={
                "real_run_name": real_run_name,
                "plz": int(plz),
                "stage": stage,
                "voltage_threshold": float(voltage_threshold),
            },
        )


def _real_cable_summary(run_ids: list[int], *, stage: str) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()
    db = SurroGridDatabase()
    query = text(
        """
        SELECT
            real_powerflow_run_id,
            cable,
            cable_loading_p50_time_percent,
            cable_loading_p90_time_percent,
            cable_loading_p95_time_percent,
            cable_loading_p99_time_percent,
            cable_loading_max_time_percent,
            cable_loading_hours_above_100,
            cable_max_i_ka,
            cable_parallel,
            cable_installed_capacity_ka
        FROM surrogrid.real_powerflow_cable_summary
        WHERE stage = :stage
          AND real_powerflow_run_id = ANY(:run_ids)
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"stage": stage, "run_ids": [int(v) for v in run_ids]})


def _line_ids_by_edge(grid, active_line_ids: list[int]) -> dict[frozenset[int], list[int]]:
    by_edge: dict[frozenset[int], list[int]] = {}
    for line_id, line in grid.line.loc[active_line_ids].iterrows():
        edge = frozenset((int(line["from_bus"]), int(line["to_bus"])))
        by_edge.setdefault(edge, []).append(int(line_id))
    return by_edge


def _active_bus_graph(grid, active_line_ids: list[int]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from([int(bus) for bus in grid.bus.index])
    for line_id, line in grid.line.loc[active_line_ids].iterrows():
        graph.add_edge(int(line["from_bus"]), int(line["to_bus"]), line_ids=[int(line_id)])
    if hasattr(grid, "switch") and not grid.switch.empty:
        switches = grid.switch
        if "closed" in switches.columns:
            switches = switches[switches["closed"].fillna(True).astype(bool)]
        if "et" in switches.columns:
            switches = switches[switches["et"].astype(str).eq("b")]
        for _, switch in switches.iterrows():
            graph.add_edge(int(switch["bus"]), int(switch["element"]), line_ids=[])
    return graph


def _path_pairs_between(active_graph: nx.Graph, from_bus: int, to_bus: int) -> list[tuple[int, int]]:
    try:
        bus_path = nx.shortest_path(active_graph, int(from_bus), int(to_bus))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return [(int(a), int(b)) for a, b in zip(bus_path[:-1], bus_path[1:])]


def _line_impedance_from_row(line: pd.Series) -> tuple[float, float, float]:
    length_km = abs(_line_float(line, "length_km", 0.0))
    parallel = _line_float(line, "parallel", 1.0)
    if not np.isfinite(parallel) or parallel <= 0:
        parallel = 1.0
    r_ohm = _line_float(line, "r_ohm_per_km", 0.0) * length_km / parallel
    x_ohm = _line_float(line, "x_ohm_per_km", 0.0) * length_km / parallel
    return r_ohm, x_ohm, float(np.hypot(r_ohm, x_ohm))


def _inactive_line_reason(grid, line_id: int) -> str:
    reasons = []
    if "in_service" in grid.line.columns and not bool(grid.line.at[line_id, "in_service"]):
        value = grid.line.at[line_id, "split_removed_reason"] if "split_removed_reason" in grid.line.columns else None
        reasons.append(str(value) if pd.notna(value) else "line_out_of_service")
    if hasattr(grid, "switch") and not grid.switch.empty and {"et", "element", "closed"}.issubset(grid.switch.columns):
        mask = (
            grid.switch["et"].astype(str).eq("l")
            & grid.switch["element"].dropna().astype(int).reindex(grid.switch.index, fill_value=-1).eq(int(line_id))
            & ~grid.switch["closed"].fillna(True).astype(bool)
        )
        if mask.any():
            reasons.append("open_line_switch")
    return "+".join(dict.fromkeys(reason for reason in reasons if reason and reason != "None")) or "inactive"


def _line_flags_from_row(line: pd.Series, *, max_i_ka: float | None = None) -> tuple[bool, bool, bool, bool]:
    name = _line_text(line, "name")
    std_type = _line_text(line, "std_type")
    text_blob = f"{name} {std_type}".lower()
    service_like = any(token in text_blob for token in ("kban", "haan", "anschluss", "service"))
    small_named = any(token in text_blob for token in ("4x35", "4_35", "3x35", "3_35", "35sm"))
    if max_i_ka is None:
        max_i_ka = _line_float(line, "max_i_ka")
    small_capacity = np.isfinite(max_i_ka) and float(max_i_ka) < 0.18
    missing_attachment = "missing" in text_blob or "synthetic" in text_blob
    return service_like, small_named, small_capacity, missing_attachment


def _path_bus_pairs(parents: dict[int, int | None], target_bus: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    bus = int(target_bus)
    seen: set[int] = set()
    while bus in parents and bus not in seen:
        seen.add(bus)
        parent = parents[bus]
        if parent is None:
            break
        pairs.append((int(parent), int(bus)))
        bus = int(parent)
    pairs.reverse()
    return pairs


def _line_float(line: pd.Series, column: str, default: float = math.nan) -> float:
    if column not in line.index:
        return default
    value = pd.to_numeric(pd.Series([line[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else default


def _line_text(line: pd.Series, column: str) -> str:
    if column not in line.index or pd.isna(line[column]):
        return ""
    return str(line[column])


def _line_record(
    *,
    lv_id: str,
    run_id: int,
    critical_bus: int,
    order: int,
    from_bus: int,
    to_bus: int,
    line_id: int,
    line: pd.Series,
    cable_summary: pd.DataFrame,
) -> dict[str, Any]:
    length_km = abs(_line_float(line, "length_km", 0.0))
    parallel = _line_float(line, "parallel", 1.0)
    if not np.isfinite(parallel) or parallel <= 0:
        parallel = 1.0
    r_ohm_per_km = _line_float(line, "r_ohm_per_km", 0.0)
    x_ohm_per_km = _line_float(line, "x_ohm_per_km", 0.0)
    r_ohm = r_ohm_per_km * length_km / parallel
    x_ohm = x_ohm_per_km * length_km / parallel
    name = _line_text(line, "name")
    std_type = _line_text(line, "std_type")
    text_blob = f"{name} {std_type}".lower()
    solver_max_i_ka = _line_float(line, "max_i_ka")
    service_like = any(token in text_blob for token in ("kban", "haan", "anschluss", "service"))
    small_named = any(token in text_blob for token in ("4x35", "4_35", "3x35", "3_35", "35sm"))
    # _prepare_real_grid relaxes grid.line.max_i_ka to 1000 for solving. The
    # stored cable summary keeps the real pre-relaxation capacity and is used
    # below to identify genuinely small-capacity path segments.
    small_capacity = False
    missing_attachment = "missing" in text_blob or "synthetic" in text_blob

    row = {
        "lv_id": lv_id,
        "real_powerflow_run_id": int(run_id),
        "critical_bus": int(critical_bus),
        "path_order": int(order),
        "line_id": int(line_id),
        "from_bus": int(from_bus),
        "to_bus": int(to_bus),
        "line_name": name,
        "std_type": std_type,
        "length_km": length_km,
        "parallel": parallel,
        "max_i_ka_from_grid": solver_max_i_ka,
        "r_ohm_per_km": r_ohm_per_km,
        "x_ohm_per_km": x_ohm_per_km,
        "z_ohm_per_km": float(np.hypot(r_ohm_per_km, x_ohm_per_km)),
        "path_r_ohm": r_ohm,
        "path_x_ohm": x_ohm,
        "path_z_ohm": float(np.hypot(r_ohm, x_ohm)),
        "is_service_like": bool(service_like),
        "is_small_named_cable": bool(small_named),
        "is_small_capacity_cable": bool(small_capacity),
        "is_missing_attachment": bool(missing_attachment),
    }
    if not cable_summary.empty:
        match = cable_summary[cable_summary["cable"].astype(int).eq(int(line_id))]
        if not match.empty:
            for column in [
                "cable_loading_p50_time_percent",
                "cable_loading_p90_time_percent",
                "cable_loading_p95_time_percent",
                "cable_loading_p99_time_percent",
                "cable_loading_max_time_percent",
                "cable_loading_hours_above_100",
                "cable_max_i_ka",
                "cable_parallel",
                "cable_installed_capacity_ka",
            ]:
                row[column] = match.iloc[0].get(column)
            stored_capacity = pd.to_numeric(pd.Series([match.iloc[0].get("cable_max_i_ka")]), errors="coerce").iloc[0]
            if pd.notna(stored_capacity):
                row["is_small_capacity_cable"] = bool(float(stored_capacity) < 0.18)
    return row


def _candidate_alternative_lines(
    *,
    lv_id: str,
    run_id: int,
    grid,
    original_line_table: pd.DataFrame,
    active_line_ids: list[int],
    path_lines: pd.DataFrame,
) -> pd.DataFrame:
    if path_lines.empty:
        return pd.DataFrame()

    active_graph = _active_bus_graph(grid, active_line_ids)
    line_ids_by_edge = _line_ids_by_edge(grid, active_line_ids)
    critical_path_line_ids = {int(value) for value in path_lines["line_id"].dropna().astype(int)}
    path_by_line = path_lines.set_index("line_id", drop=False)
    active_line_set = set(active_line_ids)
    records: list[dict[str, Any]] = []

    for line_id, line in grid.line.iterrows():
        line_id = int(line_id)
        if line_id in active_line_set:
            continue
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        if from_bus not in active_graph or to_bus not in active_graph:
            continue
        cycle_pairs = _path_pairs_between(active_graph, from_bus, to_bus)
        if not cycle_pairs:
            continue

        bypass_line_ids: list[int] = []
        for edge in cycle_pairs:
            for active_line_id in line_ids_by_edge.get(frozenset(edge), []):
                if int(active_line_id) in critical_path_line_ids:
                    bypass_line_ids.append(int(active_line_id))
        if not bypass_line_ids:
            continue

        bypass_rows = path_by_line.loc[bypass_line_ids]
        if isinstance(bypass_rows, pd.Series):
            bypass_rows = bypass_rows.to_frame().T

        original_line = original_line_table.loc[line_id] if line_id in original_line_table.index else line
        candidate_r, candidate_x, candidate_z = _line_impedance_from_row(original_line)
        candidate_max_i = _line_float(original_line, "max_i_ka")
        service_like, small_named, small_capacity, missing_attachment = _line_flags_from_row(original_line, max_i_ka=candidate_max_i)
        bypass_r = float(bypass_rows["path_r_ohm"].sum()) if "path_r_ohm" in bypass_rows else np.nan
        bypass_z = float(bypass_rows["path_z_ohm"].sum()) if "path_z_ohm" in bypass_rows else np.nan
        bypass_max_loading = (
            float(bypass_rows["cable_loading_max_time_percent"].max())
            if "cable_loading_max_time_percent" in bypass_rows
            else np.nan
        )
        records.append(
            {
                "lv_id": lv_id,
                "real_powerflow_run_id": int(run_id),
                "candidate_line_id": int(line_id),
                "candidate_line_name": _line_text(original_line, "name"),
                "candidate_std_type": _line_text(original_line, "std_type"),
                "candidate_from_bus": from_bus,
                "candidate_to_bus": to_bus,
                "inactive_reason": _inactive_line_reason(grid, line_id),
                "candidate_length_km": abs(_line_float(original_line, "length_km", 0.0)),
                "candidate_max_i_ka": candidate_max_i,
                "candidate_r_ohm": candidate_r,
                "candidate_x_ohm": candidate_x,
                "candidate_z_ohm": candidate_z,
                "candidate_is_service_like": bool(service_like),
                "candidate_is_small_named_cable": bool(small_named),
                "candidate_is_small_capacity_cable": bool(small_capacity),
                "candidate_is_missing_attachment": bool(missing_attachment),
                "active_cycle_bus_edges": len(cycle_pairs),
                "bypassed_critical_line_count": len(bypass_line_ids),
                "bypassed_critical_line_ids": ",".join(str(value) for value in sorted(set(bypass_line_ids))),
                "bypassed_path_r_ohm": bypass_r,
                "bypassed_path_z_ohm": bypass_z,
                "bypassed_max_loading_percent": bypass_max_loading,
                "bypassed_service_or_small_count": int(
                    (
                        bypass_rows.get("is_service_like", pd.Series(dtype=bool))
                        | bypass_rows.get("is_small_named_cable", pd.Series(dtype=bool))
                        | bypass_rows.get("is_small_capacity_cable", pd.Series(dtype=bool))
                    ).sum()
                ),
                "candidate_vs_bypassed_r_ratio": candidate_r / bypass_r if bypass_r else np.nan,
                "candidate_vs_bypassed_z_ratio": candidate_z / bypass_z if bypass_z else np.nan,
                "candidate_likely_relief": bool(
                    (np.isfinite(bypass_max_loading) and bypass_max_loading >= 100.0)
                    or candidate_z < bypass_z
                    or int((bypass_rows.get("is_service_like", pd.Series(dtype=bool)) | bypass_rows.get("is_small_capacity_cable", pd.Series(dtype=bool))).sum()) > 0
                ),
            }
        )

    alternatives = pd.DataFrame(records)
    if not alternatives.empty:
        alternatives = alternatives.sort_values(
            ["candidate_likely_relief", "bypassed_max_loading_percent", "candidate_vs_bypassed_z_ratio"],
            ascending=[False, False, True],
            na_position="last",
        )
    return alternatives


def _diagnosis(summary: dict[str, Any]) -> str:
    trafo = summary.get("trafo_loading_max_time_percent", np.nan)
    path_max = summary.get("critical_path_cable_loading_max_percent", np.nan)
    small = int(summary.get("critical_path_small_or_service_line_count", 0) or 0)
    missing = int(summary.get("critical_path_missing_attachment_line_count", 0) or 0)
    if np.isfinite(trafo) and trafo >= 100.0:
        return "transformer_overload"
    if np.isfinite(path_max) and path_max >= 100.0 and small > 0:
        return "service_or_small_cable_path_overload"
    if np.isfinite(path_max) and path_max >= 100.0:
        return "critical_path_cable_overload"
    if small > 0:
        return "small_or_service_like_critical_path"
    if missing > 0:
        return "synthetic_attachment_on_critical_path"
    return "weak_path_or_voltage_drop"


def audit_one_grid(
    row: dict[str, Any],
    cable_summary: pd.DataFrame,
    real_runner,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    source_file = Path(row["source_file"])
    net = pp.from_excel(source_file)
    original_line_table = net.line.copy()
    grid, _rated, _max_i, _voltage_buses, _backbone_cables, _loads, load_scope = real_runner._prepare_real_grid(net)

    pwrflw = real_runner.pwrflw
    active_line_ids = [int(value) for value in pwrflw._active_line_index(grid)]
    adjacency = pwrflw._grid_adjacency(grid)
    root_bus = int(pwrflw._root_bus(grid))
    critical_bus = int(row["critical_bus"])
    parents = pwrflw._parent_tree_from_root(adjacency, root_bus)
    path_pairs = _path_bus_pairs(parents, critical_bus)
    line_ids_by_edge = _line_ids_by_edge(grid, active_line_ids)

    line_rows: list[dict[str, Any]] = []
    path_line_ids: list[int] = []
    for order, (from_bus, to_bus) in enumerate(path_pairs, start=1):
        line_ids = line_ids_by_edge.get(frozenset((from_bus, to_bus)), [])
        for line_id in line_ids:
            path_line_ids.append(int(line_id))
            line_rows.append(
                _line_record(
                    lv_id=str(row["lv_id"]),
                    run_id=int(row["real_powerflow_run_id"]),
                    critical_bus=critical_bus,
                    order=order,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    line_id=int(line_id),
                    line=grid.line.loc[int(line_id)],
                    cable_summary=cable_summary,
                )
            )

    path_lines = pd.DataFrame(line_rows)
    if not path_lines.empty:
        path_lines["path_r_share"] = path_lines["path_r_ohm"] / path_lines["path_r_ohm"].sum() if path_lines["path_r_ohm"].sum() else np.nan
        path_lines["path_z_share"] = path_lines["path_z_ohm"] / path_lines["path_z_ohm"].sum() if path_lines["path_z_ohm"].sum() else np.nan
        worst_r = path_lines.sort_values("path_r_ohm", ascending=False).iloc[0]
        worst_loading = path_lines.sort_values("cable_loading_max_time_percent", ascending=False, na_position="last").iloc[0]
    else:
        worst_r = pd.Series(dtype=object)
        worst_loading = pd.Series(dtype=object)

    summary: dict[str, Any] = {
        **row,
        **load_scope,
        "source_file": str(source_file),
        "root_bus": root_bus,
        "path_found": bool(critical_bus in parents),
        "path_bus_edges": int(len(path_pairs)),
        "critical_path_line_rows": int(len(path_lines)),
        "critical_path_parallel_line_ids": ",".join(str(value) for value in path_line_ids),
        "critical_path_length_km": float(path_lines["length_km"].sum()) if not path_lines.empty else np.nan,
        "critical_path_r_ohm": float(path_lines["path_r_ohm"].sum()) if not path_lines.empty else np.nan,
        "critical_path_x_ohm": float(path_lines["path_x_ohm"].sum()) if not path_lines.empty else np.nan,
        "critical_path_z_ohm": float(path_lines["path_z_ohm"].sum()) if not path_lines.empty else np.nan,
        "critical_path_cable_loading_max_percent": float(path_lines["cable_loading_max_time_percent"].max()) if "cable_loading_max_time_percent" in path_lines else np.nan,
        "critical_path_cable_loading_p95_max_percent": float(path_lines["cable_loading_p95_time_percent"].max()) if "cable_loading_p95_time_percent" in path_lines else np.nan,
        "critical_path_overloaded_line_count": int((path_lines.get("cable_loading_max_time_percent", pd.Series(dtype=float)) >= 100.0).sum()),
        "critical_path_small_or_service_line_count": int((path_lines.get("is_service_like", pd.Series(dtype=bool)) | path_lines.get("is_small_named_cable", pd.Series(dtype=bool)) | path_lines.get("is_small_capacity_cable", pd.Series(dtype=bool))).sum()),
        "critical_path_missing_attachment_line_count": int(path_lines.get("is_missing_attachment", pd.Series(dtype=bool)).sum()),
        "largest_r_line_id": worst_r.get("line_id", np.nan),
        "largest_r_line_name": worst_r.get("line_name", ""),
        "largest_r_std_type": worst_r.get("std_type", ""),
        "largest_r_length_km": worst_r.get("length_km", np.nan),
        "largest_r_ohm": worst_r.get("path_r_ohm", np.nan),
        "largest_r_share": worst_r.get("path_r_share", np.nan),
        "highest_loading_line_id": worst_loading.get("line_id", np.nan),
        "highest_loading_line_name": worst_loading.get("line_name", ""),
        "highest_loading_std_type": worst_loading.get("std_type", ""),
        "highest_loading_percent": worst_loading.get("cable_loading_max_time_percent", np.nan),
    }
    alternatives = _candidate_alternative_lines(
        lv_id=str(row["lv_id"]),
        run_id=int(row["real_powerflow_run_id"]),
        grid=grid,
        original_line_table=original_line_table,
        active_line_ids=active_line_ids,
        path_lines=path_lines,
    )
    summary["alternative_line_count_touching_critical_path"] = int(len(alternatives))
    summary["likely_relief_alternative_count"] = int(alternatives["candidate_likely_relief"].sum()) if not alternatives.empty else 0
    if not alternatives.empty:
        best = alternatives.iloc[0]
        summary["best_alternative_line_id"] = best.get("candidate_line_id")
        summary["best_alternative_line_name"] = best.get("candidate_line_name")
        summary["best_alternative_std_type"] = best.get("candidate_std_type")
        summary["best_alternative_reason"] = best.get("inactive_reason")
        summary["best_alternative_z_ratio"] = best.get("candidate_vs_bypassed_z_ratio")
        summary["best_alternative_bypassed_max_loading_percent"] = best.get("bypassed_max_loading_percent")
    summary["bottleneck_diagnosis"] = _diagnosis(summary)
    return summary, path_lines, alternatives


def audit_critical_topology(
    *,
    real_run_name: str,
    plz: int,
    stage: str,
    voltage_threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    critical = critical_real_grids(
        real_run_name=real_run_name,
        plz=plz,
        stage=stage,
        voltage_threshold=voltage_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if critical.empty:
        empty_summary = pd.DataFrame()
        empty_lines = pd.DataFrame()
        empty_summary.to_csv(output_dir / "critical_grid_summary.csv", index=False)
        empty_lines.to_csv(output_dir / "critical_path_lines.csv", index=False)
        return empty_summary, empty_lines

    cable_summary = _real_cable_summary(critical["real_powerflow_run_id"].astype(int).tolist(), stage=stage)
    real_runner = _load_real_runner()
    summaries: list[dict[str, Any]] = []
    path_tables: list[pd.DataFrame] = []
    alternative_tables: list[pd.DataFrame] = []
    for row in critical.to_dict("records"):
        run_id = int(row["real_powerflow_run_id"])
        grid_cables = cable_summary[cable_summary["real_powerflow_run_id"].astype(int).eq(run_id)]
        try:
            summary, path_lines, alternatives = audit_one_grid(row, grid_cables, real_runner)
        except Exception as exc:  # keep the batch useful when one source file is malformed
            summary = {**row, "path_found": False, "audit_error": repr(exc), "bottleneck_diagnosis": "audit_failed"}
            path_lines = pd.DataFrame()
            alternatives = pd.DataFrame()
        summaries.append(summary)
        if not path_lines.empty:
            path_tables.append(path_lines)
        if not alternatives.empty:
            alternative_tables.append(alternatives)

    summary_df = pd.DataFrame(summaries)
    path_df = pd.concat(path_tables, ignore_index=True) if path_tables else pd.DataFrame()
    alternative_df = pd.concat(alternative_tables, ignore_index=True) if alternative_tables else pd.DataFrame()
    summary_df.to_csv(output_dir / "critical_grid_summary.csv", index=False)
    path_df.to_csv(output_dir / "critical_path_lines.csv", index=False)
    alternative_df.to_csv(output_dir / "critical_path_alternative_lines.csv", index=False)
    return summary_df, path_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit topology bottlenecks in critical real-grid voltage cases.")
    parser.add_argument("--real-run-name", default="real_hybrid")
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--stage", default="pre")
    parser.add_argument("--voltage-threshold", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=POSTPROCESSING_DIR / "output" / "audits" / "topology",
    )
    args = parser.parse_args()

    summary, path_lines = audit_critical_topology(
        real_run_name=args.real_run_name,
        plz=args.plz,
        stage=args.stage,
        voltage_threshold=args.voltage_threshold,
        output_dir=args.output_dir,
    )
    print(f"Critical grids below {args.voltage_threshold:.2f} p.u.: {len(summary)}")
    if not summary.empty:
        columns = [
            "lv_id",
            "voltage_min_time_pu",
            "critical_bus",
            "trafo_loading_max_time_percent",
            "critical_path_cable_loading_max_percent",
            "critical_path_small_or_service_line_count",
            "critical_path_missing_attachment_line_count",
            "bottleneck_diagnosis",
        ]
        print(summary[[col for col in columns if col in summary.columns]].to_string(index=False))
    print(f"Wrote {args.output_dir / 'critical_grid_summary.csv'}")
    alternative_path = args.output_dir / "critical_path_alternative_lines.csv"
    alternative_count = len(pd.read_csv(alternative_path)) if alternative_path.exists() and alternative_path.stat().st_size else 0
    print(f"Wrote {args.output_dir / 'critical_path_lines.csv'} ({len(path_lines)} path-line rows)")
    print(f"Wrote {alternative_path} ({alternative_count} alternative-line rows)")


if __name__ == "__main__":
    main()
