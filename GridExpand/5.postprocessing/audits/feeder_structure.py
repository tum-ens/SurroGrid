"""Compare feeder structure and cable utilization for synthetic and real grids.

The audit normalizes pandapower line tables in two steps. Parallel active line
rows between the same buses form one electrical edge. Consecutive edges through
degree-two buses without allocated demand then form one feeder section. This
keeps line-table segmentation from being mistaken for a topology difference.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402


def _normalized_ags(value: str | int) -> int:
    return int(str(value).strip().lstrip("0") or "0")


def _active_line_ids(net: pp.pandapowerNet) -> pd.Index:
    active = pd.Series(True, index=net.line.index)
    if "in_service" in net.line.columns:
        active &= net.line["in_service"].fillna(True).astype(bool)
    if not net.switch.empty and {"et", "element", "closed"}.issubset(net.switch.columns):
        open_lines = net.switch.loc[
            net.switch["et"].astype(str).eq("l")
            & ~net.switch["closed"].fillna(True).astype(bool),
            "element",
        ].dropna()
        active.loc[active.index.intersection(open_lines.astype(int))] = False
    return pd.Index(net.line.index[active], dtype=int)


def _root_bus(net: pp.pandapowerNet) -> int:
    if not net.trafo.empty and "lv_bus" in net.trafo.columns:
        roots = net.trafo.loc[
            net.trafo.get("in_service", pd.Series(True, index=net.trafo.index))
            .fillna(True)
            .astype(bool),
            "lv_bus",
        ].dropna().astype(int).unique()
        if len(roots) == 1:
            return int(roots[0])
        if len(roots) > 1:
            raise ValueError(f"Expected one LV transformer root, found {roots.tolist()}.")
    if not net.ext_grid.empty and "bus" in net.ext_grid.columns:
        return int(net.ext_grid.iloc[0]["bus"])
    raise ValueError("Grid has neither an active LV transformer nor an external-grid root.")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _line_bundle(net: pp.pandapowerNet, line_ids: list[int]) -> dict[str, Any]:
    lines = net.line.loc[line_ids]
    lengths = pd.to_numeric(lines.get("length_km"), errors="coerce").dropna()
    length_km = float(lengths.median()) if not lengths.empty else 0.0
    capacities = (
        pd.to_numeric(lines.get("max_i_ka"), errors="coerce")
        * pd.to_numeric(lines.get("parallel", 1.0), errors="coerce").fillna(1.0)
    )
    capacity_a = float(capacities.replace([np.inf, -np.inf], np.nan).sum(min_count=1) * 1000.0)

    admittance = 0.0j
    for _, line in lines.iterrows():
        row_length = max(_finite_float(line.get("length_km")), 0.0)
        parallel = max(_finite_float(line.get("parallel"), 1.0), 1.0)
        impedance = complex(
            _finite_float(line.get("r_ohm_per_km")) * row_length,
            _finite_float(line.get("x_ohm_per_km")) * row_length,
        ) / parallel
        if abs(impedance) > 0:
            admittance += 1.0 / impedance
    equivalent_impedance = abs(1.0 / admittance) if abs(admittance) > 0 else 0.0
    path_weight = equivalent_impedance if equivalent_impedance > 0 else max(length_km, 1e-9)
    return {
        "line_ids": tuple(sorted(map(int, line_ids))),
        "line_row_count": int(len(line_ids)),
        "length_km": length_km,
        "capacity_a": capacity_a,
        "impedance_ohm": float(equivalent_impedance),
        "path_weight": float(path_weight),
    }


def _electrical_graph(net: pp.pandapowerNet) -> nx.Graph:
    graph = nx.Graph()
    line_ids_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for line_id, line in net.line.loc[_active_line_ids(net)].iterrows():
        edge = tuple(sorted((int(line["from_bus"]), int(line["to_bus"]))))
        line_ids_by_edge[edge].append(int(line_id))
    for (from_bus, to_bus), line_ids in line_ids_by_edge.items():
        graph.add_edge(from_bus, to_bus, edge_kind="line", **_line_bundle(net, line_ids))

    if not net.switch.empty and {"bus", "element", "et"}.issubset(net.switch.columns):
        switches = net.switch
        if "closed" in switches.columns:
            switches = switches[switches["closed"].fillna(True).astype(bool)]
        switches = switches[switches["et"].astype(str).eq("b")]
        for _, switch in switches.iterrows():
            bus = int(switch["bus"])
            element = int(switch["element"])
            if graph.has_edge(bus, element):
                continue
            graph.add_edge(
                bus,
                element,
                edge_kind="switch",
                line_ids=(),
                line_row_count=0,
                length_km=0.0,
                capacity_a=float("nan"),
                impedance_ohm=0.0,
                path_weight=1e-12,
            )
    return graph


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((int(left), int(right))))


def _line_paths(
    graph: nx.Graph,
    root: int,
    demand_by_bus: pd.Series,
) -> tuple[dict[int, list[tuple[int, int]]], pd.Series, list[int]]:
    paths = nx.single_source_dijkstra_path(graph, root, weight="path_weight")
    line_degree = {
        bus: sum(1 for neighbor in graph.neighbors(bus) if graph[bus][neighbor]["edge_kind"] == "line")
        for bus in graph.nodes
    }
    retained_paths: dict[int, list[tuple[int, int]]] = {}
    mapped_demand: defaultdict[int, float] = defaultdict(float)
    missing_buses: list[int] = []
    for bus, demand in demand_by_bus.items():
        bus = int(bus)
        demand = float(demand)
        if demand <= 0:
            continue
        path_nodes = paths.get(bus)
        if not path_nodes:
            missing_buses.append(bus)
            continue
        mapped_bus = bus
        pairs = list(zip(path_nodes[:-1], path_nodes[1:]))
        if pairs and line_degree.get(bus, 0) <= 1 and graph[pairs[-1][0]][pairs[-1][1]]["edge_kind"] == "line":
            mapped_bus = int(pairs[-1][0])
            pairs = pairs[:-1]
        line_edges = [
            _edge_key(left, right)
            for left, right in pairs
            if graph[left][right]["edge_kind"] == "line"
        ]
        retained_paths[bus] = line_edges
        mapped_demand[mapped_bus] += demand
    return retained_paths, pd.Series(mapped_demand, dtype=float), missing_buses


def _retained_tree(
    graph: nx.Graph,
    retained_paths: Mapping[int, list[tuple[int, int]]],
) -> nx.Graph:
    retained = nx.Graph()
    for edge in {edge for path in retained_paths.values() for edge in path}:
        retained.add_edge(*edge, **graph[edge[0]][edge[1]])
    return retained


def _feeder_sections(
    tree: nx.Graph,
    root: int,
    mapped_demand: pd.Series,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    if tree.number_of_edges() == 0:
        return [], {}
    boundaries = {
        int(node)
        for node in tree.nodes
        if int(node) == root or tree.degree(node) != 2 or float(mapped_demand.get(node, 0.0)) > 0
    }
    visited: set[tuple[int, int]] = set()
    sections: list[dict[str, Any]] = []
    edge_to_section: dict[tuple[int, int], int] = {}
    for start in sorted(boundaries):
        for neighbor in sorted(tree.neighbors(start)):
            first_edge = _edge_key(start, neighbor)
            if first_edge in visited:
                continue
            nodes = [start, neighbor]
            edges = [first_edge]
            visited.add(first_edge)
            previous, current = start, neighbor
            while current not in boundaries:
                next_nodes = [node for node in tree.neighbors(current) if node != previous]
                if len(next_nodes) != 1:
                    break
                following = int(next_nodes[0])
                edge = _edge_key(current, following)
                if edge in visited:
                    break
                nodes.append(following)
                edges.append(edge)
                visited.add(edge)
                previous, current = current, following
            section_id = len(sections)
            for edge in edges:
                edge_to_section[edge] = section_id
            attributes = [tree[edge[0]][edge[1]] for edge in edges]
            sections.append(
                {
                    "section_id": section_id,
                    "from_bus": int(nodes[0]),
                    "to_bus": int(nodes[-1]),
                    "edge_count": len(edges),
                    "line_row_count": int(sum(item["line_row_count"] for item in attributes)),
                    "line_ids": tuple(line_id for item in attributes for line_id in item["line_ids"]),
                    "edge_line_ids": tuple(item["line_ids"] for item in attributes),
                    "length_km": float(sum(item["length_km"] for item in attributes)),
                    "capacity_a": float(min(item["capacity_a"] for item in attributes)),
                    "impedance_ohm": float(sum(item["impedance_ohm"] for item in attributes)),
                }
            )
    return sections, edge_to_section


def _analyze_grid(
    net: pp.pandapowerNet,
    demand_by_bus: pd.Series,
    *,
    data_source: str,
    grid: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = _root_bus(net)
    graph = _electrical_graph(net)
    if root not in graph:
        raise ValueError(f"Transformer root bus {root} is absent from the active line graph for {grid}.")
    retained_paths, mapped_demand, missing_buses = _line_paths(graph, root, demand_by_bus)
    tree = _retained_tree(graph, retained_paths)
    sections, edge_to_section = _feeder_sections(tree, root, mapped_demand)

    edge_demand: defaultdict[tuple[int, int], float] = defaultdict(float)
    feeder_demand: defaultdict[int, float] = defaultdict(float)
    depth_rows: list[tuple[float, int, int, float, float]] = []
    for bus, demand in demand_by_bus.items():
        demand = float(demand)
        line_path = retained_paths.get(int(bus))
        if demand <= 0 or line_path is None:
            continue
        for edge in line_path:
            edge_demand[edge] += demand
        section_path = list(dict.fromkeys(edge_to_section[edge] for edge in line_path if edge in edge_to_section))
        if section_path:
            feeder_demand[section_path[0]] += demand
        path_length = sum(graph[edge[0]][edge[1]]["length_km"] for edge in line_path)
        path_impedance = sum(graph[edge[0]][edge[1]]["impedance_ohm"] for edge in line_path)
        depth_rows.append((demand, len(line_path), len(section_path), path_length, path_impedance))

    section_rows = []
    for section in sections:
        section_edges = [edge for edge, section_id in edge_to_section.items() if section_id == section["section_id"]]
        downstream_demand = max((edge_demand[edge] for edge in section_edges), default=0.0)
        section_rows.append(
            {
                **section,
                "data_source": data_source,
                "grid": grid,
                "root_bus": root,
                "downstream_annual_kwh": float(downstream_demand),
                "is_outgoing_feeder": bool(section["section_id"] in feeder_demand),
                "feeder_annual_kwh": float(feeder_demand.get(section["section_id"], 0.0)),
            }
        )
    section_frame = pd.DataFrame(section_rows)

    total_demand = float(demand_by_bus.sum())
    depth = pd.DataFrame(
        depth_rows,
        columns=["demand", "edge_depth", "section_depth", "path_length_km", "path_impedance_ohm"],
    )

    def weighted_mean(column: str) -> float:
        if depth.empty or depth["demand"].sum() <= 0:
            return float("nan")
        return float(np.average(depth[column], weights=depth["demand"]))

    if section_frame.empty:
        load_weighted_capacity = float("nan")
        rows_per_section = float("nan")
        downstream_median = float("nan")
    else:
        weights = section_frame["downstream_annual_kwh"].clip(lower=0.0)
        load_weighted_capacity = (
            float(np.average(section_frame["capacity_a"], weights=weights))
            if weights.sum() > 0
            else float("nan")
        )
        rows_per_section = float(section_frame["line_row_count"].mean())
        downstream_median = float(section_frame["downstream_annual_kwh"].median())
    feeder_values = np.asarray(list(feeder_demand.values()), dtype=float)
    grid_row = {
        "data_source": data_source,
        "grid": grid,
        "root_bus": root,
        "annual_demand_kwh": total_demand,
        "allocation_buses": int((demand_by_bus > 0).sum()),
        "unmapped_allocation_buses": len(missing_buses),
        "unmapped_annual_kwh": float(demand_by_bus.reindex(missing_buses).fillna(0.0).sum()),
        "outgoing_feeders": len(feeder_demand),
        "max_feeder_demand_share": (
            float(feeder_values.max() / feeder_values.sum()) if feeder_values.sum() > 0 else float("nan")
        ),
        "mean_demand_per_feeder_kwh": (
            float(feeder_values.mean()) if feeder_values.size else float("nan")
        ),
        "backbone_line_rows": int(section_frame["line_row_count"].sum()) if not section_frame.empty else 0,
        "backbone_edges": int(tree.number_of_edges()),
        "feeder_sections": int(len(section_frame)),
        "mean_line_rows_per_section": rows_per_section,
        "median_downstream_demand_kwh": downstream_median,
        "downstream_demand_weighted_capacity_a": load_weighted_capacity,
        "demand_weighted_edge_depth": weighted_mean("edge_depth"),
        "demand_weighted_section_depth": weighted_mean("section_depth"),
        "demand_weighted_path_length_km": weighted_mean("path_length_km"),
        "demand_weighted_path_impedance_ohm": weighted_mean("path_impedance_ohm"),
    }
    return grid_row, section_frame


def _read_plan_demands(paired_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand_columns = ["residential_equivalent_hh_annual_kwh", "calibrated_annual_ghd_kwh"]
    synthetic = pd.read_csv(paired_dir / "paired_synthetic_bus_allocation_plan.csv")
    real = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    for frame in (synthetic, real):
        frame["annual_demand_kwh"] = sum(
            pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            for column in demand_columns
        )
    synthetic = (
        synthetic.groupby(["synthetic_grid_case_id", "synthetic_bus"], as_index=False, observed=True)
        ["annual_demand_kwh"].sum()
    )
    real = (
        real.groupby(["lv_id", "allocation_bus"], as_index=False, observed=True)
        ["annual_demand_kwh"].sum()
    )
    return synthetic, real


def _synthetic_grid_rows(db: SurroGridDatabase, spec: Mapping[str, str], ags: str | int) -> pd.DataFrame:
    query = text(
        """
        SELECT DISTINCT ON (pr.grid_case_id)
               pr.powerflow_run_id, pr.grid_case_id,
               gc.pylovo_grid_result_id AS grid_result_id,
               gc.ags, gc.plz, gc.kcid, gc.bcid
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        JOIN surrogrid.powerflow_summary ps USING (powerflow_run_id)
        WHERE pr.run_name = :run_name AND ps.stage = :stage AND gc.ags = :ags
        ORDER BY pr.grid_case_id, pr.created_at DESC
        """
    )
    with db.engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={"run_name": spec["run_name"], "stage": spec["stage"], "ags": _normalized_ags(ags)},
        )


def _real_grid_rows(
    db: SurroGridDatabase,
    spec: Mapping[str, str],
    plz: int | None,
    excluded_lv_ids: set[int],
) -> pd.DataFrame:
    query = text(
        """
        SELECT DISTINCT ON (rpr.real_grid_case_id)
               rpr.real_powerflow_run_id, rpr.real_grid_case_id,
               rgc.lv_id, rgc.plz, rgc.source_file
        FROM surrogrid.real_powerflow_run rpr
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        JOIN surrogrid.real_powerflow_summary rps USING (real_powerflow_run_id)
        WHERE rpr.run_name = :run_name AND rps.stage = :stage
          AND (:plz IS NULL OR rgc.plz = :plz)
        ORDER BY rpr.real_grid_case_id, rpr.created_at DESC
        """
    )
    with db.engine.connect() as conn:
        rows = pd.read_sql_query(
            query,
            conn,
            params={"run_name": spec["run_name"], "stage": spec["stage"], "plz": plz},
        )
    rows["lv_id_int"] = rows["lv_id"].astype(str).str.removeprefix("LV_").astype(int)
    return rows[~rows["lv_id_int"].isin(excluded_lv_ids)].copy()


def _cable_summaries(
    db: SurroGridDatabase,
    specs: Mapping[str, Mapping[str, str]],
    *,
    data_source: str,
    ags: str | int,
    real_plz: int | None,
    excluded_lv_ids: set[int],
) -> pd.DataFrame:
    frames = []
    synthetic_query = text(
        """
        SELECT pr.grid_case_id::TEXT AS grid, pcs.cable AS line_id,
               pcs.cable_installed_capacity_ka * 1000.0 AS installed_capacity_a,
               pcs.cable_loading_max_time_percent AS max_loading_percent
        FROM surrogrid.powerflow_cable_summary pcs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name AND pcs.stage = :stage AND gc.ags = :ags
        """
    )
    real_query = text(
        """
        SELECT CONCAT('LV_', LPAD(rgc.lv_id::TEXT, 3, '0')) AS grid,
               rpcs.cable AS line_id,
               rpcs.cable_installed_capacity_ka * 1000.0 AS installed_capacity_a,
               rpcs.cable_loading_max_time_percent AS max_loading_percent
        FROM surrogrid.real_powerflow_cable_summary rpcs
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        WHERE rpr.run_name = :run_name AND rpcs.stage = :stage
          AND (:plz IS NULL OR rgc.plz = :plz)
        """
    )
    with db.engine.connect() as conn:
        for stage_label, spec in specs.items():
            params = {"run_name": spec["run_name"], "stage": spec["stage"]}
            if data_source == "Synthetic":
                params["ags"] = _normalized_ags(ags)
                frame = pd.read_sql_query(synthetic_query, conn, params=params)
            else:
                params["plz"] = real_plz
                frame = pd.read_sql_query(real_query, conn, params=params)
                ids = frame["grid"].str.removeprefix("LV_").astype(int)
                frame = frame[~ids.isin(excluded_lv_ids)].copy()
            frame["comparison_stage"] = stage_label
            frame["data_source"] = data_source
            frame["max_current_a"] = (
                frame["installed_capacity_a"] * frame["max_loading_percent"] / 100.0
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_section_loading(sections: pd.DataFrame, cable_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    summaries = cable_summaries.copy()
    summaries["line_id"] = summaries["line_id"].astype(int)
    for section in sections.to_dict("records"):
        for stage in summaries["comparison_stage"].drop_duplicates():
            lines = summaries[
                summaries["grid"].astype(str).eq(str(section["grid"]))
                & summaries["comparison_stage"].eq(stage)
                & summaries["line_id"].isin(section["line_ids"])
            ]
            if lines.empty:
                continue
            edge_groups = []
            for line_ids in section["edge_line_ids"]:
                edge_lines = lines[lines["line_id"].isin(line_ids)]
                if edge_lines.empty:
                    continue
                capacity = float(edge_lines["installed_capacity_a"].sum())
                current = float(edge_lines["max_current_a"].sum())
                edge_groups.append(100.0 * current / capacity if capacity > 0 else float("nan"))
            rows.append(
                {
                    **section,
                    "comparison_stage": stage,
                    "matched_line_rows": int(len(lines)),
                    "section_max_loading_percent": max(edge_groups) if edge_groups else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _grid_summary(grid_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "outgoing_feeders",
        "max_feeder_demand_share",
        "mean_demand_per_feeder_kwh",
        "backbone_line_rows",
        "backbone_edges",
        "feeder_sections",
        "mean_line_rows_per_section",
        "median_downstream_demand_kwh",
        "downstream_demand_weighted_capacity_a",
        "demand_weighted_edge_depth",
        "demand_weighted_section_depth",
        "demand_weighted_path_length_km",
        "demand_weighted_path_impedance_ohm",
    ]
    rows = []
    for source, group in grid_metrics.groupby("data_source", observed=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "data_source": source,
                    "metric": metric,
                    "grids": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def build_feeder_structure_comparison(
    *,
    synthetic_specs: Mapping[str, Mapping[str, str]],
    real_specs: Mapping[str, Mapping[str, str]],
    paired_dir: str | Path,
    ags: str | int,
    real_plz: int | None,
    excluded_real_lv_ids: tuple[int, ...] = (),
) -> dict[str, pd.DataFrame]:
    """Build graph-normalized feeder, path, capacity, and loading diagnostics."""
    paired_dir = Path(paired_dir).expanduser().resolve()
    synthetic_plan, real_plan = _read_plan_demands(paired_dir)
    excluded = {int(value) for value in excluded_real_lv_ids}
    db = SurroGridDatabase()

    synthetic_pre = next(iter(synthetic_specs.values()))
    real_pre = next(iter(real_specs.values()))
    synthetic_grids = _synthetic_grid_rows(db, synthetic_pre, ags)
    real_grids = _real_grid_rows(db, real_pre, real_plz, excluded)

    grid_rows: list[dict[str, Any]] = []
    section_frames: list[pd.DataFrame] = []
    for row in synthetic_grids.to_dict("records"):
        grid_case_id = int(row["grid_case_id"])
        demand = synthetic_plan.loc[
            synthetic_plan["synthetic_grid_case_id"].astype(int).eq(grid_case_id)
        ].set_index("synthetic_bus")["annual_demand_kwh"]
        net = db.read_pandapower_grid(row)
        grid_row, sections = _analyze_grid(
            net,
            demand,
            data_source="Synthetic",
            grid=str(grid_case_id),
        )
        grid_rows.append(grid_row)
        if not sections.empty:
            section_frames.append(sections)

    for row in real_grids.to_dict("records"):
        lv_id = int(row["lv_id_int"])
        demand = real_plan.loc[real_plan["lv_id"].astype(int).eq(lv_id)].set_index("allocation_bus")[
            "annual_demand_kwh"
        ]
        net = pp.from_excel(Path(str(row["source_file"])))
        grid_row, sections = _analyze_grid(
            net,
            demand,
            data_source="Real SWF",
            grid=f"LV_{lv_id:03d}",
        )
        grid_rows.append(grid_row)
        if not sections.empty:
            section_frames.append(sections)

    grid_metrics = pd.DataFrame(grid_rows)
    sections = pd.concat(section_frames, ignore_index=True) if section_frames else pd.DataFrame()

    synthetic_cables = _cable_summaries(
        db,
        synthetic_specs,
        data_source="Synthetic",
        ags=ags,
        real_plz=real_plz,
        excluded_lv_ids=excluded,
    )
    real_cables = _cable_summaries(
        db,
        real_specs,
        data_source="Real SWF",
        ags=ags,
        real_plz=real_plz,
        excluded_lv_ids=excluded,
    )
    cable_summaries = pd.concat([synthetic_cables, real_cables], ignore_index=True)
    loading_frames = []
    for source, source_sections in sections.groupby("data_source", observed=True):
        source_cables = cable_summaries[cable_summaries["data_source"].eq(source)]
        loading_frames.append(_attach_section_loading(source_sections, source_cables))
    section_loading = pd.concat(loading_frames, ignore_index=True) if loading_frames else pd.DataFrame()
    return {
        "grid_metrics": grid_metrics,
        "sections": sections,
        "section_loading": section_loading,
        "summary": _grid_summary(grid_metrics),
    }


def export_feeder_structure_comparison(
    comparison: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the reusable audit tables to one generated-output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("grid_metrics", "sections", "section_loading", "summary"):
        path = output_dir / f"feeder_structure_{name}.csv"
        comparison[name].to_csv(path, index=False)
        paths[name] = path
    metadata_path = output_dir / "feeder_structure_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "parallel_line_handling": "rows_between_same_bus_pair_are_one_electrical_edge",
                "section_definition": "maximal_edge_chain_through_degree_two_buses_without_allocated_demand",
                "demand_basis": "paired_annual_household_plus_calibrated_ghd_demand",
                "service_connections": "terminal_line_into_load_bus_excluded",
                "section_loading": "maximum_capacity_weighted_row_peak_loading_across_section_edges",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metadata"] = metadata_path
    return paths
