"""Pandapower network preparation and evaluation scopes for Step 4.

Conventions (keep them; the power-flow results depend on them):

- Static DSO/pylovo loads are replaced by one zeroed load row per scenario
  demand bus; static generators, generators and storages are switched off.
- Thermal and voltage limits are relaxed (lines 1000 kA, loads 1000 MW, buses
  0..10 pu); the rated cable currents are kept aside for the evaluation.
- Synthetic grids: the single MV/LV transformer is replaced by a closed
  bus-bus switch and the external-grid bus gets the LV nominal voltage.
"""

from __future__ import annotations

from collections import deque

import pandapower as pp
import pandas as pd

LOAD_DEFAULTS = {
    "const_z_percent": 0.0,
    "const_i_percent": 0.0,
    "const_z_p_percent": 0.0,
    "const_z_q_percent": 0.0,
    "const_i_p_percent": 0.0,
    "const_i_q_percent": 0.0,
    "scaling": 1.0,
    "in_service": True,
}


def transformer_rating_mva(grid) -> float:
    """Station rating in MVA: ``sn_mva`` is per unit, so parallel units multiply it."""
    if "sn_mva" not in grid.trafo.columns:
        return float("nan")
    sn_mva = pd.to_numeric(grid.trafo["sn_mva"], errors="coerce")
    if "parallel" in grid.trafo.columns:
        parallel = pd.to_numeric(grid.trafo["parallel"], errors="coerce").fillna(1)
    else:
        parallel = 1
    return float((sn_mva * parallel).sum())


def set_scenario_load_buses(grid, load_buses, name_prefix="Scenario_Profile_"):
    """Replace static loads with one zeroed row per scenario demand bus.

    Static generators, generators and storages are switched off (their demand is
    part of the scenario demand).
    """
    load_buses = sorted({int(bus) for bus in load_buses})
    if not load_buses:
        raise ValueError("Scenario power flow requires at least one demand bus.")

    missing_buses = sorted(set(load_buses).difference(map(int, grid.bus.index)))
    if missing_buses:
        raise ValueError(
            "Scenario demand references buses missing from the pandapower grid: "
            f"{missing_buses[:10]}"
        )

    existing_load = grid.load.copy() if hasattr(grid, "load") else pd.DataFrame()
    template_columns = (
        list(existing_load.columns)
        if len(existing_load.columns)
        else ["bus", "p_mw", "q_mvar", "name"]
    )
    rows = [
        {"bus": bus, "p_mw": 0.0, "q_mvar": 0.0, "name": f"{name_prefix}{bus}"}
        for bus in load_buses
    ]
    grid.load = (
        pd.DataFrame(rows)
        .reindex(columns=template_columns)
        .reset_index(drop=True)
    )
    grid.load["bus"] = grid.load["bus"].astype(int)
    grid.load["p_mw"] = 0.0
    grid.load["q_mvar"] = 0.0
    grid.load["max_p_mw"] = 1000.0
    for column, value in LOAD_DEFAULTS.items():
        grid.load[column] = value
    disable_static_injections(grid)
    return grid


def disable_static_injections(grid, *, fill_missing=True):
    """Switch off static generators, generators and storages.

    Args:
        fill_missing: also replace missing ``p_mw``/``q_mvar`` by 0 and missing
            ``scaling`` by 1 (pandapower reads disabled rows too, and NaN there
            breaks the solver; real DSO grids carry NaN).
    """
    for element_name in ("sgen", "gen", "storage"):
        element = getattr(grid, element_name, None)
        if element is None or element.empty:
            continue
        element["in_service"] = False
        if fill_missing:
            for column in ("p_mw", "q_mvar"):
                if column in element.columns:
                    element[column] = element[column].fillna(0.0)
            if "scaling" in element.columns:
                element["scaling"] = element["scaling"].fillna(1.0)
    return grid


def rated_cable_currents(grid) -> pd.Series:
    """Copy of the rated line currents (``max_i_ka``) before limits are relaxed."""
    if "max_i_ka" in grid.line.columns:
        return grid.line["max_i_ka"].copy()
    return pd.Series(float("nan"), index=grid.line.index, name="max_i_ka")


def prepare_synthetic_grid(grid):
    """Relax limits and replace the single transformer by a closed bus-bus switch.

    Raises:
        ValueError: the grid does not have exactly one transformer row.
    """
    # Increase max line capacity
    df_lines = grid.line
    df_lines["max_i_ka"] = 1000
    grid.line = df_lines

    # Remove load max restrictions
    df_loads = grid.load
    df_loads["max_p_mw"] = 1000
    for column, value in {
        "const_z_percent": 0.0,
        "const_i_percent": 0.0,
        "scaling": 1.0,
        "in_service": True,
    }.items():
        if column not in df_loads.columns:
            df_loads[column] = value
    grid.load = df_loads

    # Remove voltage restrictions
    df_buses = grid.bus
    df_buses[["min_vm_pu", "max_vm_pu"]] = (0, 10)
    grid.bus = df_buses

    # Remove trafo and replace with switch
    if len(grid.trafo) != 1:
        raise ValueError(
            f"Synthetic grids must have exactly one transformer row, found {len(grid.trafo)}."
        )
    trafo_buses = grid.trafo[["hv_bus", "lv_bus"]].values[0]
    grid.trafo.drop(index=grid.trafo.index, inplace=True)

    ext_grid_bus = int(grid.ext_grid.loc[0, "bus"])  # bus which is the external import bus
    lv_bus = [bus for bus in trafo_buses if bus != ext_grid_bus][0]
    grid.bus.loc[ext_grid_bus, "vn_kv"] = grid.bus.loc[lv_bus, "vn_kv"]

    pp.create_switch(
        grid,
        bus=ext_grid_bus,
        element=lv_bus,
        et="b",
        closed=True,
        type="CB",
        name="SW_replacing_T0",
    )
    return grid


# Evaluation scopes ----------------------------------------------------------

def active_line_index(grid):
    """Lines in service and not cut by an open line switch."""
    if grid.line.empty:
        return pd.Index([], dtype=int)
    if "in_service" in grid.line.columns:
        active = grid.line["in_service"].fillna(True).astype(bool)
        line_index = pd.Index(grid.line.index[active])
    else:
        line_index = pd.Index(grid.line.index)

    if hasattr(grid, "switch") and not grid.switch.empty and {"et", "element", "closed"}.issubset(grid.switch.columns):
        open_line_switches = grid.switch[
            grid.switch["et"].astype(str).eq("l")
            & ~grid.switch["closed"].fillna(True).astype(bool)
        ]
        if not open_line_switches.empty:
            open_lines = pd.Index(open_line_switches["element"].dropna().astype(int).unique())
            line_index = line_index.difference(open_lines)
    return line_index


def grid_adjacency(grid):
    """Bus adjacency over active lines and closed bus-bus switches."""
    adjacency = {}
    for _, line in grid.line.loc[active_line_index(grid)].iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        adjacency.setdefault(from_bus, set()).add(to_bus)
        adjacency.setdefault(to_bus, set()).add(from_bus)

    if hasattr(grid, "switch") and not grid.switch.empty:
        switches = grid.switch
        if "closed" in switches.columns:
            switches = switches[switches["closed"].fillna(True).astype(bool)]
        if "et" in switches.columns:
            switches = switches[switches["et"].astype(str).eq("b")]
        for _, switch in switches.iterrows():
            bus = int(switch["bus"])
            element = int(switch["element"])
            adjacency.setdefault(bus, set()).add(element)
            adjacency.setdefault(element, set()).add(bus)
    return adjacency


def root_bus(grid):
    """External-grid bus, else the first transformer LV bus, else the first bus."""
    if hasattr(grid, "ext_grid") and not grid.ext_grid.empty and "bus" in grid.ext_grid.columns:
        return int(grid.ext_grid.iloc[0]["bus"])
    if hasattr(grid, "trafo") and not grid.trafo.empty and "lv_bus" in grid.trafo.columns:
        return int(grid.trafo.iloc[0]["lv_bus"])
    return int(grid.bus.index[0])


def parent_tree_from_root(adjacency, root):
    """Breadth-first parent of every reachable bus (neighbours in sorted order)."""
    parents = {int(root): None}
    queue = deque([int(root)])
    while queue:
        bus = queue.popleft()
        for neighbor in sorted(adjacency.get(bus, [])):
            if neighbor in parents:
                continue
            parents[neighbor] = bus
            queue.append(neighbor)
    return parents


def comparison_backbone_scope(grid, load_buses):
    """Return demand-carrying backbone cable ids and upstream voltage buses.

    The comparison scope keeps only active line rows that lie on at least one
    path from the root bus to a selected household load bus. Terminal service
    connections into selected load endpoints are excluded. If parallel line rows
    connect the same two path buses, all active parallel rows are retained.
    Voltages are evaluated at the nearest upstream bus on the retained backbone.
    """
    load_buses = {
        int(bus)
        for bus in load_buses
        if pd.notna(bus) and int(bus) in grid.bus.index
    }
    active_line_ids = active_line_index(grid)
    if len(active_line_ids) == 0 or not load_buses:
        return [], []

    lines = grid.line.loc[active_line_ids]
    line_ids_by_edge = {}
    line_neighbors = {}
    for line_id, line in lines.iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        edge = frozenset((from_bus, to_bus))
        line_ids_by_edge.setdefault(edge, []).append(int(line_id))
        line_neighbors.setdefault(from_bus, set()).add(to_bus)
        line_neighbors.setdefault(to_bus, set()).add(from_bus)

    adjacency = grid_adjacency(grid)
    parents = parent_tree_from_root(adjacency, root_bus(grid))
    retained_line_ids = set()
    voltage_buses = []

    for load_bus in sorted(load_buses):
        if load_bus not in parents:
            continue
        path_edges = []
        bus = load_bus
        seen = set()
        while bus in parents and bus not in seen:
            seen.add(bus)
            parent = parents[bus]
            if parent is None:
                break
            path_edges.append((int(parent), int(bus)))
            bus = parent

        if not path_edges:
            continue

        terminal_load_bus = len(line_neighbors.get(load_bus, set())) <= 1
        service_edge = frozenset(path_edges[0]) if terminal_load_bus else None
        mapped_voltage_bus = None

        for parent, child in path_edges:
            edge = frozenset((parent, child))
            if edge == service_edge:
                mapped_voltage_bus = int(parent)
                continue
            line_ids = line_ids_by_edge.get(edge)
            if not line_ids:
                continue
            retained_line_ids.update(line_ids)
            if mapped_voltage_bus is None:
                mapped_voltage_bus = int(child)

        if mapped_voltage_bus is None:
            mapped_voltage_bus = int(load_bus)
        voltage_buses.append(mapped_voltage_bus)

    backbone_cable_ids = pd.Index(sorted(retained_line_ids), dtype=int)
    if len(backbone_cable_ids) > 0:
        backbone_lines = grid.line.loc[backbone_cable_ids]
        backbone_buses = set(backbone_lines["from_bus"].astype(int)).union(
            set(backbone_lines["to_bus"].astype(int))
        )
        voltage_buses = [bus for bus in voltage_buses if bus in backbone_buses]

    voltage_buses = pd.Index(voltage_buses, dtype=int).drop_duplicates().tolist()
    return backbone_cable_ids.astype(int).tolist(), voltage_buses


def comparison_evaluation_scope(grid, load_buses, scope="full"):
    """Return cable and voltage-bus ids for a named comparison scope.

    ``full`` evaluates active service lines and their terminal load buses.
    ``backbone`` retains the historical comparison behavior that removes each
    terminal service edge and maps its voltage observation one bus upstream.
    """
    if scope not in {"full", "backbone"}:
        raise ValueError("scope must be either 'full' or 'backbone'.")
    if scope == "backbone":
        return comparison_backbone_scope(grid, load_buses)

    cable_ids = active_line_index(grid).astype(int).tolist()
    voltage_buses = pd.Index(
        [int(bus) for bus in load_buses if pd.notna(bus) and int(bus) in grid.bus.index],
        dtype=int,
    ).drop_duplicates().tolist()
    if not voltage_buses:
        voltage_buses = grid.bus.index.astype(int).tolist()
    return cable_ids, voltage_buses
