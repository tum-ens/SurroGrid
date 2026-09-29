"""Expansion rows of the synthetic pylovo grids.

The SQL builds the scope and the mapping to the visible pylovo lines (``selected_runs.sql``,
``component_loading.sql``, ``transformer_peak.sql``); the rules (``staged``) run here in Python, and
their decisions go to two temp tables (``expansion_component_selection``,
``expansion_station_selection``) that ``line_insert.sql`` and ``transformer_insert.sql`` join. The
rules read the pylovo network of each grid for its topology, and the Step 4 bus voltages. Every step
runs inside the caller's transaction.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from gridexpand.db import writers
from gridexpand.db.grids import read_pandapower_grid

from . import staged
from .results import grid_row_from_state, route_fields, station_fields
from .topology import build_topology

# Unmapped components of at most this length without a pylovo line (root connectors, see
# component_audit.sql) belong to the station busbar.
ROOT_CONNECTOR_MAX_KM = 0.005

COMPONENT_SELECTION_TYPES = {
    "powerflow_run_id": "bigint",
    "component_line": "integer",
    "required_parallel": "integer",
    "additional_parallel": "integer",
    "reinforcement_150_count": "integer",
    "reinforcement_185_count": "integer",
    "reinforcement_240_count": "integer",
    "reinforcement_added_capacity_ka": "double precision",
    "reinforcement_catalog": "text",
    "line_cost_eur_per_km": "double precision",
    "line_cost_basis": "text",
    "duct_cost_eur_per_km": "double precision",
    "reopen_cost_eur_per_km": "double precision",
    "existing_duct_share": "double precision",
    "trenching_share": "double precision",
    "estimated_component_cost_eur": "double precision",
    "measure": "text",
    "is_station_outlet": "boolean",
    "is_service_line": "boolean",
    "route_cable_count": "integer",
    "service_cost_eur": "double precision",
}
STATION_SELECTION_TYPES = {
    "powerflow_run_id": "bigint",
    "required_kva": "double precision",
    "estimated_cost_eur": "double precision",
    "transformer_cost_basis": "text",
    "requires_expansion": "boolean",
    "station_measure": "text",
    "station_limit_kva": "double precision",
    "excess_kva": "double precision",
    "transformer_exchange_cost_eur": "double precision",
    "load_transfer_cost_eur": "double precision",
    "new_station_cost_eur": "double precision",
    "voltage_measure": "text",
    "voltage_cost_eur": "double precision",
}


def _create_temp(conn: Connection, table: str, types: dict[str, str], frame: pd.DataFrame) -> None:
    columns = ", ".join(f'"{name}" {sql_type}' for name, sql_type in types.items())
    conn.execute(text(f"CREATE TEMP TABLE {table} ({columns}) ON COMMIT DROP"))
    if frame.empty:
        return
    prepared = writers.csv_frame(frame[list(types)], types)
    buffer = io.StringIO()
    prepared.to_csv(buffer, index=False, header=False, na_rep=writers.NULL)
    buffer.seek(0)
    names = ", ".join(f'"{name}"' for name in types)
    cursor = conn.connection.dbapi_connection.cursor()
    try:
        cursor.copy_expert(f"COPY {table} ({names}) FROM STDIN WITH (FORMAT csv, NULL '{writers.NULL}')", buffer)
    finally:
        cursor.close()
    conn.execute(text(f"ANALYZE {table}"))


def _optional_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


# Rules ------------------------------------------------------------------------------------------


def _staged(
    conn: Connection,
    engine,
    runs: pd.DataFrame,
    components: pd.DataFrame,
    peaks: pd.DataFrame,
    params: staged.StagedParameters,
    stage: str,
    grid_label: Callable[[Any], str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    run_ids = [int(value) for value in runs["powerflow_run_id"]]
    voltages = pd.read_sql_query(
        text(
            """
            SELECT powerflow_run_id, bus, voltage_min_time_pu
            FROM surrogrid.powerflow_bus_voltage_summary
            WHERE powerflow_run_id = ANY(:ids) AND stage = :stage AND voltage_min_time_pu IS NOT NULL
            """
        ),
        conn, params={"ids": run_ids, "stage": stage},
    )
    busbar = pd.read_sql_query(
        text(
            """
            SELECT powerflow_run_id, lv_busbar_vm_pu
            FROM surrogrid.powerflow_summary
            WHERE powerflow_run_id = ANY(:ids) AND stage = :stage
            """
        ),
        conn, params={"ids": run_ids, "stage": stage},
    ).set_index("powerflow_run_id")["lv_busbar_vm_pu"]
    peak_by_run = {int(p.powerflow_run_id): p for p in peaks.itertuples()}
    grids, routes_by_run = [], {}
    for run in runs.itertuples():
        run_id = int(run.powerflow_run_id)
        net = read_pandapower_grid(engine, {"grid_result_id": int(run.pylovo_grid_result_id)})
        own = components[components["powerflow_run_id"] == run_id]
        connectors = own[
            own["visible_line_id"].isna()
            & own["source_line_name"].isna()
            & (own["component_length_km"].fillna(0.0) <= ROOT_CONNECTOR_MAX_KM)
        ]
        if not connectors.empty:
            net.line.loc[connectors["line"].astype(int), "length_km"] = 0.0
        topology = build_topology(net)
        mapped = own[own["visible_line_id"].notna()]
        routes, route_above_bus = [], {}
        for c in mapped.itertuples():
            line = int(c.line)
            a, b = int(net.line.at[line, "from_bus"]), int(net.line.at[line, "to_bus"])
            parallel = int(c.component_parallel) if pd.notna(c.component_parallel) else 1
            max_i_ka = _optional_float(c.max_i_ka) or 0.0
            route = staged.RouteInput(
                key=line,
                length_km=float(c.component_length_km) if pd.notna(c.component_length_km) else 0.0,
                existing_cables=parallel,
                installed_capacity_ka=max_i_ka * parallel,
                p100_ka=float(c.max_i_from_ka),
                is_outlet=topology.is_outlet(a, b),
                is_service=topology.is_service(a, b),
            )
            routes.append(route)
            route_above_bus[topology.child(a, b)] = line
        routes_by_run[run_id] = {route.key: route for route in routes}
        peak = peak_by_run.get(run_id)
        busbar_value = busbar.get(run_id)
        grids.append(
            staged.GridInput(
                key=run_id,
                settlement_type=int(run.settlement_type) if pd.notna(run.settlement_type) else None,
                rated_kva=float(peak.rated_kva) if peak is not None else None,
                peak_kva=float(peak.s_mva) * 1000.0 if peak is not None else None,
                routes=routes,
                existing_outlet_cables=sum(route.existing_cables for route in routes if route.is_outlet),
                route_above_bus=route_above_bus,
                parent_bus=topology.parent,
                bus_distance_km=topology.distance_km,
                bus_min_voltage={
                    int(v.bus): float(v.voltage_min_time_pu)
                    for v in voltages[voltages["powerflow_run_id"] == run_id].itertuples()
                },
                lv_busbar_vm_pu=_optional_float(busbar_value),
                coordinates=topology.coordinate_array(),
            )
        )
    states = staged.run_stages(grids, params)
    selection_rows, station_rows, grid_rows = [], [], []
    labels = {int(run.powerflow_run_id): grid_label(run) for run in runs.itertuples()}
    identity = {int(run.powerflow_run_id): run._asdict() for run in runs.itertuples()}
    for state in states:
        run_id = int(state.key)
        routes = routes_by_run[run_id]
        for key, result in state.routes.items():
            fields = route_fields(result, routes[key], service_lines_in_total=params.service_lines_in_total)
            selection_rows.append({
                "powerflow_run_id": run_id,
                "component_line": int(key),
                "required_parallel": int(routes[key].existing_cables) + result.added_cables,
                "additional_parallel": result.added_cables,
                "reinforcement_150_count": fields["reinforcement_150_count"],
                "reinforcement_185_count": fields["reinforcement_185_count"],
                "reinforcement_240_count": fields["reinforcement_240_count"],
                "reinforcement_added_capacity_ka": fields["reinforcement_added_capacity_ka"],
                "reinforcement_catalog": fields["reinforcement_catalog"],
                "line_cost_eur_per_km": fields["cost_eur_per_km"],
                "line_cost_basis": fields["cost_basis"],
                "duct_cost_eur_per_km": fields["duct_cost_eur_per_km"],
                "reopen_cost_eur_per_km": fields["reopen_cost_eur_per_km"],
                "existing_duct_share": fields["line_existing_duct_share"],
                "trenching_share": fields["line_trenching_share"],
                "estimated_component_cost_eur": fields["estimated_cost_eur"],
                "measure": fields["measure"],
                "is_station_outlet": fields["is_station_outlet"],
                "is_service_line": fields["is_service_line"],
                "route_cable_count": fields["route_cable_count"],
                "service_cost_eur": fields["service_cost_eur"],
            })
        if state.station is not None:
            station_rows.append({
                "powerflow_run_id": run_id,
                "required_kva": state.station.required_kva,
                "estimated_cost_eur": state.station_cost_eur(),
                "transformer_cost_basis": state.station.cost_basis,
                "requires_expansion": state.station_cost_eur() > 0.0 or state.station_measure != "none",
                **station_fields(state),
            })
        elif state.station_cost_eur() > 0:
            print(
                f"Warning: synthetic run {run_id} has no transformer rating; its station-level cost "
                f"({state.station_cost_eur():.0f} EUR) is only in expansion_grid_result."
            )
        grid_rows.append(
            grid_row_from_state(
                state, params, identity[run_id], grid_label=labels[run_id], partner_labels=labels, real=False
            )
        )
    return (
        pd.DataFrame(selection_rows, columns=list(COMPONENT_SELECTION_TYPES)),
        pd.DataFrame(station_rows, columns=list(STATION_SELECTION_TYPES)),
        grid_rows,
    )


# Entry point ------------------------------------------------------------------------------------


def materialize_synthetic(
    conn: Connection,
    engine,
    run_id: int,
    *,
    stage: str,
    assumption: dict[str, Any],
    duct_share_override: float | None,
    sql_text: Callable[[str], str],
) -> dict[str, int]:
    """Write the line, transformer and grid rows of one synthetic analysis (temp scope tables exist).

    Args:
        conn: the materialization transaction (``expansion_selected_run`` and
            ``expansion_component_loading`` exist).
        engine: engine for reading the pylovo networks.
        run_id: ``expansion_analysis_run_id``.
        stage: power-flow stage.
        assumption: the ``expansion_cost_assumption`` row (rule set ``staged_2026``).
        duct_share_override: ``--line-existing-duct-share``.
        sql_text: loader of the ``sql/`` files.

    Raises:
        ValueError: the assumption row is not of rule set ``staged_2026``.
    """
    params = staged.StagedParameters.from_assumption(assumption, duct_share_override)
    conn.execute(text(sql_text("transformer_peak.sql")), {"stage": stage})
    runs = pd.read_sql_query(text("SELECT * FROM expansion_selected_run ORDER BY powerflow_run_id"), conn)
    components = pd.read_sql_query(
        text(
            """
            SELECT powerflow_run_id, line, visible_line_id, source_line_name, component_length_km,
                   max_i_ka, component_parallel, max_i_from_ka, settlement_type
            FROM expansion_component_loading
            ORDER BY powerflow_run_id, line
            """
        ),
        conn,
    )
    peaks = pd.read_sql_query(text("SELECT * FROM expansion_transformer_peak ORDER BY powerflow_run_id"), conn)

    def grid_label(run) -> str:
        return f"{run.plz}-{run.kcid}-{run.bcid}"

    selections, stations, grid_rows = _staged(conn, engine, runs, components, peaks, params, stage, grid_label)
    _create_temp(conn, "expansion_component_selection", COMPONENT_SELECTION_TYPES, selections)
    _create_temp(conn, "expansion_station_selection", STATION_SELECTION_TYPES, stations)
    params_sql = {"expansion_analysis_run_id": run_id}
    lines = conn.execute(text(sql_text("line_insert.sql")), params_sql).rowcount
    transformers = conn.execute(text(sql_text("transformer_insert.sql")), params_sql).rowcount
    grid_count = 0
    if grid_rows:
        frame = pd.DataFrame(grid_rows)
        frame.insert(0, "expansion_analysis_run_id", int(run_id))
        grid_count = writers.copy_frame(conn, "expansion_grid_result", frame)
    return {"line_rows": int(lines or 0), "transformer_rows": int(transformers or 0), "grid_rows": grid_count}
