"""Result-row fields of the expansion rules, shared by the synthetic and the real materialization.

- Line rows carry the route decision (``route_fields``).
- Transformer rows carry every station-level cost: exchange, load transfer, new substations and
  voltage measures (``station_fields``). The sum of ``estimated_cost_eur`` over the line and
  transformer rows is therefore the grid total.
- ``expansion_grid_result`` gets one row per grid with the decision trail (``grid_row_from_state``).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from . import staged


def route_fields(
    result: staged.RouteResult, route: staged.RouteInput, *, service_lines_in_total: bool
) -> dict[str, Any]:
    """Line-row fields of one route decision.

    A service line's cost goes to ``service_cost_eur``; it enters ``estimated_cost_eur`` only if
    ``service_lines_in_total``.
    """
    service_cost = result.cable_cost_eur if result.measure == "service" else 0.0
    in_total = result.measure != "service" or service_lines_in_total
    return {
        "line_existing_duct_share": result.existing_duct_share,
        "line_trenching_share": result.trenching_share,
        "cost_eur_per_km": result.cost_eur_per_km,
        "estimated_cost_eur": result.cable_cost_eur if in_total else 0.0,
        "cost_basis": result.cost_basis,
        "duct_cost_eur_per_km": result.duct_cost_eur_per_km,
        "reopen_cost_eur_per_km": result.reopen_cost_eur_per_km,
        "reinforcement_150_count": result.n150,
        "reinforcement_185_count": result.n185,
        "reinforcement_240_count": result.n240,
        "reinforcement_added_capacity_ka": result.added_capacity_ka,
        "reinforcement_catalog": result.catalog,
        "measure": result.measure,
        "is_station_outlet": route.is_outlet,
        "is_service_line": route.is_service,
        "route_cable_count": int(route.existing_cables),
        "service_cost_eur": service_cost,
    }


def station_fields(state: staged.GridState) -> dict[str, Any]:
    """Transformer-row breakdown of one staged grid."""
    station = state.station
    return {
        "station_measure": state.station_measure,
        "station_limit_kva": station.limit_kva if station is not None else None,
        "excess_kva": state.excess_kva if state.excess_kva > 0 else 0.0,
        "transformer_exchange_cost_eur": state.transformer_exchange_cost_eur(),
        "load_transfer_cost_eur": state.load_transfer_cost_eur,
        "new_station_cost_eur": state.new_station_cost_eur,
        "voltage_measure": state.voltage_measure,
        "voltage_cost_eur": state.voltage_cost_eur,
    }


def _identity(identity: Mapping[str, Any], *, real: bool) -> dict[str, Any]:
    if real:
        return {
            "grid_key": f"real:{int(identity['real_powerflow_run_id'])}",
            "scenario_id": int(identity["scenario_id"]),
            "powerflow_run_id": None,
            "grid_case_id": None,
            "real_powerflow_run_id": int(identity["real_powerflow_run_id"]),
            "real_grid_case_id": int(identity["real_grid_case_id"]),
            "plz": identity.get("plz"),
        }
    return {
        "grid_key": f"synthetic:{int(identity['powerflow_run_id'])}",
        "scenario_id": int(identity["scenario_id"]),
        "powerflow_run_id": int(identity["powerflow_run_id"]),
        "grid_case_id": int(identity["grid_case_id"]),
        "real_powerflow_run_id": None,
        "real_grid_case_id": None,
        "plz": identity.get("plz"),
    }


def grid_row_from_state(
    state: staged.GridState,
    params: staged.StagedParameters,
    identity: Mapping[str, Any],
    *,
    grid_label: str,
    partner_labels: Mapping[Hashable, str],
    real: bool,
) -> dict[str, Any]:
    """``expansion_grid_result`` row of one staged grid."""
    results = state.routes.values()
    station = state.station
    return {
        **_identity(identity, real=real),
        "grid_label": grid_label,
        "settlement_type": state.input.settlement_type,
        "rule_set": staged.RULE_SET,
        "transformer_rated_power_kva": station.rated_kva if station is not None else None,
        "peak_kva": station.peak_kva if station is not None else None,
        "station_limit_kva": station.limit_kva if station is not None else None,
        "excess_kva": state.excess_kva,
        "station_measure": state.station_measure,
        "escalation_reason": "|".join(state.escalation) or None,
        "transfer_kva": state.transfer_kva or None,
        "transfer_partners": "|".join(partner_labels.get(key, str(key)) for key in state.transfer_partners) or None,
        "new_station_cluster": state.new_station_cluster,
        "new_stations": state.new_stations or None,
        "existing_outlet_cables": int(state.input.existing_outlet_cables),
        "added_outlet_cables": int(state.added_outlet_cables),
        "panels_total": int(state.panels_total),
        "routes_reinforced": sum(1 for r in results if r.measure in ("local", "outlet")),
        "cables_added": sum(r.added_cables for r in results if r.measure in ("local", "outlet", "service")),
        "heavy_routes": sum(1 for r in results if r.heavy),
        "relieved_routes": sum(1 for r in results if r.measure.startswith("relieved_by")),
        "unresolved_routes": sum(1 for r in results if r.measure.startswith("unresolved")),
        "service_routes_reinforced": sum(1 for r in results if r.measure == "service"),
        "voltage_residual_buses": int(state.voltage_residual_buses),
        "voltage_measure": state.voltage_measure,
        "cable_cost_eur": state.cable_cost_eur(params),
        "service_cost_eur": state.service_cost_eur(),
        "transformer_exchange_cost_eur": state.transformer_exchange_cost_eur(),
        "load_transfer_cost_eur": state.load_transfer_cost_eur,
        "new_station_cost_eur": state.new_station_cost_eur,
        "voltage_cost_eur": state.voltage_cost_eur,
        "total_cost_eur": state.total_cost_eur(params),
    }
