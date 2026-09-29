"""Staged expansion rules (gridexpand.analysis.expansion.staged), with hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest

from gridexpand.analysis.expansion import staged


def _params(assumption, **overrides):
    return staged.StagedParameters.from_assumption({**assumption, **overrides})


def test_parameters_refuse_other_rule_sets_and_missing_values(staged_assumption):
    retired = {**staged_assumption, "assumption_key": "de_lv_heuristic_2026", "rule_set": "heuristic_2026"}
    with pytest.raises(ValueError, match="'heuristic_2026' is retired"):
        staged.StagedParameters.from_assumption(retired)
    with pytest.raises(ValueError, match="lacks"):
        _params(staged_assumption, new_station_eur=None)


def test_new_station_cost_components(staged_assumption):
    params = _params(staged_assumption)
    # 85 k station + 0.2 km x 250 k MV + 0.1 km x settlement trench cost
    assert params.new_station_cost_eur(1) == pytest.approx(85000 + 50000 + 8000)
    assert params.new_station_cost_eur(2) == pytest.approx(85000 + 50000 + 10000)
    assert params.new_station_cost_eur(3) == pytest.approx(85000 + 50000 + 16500)
    assert params.new_station_cost_eur(None) == params.new_station_cost_eur(2)


# Stage 1 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rated", "peak", "settlement", "measure", "required", "excess", "cost"),
    [
        (400, 380, 2, "none", 400, 0, 0),
        (400, 500, 2, "exchange", 630, 0, 12000),
        (400, 700, 2, "exchange", 800, 0, 12500),
        (630, 900, 2, "exchange", 1000, 0, 15000),  # limit 1000 kVA in every settlement type
        (630, 1100, 1, "over_limit", 1000, 100, 15000),  # exchange to the limit, the rest goes to stage 4
        (1260, 1300, 2, "over_limit", 1260, 40, 0),  # a unit above the limit (2 x 630 kVA) is its ceiling
        (200, 150, 1, "none", 200, 0, 0),
        (200, 210, 1, "exchange", 250, 0, 8000),  # non-standard rating
    ],
)
def test_station_decision(staged_assumption, rated, peak, settlement, measure, required, excess, cost):
    result = staged.station_decision(rated, peak, settlement, _params(staged_assumption))
    assert (result.measure, result.required_kva) == (measure, required)
    assert result.excess_kva == pytest.approx(excess)
    assert result.exchange_cost_eur == cost


def test_station_planning_limit(staged_assumption):
    result = staged.station_decision(630, 690, 2, _params(staged_assumption, transformer_planning_limit=1.1))
    assert result.measure == "none"  # 690 <= 1.1 x 630


# Stage 2 ----------------------------------------------------------------------------------------


def _route(p100, installed=0.27, cables=1, **flags):
    return staged.RouteInput(key="r", length_km=0.1, existing_cables=cables, installed_capacity_ka=installed,
                             p100_ka=p100, **flags)


def test_route_within_capacity(staged_assumption):
    assert staged.select_route(_route(0.25), 2, _params(staged_assumption)).measure == "none"


def test_route_least_cost_choice(staged_assumption):
    # 0.5 kA on one NAYY 150 (0.27 kA): one added cable does; the 150 is cheapest per km on a trenched route:
    # 25 + 0.8 x (100 - 25) = 85 kEUR/km (185: 89, 240: 94).
    result = staged.select_route(_route(0.5), 2, _params(staged_assumption))
    assert (result.n150, result.n185, result.n240) == (1, 0, 0)
    assert result.cost_eur_per_km == pytest.approx(25000 + 0.8 * (100000 - 25000))
    assert result.cable_cost_eur == pytest.approx(0.1 * 85000)
    assert result.measure == "local"


def test_existing_double_cable_counts_at_its_nominal_capacity(staged_assumption):
    # Two existing cables of 0.27 kA carry 0.54 kA (no grouping derating).
    assert staged.select_route(_route(0.5, installed=0.54, cables=2), 2, _params(staged_assumption)).measure == "none"


def test_route_cap_escalates(staged_assumption):
    # Three added NAYY 240 at most: 0.27 + 3 x 0.357 = 1.341 kA < 1.5 kA.
    result = staged.select_route(_route(1.5), 2, _params(staged_assumption))
    assert result.measure == "unresolved" and result.heavy
    assert result.missing_capacity_ka == pytest.approx(1.5 - (0.27 + 3 * 0.357))
    assert result.cable_cost_eur == 0.0



def test_duct_share_override_is_clipped(staged_assumption):
    # All of the route in existing ducts: no trench, one NAYY 150 at its duct price.
    params = staged.StagedParameters.from_assumption(staged_assumption, duct_share_override=1.5)
    result = staged.select_route(_route(0.4), 2, params)
    assert (result.existing_duct_share, result.trenching_share) == (1.0, 0.0)
    assert result.cost_eur_per_km == 25000.0


@pytest.mark.parametrize(
    "share, label",
    [(0.2, "20"), (0.8, "80"), (0.125, "13"), (0.005, "1"), (0.0, "0"), (1.0, "100"), (0.12499999999999998, "13")],
)
def test_percent_label_rounds_half_up(share, label):
    assert staged.percent_label(share) == label

def test_route_measures_follow_the_route_type(staged_assumption):
    params = _params(staged_assumption)
    assert staged.select_route(_route(0.4, is_outlet=True), 2, params).measure == "outlet"
    assert staged.select_route(_route(0.4, is_service=True), 2, params).measure == "service"
    assert staged.select_route(_route(1.5, is_service=True), 2, params).measure == "unresolved_service"


# Stages 1-4 -------------------------------------------------------------------------------------


def _grid(key, rated, peak, routes=(), *, origin=(0.0, 0.0), settlement=2, **kwargs):
    coordinates = np.array([origin, (origin[0] + 10.0, origin[1])])
    return staged.GridInput(key=key, settlement_type=settlement, rated_kva=rated, peak_kva=peak,
                            routes=list(routes), coordinates=coordinates, **kwargs)


def _heavy_route():
    # 0.8 kA on one NAYY 150 needs two added cables (cheapest: 2 x NAYY 150, 0.27 + 0.54 = 0.81 kA).
    return staged.RouteInput(key="heavy", length_km=0.2, existing_cables=1, installed_capacity_ka=0.27,
                             p100_ka=0.8, is_outlet=True)


def _light_route():
    return staged.RouteInput(key="light", length_km=0.1, existing_cables=1, installed_capacity_ka=0.27,
                             p100_ka=0.4)


def test_load_transfer_to_a_neighbour(staged_assumption):
    params = _params(staged_assumption)
    over = _grid("A", 630, 1100, [_heavy_route(), _light_route()])
    neighbour = _grid("B", 630, 300, origin=(40.0, 0.0))  # buses 30 m apart
    states = staged.run_stages([over, neighbour], params)
    a = states[0]
    assert a.measure == "transfer" and a.transfer_partners == ["B"]
    assert a.transfer_kva == pytest.approx(100)  # 1100 - 1000
    assert a.load_transfer_cost_eur == 20000
    assert a.transformer_exchange_cost_eur() == 15000  # exchange to the 1000 kVA limit
    assert a.routes["heavy"].measure == "relieved_by_transfer" and a.routes["heavy"].cable_cost_eur == 0.0
    assert a.routes["heavy"].added_cables == 0 and a.routes["heavy"].heavy  # nothing is built there
    assert a.routes["light"].measure == "local" and a.routes["light"].cable_cost_eur > 0
    assert a.added_outlet_cables == 0  # the relieved outlet needs no new panel


def test_new_station_without_neighbours(staged_assumption):
    params = _params(staged_assumption, load_transfer_adjacency_m=0.0)
    over = _grid("A", 630, 1100, [_heavy_route()])
    neighbour = _grid("B", 630, 300, origin=(40.0, 0.0))
    a = staged.run_stages([over, neighbour], params)[0]
    assert a.measure == "new_station" and a.new_stations == pytest.approx(1.0)
    assert a.new_station_cost_eur == pytest.approx(145000)
    assert a.routes["heavy"].measure == "relieved_by_new_station"
    assert a.total_cost_eur(params) == pytest.approx(145000 + 15000)


def test_neighbours_over_the_limit_share_new_stations(staged_assumption):
    params = _params(staged_assumption)
    a = _grid("A", 800, 1400, settlement=2)  # excess 400
    b = _grid("B", 800, 1400, origin=(30.0, 0.0), settlement=2)  # excess 400, adjacent, no spare
    states = staged.run_stages([a, b], params)
    # ceil(800 / 630) = 2 stations for the cluster, one per grid.
    assert {s.new_station_cluster for s in states} == {"c1"}
    assert [s.new_stations for s in states] == pytest.approx([1.0, 1.0])


def test_spare_capacity_is_used_once(staged_assumption):
    params = _params(staged_assumption)
    a = _grid("A", 800, 1100)  # excess 100
    b = _grid("B", 800, 1200, origin=(0.0, 30.0))  # excess 200
    c = _grid("C", 630, 400, origin=(30.0, 0.0))  # spare 230, adjacent to both
    states = {s.key: s for s in staged.run_stages([a, b, c], params)}
    assert states["B"].measure == "transfer"  # larger excess first: takes 200 of 230
    assert states["A"].measure == "new_station"  # 30 left < 100


def test_trench_escalation_without_station_problem(staged_assumption):
    params = _params(staged_assumption, load_transfer_adjacency_m=0.0)
    route = staged.RouteInput(key="r", length_km=0.1, existing_cables=1, installed_capacity_ka=0.27,
                              p100_ka=1.5)
    state = staged.run_stages([_grid("A", 630, 400, [route])], params)[0]
    assert state.escalation == ["trench"] and state.measure == "new_station"
    assert state.routes["r"].measure == "relieved_by_new_station"


def test_panels_reported_and_optional_trigger(staged_assumption):
    route = staged.RouteInput(key="o", length_km=0.1, existing_cables=1, installed_capacity_ka=0.27,
                              p100_ka=0.4, is_outlet=True)
    state = staged.evaluate_grid(_grid("A", 630, 400, [route], existing_outlet_cables=12), _params(staged_assumption))
    assert (state.added_outlet_cables, state.panels_total, state.escalation) == (1, 13, [])
    triggered = staged.evaluate_grid(
        _grid("A", 630, 400, [route], existing_outlet_cables=12), _params(staged_assumption, panel_trigger=True)
    )
    assert triggered.escalation == ["panels"] and triggered.excess_kva > 0


def test_service_lines_stay_out_of_the_total(staged_assumption):
    params = _params(staged_assumption)
    service = staged.RouteInput(key="s", length_km=0.02, existing_cables=1, installed_capacity_ka=0.142,
                                p100_ka=0.2, is_service=True)
    state = staged.run_stages([_grid("A", 630, 400, [service])], params)[0]
    assert state.service_cost_eur() > 0 and state.cable_cost_eur(params) == 0.0
    included = _params(staged_assumption, service_lines_in_total=True)
    assert state.cable_cost_eur(included) == state.service_cost_eur()


# Stage 5 ----------------------------------------------------------------------------------------


def _voltage_grid(vm, busbar, *, measured_route=False, rated=630, peak=400):
    # Station (bus 0) -- route "feeder" -- bus 1 -- route "branch" -- bus 2 (0.3 km from the station)
    feeder = staged.RouteInput(key="feeder", length_km=0.2, existing_cables=1, installed_capacity_ka=0.27,
                               p100_ka=0.4 if measured_route else 0.1, is_outlet=True)
    branch = staged.RouteInput(key="branch", length_km=0.1, existing_cables=1, installed_capacity_ka=0.27,
                               p100_ka=0.1)
    return _grid(
        "A", rated, peak, [feeder, branch],
        route_above_bus={1: "feeder", 2: "branch"}, parent_bus={0: None, 1: 0, 2: 1},
        bus_distance_km={0: 0.0, 1: 0.2, 2: 0.3}, bus_min_voltage={1: 0.95, 2: vm}, lv_busbar_vm_pu=busbar,
    )


@pytest.mark.parametrize(
    ("vm", "busbar", "measured", "measure", "cost"),
    [
        (0.92, None, False, "not_assessed", 0.0),  # run solved before the Step 4 station voltage
        (0.92, 0.96, False, "none", 0.0),
        (0.88, 0.96, True, "resolved_by_measures", 0.0),  # the feeder above the bus is reinforced
        # 0.88 + (0.96 x 1.10 - 0.96) >= 0.90; the rONT replaces the kept 630 kVA unit: 12 + 12 kEUR
        (0.88, 0.96, False, "ront", 24000.0),
        # 0.78 + (1.056 - 1.0105) < 0.90: split at 2/3 of 0.3 km with one NAYY 240 (94 kEUR/km)
        (0.78, 0.96 / 0.95, False, "split", 2 / 3 * 0.3 * 94000),
    ],
)
def test_residual_voltage(staged_assumption, vm, busbar, measured, measure, cost):
    state = staged.run_stages([_voltage_grid(vm, busbar, measured_route=measured)], _params(staged_assumption))[0]
    assert state.voltage_measure == measure
    assert state.voltage_cost_eur == pytest.approx(cost)


def test_ront_on_an_exchanged_transformer_costs_the_premium(staged_assumption):
    # 400 kVA at a 500 kVA peak: stage 1 buys a 630 kVA unit (12 kEUR), the rONT adds its premium only
    grid = _voltage_grid(0.88, 0.96, rated=400, peak=500)
    state = staged.run_stages([grid], _params(staged_assumption))[0]
    assert state.station.exchange_cost_eur == 12000.0
    assert (state.voltage_measure, state.voltage_cost_eur) == ("ront", 12000.0)
