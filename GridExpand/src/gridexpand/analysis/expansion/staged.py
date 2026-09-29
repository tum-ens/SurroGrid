"""Staged LV expansion rules (rule set ``staged_2026``; docs/expansion_costs.md).

The rules follow the order of measures of Niederle et al. (EnInnov 2026) and the German planning
literature. Per grid (``evaluate_grid``):

1. **Station.** The transformer is exchanged for the smallest standard size that carries the
   P100 station load, up to the station limit of the settlement type. Above the limit the grid is
   *over the limit* by its excess load.
2. **Routes.** Each overloaded route gets the least-cost catalogue combination of parallel cables,
   at most ``line_max_added_cables``, at their nominal ratings (no grouping derating, as in German
   DSO planning practice). A route that the cap cannot resolve escalates the grid (excess = the load
   that must leave the route).
3. **Panels.** Existing station outlets plus the cables added on outlet routes (one NH way each);
   reported, and an escalation only if ``panel_trigger`` is set.

Per analysis (``resolve_over_limit``):

4. **Grids over the limit or escalated.** First the excess goes to neighbouring stations with spare
   capacity (load transfer; neighbours are grids whose buses come within ``load_transfer_adjacency_m``).
   Otherwise neighbouring remaining grids form a cluster that shares whole new substations. Routes
   that need two or more added cables (and unresolved routes) count as relieved; routes with one
   added cable keep their cost.

Per grid (``resolve_voltage``):

5. **Residual voltage.** Step 4 already applies the off-load tap (``powerflow.station_voltage``).
   A bus still below the band counts as resolved if a measure touches a route on its path to the
   station or the grid gets a new substation; otherwise the grid gets an rONT, or, beyond the rONT
   range, a feeder split at 2/3 of the distance to the farthest critical bus. The rONT costs its
   premium where stage 1 buys a new transformer anyway, and a full unit (conventional price of the
   station's size plus the premium) where it replaces the existing transformer.

All functions are pure. ``real_materialization`` and ``synthetic_materialization`` build the inputs
and write the rows. The former rule set ``heuristic_2026`` was retired on 2026-09-29; its assumption row
stays for the analyses written with it, and its code is in the git history (SurroGrid ``dev`` 8005175).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np

from gridexpand.powerflow.config import config as powerflow_config

RULE_SET = "staged_2026"
STANDARD_KVA = (100.0, 160.0, 250.0, 400.0, 630.0, 800.0, 1000.0)
LV_KV = 0.4
KVA_EPSILON = 1e-9
REINFORCEMENT_CATALOG = "NAYY_4_150|NAYY_4_185|NAYY_4_240"
# Tolerance [kA] below which a capacity gap counts as covered.
CAPACITY_EPSILON = 1e-12
NO_EXPANSION_BASIS = "none_existing_capacity_sufficient"

RELIEVED_BY = {"transfer": "relieved_by_transfer", "new_station": "relieved_by_new_station"}
# Route measures that change the path of a bus (stage 5 counts its voltage as resolved).
TOUCHING_MEASURES = frozenset({"local", "outlet", "service", *RELIEVED_BY.values()})


# Parameters -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedParameters:
    """The ``staged_2026`` columns of one ``expansion_cost_assumption`` row."""

    assumption: Mapping[str, Any]
    duct_share_override: float | None
    planning_limit: float
    station_max_kva: Mapping[int, float]
    transformer_cost: tuple[tuple[float, float], ...]
    max_added_cables: int
    panel_max: int
    panel_trigger: bool
    new_station_eur: float
    mv_loop_in_km: float
    mv_cable_eur_per_km: float
    lv_connection_km: float
    new_station_kva: float
    transfer_eur: float
    adjacency_m: float
    ront_premium_eur: float
    ront_range_percent: float
    service_lines_in_total: bool

    @classmethod
    def from_assumption(cls, row: Mapping[str, Any], duct_share_override: float | None = None) -> StagedParameters:
        """Parameters of a ``staged_2026`` row.

        Raises:
            ValueError: the row is of another rule set, or a parameter is missing.
        """
        if row.get("rule_set") != RULE_SET:
            raise ValueError(
                f"Assumption {row.get('assumption_key')!r} is not of rule set {RULE_SET!r}; rule set "
                f"{row.get('rule_set')!r} is retired (its analyses stay readable, its code is in git: dev 8005175)."
            )
        required = (
            "transformer_planning_limit", "station_max_kva_rural", "station_max_kva_suburban",
            "station_max_kva_urban", "line_max_added_cables", "panel_max", "panel_trigger", "new_station_eur", "new_station_mv_loop_in_km",
            "mv_cable_eur_per_km", "new_station_lv_connection_km", "new_station_kva", "load_transfer_eur",
            "load_transfer_adjacency_m", "ront_premium_eur", "ront_control_range_percent",
            "service_lines_in_total",
        )
        missing = [name for name in required if row.get(name) is None]
        if missing:
            raise ValueError(f"Assumption {row.get('assumption_key')!r} lacks {missing}.")
        return cls(
            assumption=row,
            duct_share_override=duct_share_override,
            planning_limit=float(row["transformer_planning_limit"]),
            station_max_kva={
                1: float(row["station_max_kva_rural"]),
                2: float(row["station_max_kva_suburban"]),
                3: float(row["station_max_kva_urban"]),
            },
            transformer_cost=tuple(
                (kva, float(row[f"transformer_replace_{int(kva)}_eur"])) for kva in STANDARD_KVA
            ),
            max_added_cables=int(row["line_max_added_cables"]),
            panel_max=int(row["panel_max"]),
            panel_trigger=bool(row["panel_trigger"]),
            new_station_eur=float(row["new_station_eur"]),
            mv_loop_in_km=float(row["new_station_mv_loop_in_km"]),
            mv_cable_eur_per_km=float(row["mv_cable_eur_per_km"]),
            lv_connection_km=float(row["new_station_lv_connection_km"]),
            new_station_kva=float(row["new_station_kva"]),
            transfer_eur=float(row["load_transfer_eur"]),
            adjacency_m=float(row["load_transfer_adjacency_m"]),
            ront_premium_eur=float(row["ront_premium_eur"]),
            ront_range_percent=float(row["ront_control_range_percent"]),
            service_lines_in_total=bool(row["service_lines_in_total"]),
        )

    def station_limit_kva(self, settlement_type: int | None) -> float:
        """Largest standard transformer of one station (semi-urban if the settlement type is unknown)."""
        return self.station_max_kva.get(settlement_type, self.station_max_kva[2])

    def new_station_cost_eur(self, settlement_type: int | None) -> float:
        """All-in cost of one new substation: station, MV loop-in and LV connection."""
        reopen_cost, _ = settlement_route(settlement_type, self.assumption)
        return (
            self.new_station_eur
            + self.mv_loop_in_km * self.mv_cable_eur_per_km
            + self.lv_connection_km * reopen_cost
        )


def settlement_route(settlement_type: int | None, assumption: Mapping[str, Any]) -> tuple[float, str]:
    """Trench cost [EUR/km] and label of a pylovo settlement type (1 rural, 3 urban, else semi-urban)."""
    if settlement_type == 1:
        return float(assumption["line_reopen_rural_eur_per_km"]), "rural"
    if settlement_type == 3:
        return float(assumption["line_reopen_urban_eur_per_km"]), "urban"
    return float(assumption["line_reopen_suburban_eur_per_km"]), "semiurban"


def existing_duct_share(assumption: Mapping[str, Any], override: float | None) -> float:
    """Share of a route with existing ducts (override or assumption), clipped to [0, 1]."""
    share = float(assumption["line_existing_duct_share"]) if override is None else float(override)
    return min(max(share, 0.0), 1.0)


def percent_label(share: float) -> str:
    """``share * 100`` rounded half up at 15 significant digits (the labels of all stored analyses)."""
    value = Decimal(f"{share * 100.0:.15g}")
    return str(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def cost_basis_label(settlement_label: str, duct_share: float, n150: int, n185: int, n240: int) -> str:
    """Label of one selected cable combination, e.g. ``catalog_rural_duct20_trench80_150x1_185x0_240x0``."""
    return (
        f"catalog_{settlement_label}_duct{percent_label(duct_share)}_"
        f"trench{percent_label(1.0 - duct_share)}_"
        f"150x{n150}_185x{n185}_240x{n240}"
    )


# Stage 1: station ---------------------------------------------------------------------------------


@dataclass
class StationResult:
    """Transformer decision of one grid (stage 1; stage 4 may turn ``over_limit`` into a measure)."""

    rated_kva: float
    peak_kva: float
    limit_kva: float
    measure: str  # none | exchange | over_limit
    required_kva: float
    excess_kva: float = 0.0
    exchange_cost_eur: float = 0.0
    cost_basis: str = NO_EXPANSION_BASIS


def _exchange_cost(size_kva: float, params: StagedParameters) -> float:
    for kva, eur in params.transformer_cost:
        if abs(kva - size_kva) < KVA_EPSILON:
            return eur
    raise ValueError(f"No transformer cost for {size_kva} kVA.")


def station_decision(
    rated_kva: float, peak_kva: float, settlement_type: int | None, params: StagedParameters
) -> StationResult:
    """Stage 1: keep, exchange, or over the limit.

    A station whose unit is already above the limit keeps it as its ceiling.
    """
    tau = params.planning_limit
    limit = params.station_limit_kva(settlement_type)
    ceiling = max(rated_kva, limit)
    if peak_kva <= tau * rated_kva + KVA_EPSILON:
        return StationResult(rated_kva, peak_kva, limit, "none", rated_kva)
    if peak_kva <= tau * ceiling + KVA_EPSILON:
        size = min(
            kva for kva in STANDARD_KVA
            if kva <= ceiling + KVA_EPSILON and tau * kva + KVA_EPSILON >= peak_kva and kva > rated_kva
        )
        return StationResult(
            rated_kva, peak_kva, limit, "exchange", size,
            exchange_cost_eur=_exchange_cost(size, params), cost_basis=f"exchange_to_{int(size)}kva",
        )
    excess = peak_kva - tau * ceiling
    if rated_kva + KVA_EPSILON < limit:
        size = max(kva for kva in STANDARD_KVA if kva <= limit + KVA_EPSILON)
        return StationResult(
            rated_kva, peak_kva, limit, "over_limit", size, excess_kva=excess,
            exchange_cost_eur=_exchange_cost(size, params), cost_basis=f"exchange_to_{int(size)}kva_limit",
        )
    return StationResult(rated_kva, peak_kva, limit, "over_limit", rated_kva, excess_kva=excess,
                         cost_basis="at_limit")


# Stage 2: routes ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteInput:
    """One cable route (a synthetic line component or a real corridor)."""

    key: Hashable
    length_km: float
    existing_cables: int
    installed_capacity_ka: float
    p100_ka: float
    is_outlet: bool = False
    is_service: bool = False


@dataclass
class RouteResult:
    """Decision and cost of one route."""

    key: Hashable
    measure: str  # none | local | outlet | service | unresolved | unresolved_service | relieved_by_*
    n150: int = 0
    n185: int = 0
    n240: int = 0
    added_capacity_ka: float = 0.0
    existing_duct_share: float = 0.0
    trenching_share: float = 1.0
    cost_eur_per_km: float = 0.0
    duct_cost_eur_per_km: float = 0.0
    reopen_cost_eur_per_km: float = 0.0
    cost_basis: str = NO_EXPANSION_BASIS
    cable_cost_eur: float = 0.0  # cost of the selected cables (0 when relieved)
    missing_capacity_ka: float = 0.0  # unresolved: capacity missing even with the maximum cables
    catalog: str = REINFORCEMENT_CATALOG

    @property
    def added_cables(self) -> int:
        return self.n150 + self.n185 + self.n240

    @property
    def heavy(self) -> bool:
        """Needs two or more added cables, or cannot be resolved: relieved by a stage-4 measure."""
        return (
            self.added_cables >= 2
            or self.measure in ("unresolved", "unresolved_service")
            or self.measure.startswith("relieved_by")
        )

    def relieve(self, measure: str) -> None:
        """A stage-4 measure takes the load off this route: no cable is built."""
        self.measure = measure
        self.n150 = self.n185 = self.n240 = 0
        self.added_capacity_ka = 0.0
        self.cost_eur_per_km = 0.0
        self.duct_cost_eur_per_km = 0.0
        self.cable_cost_eur = 0.0
        self.cost_basis = measure


def select_route(route: RouteInput, settlement_type: int | None, params: StagedParameters) -> RouteResult:
    """Stage 2: least-cost catalogue cables for one route under the cap (nominal ratings).

    Tie-breaks: fewer cables, less excess capacity, more 240 mm2, more 185 mm2 cables. The trench is
    charged once per route: the most expensive selected cable is replaced by the settlement's trench
    cost for the trenching share of the route.
    """
    assumption = params.assumption
    reopen_cost, settlement_label = settlement_route(settlement_type, assumption)
    duct_share = existing_duct_share(assumption, params.duct_share_override)
    trenching_share = 1.0 - duct_share
    base = RouteResult(
        route.key, "none", existing_duct_share=duct_share,
        trenching_share=trenching_share, reopen_cost_eur_per_km=reopen_cost,
    )
    if route.p100_ka <= route.installed_capacity_ka + CAPACITY_EPSILON:
        return base
    catalog = (
        (float(assumption["line_reinforcement_150_max_i_ka"]), float(assumption["line_parallel_150_eur_per_km"])),
        (float(assumption["line_reinforcement_185_max_i_ka"]), float(assumption["line_parallel_185_eur_per_km"])),
        (float(assumption["line_reinforcement_240_max_i_ka"]), float(assumption["line_parallel_240_eur_per_km"])),
    )
    cap = params.max_added_cables
    best_key = None
    best = None
    for n150 in range(cap + 1):
        for n185 in range(cap + 1 - n150):
            for n240 in range(cap + 1 - n150 - n185):
                count = n150 + n185 + n240
                if count == 0:
                    continue
                counts = (n150, n185, n240)
                added = sum(n * capacity for n, (capacity, _) in zip(counts, catalog))
                capacity_ka = route.installed_capacity_ka + added
                if capacity_ka + CAPACITY_EPSILON < route.p100_ka:
                    continue
                duct_total = sum(n * cost for n, (_, cost) in zip(counts, catalog))
                primary = max(cost for n, (_, cost) in zip(counts, catalog) if n > 0)
                cost_per_km = duct_total + trenching_share * (reopen_cost - primary)
                key = (cost_per_km, count, capacity_ka - route.p100_ka, -n240, -n185)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (n150, n185, n240, added, duct_total, cost_per_km)
    if best is None:
        full = route.installed_capacity_ka + cap * catalog[2][0]
        base.measure = "unresolved_service" if route.is_service else "unresolved"
        base.missing_capacity_ka = max(route.p100_ka - full, 0.0)
        base.cost_basis = f"unresolved_within_{cap}_cables"
        return base
    n150, n185, n240, added, duct_total, cost_per_km = best
    base.measure = "service" if route.is_service else ("outlet" if route.is_outlet else "local")
    base.n150, base.n185, base.n240 = n150, n185, n240
    base.added_capacity_ka = added
    base.cost_eur_per_km = cost_per_km
    base.duct_cost_eur_per_km = duct_total
    base.cost_basis = cost_basis_label(settlement_label, duct_share, n150, n185, n240)
    base.cable_cost_eur = route.length_km * cost_per_km
    return base


# Grid inputs and state ----------------------------------------------------------------------------


@dataclass
class GridInput:
    """Everything the stages need about one grid."""

    key: Hashable
    settlement_type: int | None
    rated_kva: float | None
    peak_kva: float | None
    routes: Sequence[RouteInput]
    existing_outlet_cables: int = 0
    # Topology for stage 5: route key of the edge above each bus, the parent bus, the distance to
    # the station along the tree, and the outlet route of each bus.
    route_above_bus: Mapping[int, Hashable] = field(default_factory=dict)
    parent_bus: Mapping[int, int | None] = field(default_factory=dict)
    bus_distance_km: Mapping[int, float] = field(default_factory=dict)
    bus_min_voltage: Mapping[int, float] = field(default_factory=dict)
    lv_busbar_vm_pu: float | None = None  # None: run solved before the Step 4 station voltage
    coordinates: np.ndarray | None = None  # (n, 2) in metres, for the neighbourhood


@dataclass
class GridState:
    """Stage results of one grid."""

    input: GridInput
    station: StationResult | None
    routes: dict[Hashable, RouteResult]
    escalation: list[str] = field(default_factory=list)
    excess_kva: float = 0.0
    added_outlet_cables: int = 0
    panels_total: int = 0
    measure: str = "none"  # none | transfer | new_station
    transfer_kva: float = 0.0
    transfer_partners: list[Hashable] = field(default_factory=list)
    new_station_cluster: str | None = None
    new_stations: float = 0.0
    load_transfer_cost_eur: float = 0.0
    new_station_cost_eur: float = 0.0
    voltage_residual_buses: int = 0
    voltage_measure: str = "not_assessed"
    voltage_cost_eur: float = 0.0

    @property
    def key(self) -> Hashable:
        return self.input.key

    @property
    def over_limit(self) -> bool:
        return self.excess_kva > KVA_EPSILON

    def cable_cost_eur(self, params: StagedParameters) -> float:
        """Route costs in the total (service lines only if ``service_lines_in_total``)."""
        return sum(
            result.cable_cost_eur for result in self.routes.values()
            if params.service_lines_in_total or result.measure != "service"
        )

    def service_cost_eur(self) -> float:
        return sum(result.cable_cost_eur for result in self.routes.values() if result.measure == "service")

    def transformer_exchange_cost_eur(self) -> float:
        return self.station.exchange_cost_eur if self.station is not None else 0.0

    def station_cost_eur(self) -> float:
        """Everything the transformer row carries: exchange, transfer, new substations, voltage."""
        return (
            self.transformer_exchange_cost_eur() + self.load_transfer_cost_eur
            + self.new_station_cost_eur + self.voltage_cost_eur
        )

    def total_cost_eur(self, params: StagedParameters) -> float:
        return self.cable_cost_eur(params) + self.station_cost_eur()

    @property
    def station_measure(self) -> str:
        if self.measure != "none":
            return self.measure
        if self.station is None:
            return "no_rating"
        return self.station.measure


def _kva(current_ka: float) -> float:
    return math.sqrt(3.0) * LV_KV * current_ka * 1000.0


def evaluate_grid(grid: GridInput, params: StagedParameters) -> GridState:
    """Stages 1-3 of one grid."""
    station = None
    if grid.rated_kva is not None and grid.rated_kva > 0 and grid.peak_kva is not None:
        station = station_decision(float(grid.rated_kva), float(grid.peak_kva), grid.settlement_type, params)
    routes = {route.key: select_route(route, grid.settlement_type, params) for route in grid.routes}
    state = GridState(grid, station, routes)
    if station is not None and station.measure == "over_limit":
        state.escalation.append("station_limit")
        state.excess_kva = station.excess_kva
    unresolved = [
        result.missing_capacity_ka for result in routes.values() if result.measure == "unresolved"
    ]
    if unresolved:
        state.escalation.append("trench")
        state.excess_kva = max(state.excess_kva, _kva(max(unresolved)))
    by_key = {route.key: route for route in grid.routes}
    state.added_outlet_cables = sum(
        result.added_cables for key, result in routes.items() if by_key[key].is_outlet
    )
    state.panels_total = int(grid.existing_outlet_cables) + state.added_outlet_cables
    allowed = max(params.panel_max, int(grid.existing_outlet_cables))
    if params.panel_trigger and state.panels_total > allowed:
        state.escalation.append("panels")
        surplus_cables = state.panels_total - allowed
        state.excess_kva = max(
            state.excess_kva,
            _kva(surplus_cables * float(params.assumption["line_reinforcement_240_max_i_ka"])),
        )
    return state


# Stage 4: grids over the limit ----------------------------------------------------------------------


def neighbours(states: Sequence[GridState], adjacency_m: float) -> dict[Hashable, list[Hashable]]:
    """Grids whose buses come within ``adjacency_m`` of each other (none if the distance is <= 0)."""
    result: dict[Hashable, list[Hashable]] = {state.key: [] for state in states}
    if adjacency_m <= 0:
        return result
    from scipy.spatial import cKDTree

    located = [state for state in states if state.input.coordinates is not None and len(state.input.coordinates)]
    trees = {state.key: cKDTree(state.input.coordinates) for state in located}
    boxes = {
        state.key: (state.input.coordinates.min(axis=0), state.input.coordinates.max(axis=0)) for state in located
    }
    for i, left in enumerate(located):
        for right in located[i + 1:]:
            (left_min, left_max), (right_min, right_max) = boxes[left.key], boxes[right.key]
            if np.any(left_min - adjacency_m > right_max) or np.any(right_min - adjacency_m > left_max):
                continue
            distance, _ = trees[right.key].query(left.input.coordinates, k=1, distance_upper_bound=adjacency_m)
            if np.isfinite(distance).any() and float(np.min(distance)) <= adjacency_m:
                result[left.key].append(right.key)
                result[right.key].append(left.key)
    return result


def _spare_kva(state: GridState, params: StagedParameters) -> float:
    """Spare capacity of a station after its own stage 1 (none for grids that need stage 4)."""
    if state.station is None or state.over_limit:
        return 0.0
    rating = state.station.required_kva if state.station.measure == "exchange" else state.station.rated_kva
    return max(params.planning_limit * rating - state.station.peak_kva, 0.0)


def resolve_over_limit(states: Sequence[GridState], params: StagedParameters) -> None:
    """Stage 4 over all grids of an analysis: load transfer first, then shared whole new substations."""
    candidates = [state for state in states if state.over_limit]
    if not candidates:
        return
    adjacency = neighbours(states, params.adjacency_m)
    spare = {state.key: _spare_kva(state, params) for state in states}
    by_key = {state.key: state for state in states}
    for state in sorted(candidates, key=lambda s: (-s.excess_kva, str(s.key))):
        partners = sorted((key for key in adjacency[state.key] if spare[key] > KVA_EPSILON),
                          key=lambda key: (-spare[key], str(key)))
        if sum(spare[key] for key in partners) + KVA_EPSILON < state.excess_kva:
            continue
        remaining = state.excess_kva
        for key in partners:
            take = min(spare[key], remaining)
            spare[key] -= take
            remaining -= take
            state.transfer_partners.append(key)
            if remaining <= KVA_EPSILON:
                break
        state.measure = "transfer"
        state.transfer_kva = state.excess_kva
        state.load_transfer_cost_eur = params.transfer_eur
    remaining_states = [state for state in candidates if state.measure != "transfer"]
    remaining_keys = {state.key for state in remaining_states}
    seen: set[Hashable] = set()
    cluster_index = 0
    for state in sorted(remaining_states, key=lambda s: str(s.key)):
        if state.key in seen:
            continue
        cluster_index += 1
        members = []
        stack = [state.key]
        seen.add(state.key)
        while stack:
            key = stack.pop()
            members.append(by_key[key])
            for other in adjacency[key]:
                if other in remaining_keys and other not in seen:
                    seen.add(other)
                    stack.append(other)
        total_excess = sum(member.excess_kva for member in members)
        n_new = math.ceil(total_excess / (params.planning_limit * params.new_station_kva) - 1e-9)
        n_new = max(n_new, 1)
        for member in members:
            share = n_new * member.excess_kva / total_excess
            member.measure = "new_station"
            member.new_station_cluster = f"c{cluster_index}"
            member.new_stations = share
            member.new_station_cost_eur = share * params.new_station_cost_eur(member.input.settlement_type)
    for state in candidates:
        relieved = RELIEVED_BY[state.measure]
        outlets = {route.key for route in state.input.routes if route.is_outlet}
        for result in state.routes.values():
            if result.heavy and result.measure not in ("service", "unresolved_service"):
                result.relieve(relieved)
        state.added_outlet_cables = sum(
            result.added_cables for key, result in state.routes.items()
            if key in outlets and result.measure == "outlet"
        )
        state.panels_total = int(state.input.existing_outlet_cables) + state.added_outlet_cables


# Stage 5: residual voltage ----------------------------------------------------------------------------


def _path_routes(bus: int, grid: GridInput) -> list[Hashable]:
    routes = []
    seen = set()
    while bus is not None and bus not in seen:
        seen.add(bus)
        route = grid.route_above_bus.get(bus)
        if route is not None:
            routes.append(route)
        bus = grid.parent_bus.get(bus)
    return routes


def _outlet_route(bus: int, grid: GridInput) -> Hashable | None:
    path = _path_routes(bus, grid)
    return path[-1] if path else None


def _ront_cost(state: GridState, params: StagedParameters) -> float:
    """rONT price: the premium on a transformer bought anyway, else a full unit replacing the old one.

    A station without a rating is priced at the size of a new station.
    """
    station = state.station
    if station is not None and station.exchange_cost_eur > 0.0:
        return params.ront_premium_eur
    rated = station.rated_kva if station is not None else params.new_station_kva
    size = next((kva for kva in STANDARD_KVA if kva + KVA_EPSILON >= rated), STANDARD_KVA[-1])
    return _exchange_cost(size, params) + params.ront_premium_eur


def resolve_voltage(state: GridState, params: StagedParameters) -> None:
    """Stage 5 of one grid (after stage 4)."""
    grid = state.input
    if grid.lv_busbar_vm_pu is None:
        state.voltage_measure = "not_assessed"
        return
    minimum = powerflow_config.MIN_VM_PU
    residual = {bus: vm for bus, vm in grid.bus_min_voltage.items() if vm < minimum - 1e-9}
    state.voltage_residual_buses = len(residual)
    if not residual:
        state.voltage_measure = "none"
        return
    if state.measure == "new_station":
        state.voltage_measure = "resolved_by_measures"
        return
    touched = {key for key, result in state.routes.items() if result.measure in TOUCHING_MEASURES}
    unresolved = {bus: vm for bus, vm in residual.items() if not touched.intersection(_path_routes(bus, grid))}
    if not unresolved:
        state.voltage_measure = "resolved_by_measures"
        return
    ceiling = powerflow_config.LV_REFERENCE_VOLTAGE_PU * (1.0 + params.ront_range_percent / 100.0)
    headroom = max(ceiling - float(grid.lv_busbar_vm_pu), 0.0)
    beyond = {bus: vm for bus, vm in unresolved.items() if vm + headroom < minimum - 1e-9}
    if not beyond:
        state.voltage_measure = "ront"
        state.voltage_cost_eur = _ront_cost(state, params)
        return
    reopen_cost, _ = settlement_route(grid.settlement_type, params.assumption)
    duct_share = existing_duct_share(params.assumption, params.duct_share_override)
    cable_240 = float(params.assumption["line_parallel_240_eur_per_km"])
    cost_per_km = cable_240 + (1.0 - duct_share) * (reopen_cost - cable_240)
    farthest: dict[Hashable, float] = {}
    for bus in beyond:
        outlet = _outlet_route(bus, grid)
        farthest[outlet] = max(farthest.get(outlet, 0.0), float(grid.bus_distance_km.get(bus, 0.0)))
    state.voltage_measure = "split"
    state.voltage_cost_eur = sum(2.0 / 3.0 * distance * cost_per_km for distance in farthest.values())


def run_stages(grids: Iterable[GridInput], params: StagedParameters) -> list[GridState]:
    """All stages for the grids of one analysis."""
    states = [evaluate_grid(grid, params) for grid in grids]
    resolve_over_limit(states, params)
    for state in states:
        resolve_voltage(state, params)
    return states
