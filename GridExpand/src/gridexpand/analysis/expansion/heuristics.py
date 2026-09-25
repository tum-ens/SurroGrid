"""LV reinforcement heuristic shared by the synthetic (SQL) and real (Python) paths.

The synthetic path evaluates the same rules in SQL (``sql/line_insert.sql``,
``sql/transformer_insert.sql``); ``tests/analysis/test_heuristics_sql_parity.py``
checks that both give identical selections, costs and labels. Cost values come
from one ``surrogrid.expansion_cost_assumption`` row (``assumption``).

Cable rule: the least-cost combination of parallel NAYY 4x150/185/240 cables whose
added ampacity covers the missing capacity (full integer search, ties broken by
fewer cables, less excess capacity, more 240 mm², more 185 mm²). The trench is
charged once per route: the most expensive selected cable type is replaced by the
settlement's reopening cost for the trenching share of the route.

Transformer rule: P100 apparent power rounded up to the capacity step
(``transformer_capacity_step_kva``) and priced with all-in replacement bins.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping

REINFORCEMENT_CATALOG = "NAYY_4_150|NAYY_4_185|NAYY_4_240"
# Tolerance [kA] / [steps] below which a capacity gap counts as covered.
CAPACITY_EPSILON = 1e-12
NO_EXPANSION_BASIS = "none_existing_capacity_sufficient"
TRANSFORMER_BINS = (
    (100.0, "transformer_replace_100_eur"),
    (160.0, "transformer_replace_160_eur"),
    (250.0, "transformer_replace_250_eur"),
    (400.0, "transformer_replace_400_eur"),
    (630.0, "transformer_replace_630_eur"),
    (800.0, "transformer_replace_800_eur"),
    (1000.0, "transformer_replace_1000_eur"),
)
STATION_REBUILD_BASIS = "station_rebuild_boundary_case_gt_1000kva"


def settlement_route(settlement_type: int | None, assumption: Mapping[str, Any]) -> tuple[float, str]:
    """Reopening cost [EUR/km] and label of a pylovo settlement type (1 rural, 3 urban, else semi-urban)."""
    if settlement_type == 1:
        return float(assumption["line_reopen_rural_eur_per_km"]), "rural"
    if settlement_type == 3:
        return float(assumption["line_reopen_urban_eur_per_km"]), "urban"
    return float(assumption["line_reopen_suburban_eur_per_km"]), "semiurban"


def existing_duct_share(assumption: Mapping[str, Any], override: float | None) -> float:
    """Share of the route with existing ducts (override or assumption), clipped to [0, 1]."""
    share = float(assumption["line_existing_duct_share"]) if override is None else float(override)
    return min(max(share, 0.0), 1.0)


def percent_label(share: float) -> str:
    """``share * 100`` rounded like PostgreSQL ``ROUND(x::NUMERIC)``.

    PostgreSQL converts a double to numeric with 15 significant digits and rounds
    halves away from zero (Python's ``round`` rounds halves to even).
    """
    value = Decimal(f"{share * 100.0:.15g}")
    return str(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def cost_basis_label(settlement_label: str, duct_share: float, n150: int, n185: int, n240: int) -> str:
    """Label of one selected cable combination, e.g. ``catalog_rural_duct20_trench80_150x1_185x0_240x0``."""
    return (
        f"catalog_{settlement_label}_duct{percent_label(duct_share)}_"
        f"trench{percent_label(1.0 - duct_share)}_"
        f"150x{n150}_185x{n185}_240x{n240}"
    )


def select_cable_reinforcement(
    *,
    required_added_capacity_ka: float,
    settlement_type: int | None,
    length_km: float,
    assumption: Mapping[str, Any],
    duct_share_override: float | None = None,
) -> dict[str, Any]:
    """Least-cost adequate combination of parallel reinforcement cables for one route.

    Args:
        required_added_capacity_ka: missing capacity (peak current minus installed capacity).
        settlement_type: pylovo settlement type of the postcode (None: semi-urban).
        length_km: route length that the per-km cost applies to.
        assumption: one ``expansion_cost_assumption`` row.
        duct_share_override: existing-duct share instead of the assumption's value.

    Returns:
        Shares, per-km and total cost, cost basis label, cable counts and added capacity.
    """
    reopen_cost, settlement_label = settlement_route(settlement_type, assumption)
    duct_share = existing_duct_share(assumption, duct_share_override)
    trenching_share = 1.0 - duct_share
    catalog = (
        (float(assumption["line_reinforcement_150_max_i_ka"]), float(assumption["line_parallel_150_eur_per_km"])),
        (float(assumption["line_reinforcement_185_max_i_ka"]), float(assumption["line_parallel_185_eur_per_km"])),
        (float(assumption["line_reinforcement_240_max_i_ka"]), float(assumption["line_parallel_240_eur_per_km"])),
    )
    required = max(float(required_added_capacity_ka), 0.0)
    if required <= CAPACITY_EPSILON:
        return {
            "line_existing_duct_share": duct_share,
            "line_trenching_share": trenching_share,
            "cost_eur_per_km": 0.0,
            "estimated_cost_eur": 0.0,
            "cost_basis": NO_EXPANSION_BASIS,
            "duct_cost_eur_per_km": 0.0,
            "reopen_cost_eur_per_km": reopen_cost,
            "reinforcement_150_count": 0,
            "reinforcement_185_count": 0,
            "reinforcement_240_count": 0,
            "reinforcement_added_capacity_ka": 0.0,
            "reinforcement_catalog": REINFORCEMENT_CATALOG,
        }

    limit = int(math.ceil(required / min(capacity for capacity, _ in catalog)))
    best_key: tuple[float, int, float, int, int] | None = None
    best: tuple[int, int, int, float, float, float] | None = None
    for n150 in range(limit + 1):
        for n185 in range(limit + 1):
            for n240 in range(limit + 1):
                count = n150 + n185 + n240
                if count == 0:
                    continue
                counts = (n150, n185, n240)
                added_capacity = sum(n * capacity for n, (capacity, _) in zip(counts, catalog))
                if added_capacity + CAPACITY_EPSILON < required:
                    continue
                total_duct_cost = sum(n * cost for n, (_, cost) in zip(counts, catalog))
                primary_duct_cost = max(cost for n, (_, cost) in zip(counts, catalog) if n > 0)
                cost_per_km = total_duct_cost + trenching_share * (reopen_cost - primary_duct_cost)
                key = (cost_per_km, count, added_capacity - required, -n240, -n185)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (n150, n185, n240, added_capacity, total_duct_cost, cost_per_km)
    if best is None:
        raise RuntimeError(f"No reinforcement combination covers {required:.6f} kA.")
    n150, n185, n240, added_capacity, total_duct_cost, cost_per_km = best
    return {
        "line_existing_duct_share": duct_share,
        "line_trenching_share": trenching_share,
        "cost_eur_per_km": cost_per_km,
        "estimated_cost_eur": length_km * cost_per_km,
        "cost_basis": cost_basis_label(settlement_label, duct_share, n150, n185, n240),
        "duct_cost_eur_per_km": total_duct_cost,
        "reopen_cost_eur_per_km": reopen_cost,
        "reinforcement_150_count": n150,
        "reinforcement_185_count": n185,
        "reinforcement_240_count": n240,
        "reinforcement_added_capacity_ka": added_capacity,
        "reinforcement_catalog": REINFORCEMENT_CATALOG,
    }


def required_transformer_kva(max_s_mva: float, step_kva: float) -> float:
    """P100 apparent power [kVA] rounded up to the capacity step (float noise tolerated)."""
    step = float(step_kva)
    return math.ceil(max_s_mva * 1000.0 / step - CAPACITY_EPSILON) * step


def transformer_upgrade_cost(
    required_kva: float, rated_kva: float, assumption: Mapping[str, Any]
) -> tuple[float, str]:
    """All-in replacement cost [EUR] and cost basis for a required transformer size."""
    if required_kva <= rated_kva:
        return 0.0, NO_EXPANSION_BASIS
    for capacity, column in TRANSFORMER_BINS:
        if required_kva <= capacity:
            return float(assumption[column]), f"all_in_replacement_to_{int(capacity)}kva"
    return float(assumption["transformer_station_rebuild_boundary_eur"]), STATION_REBUILD_BASIS
