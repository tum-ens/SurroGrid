"""Pure reinforcement heuristic (gridexpand.analysis.expansion.heuristics)."""

from __future__ import annotations

import pytest

from gridexpand.analysis.expansion import heuristics as h

# (required kA, settlement type) -> ((n150, n185, n240), EUR/km, basis), pinned from the
# pre-refactor real_materialization._line_cost with the seeded assumption row.
GOLDEN = [
    (0.0, 2, (0, 0, 0), 0.0, "none_existing_capacity_sufficient"),
    (0.05, 1, (1, 0, 0), 77000.0, "catalog_rural_duct20_trench80_150x1_185x0_240x0"),
    (0.28, 2, (0, 1, 0), 89000.0, "catalog_semiurban_duct20_trench80_150x0_185x1_240x0"),
    (0.30, 3, (0, 1, 0), 141000.0, "catalog_urban_duct20_trench80_150x0_185x1_240x0"),
    (0.34, 2, (0, 0, 1), 94000.0, "catalog_semiurban_duct20_trench80_150x0_185x0_240x1"),
    (0.36, None, (2, 0, 0), 110000.0, "catalog_semiurban_duct20_trench80_150x2_185x0_240x0"),
    (0.6, 1, (1, 0, 1), 111000.0, "catalog_rural_duct20_trench80_150x1_185x0_240x1"),
    (0.9, 3, (4, 0, 0), 212000.0, "catalog_urban_duct20_trench80_150x4_185x0_240x0"),
    (1.3, 2, (5, 0, 0), 185000.0, "catalog_semiurban_duct20_trench80_150x5_185x0_240x0"),
]


@pytest.mark.parametrize("required, settlement, counts, cost, basis", GOLDEN)
def test_cable_selection_golden(assumption, required, settlement, counts, cost, basis):
    result = h.select_cable_reinforcement(
        required_added_capacity_ka=required, settlement_type=settlement, length_km=0.5, assumption=assumption
    )
    assert (
        result["reinforcement_150_count"],
        result["reinforcement_185_count"],
        result["reinforcement_240_count"],
    ) == counts
    assert result["cost_eur_per_km"] == cost
    assert result["estimated_cost_eur"] == 0.5 * cost
    assert result["cost_basis"] == basis


def test_tiny_gap_counts_as_covered(assumption):
    result = h.select_cable_reinforcement(
        required_added_capacity_ka=1e-13, settlement_type=1, length_km=1.0, assumption=assumption
    )
    assert result["cost_basis"] == h.NO_EXPANSION_BASIS
    assert result["reopen_cost_eur_per_km"] == 90000.0


def test_duct_share_override_is_clipped(assumption):
    result = h.select_cable_reinforcement(
        required_added_capacity_ka=0.1, settlement_type=2, length_km=1.0, assumption=assumption,
        duct_share_override=1.5,
    )
    assert result["line_existing_duct_share"] == 1.0
    assert result["line_trenching_share"] == 0.0
    assert result["cost_eur_per_km"] == 25000.0


@pytest.mark.parametrize(
    "share, label",
    [(0.2, "20"), (0.8, "80"), (0.125, "13"), (0.005, "1"), (0.0, "0"), (1.0, "100"), (0.12499999999999998, "13")],
)
def test_percent_label_rounds_like_postgres(share, label):
    # PostgreSQL: ROUND((share * 100.0)::NUMERIC), 15 significant digits, halves away from zero.
    assert h.percent_label(share) == label


@pytest.mark.parametrize(
    "s_mva, rated, required, cost, basis",
    [
        (0.1, 250.0, 100.0, 0.0, "none_existing_capacity_sufficient"),
        (0.251, 250.0, 300.0, 33000.0, "all_in_replacement_to_400kva"),
        (0.25000000000000006, 250.0, 250.0, 0.0, "none_existing_capacity_sufficient"),
        (0.63, 400.0, 650.0, 42000.0, "all_in_replacement_to_800kva"),
        (1.2, 630.0, 1200.0, 100000.0, "station_rebuild_boundary_case_gt_1000kva"),
    ],
)
def test_transformer_rule(assumption, s_mva, rated, required, cost, basis):
    required_kva = h.required_transformer_kva(s_mva, assumption["transformer_capacity_step_kva"])
    assert required_kva == required
    assert h.transformer_upgrade_cost(max(rated, required_kva), rated, assumption) == (cost, basis)
