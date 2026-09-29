"""Fixtures of the analysis tests (no database)."""

from __future__ import annotations

import pytest

# The seeded ``de_lv_staged_2026`` row (db/sql/0007_staged_expansion.sql); test_staged_assumption.py
# checks that these values equal the migration.
STAGED_ASSUMPTION = {
    "assumption_key": "de_lv_staged_2026",
    "rule_set": "staged_2026",
    "line_parallel_150_eur_per_km": 25000.0,
    "line_parallel_185_eur_per_km": 45000.0,
    "line_parallel_240_eur_per_km": 70000.0,
    "line_reinforcement_150_max_i_ka": 0.270,
    "line_reinforcement_185_max_i_ka": 0.313,
    "line_reinforcement_240_max_i_ka": 0.357,
    "line_existing_duct_share": 0.20,
    "line_reopen_rural_eur_per_km": 80000.0,
    "line_reopen_suburban_eur_per_km": 100000.0,
    "line_reopen_urban_eur_per_km": 165000.0,
    "transformer_replace_100_eur": 8000.0,
    "transformer_replace_160_eur": 8000.0,
    "transformer_replace_250_eur": 8000.0,
    "transformer_replace_400_eur": 10000.0,
    "transformer_replace_630_eur": 12000.0,
    "transformer_replace_800_eur": 12500.0,
    "transformer_replace_1000_eur": 15000.0,
    "transformer_planning_limit": 1.0,
    "station_max_kva_rural": 1000.0,
    "station_max_kva_suburban": 1000.0,
    "station_max_kva_urban": 1000.0,
    "line_max_added_cables": 3,
    "panel_max": 12,
    "panel_trigger": False,
    "new_station_eur": 85000.0,
    "new_station_mv_loop_in_km": 0.2,
    "mv_cable_eur_per_km": 250000.0,
    "new_station_lv_connection_km": 0.1,
    "new_station_kva": 630.0,
    "load_transfer_eur": 20000.0,
    "load_transfer_adjacency_m": 50.0,
    "ront_premium_eur": 12000.0,
    "ront_control_range_percent": 10.0,
    "service_lines_in_total": False,
}


@pytest.fixture()
def staged_assumption() -> dict:
    return dict(STAGED_ASSUMPTION)
