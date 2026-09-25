"""The SQL twin of the reinforcement heuristic equals the Python helper (opt-in, see conftest.py).

Evaluates the production SQL fragments ``cable_selection.sql`` and
``transformer_cost.sql`` over VALUES lists (no table is read) and compares every
selection, cost and label with ``gridexpand.analysis.expansion.heuristics``.
"""

from __future__ import annotations

import itertools

import pytest
from sqlalchemy import text

from gridexpand.analysis.expansion import heuristics as h
from gridexpand.analysis.expansion.grid_expansion import sql_text

REQUIRED_KA = [0.0, 1e-13, 1e-12, 0.01, 0.05, 0.1, 0.2, 0.27, 0.2700000000001, 0.28, 0.3, 0.313, 0.34, 0.357,
               0.36, 0.5, 0.54, 0.6, 0.626, 0.7, 0.9, 1.07, 1.3, 1.6]
SETTLEMENTS = [1, 2, 3, None]
DUCT_OVERRIDES = [None, 0.0, 0.125, 0.3, 0.625, 1.0, 1.4, -0.1]


def _assumption_cte(assumption: dict) -> str:
    columns = list(assumption)
    values = ", ".join(
        f"{value}::integer" if column == "transformer_capacity_step_kva" else f"{float(value)!r}::double precision"
        for column, value in assumption.items()
    )
    return f"assumption AS (SELECT * FROM (VALUES ({values})) AS a({', '.join(columns)}))"


def _sql_number(value) -> str:
    return "NULL::integer" if value is None else str(int(value))


@pytest.mark.parametrize("duct_override", DUCT_OVERRIDES)
def test_cable_selection_sql_equals_python(sandbox_engine, assumption, duct_override):
    cases = list(itertools.product(REQUIRED_KA, SETTLEMENTS))
    rows = ", ".join(
        f"({i}, {required!r}::double precision, {_sql_number(settlement)}, 0.37::double precision, 1)"
        for i, (required, settlement) in enumerate(cases)
    )
    query = f"""
        WITH {_assumption_cte(assumption)},
        component_loading AS (
            SELECT * FROM (VALUES {rows})
                AS v(case_id, required_added_capacity_ka, settlement_type, component_length_km, component_parallel)
        ),
        selected AS ({sql_text("cable_selection.sql")})
        SELECT * FROM selected ORDER BY case_id
    """
    with sandbox_engine.connect() as conn:
        result = conn.execute(text(query), {"line_existing_duct_share": duct_override}).mappings().all()
    assert len(result) == len(cases)
    for row, (required, settlement) in zip(result, cases):
        expected = h.select_cable_reinforcement(
            required_added_capacity_ka=required,
            settlement_type=settlement,
            length_km=0.37,
            assumption=assumption,
            duct_share_override=duct_override,
        )
        assert (
            row["reinforcement_150_count"],
            row["reinforcement_185_count"],
            row["reinforcement_240_count"],
        ) == (
            expected["reinforcement_150_count"],
            expected["reinforcement_185_count"],
            expected["reinforcement_240_count"],
        ), (required, settlement)
        assert row["line_cost_eur_per_km"] == expected["cost_eur_per_km"]
        assert row["estimated_component_cost_eur"] == expected["estimated_cost_eur"]
        assert row["line_cost_basis"] == expected["cost_basis"]
        assert row["duct_cost_eur_per_km"] == expected["duct_cost_eur_per_km"]
        assert row["reopen_cost_eur_per_km"] == expected["reopen_cost_eur_per_km"]
        assert row["reinforcement_added_capacity_ka"] == expected["reinforcement_added_capacity_ka"]
        assert row["existing_duct_share"] == expected["line_existing_duct_share"]
        assert row["trenching_share"] == expected["line_trenching_share"]


def test_transformer_cost_sql_equals_python(sandbox_engine, assumption):
    s_values = [0.0, 0.05, 0.1, 0.1000000000000001, 0.16, 0.2, 0.25, 0.25000000000000006, 0.251, 0.4, 0.45,
                0.63, 0.7, 0.8, 0.95, 1.0, 1.0000000000000002, 1.2]
    ratings = [100.0, 160.0, 250.0, 400.0, 630.0]
    cases = list(itertools.product(s_values, ratings))
    rows = ", ".join(
        f"({i}, {s!r}::double precision, {rated!r}::double precision)" for i, (s, rated) in enumerate(cases)
    )
    query = f"""
        WITH {_assumption_cte(assumption)},
        transformer_peak AS (SELECT * FROM (VALUES {rows}) AS v(case_id, s_mva, rated_kva)),
        estimated AS ({sql_text("transformer_cost.sql")})
        SELECT case_id, GREATEST(required_kva, rated_kva) AS required_transformer_kva,
               estimated_cost_eur, transformer_cost_basis
        FROM estimated ORDER BY case_id
    """
    with sandbox_engine.connect() as conn:
        result = conn.execute(text(query)).mappings().all()
    for row, (s_mva, rated) in zip(result, cases):
        required = max(rated, h.required_transformer_kva(s_mva, assumption["transformer_capacity_step_kva"]))
        cost, basis = h.transformer_upgrade_cost(required, rated, assumption)
        assert row["required_transformer_kva"] == required, (s_mva, rated)
        assert (row["estimated_cost_eur"], row["transformer_cost_basis"]) == (cost, basis), (s_mva, rated)
