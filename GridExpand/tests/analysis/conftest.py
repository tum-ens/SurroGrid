"""Fixtures of the analysis tests.

The SQL parity tests need a PostgreSQL connection and run only when you name the
sandbox database of ``GridExpand/.env``::

    GRIDEXPAND_ANALYSIS_TEST_DATABASE=sg_impl_post uv run pytest tests/analysis

They only evaluate SELECT statements over VALUES lists (no tables are read or written).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parents[2]
REAL_DATABASE_PORT = "54327"  # the user's InfDB: never used by tests

# Values of the seeded ``de_lv_heuristic_2026`` row (db/sql/0001_baseline.sql).
ASSUMPTION = {
    "line_parallel_150_eur_per_km": 25000.0,
    "line_parallel_185_eur_per_km": 45000.0,
    "line_parallel_240_eur_per_km": 70000.0,
    "line_reinforcement_150_max_i_ka": 0.270,
    "line_reinforcement_185_max_i_ka": 0.313,
    "line_reinforcement_240_max_i_ka": 0.357,
    "line_existing_duct_share": 0.20,
    "line_reopen_rural_eur_per_km": 90000.0,
    "line_reopen_suburban_eur_per_km": 100000.0,
    "line_reopen_urban_eur_per_km": 165000.0,
    "transformer_replace_100_eur": 28000.0,
    "transformer_replace_160_eur": 28800.0,
    "transformer_replace_250_eur": 30000.0,
    "transformer_replace_400_eur": 33000.0,
    "transformer_replace_630_eur": 38000.0,
    "transformer_replace_800_eur": 42000.0,
    "transformer_replace_1000_eur": 48000.0,
    "transformer_station_rebuild_boundary_eur": 100000.0,
    "transformer_capacity_step_kva": 50,
}


@pytest.fixture()
def assumption() -> dict:
    return dict(ASSUMPTION)


@pytest.fixture(scope="session")
def sandbox_engine():
    """Engine of the sandbox database named in GRIDEXPAND_ANALYSIS_TEST_DATABASE (else skip)."""
    wanted = os.getenv("GRIDEXPAND_ANALYSIS_TEST_DATABASE")
    if not wanted:
        pytest.skip("set GRIDEXPAND_ANALYSIS_TEST_DATABASE=<sandbox db> to run the SQL parity tests")
    values = dotenv_values(PROJECT_DIR / ".env")
    port = str(values.get("DB_PORT", "")).strip('"')
    if port == REAL_DATABASE_PORT or values.get("DB_NAME") != wanted or not wanted.startswith("sg_"):
        pytest.skip("GridExpand/.env does not name this sandbox database")
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=values.get("DB_USER"),
            password=values.get("DB_PASSWORD"),
            host=values.get("DB_HOST"),
            port=int(port),
            database=wanted,
        )
    )
    yield engine
    engine.dispose()
