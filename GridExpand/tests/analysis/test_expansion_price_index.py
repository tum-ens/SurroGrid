"""scripts/expansion_price_index.py and migration 0011 (unit costs at 2025 prices)."""

from __future__ import annotations

import importlib.util

import pytest

from gridexpand.paths import PROJECT_DIR

spec = importlib.util.spec_from_file_location("expansion_price_index", PROJECT_DIR / "scripts" / "expansion_price_index.py")
price_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(price_index)

SOURCES, INDICES = price_index.load_sources(), price_index.load_indices()


def test_migration_is_generated_from_the_data():
    assert price_index.MIGRATION.read_text(encoding="utf-8") == price_index.migration_sql()


def test_index_series_cover_the_source_years_and_are_based_on_2021():
    for series in INDICES.values():
        assert series[2021] == pytest.approx(100.0, abs=0.05)
        assert set(range(2007, 2026)) <= set(series)
    years = {row["price_year"] for rows in SOURCES.values() for row in rows}
    assert min(years) >= 2007 and max(years) <= price_index.PRICE_YEAR


def test_the_rules_reproduce_the_values_of_0007_and_0010(staged_assumption):
    nominal = price_index.indexed_defaults(SOURCES, INDICES, indexed=False)
    expected = {**{p: staged_assumption[p] for p in nominal if p in staged_assumption}, "station_replacement_eur": 70000.0}
    differences = {p: (nominal[p], expected[p]) for p in nominal if nominal[p] != expected[p]}
    # documented changes: 400 kVA is the upper source value (0007 rounded it down); the parallel cables lose
    # the +5/+5/+10 kEUR/km margin that the index replaces
    assert differences == {
        "transformer_replace_400_eur": (10500.0, 10000.0),
        "line_parallel_150_eur_per_km": (20000.0, 25000.0),
        "line_parallel_185_eur_per_km": (40000.0, 45000.0),
        "line_parallel_240_eur_per_km": (60000.0, 70000.0),
    }


def test_indexed_defaults():
    values = price_index.indexed_defaults(SOURCES, INDICES)
    sizes = [values[f"transformer_replace_{kva}_eur"] for kva in (100, 160, 250, 400, 630, 800, 1000)]
    assert sizes == sorted(sizes)
    assert values["station_replacement_eur"] == values["new_station_eur"] - values["transformer_replace_1000_eur"]
    assert values["line_reopen_rural_eur_per_km"] < values["line_reopen_suburban_eur_per_km"] < values["line_reopen_urban_eur_per_km"]
    # PuBStadt 2021 parallel cable x construction index 2025/2021
    assert values["line_parallel_150_eur_per_km"] == round(20000 * INDICES["civil"][2025] / INDICES["civil"][2021], -3)


def test_transformer_index_before_2010_is_chained_with_gp19_27():
    trafo = INDICES["transformer"]
    assert trafo[2007] / trafo[2010] == pytest.approx(86.4 / 88.5)
