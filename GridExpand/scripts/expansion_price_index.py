"""Unit costs of the staged expansion rules at one price year (offline build of migration 0011).

Every source value behind the monetary parameters of ``de_lv_staged_2026`` (``data/expansion_costs/source_values.csv``)
is escalated from its price year to ``PRICE_YEAR`` with Destatis indices (``data/expansion_costs/price_indices.csv``),
and each parameter is recomputed with the rule of migration 0007/0010 (median, upper value, single source or
derived; see ``RULES``):

- civil works (trenches, parallel cable laying, MV cable, LV links, load transfer): construction price index
  "Ortskanaele" (GENESIS 61261-0003, without VAT);
- transformers and the rONT premium: producer price index GP19-27114 "Elektrische Transformatoren"; 2007-2009 are
  chained with GP19-27 "Elektrische Ausruestungen" (GENESIS 61241-0003);
- new stations: the mean of both factors (about half housing and civil works, half equipment).

A source without a stated price base has the price year "publication year - 1" (recorded per row). The sensitivity
rows keep their spread: an override is scaled with the factor of its default parameter (``SENSITIVITIES``).

Usage::

    uv run python scripts/expansion_price_index.py            # print the table (nominal rules vs indexed)
    uv run python scripts/expansion_price_index.py --write    # (re)write src/gridexpand/db/sql/0011_price_year_2025.sql

The migration is generated, not edited by hand; ``tests/analysis/test_expansion_price_index.py`` checks that it
matches this script and the data files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "expansion_costs"
SQL_DIR = PROJECT_DIR / "src" / "gridexpand" / "db" / "sql"
MIGRATION = SQL_DIR / "0011_price_year_2025.sql"
PRICE_YEAR = 2025

INDEX_NAMES = {
    "civil": "Destatis construction price index Ortskanaele (61261-0003, without VAT)",
    "transformer": "Destatis producer price index GP19-27114 Elektrische Transformatoren",
    "station": "mean of the construction (Ortskanaele) and transformer (GP19-27114) factors",
}

# parameter: (rule, rounding step in EUR or EUR/km). Rules: max / median of the source values, the single source,
# or "same:<parameter>"; load_transfer_eur and station_replacement_eur are derived in ``indexed_defaults``.
RULES = {
    "transformer_replace_250_eur": ("max", 500),
    "transformer_replace_100_eur": ("same:transformer_replace_250_eur", 500),
    "transformer_replace_160_eur": ("same:transformer_replace_250_eur", 500),
    "transformer_replace_400_eur": ("max", 500),
    "transformer_replace_630_eur": ("median", 500),
    "transformer_replace_800_eur": ("median", 500),
    "transformer_replace_1000_eur": ("max", 500),
    "line_parallel_150_eur_per_km": ("single", 1000),
    "line_parallel_185_eur_per_km": ("single", 1000),
    "line_parallel_240_eur_per_km": ("single", 1000),
    "line_reopen_rural_eur_per_km": ("single", 5000),
    "line_reopen_suburban_eur_per_km": ("median", 5000),
    "line_reopen_urban_eur_per_km": ("median", 5000),
    "new_station_eur": ("median", 5000),
    "mv_cable_eur_per_km": ("median", 10000),
    "ront_premium_eur": ("median", 500),
}
LOAD_TRANSFER_LINK_KM = 0.05
LOAD_TRANSFER_STEP = 10000  # rounded up

# Method texts where the rule differs from the 0007/0010 text; the other parameters keep their text plus a note.
METHODS = {
    "transformer_replace_400_eur": "upper value of the four sources, each indexed to {year} prices, rounded to 500 EUR",
    "line_parallel_150_eur_per_km": "PuBStadt value ('Kabel und Montage', same trench) indexed to {year} prices, rounded to 1 kEUR/km "
                                    "(the +5 kEUR/km margin of 0007 is dropped: the index covers the price increase since 2021)",
    "line_parallel_185_eur_per_km": "PuBStadt value ('Kabel und Montage', same trench) indexed to {year} prices, rounded to 1 kEUR/km "
                                    "(the +5 kEUR/km margin of 0007 is dropped: the index covers the price increase since 2021)",
    "line_parallel_240_eur_per_km": "PuBStadt value ('Kabel und Montage', same trench) indexed to {year} prices, rounded to 1 kEUR/km "
                                    "(the +10 kEUR/km margin of 0007 is dropped: the index covers the price increase since 2021)",
    "load_transfer_eur": "derived: an LV link of 50 m at line_reopen_urban_eur_per_km plus a cable distribution cabinet "
                         "(PuBStadt 2021: 5 kEUR, indexed), rounded up to 10 kEUR for planning and switching; {year} prices",
    "station_replacement_eur": "derived: new_station_eur minus transformer_replace_1000_eur (the exchange pays the "
                               "transformer), both at {year} prices",
}

# Sensitivity rows: overrides scaled with the factor of their default parameter (nominal values of 0007), and
# parameters that keep the row's own value.
SENSITIVITIES = {
    "de_lv_staged_2026_low": {"scaled": {"new_station_eur": 60000.0, "mv_cable_eur_per_km": 200000.0, "load_transfer_eur": 10000.0}},
    "de_lv_staged_2026_high": {"scaled": {"new_station_eur": 150000.0, "mv_cable_eur_per_km": 400000.0, "load_transfer_eur": 30000.0}},
    "de_lv_staged_2026_trafo_allin": {"keep": [f"transformer_replace_{kva}_eur" for kva in (100, 160, 250, 400, 630, 800, 1000)]},
    "de_lv_staged_2026_station_800": {},
    "de_lv_staged_2026_no_transfer": {},
    "de_lv_staged_2026_no_station_replacement": {"keep": ["station_replacement_eur"]},
    "de_lv_staged_2026_line60": {},
    "de_lv_staged_2026_line80": {},
}


def load_indices(path: Path = DATA_DIR / "price_indices.csv") -> dict[str, dict[int, float]]:
    """Index series by year; GP19-27114 before 2010 chained with GP19-27."""
    series: dict[str, dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row["series"], {})[int(row["year"])] = float(row["value"])
    trafo, g27 = series["ppi_gp19_27114"], series["ppi_gp19_27"]
    first = min(trafo)
    for year in g27:
        if year < first:
            trafo[year] = trafo[first] * g27[year] / g27[first]
    return {"civil": series["construction_ortskanaele"], "transformer": trafo}


def factor(indices: dict[str, dict[int, float]], cost_type: str, year: int, price_year: int = PRICE_YEAR) -> float:
    civil = indices["civil"][price_year] / indices["civil"][year]
    trafo = indices["transformer"][price_year] / indices["transformer"][year]
    return {"civil": civil, "transformer": trafo, "station": 0.5 * civil + 0.5 * trafo}[cost_type]


def load_sources(path: Path = DATA_DIR / "source_values.csv") -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["value_eur"], row["price_year"] = float(row["value_eur"]), int(row["price_year"])
            sources.setdefault(row["parameter"], []).append(row)
    return sources


def _round(value: float, step: float) -> float:
    return float(step * math.floor(value / step + 0.5))


def indexed_defaults(sources, indices, price_year: int = PRICE_YEAR, indexed: bool = True) -> dict[str, float]:
    """The monetary parameters of the default row; ``indexed=False`` applies the rules to the nominal values."""
    def values(parameter):
        return [row["value_eur"] * (factor(indices, row["cost_type"], row["price_year"], price_year) if indexed else 1.0)
                for row in sources[parameter]]

    result: dict[str, float] = {}
    for parameter, (rule, step) in RULES.items():
        if rule.startswith("same:"):
            continue
        vals = values(parameter)
        value = {"max": max, "median": statistics.median, "single": lambda v: v[0]}[rule](vals)
        result[parameter] = _round(value, step)
    for parameter, (rule, _) in RULES.items():
        if rule.startswith("same:"):
            result[parameter] = result[rule.split(":", 1)[1]]
    cabinet = values("load_transfer_eur")[0]
    link = LOAD_TRANSFER_LINK_KM * result["line_reopen_urban_eur_per_km"]
    result["load_transfer_eur"] = float(LOAD_TRANSFER_STEP * math.ceil((link + cabinet) / LOAD_TRANSFER_STEP - 1e-9))
    result["station_replacement_eur"] = result["new_station_eur"] - result["transformer_replace_1000_eur"]
    return result


def _step(parameter: str) -> float:
    return LOAD_TRANSFER_STEP if parameter == "load_transfer_eur" else RULES[parameter][1]


# Provenance of 0007/0010 -------------------------------------------------------------------------------------------

def _default_provenance() -> dict[str, dict]:
    sql7 = (SQL_DIR / "0007_staged_expansion.sql").read_text(encoding="utf-8")
    provenance = json.loads(sql7[sql7.index("$prov$") + len("$prov$"):sql7.index("$prov$::jsonb")])
    sql10 = (SQL_DIR / "0010_line_limit_station_replacement.sql").read_text(encoding="utf-8")
    update = sql10[sql10.index("parameter_provenance || '") + len("parameter_provenance || '"):sql10.index("'::jsonb\nWHERE rule_set")]
    provenance.update(json.loads(update))
    return provenance


def _sensitivity_provenance(key: str) -> dict[str, dict]:
    sql7 = (SQL_DIR / "0007_staged_expansion.sql").read_text(encoding="utf-8")
    block = sql7[sql7.index(f"'assumption_key', '{key}'"):]
    block = block[block.index("a.parameter_provenance || '") + len("a.parameter_provenance || '"):]
    return json.loads(block[:block.index("'::jsonb")])


def default_entries(sources, indices, values, price_year: int = PRICE_YEAR) -> dict[str, dict]:
    """New provenance entries of the monetary parameters (value, method, sources, price year, indexing list)."""
    old = _default_provenance()
    entries = {}
    for parameter, value in values.items():
        entry = dict(old[parameter])
        method = METHODS.get(parameter)
        entry["method"] = (method.format(year=price_year) if method else
                           f"{entry['method']}; each source value indexed to {price_year} prices")
        entry["value"] = value
        entry["price_year"] = price_year
        if parameter in sources:
            entry["indexing"] = [
                {"source": row["source"], "value_eur": row["value_eur"], "price_year": row["price_year"],
                 "price_year_basis": row["price_year_basis"], "index": INDEX_NAMES[row["cost_type"]],
                 "factor": round(factor(indices, row["cost_type"], row["price_year"], price_year), 4),
                 f"value_{price_year}_eur": round(row["value_eur"] * factor(indices, row["cost_type"], row["price_year"], price_year))}
                for row in sources[parameter]
            ]
        entries[parameter] = entry
    return entries


# Migration --------------------------------------------------------------------------------------------------------

def _set_clause(values: dict[str, float]) -> str:
    return ",\n    ".join(f"{parameter} = {value:.1f}" for parameter, value in values.items())


def _update(key: str, values: dict[str, float], entries: dict[str, dict], price_year: int) -> str:
    provenance = json.dumps(entries, ensure_ascii=True, indent=None, sort_keys=True)
    return (
        "UPDATE surrogrid.expansion_cost_assumption\n"
        f"SET {_set_clause(values)},\n"
        f"    price_year = {price_year},\n"
        f"    parameter_provenance = parameter_provenance || $prov${provenance}$prov$::jsonb,\n"
        "    updated_at = now()\n"
        f"WHERE assumption_key = '{key}';\n"
    )


def migration_sql(price_year: int = PRICE_YEAR) -> str:
    sources, indices = load_sources(), load_indices()
    nominal = indexed_defaults(sources, indices, price_year, indexed=False)
    values = indexed_defaults(sources, indices, price_year)
    entries = default_entries(sources, indices, values, price_year)
    parts = [
        f"-- 0011 unit costs of the staged rules at {price_year} prices. GENERATED by scripts/expansion_price_index.py from\n"
        "-- data/expansion_costs/{source_values,price_indices}.csv; do not edit by hand.\n"
        "--\n"
        "-- Each source value is indexed from its price year with Destatis indices (civil works: construction price\n"
        "-- index Ortskanaele; transformers and rONT: producer price index GP19-27114; stations: the mean of both) and\n"
        "-- each parameter is recomputed with its rule of 0007/0010. de_lv_staged_2026_nominal keeps the values before\n"
        "-- this migration; the sensitivity rows scale their overrides with the factor of the default parameter.\n\n"
        "ALTER TABLE surrogrid.expansion_cost_assumption\n"
        "    ADD COLUMN IF NOT EXISTS price_year integer;\n\n"
        "INSERT INTO surrogrid.expansion_cost_assumption\n"
        "SELECT (jsonb_populate_record(a, jsonb_build_object(\n"
        "        'assumption_key', 'de_lv_staged_2026_nominal',\n"
        "        'description', 'Sensitivity of de_lv_staged_2026: unit costs at the price base of their sources "
        "(the values of 0007/0010, not indexed).',\n"
        "        'price_year', NULL,\n"
        "        'created_at', now(), 'updated_at', now()))).*\n"
        "FROM surrogrid.expansion_cost_assumption a\n"
        "WHERE a.assumption_key = 'de_lv_staged_2026'\n"
        "ON CONFLICT (assumption_key) DO NOTHING;\n\n",
        _update("de_lv_staged_2026", values, entries, price_year),
    ]
    for key, spec in SENSITIVITIES.items():
        row_values = {p: v for p, v in values.items() if p not in spec.get("keep", ())}
        row_entries = {p: e for p, e in entries.items() if p not in spec.get("keep", ())}
        if "scaled" in spec:
            old = _sensitivity_provenance(key)
        for parameter, nominal_value in spec.get("scaled", {}).items():
            scale = values[parameter] / nominal[parameter]
            row_values[parameter] = _round(nominal_value * scale, _step(parameter))
            entry = dict(old[parameter])
            entry["value"] = row_values[parameter]
            entry["method"] = (f"{entry['method']}; scaled with the {price_year} index factor of the default parameter "
                               f"({values[parameter]:.0f}/{nominal[parameter]:.0f})")
            entry["price_year"] = price_year
            row_entries[parameter] = entry
        parts.append("\n" + _update(key, row_values, row_entries, price_year))
    return "".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help=f"write {MIGRATION.relative_to(PROJECT_DIR)}")
    args = parser.parse_args(argv)
    sources, indices = load_sources(), load_indices()
    nominal = indexed_defaults(sources, indices, indexed=False)
    values = indexed_defaults(sources, indices)
    print(f"{'parameter':34} {'nominal rules':>14} {PRICE_YEAR:>10} {'change':>7}")
    for parameter, value in values.items():
        print(f"{parameter:34} {nominal[parameter]:>14,.0f} {value:>10,.0f} {100 * (value / nominal[parameter] - 1):>6.0f}%")
    if args.write:
        MIGRATION.write_text(migration_sql(), encoding="utf-8")
        print(f"written: {MIGRATION}")


if __name__ == "__main__":
    main()
