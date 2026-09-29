"""The seeded staged assumption rows document every parameter (db/sql/0007_staged_expansion.sql)."""

from __future__ import annotations

import json
import re

from gridexpand.paths import SQL_DIR

SQL = (SQL_DIR / "0007_staged_expansion.sql").read_text(encoding="utf-8")


def _default_row() -> tuple[dict[str, object], dict]:
    insert = SQL[SQL.index("INSERT INTO surrogrid.expansion_cost_assumption (\n    assumption_key"):]
    columns = re.findall(r"\w+", insert[insert.index("(") + 1:insert.index(")\nVALUES")])
    values = insert[insert.index("'staged_2026',\n") + len("'staged_2026',\n"):insert.index("    'Staged rules")]
    tokens = [token for token in re.split(r"[\s,]+", values) if token]
    parsed = [
        True if token == "TRUE" else False if token == "FALSE" else float(token) for token in tokens
    ]
    row = dict(zip(columns[3:3 + len(parsed)], parsed))
    assert columns[3 + len(parsed):] == ["source_note", "parameter_provenance"]
    provenance = json.loads(insert[insert.index("$prov$") + len("$prov$"):insert.index("$prov$::jsonb")])
    return row, provenance


def test_every_parameter_has_its_provenance_and_value():
    row, provenance = _default_row()
    assert set(row) | {"rule_set"} == set(provenance)
    for column, value in row.items():
        entry = provenance[column]
        assert entry["value"] == value, column
        assert entry["method"], column
        assert "sources" in entry, column


def test_test_fixture_equals_the_migration(staged_assumption):
    row, _ = _default_row()
    for column, value in row.items():
        assert staged_assumption[column] == value, column


def test_sensitivity_rows_override_documented_parameters():
    _, provenance = _default_row()
    keys = re.findall(r"'assumption_key', '(de_lv_staged_2026_\w+)'", SQL)
    assert keys == [
        "de_lv_staged_2026_low", "de_lv_staged_2026_high", "de_lv_staged_2026_trafo_allin",
        "de_lv_staged_2026_station_800", "de_lv_staged_2026_no_transfer",
    ]
    for block in SQL.split("INSERT INTO surrogrid.expansion_cost_assumption\nSELECT")[1:]:
        overridden = set(re.findall(r'"(\w+)": \{"value"', block)) | set(re.findall(r"'(\w+)', '\{\"value\"", block))
        assert overridden and overridden <= set(provenance), overridden
