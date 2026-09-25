"""The migration files, the baseline parser and the scenario-tree table list (no database)."""

from __future__ import annotations

import re

import pytest

from gridexpand.db import maintenance, schema
from gridexpand.paths import SQL_DIR


def test_migrations_are_numbered_without_gaps() -> None:
    labels = [migration.label for migration in schema.migrations()]
    assert labels[0] == "0001_baseline"
    assert [int(label[:4]) for label in labels] == list(range(1, len(labels) + 1))


def test_baseline_tables_parse_every_table() -> None:
    tables = schema.baseline_tables()
    sql = schema.migrations()[0].sql
    assert set(tables) == set(re.findall(r"^CREATE TABLE surrogrid\.(\w+) \(", sql, re.M))
    assert {name for name, table in tables.items() if table.hypertable} == set(maintenance.COMPRESSION_SETTINGS)
    grid_case = dict((name, (sql_type, not_null)) for name, sql_type, not_null in tables["grid_case"].columns)
    assert grid_case["grid_case_id"] == ("bigint", True)
    assert grid_case["pylovo_version_id"] == ("character varying(10)", True)
    assert grid_case["created_at"] == ("timestamp with time zone", True)
    geom = dict((name, sql_type) for name, sql_type, _ in tables["expansion_real_line_result"].columns)["geom"]
    assert geom == "geometry(LineString,25832)"
    # Unique keys used by ON CONFLICT are required; keys added later are not.
    assert frozenset({"ags", "plz", "kcid", "bcid", "pylovo_grid_result_id"}) in tables["grid_case"].unique_keys
    assert frozenset({"ags", "pylovo_version_id", "plz", "kcid", "bcid"}) not in tables["grid_case"].unique_keys
    assert frozenset({"scenario_key"}) in tables["scenario"].unique_keys
    assert frozenset({"analysis_key"}) in tables["expansion_analysis_run"].unique_keys


def test_baseline_parser_rejects_unknown_lines() -> None:
    with pytest.raises(ValueError, match="Unrecognised line"):
        schema.baseline_tables("CREATE TABLE surrogrid.t (\n    a numeric(3)\n);")


def test_later_migrations_are_guarded() -> None:
    """0002.. must be no-ops on a database created from 0001 (idempotent statements)."""
    for migration in schema.migrations()[1:]:
        sql = "\n".join(line for line in migration.sql.splitlines() if not line.strip().startswith("--"))
        assert not re.search(r"DROP (INDEX|TABLE|CONSTRAINT)(?! IF EXISTS)", sql), migration.label
        assert not re.search(r"CREATE (UNIQUE )?INDEX(?! IF NOT EXISTS)", sql), migration.label
        assert not re.search(r"ADD CONSTRAINT", sql.replace("ADD CONSTRAINT %I", "")), migration.label


def test_views_sql_is_rerunnable() -> None:
    sql = (SQL_DIR / "views.sql").read_text(encoding="utf-8")
    assert "DROP " not in sql
    assert all(
        statement.startswith(("CREATE OR REPLACE VIEW", "CREATE MATERIALIZED VIEW IF NOT EXISTS", "CREATE UNIQUE INDEX IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS"))
        for statement in re.findall(r"^CREATE [A-Z ]+", sql, re.M)
    )
    assert set(re.findall(r"surrogrid\.(\w+) AS", sql)) == {name.split(".")[1] for name in schema.VIEWS}


def _references() -> dict[str, set[str]]:
    sql = "\n".join(migration.sql for migration in schema.migrations())
    return {
        match.group(1): set(re.findall(r"REFERENCES surrogrid\.(\w+)", match.group(2)))
        for match in re.finditer(r"^CREATE TABLE surrogrid\.(\w+) \(\n(.*?)\n\);", sql, re.M | re.S)
    }


def test_scenario_tree_lists_every_dependent_table() -> None:
    """Scenario counts/deletes cover every table with a foreign-key path to scenario."""
    references = _references()
    tree = {"scenario"}
    changed = True
    while changed:
        changed = False
        for table, parents in references.items():
            if table not in tree and parents & tree:
                tree.add(table)
                changed = True
    assert tree == set(maintenance.scenario_tree_tables())


def test_run_children_match_foreign_keys() -> None:
    references = _references()
    for run_table, children in maintenance.SCENARIO_RUNS.items():
        dependents = {table for table, parents in references.items() if run_table in parents}
        expansion = set(maintenance.EXPANSION_RESULT_TABLES)
        assert dependents - expansion == set(children), run_table


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"new_id": None, "n_natural": 1, "has_audit": True, "buildings_match": None, "has_runs": True, "old_id": 5}, "rejected"),
        ({"new_id": 7, "n_natural": 2, "has_audit": True, "buildings_match": True, "has_runs": True, "old_id": 5}, "rejected"),
        ({"new_id": 7, "n_natural": 1, "has_audit": True, "buildings_match": False, "has_runs": True, "old_id": 5}, "rejected"),
        ({"new_id": 7, "n_natural": 1, "has_audit": False, "buildings_match": None, "has_runs": True, "old_id": 5}, "rejected"),
        ({"new_id": 7, "n_natural": 1, "has_audit": False, "buildings_match": None, "has_runs": False, "old_id": 5}, "relink"),
        ({"new_id": 7, "n_natural": 1, "has_audit": True, "buildings_match": True, "has_runs": True, "old_id": 5}, "relink"),
        ({"new_id": 5, "n_natural": 1, "has_audit": True, "buildings_match": True, "has_runs": True, "old_id": 5}, "unchanged"),
    ],
)
def test_classify_relink(row: dict, expected: str) -> None:
    assert maintenance.classify_relink(row)[0] == expected


def test_classify_relink_accepts_unverified_on_request() -> None:
    row = {"new_id": 7, "n_natural": 1, "has_audit": False, "buildings_match": None, "has_runs": True, "old_id": 5}
    assert maintenance.classify_relink(row, accept_unverified=True)[0] == "relink"


def test_db_cli_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        maintenance.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in maintenance.COMMANDS:
        assert command in out
