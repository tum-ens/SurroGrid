"""SurroGrid database maintenance (``gridexpand db <command>``).

None of these commands runs automatically; each defaults to a dry run.

- ``init-schema``: create the schema on a database without one.
- ``migrate --plan | --apply``: check and migrate an existing database.
- ``compress --plan | --apply``: TimescaleDB compression of the raw hourly
  tables, one transaction per chunk (needs the ``timescale`` licence).
- ``relink-pylovo --plan | --apply``: point grid cases at the current pylovo
  ``grid_result`` ids after a pylovo re-generation, then restore the RESTRICT
  foreign key and the views.
- ``delete-scenario <scenario_key> [--execute]``: delete one scenario's rows
  and Step 3 files. ``gridexpand db <scenario_key> ...`` is the old spelling.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from gridexpand.db import schema
from gridexpand.db.engine import get_engine
from gridexpand.db.runs import (
    DEMAND_ALLOCATION_CHILDREN,
    POWERFLOW_CHILDREN,
    REAL_POWERFLOW_CHILDREN,
)
from gridexpand.paths import (
    OPTIMIZATION_INPUT_DIR,
    OPTIMIZATION_LOGS_DIR,
    OPTIMIZATION_RESULT_DIR,
)

STEP3_ARTIFACT_DIRS = (
    OPTIMIZATION_INPUT_DIR,
    OPTIMIZATION_RESULT_DIR,
    OPTIMIZATION_LOGS_DIR,
    OPTIMIZATION_LOGS_DIR / "gurobi",
)
COMMANDS = ("delete-scenario", "init-schema", "migrate", "compress", "relink-pylovo")
Echo = Callable[[str], None]


# Scenario deletion ----------------------------------------------------------------

EXPANSION_RESULT_TABLES = (
    "expansion_line_result",
    "expansion_transformer_result",
    "expansion_real_grid_status",
    "expansion_real_line_result",
    "expansion_real_transformer_result",
)
QGIS_VIEW_TABLES = ("expansion_line_qgis_mv", "expansion_transformer_qgis_mv")
DEMAND_TABLES = ("scenario", "pipeline_run", "demand_allocation_run", *DEMAND_ALLOCATION_CHILDREN)
# Run tables of a scenario and their result tables (deleted per run).
SCENARIO_RUNS = {
    "demand_allocation_run": DEMAND_ALLOCATION_CHILDREN,
    "powerflow_run": POWERFLOW_CHILDREN,
    "real_powerflow_run": REAL_POWERFLOW_CHILDREN,
}
# Analyses of a scenario: its scenario_id, or none (never set before 0003)
# while every result row belongs to the scenario.
_ANALYSES_SQL = """
SELECT ar.expansion_analysis_run_id
FROM surrogrid.expansion_analysis_run ar
WHERE ar.scenario_id = :scenario_id
   OR (ar.scenario_id IS NULL AND EXISTS (
        SELECT 1 FROM ({results}) r
        WHERE r.expansion_analysis_run_id = ar.expansion_analysis_run_id
        GROUP BY r.expansion_analysis_run_id
        HAVING bool_and(r.scenario_id = :scenario_id)))
"""
_RESULTS_UNION = " UNION ALL ".join(
    f"SELECT expansion_analysis_run_id, scenario_id FROM surrogrid.{table}" for table in EXPANSION_RESULT_TABLES
)


def scenario_tree_tables() -> tuple[str, ...]:
    """Every table whose rows belong to a scenario (in counting order)."""
    tables = ["scenario", "pipeline_run"]
    for run_table, children in SCENARIO_RUNS.items():
        tables += [run_table, *children]
    return (*tables, "expansion_analysis_run", *EXPANSION_RESULT_TABLES)


def _scenario_id(conn: Connection, scenario_key: str) -> int:
    value = conn.execute(
        text("SELECT scenario_id FROM surrogrid.scenario WHERE scenario_key = :key"), {"key": scenario_key}
    ).scalar_one_or_none()
    if value is None:
        raise ValueError(f"No surrogrid.scenario row found for scenario_key={scenario_key!r}.")
    return int(value)


def _analysis_ids(conn: Connection, scenario_id: int) -> list[int]:
    rows = conn.execute(text(_ANALYSES_SQL.format(results=_RESULTS_UNION)), {"scenario_id": scenario_id})
    return [int(row[0]) for row in rows]


def _count(conn: Connection, sql: str, **params: Any) -> int:
    return int(conn.execute(text(sql), params).scalar_one())


def _scenario_counts(conn: Connection, scenario_id: int, analyses: list[int]) -> dict[str, int]:
    params = {"scenario_id": scenario_id, "analyses": analyses}
    counts = {"scenario": 1}
    counts["pipeline_run"] = _count(
        conn, "SELECT count(*) FROM surrogrid.pipeline_run WHERE scenario_id = :scenario_id", **params
    )
    for run_table, children in SCENARIO_RUNS.items():
        counts[run_table] = _count(
            conn, f"SELECT count(*) FROM surrogrid.{run_table} WHERE scenario_id = :scenario_id", **params
        )
        for child in children:
            counts[child] = _count(
                conn,
                f"""SELECT count(*) FROM surrogrid.{child} c
                    JOIN surrogrid.{run_table} r USING ({run_table}_id)
                    WHERE r.scenario_id = :scenario_id""",
                **params,
            )
    counts["expansion_analysis_run"] = len(analyses)
    for table in EXPANSION_RESULT_TABLES:
        counts[table] = _count(
            conn,
            f"""SELECT count(*) FROM surrogrid.{table}
                WHERE scenario_id = :scenario_id OR expansion_analysis_run_id = ANY(:analyses)""",
            **params,
        )
    for view in QGIS_VIEW_TABLES:
        if conn.execute(text("SELECT to_regclass(:name)"), {"name": f"surrogrid.{view}"}).scalar() is not None:
            counts[view] = _count(
                conn,
                f"""SELECT count(*) FROM surrogrid.{view}
                    WHERE scenario_id = :scenario_id OR expansion_analysis_run_id = ANY(:analyses)""",
                **params,
            )
    return counts


def count_scenario_data(engine: Engine, scenario_key: str) -> dict[str, int]:
    """Rows per table that deleting ``scenario_key`` would remove."""
    schema.ensure_schema(engine)
    with engine.connect() as conn:
        scenario_id = _scenario_id(conn, scenario_key)
        return _scenario_counts(conn, scenario_id, _analysis_ids(conn, scenario_id))


def delete_scenario_data(
    engine: Engine,
    scenario_key: str,
    *,
    keep_demands: bool = False,
    dry_run: bool = True,
    refresh_views: bool = True,
) -> dict[str, int]:
    """Delete the rows of one scenario key; returns the counts before deletion.

    Deletes the scenario's expansion analyses, then each power-flow and
    real-grid power-flow run in its own transaction (bounded transactions on
    large raw tables), then (unless ``keep_demands``) each Step 2 run and the
    scenario row. ``keep_demands`` keeps the scenario, pipeline and Step 2
    rows and deletes all downstream results (real-grid runs included).
    """
    schema.ensure_schema(engine)
    with engine.connect() as conn:
        scenario_id = _scenario_id(conn, scenario_key)
        analyses = _analysis_ids(conn, scenario_id)
        counts = _scenario_counts(conn, scenario_id, analyses)
        run_ids = {
            run_table: [
                int(row[0])
                for row in conn.execute(
                    text(f"SELECT {run_table}_id FROM surrogrid.{run_table} WHERE scenario_id = :sid ORDER BY 1"),
                    {"sid": scenario_id},
                )
            ]
            for run_table in SCENARIO_RUNS
        }
    if dry_run:
        return counts
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM surrogrid.expansion_analysis_run WHERE expansion_analysis_run_id = ANY(:ids)"),
            {"ids": analyses},
        )
    run_tables = ["powerflow_run", "real_powerflow_run"] + ([] if keep_demands else ["demand_allocation_run"])
    for run_table in run_tables:
        for run_id in run_ids[run_table]:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM surrogrid.{run_table} WHERE {run_table}_id = :id"), {"id": run_id})
    if not keep_demands:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM surrogrid.scenario WHERE scenario_id = :sid"), {"sid": scenario_id})
    if refresh_views:
        schema.refresh_qgis_views(engine)
    return counts


def _step3_artifact_names(engine: Engine, scenario_key: str) -> tuple[set[str], set[str]]:
    query = text(
        """
        WITH selected_scenario AS (
            SELECT scenario_id FROM surrogrid.scenario WHERE scenario_key = :scenario_key
        ), artifact_names AS (
            SELECT bridge_filename AS filename
            FROM surrogrid.demand_allocation_run
            WHERE scenario_id = (SELECT scenario_id FROM selected_scenario)
            UNION
            SELECT urbs_input_file AS filename
            FROM surrogrid.powerflow_run
            WHERE scenario_id = (SELECT scenario_id FROM selected_scenario)
        )
        SELECT filename FROM artifact_names WHERE filename IS NOT NULL AND filename <> ''
        """
    )
    schema.ensure_schema(engine)
    with engine.connect() as conn:
        filenames = {Path(str(row.filename)).name for row in conn.execute(query, {"scenario_key": scenario_key})}
    log_prefixes = {f"{Path(filename).stem}_PV" for filename in filenames}
    log_prefixes.update(f"{Path(filename).stem}_" for filename in filenames if "_PV" in Path(filename).stem)
    return filenames, log_prefixes


def _step3_artifacts_for_scenario(engine: Engine, scenario_key: str) -> list[Path]:
    exact_filenames, log_prefixes = _step3_artifact_names(engine, scenario_key)
    matches: list[Path] = []
    directories = list(STEP3_ARTIFACT_DIRS)
    if Path(scenario_key).name == scenario_key and scenario_key not in {".", ".."}:
        # Step 3 writes its results to result/<scenario_key>/.
        directories.append(OPTIMIZATION_RESULT_DIR / scenario_key)
    for directory in directories:
        if not directory.exists():
            continue
        is_log_dir = directory.name == "logs" or directory.parent.name == "logs"
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.name in exact_filenames:
                matches.append(path)
            elif is_log_dir and any(path.name.startswith(prefix) for prefix in log_prefixes):
                matches.append(path)
    return sorted(set(matches))


# Compression ------------------------------------------------------------------------

# table -> (segmentby, orderby): one batch per run and stage, ordered by asset and time.
COMPRESSION_SETTINGS = {
    "powerflow_bus_voltage": ("powerflow_run_id, stage", "bus, t_index"),
    "powerflow_line_result": ("powerflow_run_id, stage", "line, t_index"),
    "powerflow_demand": ("powerflow_run_id, stage", "bus, t_index"),
    "powerflow_import": ("powerflow_run_id, stage", "t_index"),
    "powerflow_reactive_component": ("powerflow_run_id", "bus, component, source, t_index"),
    "allocated_demand": ("demand_allocation_run_id, commodity", "bus, t_index"),
    "allocated_eff_factor": ("demand_allocation_run_id, component", "bus, t_index"),
}


def compress_raw_chunks(
    engine: Engine,
    *,
    tables: tuple[str, ...] | None = None,
    apply: bool = False,
    echo: Echo = print,
) -> int:
    """Enable compression on the raw hypertables and compress their chunks.

    One transaction per chunk; already compressed chunks are skipped and
    partially compressed ones (rows written after compression) recompressed.
    Returns a process exit code.
    """
    tables = tuple(COMPRESSION_SETTINGS) if not tables else tables
    unknown = sorted(set(tables) - set(COMPRESSION_SETTINGS))
    if unknown:
        echo(f"Unknown raw tables: {unknown}")
        return 2
    schema.ensure_schema(engine)
    with engine.connect() as conn:
        licence = conn.execute(text("SHOW timescaledb.license")).scalar()
        if licence != "timescale":
            echo(f"timescaledb.license is {licence!r}; compression needs the 'timescale' licence.")
            return 1
        plan = []
        for table in tables:
            enabled = bool(
                conn.execute(
                    text(
                        "SELECT compression_enabled FROM timescaledb_information.hypertables "
                        "WHERE hypertable_schema = 'surrogrid' AND hypertable_name = :t"
                    ),
                    {"t": table},
                ).scalar()
            )
            chunks = conn.execute(
                text(
                    "SELECT format('%I.%I', chunk_schema, chunk_name), is_compressed "
                    "FROM timescaledb_information.chunks "
                    "WHERE hypertable_schema = 'surrogrid' AND hypertable_name = :t ORDER BY range_start"
                ),
                {"t": table},
            ).all()
            plan.append((table, enabled, chunks))
        conn.commit()
    for table, enabled, chunks in plan:
        pending = sum(1 for _, compressed in chunks if not compressed)
        segmentby, orderby = COMPRESSION_SETTINGS[table]
        echo(
            f"surrogrid.{table}: compression {'enabled' if enabled else 'to enable'} "
            f"(segmentby {segmentby}; orderby {orderby}); {len(chunks)} chunks, {pending} uncompressed"
        )
    if not apply:
        echo("Dry run (--plan): nothing was changed.")
        return 0
    for table, enabled, chunks in plan:
        segmentby, orderby = COMPRESSION_SETTINGS[table]
        if not enabled:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"ALTER TABLE surrogrid.{table} SET (timescaledb.compress, "
                        f"timescaledb.compress_segmentby = '{segmentby}', "
                        f"timescaledb.compress_orderby = '{orderby}')"
                    )
                )
        for chunk, _ in chunks:
            with engine.begin() as conn:
                conn.execute(text("SELECT compress_chunk(CAST(:c AS regclass), if_not_compressed => true)"), {"c": chunk})
        with engine.connect() as conn:
            stats = conn.execute(
                text(
                    "SELECT before_compression_total_bytes, after_compression_total_bytes "
                    "FROM hypertable_compression_stats(CAST(:t AS regclass))"
                ),
                {"t": f"surrogrid.{table}"},
            ).first()
        if stats and stats[0]:
            echo(f"surrogrid.{table}: {stats[0] / 1e6:.1f} MB -> {stats[1] / 1e6:.1f} MB")
    return 0


# pylovo relink ------------------------------------------------------------------------

RELINK_PLAN_SQL = """
WITH latest_audit AS (
    SELECT DISTINCT ON (dar.grid_case_id) dar.grid_case_id, dar.demand_allocation_run_id
    FROM surrogrid.demand_allocation_run dar
    WHERE EXISTS (
        SELECT 1 FROM surrogrid.demand_component_audit a
        WHERE a.demand_allocation_run_id = dar.demand_allocation_run_id
    )
    ORDER BY dar.grid_case_id, dar.updated_at DESC, dar.demand_allocation_run_id DESC
),
audit AS (
    SELECT la.grid_case_id, array_agg(DISTINCT a.objectid ORDER BY a.objectid) AS objectids
    FROM latest_audit la
    JOIN surrogrid.demand_component_audit a USING (demand_allocation_run_id)
    GROUP BY la.grid_case_id
),
target AS (
    SELECT
        gc.grid_case_id, gc.ags, gc.plz, gc.kcid, gc.bcid, gc.pylovo_version_id,
        gc.pylovo_grid_result_id AS old_id,
        gr.grid_result_id AS new_id,
        count(*) OVER (PARTITION BY gc.ags, gc.pylovo_version_id, gc.plz, gc.kcid, gc.bcid) AS n_natural
    FROM surrogrid.grid_case gc
    LEFT JOIN pylovo.grid_result gr
      ON gr.version_id::text = gc.pylovo_version_id::text
     AND gr.plz = gc.plz AND gr.kcid = gc.kcid AND gr.bcid = gc.bcid
)
SELECT
    t.*,
    a.objectids IS NOT NULL AS has_audit,
    a.objectids = (
        SELECT array_agg(b.objectid::text ORDER BY b.objectid::text)
        FROM pylovo.buildings_result b
        WHERE b.grid_result_id = t.new_id AND b.version_id::text = t.pylovo_version_id::text
    ) AS buildings_match,
    EXISTS (SELECT 1 FROM surrogrid.pipeline_run p WHERE p.grid_case_id = t.grid_case_id) AS has_runs
FROM target t
LEFT JOIN audit a USING (grid_case_id)
ORDER BY t.grid_case_id
"""


def relink_plan(conn: Connection, *, accept_unverified: bool = False) -> list[dict[str, Any]]:
    """Classify every grid case: ``unchanged``, ``relink`` or ``rejected`` (with reason).

    A grid case is matched to the current pylovo grid with the same version,
    plz, kcid and bcid; it is accepted only if the building set of its newest
    Step 2 component audit equals that grid's buildings (grid cases without
    any run need no evidence).
    """
    rows = [dict(row) for row in conn.execute(text(RELINK_PLAN_SQL)).mappings()]
    for row in rows:
        row["status"], row["reason"] = classify_relink(row, accept_unverified=accept_unverified)
    return rows


def classify_relink(row: dict[str, Any], *, accept_unverified: bool = False) -> tuple[str, str]:
    """``(status, reason)`` of one grid case of :data:`RELINK_PLAN_SQL`."""
    if row["new_id"] is None:
        return "rejected", "no pylovo grid with this version/plz/kcid/bcid"
    if row["n_natural"] > 1:
        return "rejected", "several grid cases share this (ags, version, plz, kcid, bcid)"
    if row["has_audit"] and not row["buildings_match"]:
        return "rejected", "building set differs from the grid case's Step 2 audit"
    if not row["has_audit"] and row["has_runs"] and not accept_unverified:
        return "rejected", "runs exist but no Step 2 component audit to verify the building set"
    return ("unchanged" if int(row["old_id"]) == int(row["new_id"]) else "relink"), ""


def _constraint(conn: Connection, table: str, name: str) -> tuple[bool, bool]:
    row = conn.execute(
        text("SELECT convalidated FROM pg_constraint WHERE conrelid = CAST(:t AS regclass) AND conname = :n"),
        {"t": table, "n": name},
    ).first()
    return (row is not None, bool(row[0]) if row is not None else False)


def relink_pylovo(engine: Engine, *, apply: bool = False, accept_unverified: bool = False, echo: Echo = print) -> int:
    """Relink grid cases to current pylovo ids, then restore the FK and views.

    Relinked rows are recorded in ``surrogrid.grid_case_relink_backup``.
    Returns a process exit code (1 if grid cases were rejected).
    """
    with engine.connect() as conn:
        state = schema.inspect_schema(conn)
        if state.kind != "managed" or state.pending(schema.migrations()):
            echo("Migrate the schema first: gridexpand db migrate --plan / --apply")
            return 1
        plan = relink_plan(conn, accept_unverified=accept_unverified)
        fk_exists, fk_valid = _constraint(conn, "surrogrid.grid_case", schema.PYLOVO_FOREIGN_KEY)
        natural_exists, _ = _constraint(conn, "surrogrid.grid_case", "uq_grid_case_natural")
        conn.commit()
    counts = {status: sum(row["status"] == status for row in plan) for status in ("unchanged", "relink", "rejected")}
    echo(f"grid cases: {len(plan)} ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    for row in plan:
        if row["status"] != "unchanged":
            echo(
                f"  {row['status']}: grid_case {row['grid_case_id']} (v{row['pylovo_version_id']} "
                f"{row['plz']}/{row['kcid']}/{row['bcid']}) {row['old_id']} -> {row['new_id']} {row['reason']}"
            )
    echo(
        f"foreign key {schema.PYLOVO_FOREIGN_KEY}: "
        + ("validated" if fk_valid else "NOT VALID" if fk_exists else "missing")
        + f"; uq_grid_case_natural: {'present' if natural_exists else 'missing'}"
    )
    if not apply:
        echo("Dry run (--plan): nothing was changed.")
        return 1 if counts["rejected"] else 0
    relink = [row for row in plan if row["status"] == "relink"]
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('lock_timeout', '5s', true)"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surrogrid.grid_case_relink_backup (
                    grid_case_id bigint NOT NULL,
                    old_pylovo_grid_result_id bigint NOT NULL,
                    new_pylovo_grid_result_id bigint NOT NULL,
                    relinked_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        )
        for row in relink:
            params = {"id": row["grid_case_id"], "old": row["old_id"], "new": row["new_id"]}
            conn.execute(
                text(
                    "INSERT INTO surrogrid.grid_case_relink_backup "
                    "(grid_case_id, old_pylovo_grid_result_id, new_pylovo_grid_result_id) VALUES (:id, :old, :new)"
                ),
                params,
            )
            for table in ("grid_case", "expansion_line_result", "expansion_transformer_result"):
                conn.execute(
                    text(f"UPDATE surrogrid.{table} SET pylovo_grid_result_id = :new WHERE grid_case_id = :id"),
                    params,
                )
        duplicates = any(row["n_natural"] > 1 for row in plan)
        if not natural_exists and not duplicates:
            conn.execute(
                text(
                    "ALTER TABLE surrogrid.grid_case ADD CONSTRAINT uq_grid_case_natural "
                    "UNIQUE (ags, pylovo_version_id, plz, kcid, bcid)"
                )
            )
        if not fk_exists:
            conn.execute(
                text(
                    f"ALTER TABLE surrogrid.grid_case ADD CONSTRAINT {schema.PYLOVO_FOREIGN_KEY} "
                    "FOREIGN KEY (pylovo_grid_result_id) REFERENCES pylovo.grid_result (grid_result_id) "
                    "ON DELETE RESTRICT NOT VALID"
                )
            )
    echo(f"relinked {len(relink)} grid cases (old ids in surrogrid.grid_case_relink_backup)")
    if counts["rejected"]:
        echo("Rejected grid cases stay unchanged; the foreign key stays NOT VALID and views are not created.")
        return 1
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE surrogrid.grid_case VALIDATE CONSTRAINT {schema.PYLOVO_FOREIGN_KEY}"))
    echo(f"foreign key {schema.PYLOVO_FOREIGN_KEY} validated")
    with engine.connect() as conn:
        status = schema.view_status(conn)
        conn.commit()
    if status.missing or status.outdated:
        with engine.connect() as conn:
            schema.create_views(conn)
        echo("views created/updated from views.sql")
    schema.refresh_qgis_views(engine)
    echo("QGIS materialized views refreshed")
    return 0


# CLI ------------------------------------------------------------------------------------


def _add_plan_apply(parser: argparse.ArgumentParser, what: str) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help=f"Show what {what} would do (default).")
    mode.add_argument("--apply", action="store_true", help=f"Run {what}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gridexpand db", description="SurroGrid database maintenance.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "init-schema",
        help="Create the surrogrid schema on a database without one (never migrates an existing one).",
    )
    migrate = commands.add_parser(
        "migrate",
        help="Check and migrate an existing surrogrid schema (dry run by default).",
        description="Shape check, stamp of pre-migration schemas as version 1, pending migrations, views.",
    )
    _add_plan_apply(migrate, "the migration")
    migrate.add_argument("--lock-timeout", default="5s", help="lock_timeout per migration (default 5s).")
    compress = commands.add_parser(
        "compress", help="TimescaleDB compression of the raw hourly tables (dry run by default)."
    )
    _add_plan_apply(compress, "the compression")
    compress.add_argument(
        "--table", action="append", choices=tuple(COMPRESSION_SETTINGS), help="Only this table; repeatable."
    )
    relink = commands.add_parser(
        "relink-pylovo",
        help="Point grid cases at the current pylovo grid ids (after a pylovo re-generation).",
    )
    _add_plan_apply(relink, "the relink")
    relink.add_argument(
        "--accept-unverified",
        action="store_true",
        help="Also relink grid cases whose runs have no Step 2 component audit to verify the buildings.",
    )
    delete = commands.add_parser(
        "delete-scenario",
        help="Delete SurroGrid data for one scenario_key. Defaults to dry-run.",
        description="Delete SurroGrid data for one scenario_key. Defaults to dry-run.",
    )
    delete.add_argument("scenario_key", help="Readable surrogrid.scenario.scenario_key to clean up.")
    delete.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows and Step 3 files. Without this flag, only counts are printed.",
    )
    delete.add_argument(
        "--keep-demands",
        action="store_true",
        help="Keep the scenario, pipeline, and Step 2 rows; delete power-flow and expansion results only.",
    )
    delete.add_argument(
        "--no-refresh-expansion-views",
        action="store_true",
        help="Skip refreshing Step 5 QGIS materialized views after deletion.",
    )
    return parser


def _print_counts(counts: dict[str, int], *, keep_demands: bool, dry_run: bool) -> None:
    action = "Would delete" if dry_run else "Deleted"
    kept = set(DEMAND_TABLES) if keep_demands else set()
    print(f"{action} scenario-related SurroGrid rows:")
    for table, count in counts.items():
        print(f"  {table}: {count}{' (kept by --keep-demands)' if table in kept else ''}")


def _print_files(paths: list[Path], *, dry_run: bool) -> None:
    print(f"{'Would delete' if dry_run else 'Deleted'} Step 3 files: {len(paths)}")
    for path in paths:
        print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    """Run ``gridexpand db``; see the module docstring for the commands."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "delete-scenario")
    args = _build_parser().parse_args(argv)
    engine = get_engine()
    if args.command == "init-schema":
        schema.ensure_schema(engine)
        print("surrogrid schema is current.")
        return 0
    if args.command == "migrate":
        return schema.migrate(engine, apply=args.apply, lock_timeout=args.lock_timeout)
    if args.command == "compress":
        return compress_raw_chunks(engine, tables=tuple(args.table or ()), apply=args.apply)
    if args.command == "relink-pylovo":
        return relink_pylovo(engine, apply=args.apply, accept_unverified=args.accept_unverified)

    dry_run = not args.execute
    step3_files = _step3_artifacts_for_scenario(engine, args.scenario_key)
    counts = delete_scenario_data(
        engine,
        args.scenario_key,
        keep_demands=args.keep_demands,
        dry_run=dry_run,
        refresh_views=not args.no_refresh_expansion_views,
    )
    if not dry_run:
        for path in step3_files:
            path.unlink()
    _print_counts(counts, keep_demands=args.keep_demands, dry_run=dry_run)
    _print_files(step3_files, dry_run=dry_run)
    if dry_run:
        print("Dry run only. Re-run with --execute to delete these rows and files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
