"""Versioned ``surrogrid`` schema: numbered migrations, views, startup check.

The schema is defined by ``sql/NNNN_<name>.sql`` (applied in order and recorded
in ``surrogrid.schema_migration``) plus the re-runnable ``sql/views.sql``.

- A database without ``surrogrid`` tables is initialised automatically on first
  use (all migrations, then the views).
- An existing database is never migrated by a pipeline run. ``ensure_schema``
  raises :class:`SchemaMigrationRequired` if it predates the migrations or has
  pending ones; ``gridexpand db migrate --plan`` / ``--apply`` checks the shape
  of a pre-migration database, stamps it as version 1 and applies the rest.
- Views join ``grid_case`` to pylovo by id; they are created only while the
  ``grid_case -> pylovo.grid_result`` foreign key exists and is validated.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from gridexpand.db.engine import get_engine
from gridexpand.paths import SQL_DIR

MIGRATION_FILE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
VIEWS_SQL_PATH = SQL_DIR / "views.sql"
# Same key as the pre-migration schema code, so old and new processes serialise.
LOCK_KEY = 916200005
VIEWS = (
    "surrogrid.grid_building_bus",
    "surrogrid.grid_building_component",
    "surrogrid.expansion_line_qgis_mv",
    "surrogrid.expansion_transformer_qgis_mv",
)
QGIS_VIEWS = VIEWS[2:]
VIEWS_COMMENT_TARGET = "surrogrid.grid_building_component"
PYLOVO_FOREIGN_KEY = "fk_grid_case_pylovo_grid_result"
# Unique keys of 0001 that pre-migration databases do not have yet: added by
# 0003 (identity keys) or by `gridexpand db relink-pylovo` (natural key).
KEYS_ADDED_LATER = frozenset(
    {
        "uq_grid_case_natural",
        "uq_pipeline_run_identity",
        "uq_powerflow_run_identity",
        "uq_real_powerflow_run_identity",
    }
)
_BOOTSTRAP_SQL = """
DO $$
BEGIN
    IF to_regnamespace('surrogrid') IS NULL THEN
        CREATE SCHEMA surrogrid;
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS surrogrid.schema_migration (
    version integer PRIMARY KEY,
    name text NOT NULL,
    checksum text NOT NULL,
    mode text NOT NULL DEFAULT 'applied' CHECK (mode IN ('applied', 'stamped')),
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""

_checked: set[str] = set()


class SchemaMigrationRequired(RuntimeError):
    """The database schema must be migrated (or relinked) explicitly first."""


@dataclass(frozen=True)
class Migration:
    """One numbered SQL migration file."""

    version: int
    name: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def migrations(directory: Path = SQL_DIR) -> list[Migration]:
    """Return the migrations in ``directory``, ordered by version (1..N)."""
    found = [
        Migration(int(match.group(1)), match.group(2), path)
        for path in sorted(directory.glob("*.sql"))
        if (match := MIGRATION_FILE.match(path.name))
    ]
    versions = [migration.version for migration in found]
    if versions != list(range(1, len(found) + 1)):
        raise RuntimeError(f"Migration versions must run 1..N without gaps: {versions}")
    return found


@dataclass
class SchemaState:
    """What a database's ``surrogrid`` schema looks like to the migrator.

    ``kind`` is ``fresh`` (no surrogrid relations), ``legacy`` (relations but no
    ``schema_migration``) or ``managed``; ``applied`` maps version -> checksum.
    """

    kind: str
    relations: int
    applied: dict[int, str] = field(default_factory=dict)

    def pending(self, known: list[Migration]) -> list[Migration]:
        return [m for m in known if m.version not in self.applied]

    def unknown(self, known: list[Migration]) -> list[int]:
        versions = {m.version for m in known}
        return sorted(v for v in self.applied if v not in versions)

    def modified(self, known: list[Migration]) -> list[Migration]:
        return [m for m in known if self.applied.get(m.version, m.checksum) != m.checksum]


def inspect_schema(conn: Connection) -> SchemaState:
    """Classify the ``surrogrid`` schema of the connected database."""
    has_table, relations = conn.execute(
        text(
            """
            SELECT to_regclass('surrogrid.schema_migration') IS NOT NULL,
                   (SELECT count(*) FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'surrogrid' AND c.relkind IN ('r', 'p', 'v', 'm'))
            """
        )
    ).one()
    if has_table:
        rows = conn.execute(text("SELECT version, checksum FROM surrogrid.schema_migration"))
        return SchemaState("managed", int(relations), {int(v): str(c) for v, c in rows})
    return SchemaState("legacy" if relations else "fresh", int(relations))


def _database_label(engine: Engine) -> str:
    url = engine.url
    return f"{url.username}@{url.host}:{url.port}/{url.database}"


def _migration_problem(state: SchemaState, known: list[Migration], database: str) -> str | None:
    commands = "  gridexpand db migrate --plan\n  gridexpand db migrate --apply"
    if state.kind == "legacy":
        return (
            f"The surrogrid schema of {database} predates versioned migrations "
            "(no surrogrid.schema_migration). Pipeline runs never migrate an existing "
            f"database. Back it up, then check and migrate it explicitly:\n{commands}"
        )
    unknown = state.unknown(known)
    if unknown:
        return (
            f"The surrogrid schema of {database} has migrations {unknown} that this "
            "GridExpand version does not know; update GridExpand."
        )
    pending = state.pending(known)
    if pending:
        labels = ", ".join(m.label for m in pending)
        return (
            f"The surrogrid schema of {database} has pending migrations ({labels}). "
            f"Back it up, then run:\n{commands}"
        )
    return None


def run_sql(conn: Connection, sql: str) -> None:
    """Execute a whole SQL script (several statements, ``%`` taken literally)."""
    conn.exec_driver_sql(sql, execution_options={"no_parameters": True})


def _record(conn: Connection, migration: Migration, mode: str = "applied") -> None:
    conn.execute(
        text(
            "INSERT INTO surrogrid.schema_migration (version, name, checksum, mode) "
            "VALUES (:version, :name, :checksum, :mode)"
        ),
        {"version": migration.version, "name": migration.name, "checksum": migration.checksum, "mode": mode},
    )


def _apply(
    conn: Connection,
    migration: Migration,
    *,
    lock_timeout: str | None = None,
    stamp: bool = False,
) -> bool:
    """Apply (or stamp) one migration in its own transaction; False if done."""
    with conn.begin():
        if lock_timeout:
            conn.execute(text("SELECT set_config('lock_timeout', :value, true)"), {"value": lock_timeout})
        run_sql(conn, _BOOTSTRAP_SQL)
        done = conn.execute(
            text("SELECT 1 FROM surrogrid.schema_migration WHERE version = :version"),
            {"version": migration.version},
        ).first()
        if done:
            return False
        if not stamp:
            run_sql(conn, migration.sql)
        _record(conn, migration, "stamped" if stamp else "applied")
    return True


def _advisory_lock(conn: Connection, *, wait: bool) -> bool:
    """Take the session-level schema lock (commits the check transaction)."""
    if wait:
        conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": LOCK_KEY})
        acquired = True
    else:
        acquired = bool(conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": LOCK_KEY}).scalar())
    conn.commit()
    return acquired


def _advisory_unlock(conn: Connection) -> None:
    try:
        conn.rollback()
        conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": LOCK_KEY})
        conn.commit()
    except Exception:  # noqa: BLE001 - closing the session releases the lock too
        conn.invalidate()


def pylovo_link_problem(conn: Connection) -> str | None:
    """Why views over pylovo must not be created now, or None if they may."""
    row = conn.execute(
        text(
            """
            SELECT convalidated FROM pg_constraint
            WHERE conrelid = to_regclass('surrogrid.grid_case') AND conname = :name
            """
        ),
        {"name": PYLOVO_FOREIGN_KEY},
    ).first()
    if row is None:
        return (
            f"surrogrid.grid_case has no foreign key {PYLOVO_FOREIGN_KEY} to "
            "pylovo.grid_result, so its pylovo ids may be stale"
        )
    if not row[0]:
        return f"the foreign key {PYLOVO_FOREIGN_KEY} is not validated"
    return None


def views_checksum() -> str:
    """sha256 of ``views.sql``; stored as a comment on the component view."""
    return hashlib.sha256(VIEWS_SQL_PATH.read_bytes()).hexdigest()


@dataclass
class ViewStatus:
    """Missing views, whether ``views.sql`` changed, and the pylovo link check."""

    missing: list[str]
    outdated: bool
    link_problem: str | None


def view_status(conn: Connection) -> ViewStatus:
    missing = [
        name
        for name in VIEWS
        if conn.execute(text("SELECT to_regclass(:name)"), {"name": name}).scalar() is None
    ]
    comment = None
    if VIEWS_COMMENT_TARGET not in missing:
        comment = conn.execute(
            text("SELECT obj_description(to_regclass(:name), 'pg_class')"),
            {"name": VIEWS_COMMENT_TARGET},
        ).scalar()
    outdated = VIEWS_COMMENT_TARGET not in missing and comment != _views_comment()
    return ViewStatus(missing, outdated, pylovo_link_problem(conn))


def _views_comment() -> str:
    return f"gridexpand views.sql sha256:{views_checksum()}"


def create_views(conn: Connection) -> None:
    """Run ``views.sql`` and record its checksum (``conn`` must be outside a transaction)."""
    with conn.begin():
        run_sql(conn, VIEWS_SQL_PATH.read_text(encoding="utf-8"))
        # COMMENT takes no bind parameters; the checksum is hexadecimal.
        run_sql(conn, f"COMMENT ON VIEW {VIEWS_COMMENT_TARGET} IS '{_views_comment()}'")


def ensure_views(engine: Engine | None = None) -> bool:
    """Create missing views from ``views.sql``; True if it ran.

    Raises:
        SchemaMigrationRequired: a view is missing and the grid_case -> pylovo
            foreign key is missing or not validated.
    """
    engine = get_engine() if engine is None else engine
    with engine.connect() as conn:
        if not view_status(conn).missing:
            return False
        _advisory_lock(conn, wait=True)
        try:
            status = view_status(conn)
            conn.commit()
            if not status.missing:
                return False
            if status.link_problem:
                raise SchemaMigrationRequired(
                    f"surrogrid views are missing ({', '.join(status.missing)}) and cannot "
                    f"be created: {status.link_problem}. Relink the grid cases first:\n"
                    "  gridexpand db relink-pylovo --plan\n  gridexpand db relink-pylovo --apply"
                )
            create_views(conn)
            return True
        finally:
            _advisory_unlock(conn)


def ensure_schema(engine: Engine | None = None) -> None:
    """Check the schema once per process; initialise a database without one.

    Raises:
        SchemaMigrationRequired: the database predates the migrations, has
            pending (or unknown) migrations, or its views cannot be created.
    """
    engine = get_engine() if engine is None else engine
    key = str(engine.url)
    if key in _checked:
        return
    known = migrations()
    with engine.connect() as conn:
        state = inspect_schema(conn)
        conn.commit()
        if state.kind == "fresh":
            _advisory_lock(conn, wait=True)
            try:
                state = inspect_schema(conn)
                conn.commit()
                if state.kind != "legacy":
                    for migration in state.pending(known):
                        _apply(conn, migration)
                    state = inspect_schema(conn)
                    conn.commit()
            finally:
                _advisory_unlock(conn)
    problem = _migration_problem(state, known, _database_label(engine))
    if problem:
        raise SchemaMigrationRequired(problem)
    ensure_views(engine)
    _checked.add(key)


def refresh_qgis_views(engine: Engine | None = None) -> None:
    """Refresh the QGIS materialized views (all analyses; blocks their readers)."""
    engine = get_engine() if engine is None else engine
    with engine.begin() as conn:
        for name in QGIS_VIEWS:
            if conn.execute(text("SELECT to_regclass(:name)"), {"name": name}).scalar() is not None:
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {name}"))


# Baseline shape (for stamping pre-migration databases) ------------------------

_TYPE_ALIASES = {
    "bigserial": "bigint",
    "timestamptz": "timestamp with time zone",
}
_COLUMN_LINE = re.compile(
    r"^(?P<name>[a-z_][a-z0-9_]*) "
    r"(?P<type>bigserial|bigint|integer|text|double precision|boolean|jsonb|timestamptz"
    r"|varchar\(\d+\)|geometry\((?:LineString|Point), \d+\))(?P<rest>.*)$"
)


@dataclass(frozen=True)
class BaselineTable:
    """Columns ``(name, type, not_null)``, required unique keys and hypertable flag."""

    name: str
    columns: tuple[tuple[str, str, bool], ...]
    unique_keys: tuple[frozenset[str], ...]
    hypertable: bool


def _canonical_type(sql_type: str) -> str:
    if sql_type.startswith("varchar("):
        return "character varying" + sql_type[len("varchar"):]
    if sql_type.startswith("geometry("):
        return sql_type.replace(", ", ",")
    return _TYPE_ALIASES.get(sql_type, sql_type)


def baseline_tables(sql: str | None = None) -> dict[str, BaselineTable]:
    """Parse the tables of ``0001_baseline.sql`` (one column per line).

    Unique keys named in :data:`KEYS_ADDED_LATER` are not required.

    Raises:
        ValueError: a line of a CREATE TABLE block is neither a column nor a
            table constraint.
    """
    sql = migrations()[0].sql if sql is None else sql
    hypertables = set(re.findall(r"create_hypertable\('surrogrid\.(\w+)'", sql))
    tables: dict[str, BaselineTable] = {}
    for match in re.finditer(r"^CREATE TABLE surrogrid\.(\w+) \(\n(.*?)\n\);", sql, re.M | re.S):
        name, body = match.group(1), match.group(2)
        columns: list[tuple[str, str, bool]] = []
        keys: list[frozenset[str]] = []
        for raw in body.splitlines():
            line = raw.strip().rstrip(",")
            if not line or line.startswith("--") or line.startswith("REFERENCES "):
                continue
            if line.startswith(("CONSTRAINT ", "PRIMARY KEY")):
                constraint = re.match(r"^CONSTRAINT (\w+) (.*)$", line)
                constraint_name, definition = constraint.groups() if constraint else ("", line)
                key = re.match(r"^(?:UNIQUE|PRIMARY KEY) \(([^)]*)\)$", definition)
                if key and constraint_name not in KEYS_ADDED_LATER:
                    keys.append(frozenset(part.strip() for part in key.group(1).split(",")))
                continue
            column = _COLUMN_LINE.match(line)
            if column is None:
                raise ValueError(f"Unrecognised line in CREATE TABLE surrogrid.{name}: {raw!r}")
            rest = column.group("rest")
            sql_type = column.group("type")
            not_null = "NOT NULL" in rest or "PRIMARY KEY" in rest or sql_type == "bigserial"
            columns.append((column.group("name"), _canonical_type(sql_type), not_null))
            if re.search(r"\b(PRIMARY KEY|UNIQUE)\b", rest):
                keys.append(frozenset({column.group("name")}))
        tables[name] = BaselineTable(name, tuple(columns), tuple(keys), name in hypertables)
    return tables


def shape_problems(conn: Connection) -> list[str]:
    """Differences between a pre-migration schema and the baseline tables.

    Checks every table, column (type, NOT NULL, leftovers), required unique key
    and hypertable of ``0001_baseline.sql``. An empty list means the database
    can be stamped as version 1 without executing DDL.
    """
    expected = baseline_tables()
    actual_columns: dict[str, dict[str, tuple[str, bool]]] = {}
    for table, column, sql_type, not_null in conn.execute(
        text(
            """
            SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'surrogrid' AND c.relkind IN ('r', 'p')
              AND a.attnum > 0 AND NOT a.attisdropped
            """
        )
    ):
        actual_columns.setdefault(table, {})[column] = (sql_type, bool(not_null))
    unique_keys: dict[str, set[frozenset[str]]] = {}
    for table, columns in conn.execute(
        text(
            """
            SELECT c.relname, array_agg(a.attname::text)
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
            WHERE n.nspname = 'surrogrid' AND i.indisunique
              AND i.indpred IS NULL AND i.indexprs IS NULL
            GROUP BY c.relname, i.indexrelid
            """
        )
    ):
        unique_keys.setdefault(table, set()).add(frozenset(columns))
    hypertables = {
        row[0]
        for row in conn.execute(
            text(
                "SELECT hypertable_name FROM timescaledb_information.hypertables "
                "WHERE hypertable_schema = 'surrogrid'"
            )
        )
    }

    problems: list[str] = []
    if "scenario_int" in actual_columns:
        problems.append("surrogrid.scenario_int exists (unfinished scenario migration of the old code)")
    for name, table in expected.items():
        columns = actual_columns.get(name)
        if columns is None:
            problems.append(f"missing table surrogrid.{name}")
            continue
        for column, sql_type, not_null in table.columns:
            if column not in columns:
                problems.append(f"surrogrid.{name}: missing column {column}")
                continue
            actual_type, actual_not_null = columns[column]
            if actual_type != sql_type:
                problems.append(f"surrogrid.{name}.{column}: type {actual_type}, expected {sql_type}")
            if actual_not_null != not_null:
                expected_null = "NOT NULL" if not_null else "nullable"
                problems.append(f"surrogrid.{name}.{column}: expected {expected_null}")
        extra = sorted(set(columns) - {column for column, _, _ in table.columns})
        if extra:
            problems.append(f"surrogrid.{name}: columns not in the baseline: {', '.join(extra)}")
        for key in table.unique_keys:
            if key not in unique_keys.get(name, set()):
                problems.append(f"surrogrid.{name}: no unique index on ({', '.join(sorted(key))})")
        if table.hypertable and name not in hypertables:
            problems.append(f"surrogrid.{name} is not a hypertable")
    return problems


# gridexpand db migrate ---------------------------------------------------------


@dataclass
class MigrationPlan:
    """What ``gridexpand db migrate --apply`` would do."""

    state: SchemaState
    stamp: Migration | None
    apply: list[Migration]
    views: ViewStatus | None
    errors: list[str]
    notes: list[str]

    @property
    def run_views(self) -> bool:
        views = self.views
        will_have_link = views is None or views.link_problem is None or self.state.kind == "fresh"
        needs = views is None or bool(views.missing) or views.outdated
        return needs and will_have_link


def migration_plan(conn: Connection) -> MigrationPlan:
    """Inspect the database and plan the migration (read-only)."""
    known = migrations()
    state = inspect_schema(conn)
    errors: list[str] = []
    notes: list[str] = []
    stamp = None
    if state.kind == "fresh":
        pending = known
        views = None
    elif state.kind == "legacy":
        errors = [f"shape check: {problem}" for problem in shape_problems(conn)]
        stamp, pending = known[0], known[1:]
        views = view_status(conn)
    else:
        pending = state.pending(known)
        views = view_status(conn)
        unknown = state.unknown(known)
        if unknown:
            errors.append(f"migrations {unknown} are unknown to this GridExpand version")
        for migration in state.modified(known):
            notes.append(f"{migration.label}.sql changed after it was applied (checksum differs)")
    if views is not None and views.link_problem and state.kind == "legacy":
        # 0003/0004 turn an existing CASCADE key to pylovo into the RESTRICT key.
        legacy_key = conn.execute(
            text(
                "SELECT 1 FROM pg_constraint WHERE conrelid = 'surrogrid.grid_case'::regclass "
                "AND conname = 'grid_case_pylovo_grid_result_id_fkey'"
            )
        ).first()
        if legacy_key:
            views = ViewStatus(views.missing, views.outdated, None)
    return MigrationPlan(state, stamp, pending, views, errors, notes)


def _describe(plan: MigrationPlan, database: str) -> list[str]:
    state = plan.state
    lines = [f"Database: {database}"]
    if state.kind == "fresh":
        lines.append("State: no surrogrid tables (fresh database).")
    elif state.kind == "legacy":
        lines.append(f"State: pre-migration surrogrid schema ({state.relations} relations, no schema_migration).")
        lines.append("Shape check against 0001_baseline: " + ("FAILED" if plan.errors else "OK"))
    else:
        lines.append(f"State: versioned schema, applied {sorted(state.applied)}.")
    lines += [f"  error: {error}" for error in plan.errors]
    lines += [f"  note: {note}" for note in plan.notes]
    steps = []
    if plan.stamp is not None:
        steps.append(f"stamp {plan.stamp.label} (no DDL; the tables match the baseline)")
    steps += [f"apply {migration.label}" for migration in plan.apply]
    views = plan.views
    if plan.run_views:
        what = "create" if views is None or views.missing else "update"
        steps.append(f"views: {what} from views.sql")
    elif views is not None and (views.missing or views.outdated):
        steps.append(
            f"views: NOT created ({views.link_problem}); run `gridexpand db relink-pylovo` afterwards"
        )
    lines.append("Plan:" if steps else "Plan: nothing to do (schema and views are current).")
    lines += [f"  {step}" for step in steps]
    return lines


def migrate(
    engine: Engine | None = None,
    *,
    apply: bool,
    lock_timeout: str = "5s",
    echo: Callable[[str], None] = print,
) -> int:
    """Plan (``apply=False``: print the SQL) or apply pending migrations.

    Each migration runs in its own transaction with ``lock_timeout``; a
    pre-migration database is stamped as version 1 only if its shape check
    passes. Returns a process exit code.
    """
    engine = get_engine() if engine is None else engine
    database = _database_label(engine)
    with engine.connect() as conn:
        plan = migration_plan(conn)
        conn.commit()
        for line in _describe(plan, database):
            echo(line)
        if plan.errors:
            echo("Nothing was changed. Fix the errors above (or ask for a migration that handles them).")
            return 1
        if not apply:
            for migration in plan.apply:
                echo(f"\n-- ==== {migration.label}.sql ====\n{migration.sql}")
            if plan.run_views:
                echo(f"\n-- ==== views.sql ====\n{VIEWS_SQL_PATH.read_text(encoding='utf-8')}")
            echo("Dry run (--plan): nothing was changed.")
            return 0
        if not _advisory_lock(conn, wait=False):
            echo("Another process holds the schema lock (migration running?); try again later.")
            return 1
        try:
            if plan.stamp is not None:
                if shape_problems(conn):
                    echo("The schema changed since the check; nothing was changed.")
                    return 1
                conn.commit()
                _apply(conn, plan.stamp, lock_timeout=lock_timeout, stamp=True)
                echo(f"stamped {plan.stamp.label}")
            for migration in plan.apply:
                started = time.monotonic()
                if _apply(conn, migration, lock_timeout=lock_timeout):
                    echo(f"applied {migration.label} ({time.monotonic() - started:.1f} s)")
            status = view_status(conn)
            conn.commit()
            if status.missing or status.outdated:
                if status.link_problem:
                    echo(
                        f"views not created: {status.link_problem}.\n"
                        "Relink the grid cases, which also creates the views:\n"
                        "  gridexpand db relink-pylovo --plan\n  gridexpand db relink-pylovo --apply"
                    )
                else:
                    create_views(conn)
                    echo("views created/updated from views.sql")
        finally:
            _advisory_unlock(conn)
    _checked.discard(str(engine.url))
    echo("surrogrid schema is current.")
    return 0
