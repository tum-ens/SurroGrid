"""Read-only database access of the service.

The connection target is the one of :class:`gridexpand.db.SurroGridDatabase` (the
``.env`` file of :data:`gridexpand.paths.ENV_FILE`), so the service always reads the
database its jobs write to. The service's own engine only differs in how it connects:
short connect timeout, statement timeout and read-only transactions. The service never
writes to the database; only its ``gridexpand`` jobs do.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, OperationalError

APPLICATION_NAME = "gridexpand-service"
CONNECT_TIMEOUT_S = 5
STATEMENT_TIMEOUT_MS = 60_000

_engine: Engine | None = None
_lock = threading.Lock()


class DatabaseUnavailable(RuntimeError):
    """The configured database cannot be reached."""


def engine() -> Engine:
    """Return the service engine (created on first use)."""
    global _engine
    with _lock:
        if _engine is None:
            from gridexpand.db import SurroGridDatabase
            from gridexpand.db.engine import DatabaseNotConfigured

            try:
                url = SurroGridDatabase().engine.url
            except DatabaseNotConfigured as exc:
                raise DatabaseUnavailable(str(exc)) from exc
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=4,
                max_overflow=4,
                connect_args={
                    "connect_timeout": CONNECT_TIMEOUT_S,
                    "application_name": APPLICATION_NAME,
                    "options": (
                        "-c default_transaction_read_only=on "
                        f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"
                    ),
                },
            )
        return _engine


def reset_engine() -> None:
    """Dispose the engine (tests, or after the ``.env`` changed)."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None


def connection_info() -> dict[str, Any]:
    """Host, port, database and user of the configured database (never the password).

    All values are ``None`` when no database is configured.
    """
    try:
        url = engine().url
    except DatabaseUnavailable:
        return {"host": None, "port": None, "database": None, "user": None}
    return {"host": url.host, "port": str(url.port or 5432), "database": url.database, "user": url.username}


@contextmanager
def connect() -> Iterator[Connection]:
    """A read-only connection; connection problems raise :class:`DatabaseUnavailable`."""
    try:
        conn = engine().connect()
    except OperationalError as exc:
        raise DatabaseUnavailable(str(exc.orig).strip().splitlines()[0] if exc.orig else str(exc)) from exc
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, conn: Connection | None = None, **params: Any) -> list[dict[str, Any]]:
    """Run one query and return its rows as dictionaries."""
    if conn is not None:
        return [dict(row) for row in conn.execute(text(sql), params).mappings()]
    with connect() as own:
        return [dict(row) for row in own.execute(text(sql), params).mappings()]


def fetch_one(sql: str, conn: Connection | None = None, **params: Any) -> dict[str, Any] | None:
    """Run one query and return its first row (or ``None``)."""
    rows = fetch_all(sql, conn, **params)
    return rows[0] if rows else None


def error_message(exc: DBAPIError) -> str:
    """First line of a database error, for API responses."""
    return str(exc.orig or exc).strip().splitlines()[0]
