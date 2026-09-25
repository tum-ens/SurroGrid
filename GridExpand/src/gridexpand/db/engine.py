"""Connection settings and one cached SQLAlchemy engine per database URL.

The credentials come from ``GridExpand/.env`` (or ``$GRIDEXPAND_ENV_FILE``),
loaded once per process; values in the file override variables of the same name
in the environment. Engines check connections before use and are disposed in
forked children (Step 2 and Step 4 use ``multiprocessing``), so a child never
reuses a connection of its parent.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from gridexpand.paths import ENV_FILE

_env_loaded = False
_engines: dict[URL, Engine] = {}
_lock = threading.Lock()


def load_env() -> None:
    """Load the ``.env`` file into ``os.environ`` (once per process)."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv(ENV_FILE, override=True)
        _env_loaded = True


def database_url() -> URL:
    """Return the configured PostgreSQL URL.

    ``URL.create`` escapes the credentials, so passwords with ``@``, ``:``,
    ``/`` or ``%`` work.
    """
    load_env()
    port = (os.getenv("DB_PORT") or "").strip() or "5432"
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(port),
        database=os.getenv("DB_NAME"),
    )


def application_name(argv0: str | None = None) -> str:
    """Name shown in ``pg_stat_activity``: ``gridexpand-<entry point>``."""
    raw = sys.argv[0] if argv0 is None else argv0
    name = Path(raw).name.removesuffix(".py") if raw and not raw.startswith("-") else ""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    if name.startswith("gridexpand"):
        name = name.removeprefix("gridexpand").lstrip("-_")
    return f"gridexpand-{name}"[:63] if name else "gridexpand"


def get_engine(url: URL | None = None) -> Engine:
    """Return the process-wide engine for ``url`` (default: ``database_url()``)."""
    url = database_url() if url is None else url
    with _lock:
        engine = _engines.get(url)
        if engine is None:
            engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"application_name": application_name()},
            )
            _engines[url] = engine
    return engine


def _after_fork_in_child() -> None:
    global _lock
    _lock = threading.Lock()
    for engine in _engines.values():
        # Drop the inherited pool without closing the parent's connections.
        engine.dispose(close=False)


os.register_at_fork(after_in_child=_after_fork_in_child)
