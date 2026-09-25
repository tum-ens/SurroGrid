"""Fixtures of the service tests.

Database tests run only when you name the sandbox database of ``GridExpand/.env``::

    GRIDEXPAND_SERVICE_TEST_DATABASE=sg_impl_service_demo uv run pytest tests/service

They only read (the engine uses read-only transactions); no job is started against the
database. Everything else uses the unreachable database of ``tests/conftest.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parents[2]
REAL_DATABASE_PORT = "54327"  # the user's InfDB: never used by tests


@pytest.fixture()
def settings(tmp_path: Path):
    from gridexpand.paths import SCENARIO_CONFIG_DIR
    from gridexpand.service.settings import ServiceSettings

    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    shutil.copy(SCENARIO_CONFIG_DIR / "schweinfurt_2045.yaml", scenarios / "schweinfurt_2045.yaml")
    (scenarios / "broken.yaml").write_text("scenario: [1, 2\n", encoding="utf-8")
    return ServiceSettings(port=18766, allowed_hosts=frozenset({"testserver"}), scenario_dirs=(scenarios,),
                           user_scenario_dir=tmp_path / "user_scenarios", state_dir=tmp_path / "state",
                           runs_dir=tmp_path / "runs")


@pytest.fixture()
def fake_solvers(monkeypatch):
    """No Gurobi licence check (a subprocess) in unit tests."""
    from gridexpand.service import environment

    info = {"installed": False, "package_version": None, "usable": False, "detail": "test", "checked_at": 0}
    monkeypatch.setattr(environment, "check_gurobi", lambda: info)
    return info


@pytest.fixture()
def client(settings, fake_solvers):
    from fastapi.testclient import TestClient

    from gridexpand.service.app import create_app

    with TestClient(create_app(settings), headers={"X-GridExpand-UI": "1"}) as test_client:
        yield test_client


@pytest.fixture()
def sandbox_db(monkeypatch):
    """Point the service (and get_candidates) at the sandbox database of GridExpand/.env."""
    wanted = os.getenv("GRIDEXPAND_SERVICE_TEST_DATABASE")
    if not wanted:
        pytest.skip("set GRIDEXPAND_SERVICE_TEST_DATABASE=<sandbox db of GridExpand/.env> to run database tests")
    env = dotenv_values(PROJECT_DIR / ".env")
    if env.get("DB_NAME") != wanted or env.get("DB_PORT") == REAL_DATABASE_PORT:
        pytest.skip(f"GRIDEXPAND_SERVICE_TEST_DATABASE={wanted} is not the sandbox database of GridExpand/.env")
    from sqlalchemy import URL, create_engine

    from gridexpand.db import SurroGridDatabase
    from gridexpand.scenario import synthetic_ags_runner
    from gridexpand.service import db as service_db

    url = URL.create("postgresql+psycopg2", username=env["DB_USER"], password=env["DB_PASSWORD"],
                     host=env["DB_HOST"], port=int(env["DB_PORT"]), database=env["DB_NAME"])
    engine = create_engine(url, connect_args={"connect_timeout": 5,
                                              "options": "-c default_transaction_read_only=on"})

    class SandboxDatabase(SurroGridDatabase):
        def __init__(self) -> None:  # no .env loading: the test engine only
            self.engine = engine
            self.pylovo_version_id = None

    monkeypatch.setattr(service_db, "_engine", engine)
    monkeypatch.setattr(synthetic_ags_runner, "SurroGridDatabase", SandboxDatabase)
    yield engine
    engine.dispose()
