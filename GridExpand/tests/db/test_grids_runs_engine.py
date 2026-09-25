"""Pure helpers of db.grids, db.runs and db.engine (no database)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sqlalchemy.engine import make_url

from gridexpand.db import engine, grids, runs


def test_parse_grid_filename() -> None:
    assert grids.parse_grid_filename("09278140-03_94342_1_-1.h5") == {
        "ags": 9278140, "candidate_index": 3, "cell_id": "09278140-03", "plz": 94342, "kcid": 1, "bcid": -1,
    }
    with pytest.raises(ValueError):
        grids.parse_grid_filename("x_1_2.h5")


def test_format_grid_ref() -> None:
    ref = grids.format_grid_ref(
        ags=9184137, row={"grid_result_id": 4, "version_id": 1, "plz": 85653, "kcid": 1, "bcid": 4}, candidate_index=3
    )
    assert ref == {
        "ags": 9184137, "candidate_index": 3, "cell_id": "9184137-03", "bridge_filename": "9184137-03_85653_1_4.h5",
        "grid_result_id": 4, "version_id": "1", "plz": 85653, "kcid": 1, "bcid": 4,
    }
    assert tuple(ref) == grids.GRID_REF_KEYS


def test_candidate_sql_orders_versions_numerically() -> None:
    assert "::numeric END DESC NULLS LAST" in grids.CANDIDATES_SQL
    assert "ROW_NUMBER() OVER (ORDER BY plz, kcid, bcid) - 1 AS candidate_index" in grids.CANDIDATES_SQL


def test_normalize_ags() -> None:
    assert grids.normalize_ags("09184137") == 9184137
    assert grids.normalize_ags(" 0 ") == 0


def test_scenario_level_assumptions_keep_shared_keys_only() -> None:
    shared = runs.scenario_level_assumptions(
        {"timeframe_mode": "max_base_electricity_demand_week", "timeframe_start": "2009-01-13", "scenario_hash": "h",
         "model_case": "pre"}
    )
    assert shared["timeframe_mode"] == "max_base_electricity_demand_week"
    assert shared["timeframe_start"] is None
    assert shared["scenario_hash"] == "h"
    assert "model_case" not in shared


def test_json_safe() -> None:
    value = runs.json_safe({"a": np.float64(math.nan), "b": [np.int64(2), math.inf, True], 3: "x"})
    assert value == {"a": None, "b": [2, None, True], "3": "x"}


def test_default_run_names() -> None:
    assert runs.default_pipeline_run_name("s") == "s_pipeline"
    assert runs.default_demand_allocation_run_name("s", "all", "pool") == "s_all_pool_demand_allocation"
    assert runs.default_powerflow_run_name("s", True) == "s_pre_powerflow"
    assert runs.default_powerflow_run_name("s", False) == "s_full_powerflow"


def test_database_url_escapes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_env_loaded", True)
    for name, value in {
        "DB_USER": "user", "DB_PASSWORD": "p@ss:w/rd%", "DB_HOST": "db.example", "DB_PORT": "5433", "DB_NAME": "x",
    }.items():
        monkeypatch.setenv(name, value)
    url = engine.database_url()
    assert url.password == "p@ss:w/rd%" and url.host == "db.example" and url.port == 5433
    assert make_url(url.render_as_string(hide_password=False)).password == "p@ss:w/rd%"


def test_get_engine_is_cached_per_url() -> None:
    url = make_url("postgresql+psycopg2://a:b@127.0.0.1:9/one")
    assert engine.get_engine(url) is engine.get_engine(url)
    assert engine.get_engine(url) is not engine.get_engine(url.set(database="two"))


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/x/gridexpand/powerflow/run_pwrflw.py", "gridexpand-run_pwrflw"),
        ("gridexpand synthetic", "gridexpand-synthetic"),
        ("-c", "gridexpand"),
        ("", "gridexpand"),
    ],
)
def test_application_name(argv0: str, expected: str) -> None:
    assert engine.application_name(argv0) == expected
