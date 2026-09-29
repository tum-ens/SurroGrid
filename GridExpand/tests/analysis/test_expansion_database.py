"""End-to-end expansion materialization on a sandbox database (opt-in).

Runs only when GRIDEXPAND_EXPANSION_TEST_ENV names an env file (DB_HOST, DB_PORT, DB_NAME, DB_USER,
DB_PASSWORD) of a sandbox database whose name starts with ``sg_`` and whose port is not the InfDB's
54327 (the test suite otherwise points at an unreachable database, see tests/conftest.py). The test
DROPS and recreates the ``pylovo`` and ``surrogrid`` schemas of that database::

    GRIDEXPAND_EXPANSION_TEST_ENV=/path/sandbox.env uv run pytest tests/analysis/test_expansion_database.py

It checks the synthetic path (SQL scope and mapping, rules in Python, temp tables, inserts) on two
small pylovo-like grids, and that the retired rule set is refused.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

REAL_DATABASE_PORT = 54327


@pytest.fixture(scope="module")
def sandbox_db():
    env_file = os.getenv("GRIDEXPAND_EXPANSION_TEST_ENV")
    if not env_file:
        pytest.skip("set GRIDEXPAND_EXPANSION_TEST_ENV=<env file of an sg_* sandbox database> to run")
    from dotenv import dotenv_values
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    from gridexpand.db.database import SurroGridDatabase

    values = dotenv_values(env_file)
    name, port = str(values.get("DB_NAME") or ""), int(values.get("DB_PORT") or 5432)
    if not name.startswith("sg_") or port == REAL_DATABASE_PORT:
        pytest.skip("GRIDEXPAND_EXPANSION_TEST_ENV does not name an sg_* sandbox database")
    engine = create_engine(URL.create(
        "postgresql+psycopg2", username=values.get("DB_USER"), password=values.get("DB_PASSWORD"),
        host=values.get("DB_HOST"), port=port, database=name,
    ))
    yield SurroGridDatabase(engine)
    engine.dispose()


def _fixture(db):
    import pandapower as pp
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    from gridexpand.db import schema

    with db.engine.begin() as conn:
        conn.execute(text("""
            DROP SCHEMA IF EXISTS pylovo CASCADE; DROP SCHEMA IF EXISTS surrogrid CASCADE;
            CREATE SCHEMA pylovo;
            CREATE TABLE pylovo.grid_result (grid_result_id bigint PRIMARY KEY, version_id varchar(10), kcid int,
                bcid int, plz int, transformer_rated_power double precision, transformer_equipment_name text, grid jsonb);
            CREATE TABLE pylovo.pandapower_line (grid_result_id bigint, pp_index int, name text, std_type text,
                length_km double precision, max_i_ka double precision, parallel int, from_bus int, to_bus int);
            CREATE TABLE pylovo.lines_result (grid_result_id bigint, line_name text, geom geometry(LineString, 25832));
            CREATE TABLE pylovo.lines_result_view (id bigint, grid_result_id bigint, version_id varchar(10), plz int,
                kcid int, bcid int, line_name text, std_type text, is_helper boolean, helper_type text, from_bus int,
                to_bus int, length_km double precision, geom geometry(LineString, 25832));
            CREATE TABLE pylovo.postcode_result (version_id varchar(10), postcode_result_plz int, settlement_type int);
            INSERT INTO pylovo.postcode_result VALUES ('a1', 99999, 2);
        """))
    schema._checked.discard(str(db.engine.url))
    try:
        schema.ensure_schema(db.engine)
    except ProgrammingError:  # the QGIS views need pylovo tables this fixture lacks; migrations are applied
        pass
    schema._checked.add(str(db.engine.url))
    for gid, lon0, rated, heavy in ((1, 12.0, 400.0, True), (2, 12.0045, 630.0, False)):
        net = pp.create_empty_network()
        mv = pp.create_bus(net, vn_kv=20.0, geodata=(lon0, 48.0))
        buses = [pp.create_bus(net, vn_kv=0.4, geodata=(lon0 + dx, 48.0 + dy))
                 for dx, dy in ((0.0, 0.0), (0.0013, 0.0), (0.0033, 0.0), (0.0013, 0.0002), (0.0033, 0.0002))]
        lv, n1, n2, h1, h2 = buses
        pp.create_ext_grid(net, mv)
        pp.create_transformer(net, mv, lv, std_type="0.4 MVA 20/0.4 kV")
        lines = [(lv, n1, 0.1, "Line to 1"), (n1, n2, 0.15, "Line to 2"), (n1, h1, 0.02, "Line to 3"),
                 (n2, h2, 0.02, "Line to 4")]
        for a, b, length, name in lines:
            pp.create_line(net, a, b, length_km=length, std_type="NAYY 4x150 SE", name=name)
        pp.create_load(net, h1, p_mw=0.0)
        pp.create_load(net, h2, p_mw=0.0)
        with db.engine.begin() as conn:
            conn.execute(text("INSERT INTO pylovo.grid_result VALUES (:i, 'a1', :i, 1, 99999, :r, 'Tr', CAST(:g AS jsonb))"),
                         {"i": gid, "r": rated, "g": pp.to_json(net)})
            for idx, (a, b, length, name) in zip(net.line.index, lines):
                geom = f"LINESTRING({700000 + gid * 1000 + int(idx) * 10} 5300000, {700005 + gid * 1000 + int(idx) * 10} 5300005)"
                params = {"g": gid, "i": int(idx), "n": name, "ln": name.replace("Line to ", "L"), "l": length,
                          "a": int(a), "b": int(b), "w": geom, "id": gid * 100 + int(idx)}
                conn.execute(text("INSERT INTO pylovo.pandapower_line VALUES (:g, :i, :n, 'NAYY 4x150 SE', :l, 0.27, 1, :a, :b)"), params)
                conn.execute(text("INSERT INTO pylovo.lines_result VALUES (:g, :ln, ST_GeomFromText(:w, 25832))"), params)
                conn.execute(text("""INSERT INTO pylovo.lines_result_view VALUES (:id, :g, 'a1', 99999, :g, 1, :ln,
                                  'NAYY 4x150 SE', FALSE, NULL, :a, :b, :l, ST_GeomFromText(:w, 25832))"""), params)
        ref = {"ags": 9999999, "plz": 99999, "kcid": gid, "bcid": 1, "grid_result_id": gid, "version_id": "a1",
               "cell_id": f"test-{gid}"}
        run_id = db.create_powerflow_run(ref, urbs_input_file="test.h5", pre_only=False, scenario_key="test_staged",
                                         run_name="test_staged_post",
                                         assumptions={"timeframe_start": "2009-01-01 00:00:00+00:00"})
        loading = [300.0, 160.0, 70.0, 180.0] if heavy else [60.0, 40.0, 30.0, 30.0]
        db.write_powerflow_summary(run_id, "post", {
            "grid_summary": {"n_timesteps": 24, "n_converged_timesteps": 24, "n_failed_timesteps": 0, "n_voltage_buses": 2,
                             "n_cables": 4, "transformer_s_rated_mva": rated / 1000, "trafo_max_s_mva": 1.1 if heavy else 0.3,
                             "trafo_max_p_mw": 1.08 if heavy else 0.29, "trafo_max_q_mvar": 0.1, "trafo_critical_t_index": 5,
                             "lv_busbar_vm_pu": 0.96, "tap_steps": 0},
            "cable_summary": pd.DataFrame({"cable": [0, 1, 2, 3], "cable_max_i_ka": [0.27] * 4, "cable_parallel": [1.0] * 4,
                                           "cable_installed_capacity_ka": [0.27] * 4,
                                           "cable_loading_max_time_percent": loading,
                                           "cable_loading_max_t_index": [5] * 4}),
            "bus_voltage_summary": pd.DataFrame({"bus": [4, 5], "voltage_min_time_pu": [0.93, 0.88] if heavy else [0.95, 0.94],
                                                 "voltage_p05_time_pu": [0.95, 0.95]}),
        })


def _materialize(db, key):
    from sqlalchemy import text

    from gridexpand.analysis.expansion import grid_expansion as ge

    args = ge._build_parser().parse_args(["--run-name", "test_staged_post", "--stage", "post", "--assumption-key", key,
                                          "--analysis-key", f"test_{key}", "--no-refresh", "--replace"])
    ge.materialize(db, args)
    with db.engine.connect() as conn:
        read = lambda sql: pd.read_sql_query(text(sql), conn, params={"k": f"test_{key}"})  # noqa: E731
        lines = read("""SELECT l.powerflow_run_id, l.visible_line_name, l.measure, l.additional_parallel,
                               l.estimated_cost_eur, l.service_cost_eur FROM surrogrid.expansion_line_result l
                        JOIN surrogrid.expansion_analysis_run a USING (expansion_analysis_run_id)
                        WHERE a.analysis_key = :k ORDER BY 1, 2""")
        trafos = read("""SELECT t.station_measure, t.estimated_cost_eur, t.new_station_cost_eur
                         FROM surrogrid.expansion_transformer_result t
                         JOIN surrogrid.expansion_analysis_run a USING (expansion_analysis_run_id)
                         WHERE a.analysis_key = :k ORDER BY t.powerflow_run_id""")
        grids = read("""SELECT g.grid_label, g.total_cost_eur FROM surrogrid.expansion_grid_result g
                        JOIN surrogrid.expansion_analysis_run a USING (expansion_analysis_run_id)
                        WHERE a.analysis_key = :k ORDER BY 1""")
    return lines, trafos, grids


def test_synthetic_materialization(sandbox_db):
    _fixture(sandbox_db)
    with pytest.raises(ValueError, match="retired"):
        _materialize(sandbox_db, "de_lv_heuristic_2026")

    lines, trafos, grids = _materialize(sandbox_db, "de_lv_staged_2026")
    grid1 = lines[lines["powerflow_run_id"] == lines["powerflow_run_id"].min()]
    assert grid1["measure"].tolist() == ["relieved_by_new_station", "local", "none", "service"]
    assert grid1["additional_parallel"].tolist() == [0, 1, 0, 1]
    # 0.81 kA outlet: 2 x NAYY 150, relieved by the new substation; 0.432 kA feeder: 1 x NAYY 150
    # (85 kEUR/km x 0.15 km); 0.486 kA service line: 1 x NAYY 150 (85 kEUR/km x 0.02 km), not in the total.
    assert grid1["estimated_cost_eur"].round(2).tolist() == [0.0, 12750.0, 0.0, 0.0]
    assert grid1["service_cost_eur"].round(2).tolist() == [0.0, 0.0, 0.0, 1700.0]
    # 1100 kVA > 1000 kVA limit; the neighbour is about 90 m away (> 50 m): one new substation
    # (85 + 0.2 x 250 + 0.1 x 100 = 145 kEUR) plus the exchange to 1000 kVA (15 kEUR).
    assert trafos["station_measure"].tolist() == ["new_station", "none"]
    assert trafos["estimated_cost_eur"].tolist() == [160000.0, 0.0]
    assert grids["total_cost_eur"].round(2).tolist() == [172750.0, 0.0]
