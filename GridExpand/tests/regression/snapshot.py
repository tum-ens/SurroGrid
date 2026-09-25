"""Snapshot every table of the surrogrid schema of one SANDBOX database.

Usage: python snapshot.py <dbname> <out.pkl>

Surrogate ids are replaced by natural keys, timestamps and absolute paths are
dropped, rows are sorted. Read-only. Connects to the sandbox given by
HARNESS_DB_HOST/PORT/USER/PASSWORD (defaults: 127.0.0.1:55439, sandbox/sandbox)
and refuses database names without the HARNESS_DB_PREFIX prefix (default "sg_").
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DB = sys.argv[1]
OUT = Path(sys.argv[2])
PREFIX = os.environ.get("HARNESS_DB_PREFIX", "sg_")
if not DB.startswith(PREFIX):
    raise SystemExit(f"refusing: database {DB!r} does not start with {PREFIX!r}")
engine = create_engine(
    "postgresql+psycopg2://"
    f"{os.environ.get('HARNESS_DB_USER', 'sandbox')}:{os.environ.get('HARNESS_DB_PASSWORD', 'sandbox')}"
    f"@{os.environ.get('HARNESS_DB_HOST', '127.0.0.1')}:{os.environ.get('HARNESS_DB_PORT', '55439')}/{DB}"
)

DROP = {"created_at", "updated_at", "hash", "computed_at", "refreshed_at"}
PATH_COLUMNS = {"urbs_input_file", "bridge_filename", "source_file", "source_path"}


def read(conn, sql):
    return pd.read_sql_query(text(sql), conn)


with engine.connect() as conn:
    relations = read(
        conn,
        """
        SELECT c.relname AS name, c.relkind
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'surrogrid' AND c.relkind IN ('r', 'm', 'v', 'p')
        ORDER BY 1
        """,
    )
    maps: dict[str, dict] = {}
    gc = read(conn, "SELECT grid_case_id, plz, kcid, bcid FROM surrogrid.grid_case")
    maps["grid_case_id"] = {
        r.grid_case_id: f"{r.plz}/{r.kcid}/{r.bcid}" for r in gc.itertuples()
    }
    sc = read(conn, "SELECT scenario_id, scenario_key FROM surrogrid.scenario")
    maps["scenario_id"] = dict(zip(sc.scenario_id, sc.scenario_key))
    for table, key in (
        ("pipeline_run", "pipeline_run_id"),
        ("demand_allocation_run", "demand_allocation_run_id"),
        ("powerflow_run", "powerflow_run_id"),
    ):
        df = read(conn, f"SELECT {key}, grid_case_id, run_name FROM surrogrid.{table}")
        maps[key] = {
            r[0]: f"{maps['grid_case_id'].get(r[1])}|{r[2]}"
            for r in df.itertuples(index=False)
        }
    for table, key, natural in (
        ("expansion_analysis_run", "expansion_analysis_run_id", "analysis_key"),
        ("real_grid_case", "real_grid_case_id", "lv_id"),
        ("real_powerflow_run", "real_powerflow_run_id", "run_name"),
        ("expansion_cost_assumption", "cost_assumption_id", "cost_assumption_key"),
    ):
        try:
            df = read(conn, f"SELECT {key}, {natural} FROM surrogrid.{table}")
            maps[key] = dict(zip(df[key], df[natural]))
        except Exception:
            conn.rollback()

    snapshot: dict[str, pd.DataFrame] = {}
    for rel in relations.itertuples():
        if rel.name == "schema_marker":
            continue
        df = read(conn, f'SELECT * FROM surrogrid."{rel.name}"')
        df = df.drop(columns=[c for c in df.columns if c in DROP], errors="ignore")
        for column in list(df.columns):
            if column in maps:
                df[column] = df[column].map(maps[column])
            elif column.endswith("_id") and column in {
                "cost_assumption_id",
            }:
                df[column] = df[column].map(maps.get(column, {}))
            if column in PATH_COLUMNS:
                df[column] = df[column].astype(str).map(lambda p: Path(p).name)
            if df[column].dtype == object:
                df[column] = df[column].map(
                    lambda v: repr(sorted(v.items())) if isinstance(v, dict)
                    else repr(v) if isinstance(v, list) else v
                )
            if column == "geom" or column.endswith("_geom"):
                df[column] = df[column].astype(str)
        # Drop remaining serial primary keys that have no natural mapping.
        pk_like = [c for c in df.columns if c.endswith("_id") and c not in maps and df[c].dtype.kind in "iu" and df[c].is_unique and len(df) > 1]
        df = df.drop(columns=pk_like)
        df = df.reindex(sorted(df.columns), axis=1)
        if len(df):
            keys = [c for c in df.columns if df[c].dtype.kind != "f"]
            keys += [c for c in df.columns if df[c].dtype.kind == "f"]
            df = df.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)
        snapshot[rel.name] = df

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("wb") as handle:
    pickle.dump(snapshot, handle)
print({name: len(df) for name, df in snapshot.items() if len(df)})
