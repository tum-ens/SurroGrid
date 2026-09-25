"""Materialize grid-expansion estimates from compact power-flow summaries.

One analysis (``surrogrid.expansion_analysis_run``) reduces the peak loading of
one power-flow run name and stage to reinforcement needs and costs per visible
pylovo cable and per transformer (synthetic grids, SQL in ``sql/``) or per
cable corridor and transformer of the real SWF/ÜZW grids (``real_materialization``).
The rules are documented in ``heuristics``. Each analysis is written in one
transaction: a failure leaves no partial analysis behind.

``gridexpand expansion --help`` lists the options. Batch callers pass
``--no-refresh`` and refresh the QGIS views once at the end with
``gridexpand.db.refresh_qgis_views()``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Connection

from gridexpand.db import refresh_qgis_views
from gridexpand.db.database import SurroGridDatabase, normalize_ags

from .real_materialization import insert_real_results, prepare_real_results

SQL_DIR = Path(__file__).with_name("sql")
DATA_SOURCE_LABELS = {
    "synthetic": "Synthetic",
    "real_swf": "Real SWF",
    "real_uzw": "Real ÜZW",
}


@cache
def sql_text(name: str) -> str:
    """SQL of ``sql/<name>`` with its fragments (``/*CABLE_SELECTION*/``, ``/*TRANSFORMER_COST*/``) inlined."""
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    for marker, fragment in (
        ("/*CABLE_SELECTION*/", "cable_selection.sql"),
        ("/*TRANSFORMER_COST*/", "transformer_cost.sql"),
    ):
        if marker in sql:
            sql = sql.replace(marker, sql_text(fragment))
    return sql


def _ags_values(args: argparse.Namespace) -> list[int] | None:
    return [normalize_ags(value) for value in args.ags] if args.ags else None


def _plz_values(args: argparse.Namespace) -> list[int] | None:
    return [int(value) for value in args.plz] if args.plz else None


def _single(values: list[int] | None) -> int | None:
    return values[0] if values is not None and len(values) == 1 else None


def _synthetic_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "run_name": args.run_name,
        "stage": args.stage,
        "scenario_id": args.scenario_id,
        "ags": _ags_values(args),
        "plz": _plz_values(args),
        "pylovo_version_id": args.pylovo_version_id,
        "assumption_key": args.assumption_key,
        "line_existing_duct_share": args.line_existing_duct_share,
    }


def _analysis_key(args: argparse.Namespace) -> str:
    if args.analysis_key:
        return args.analysis_key
    ags = _single(_ags_values(args))
    scope = str(ags).zfill(8) if ags is not None else "all"
    run = args.run_name.replace("baseline_static_", "").replace("_powerflow", "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scope}_{run}_{args.stage}_{stamp}"


def analysis_identity(args: argparse.Namespace) -> dict[str, object]:
    """The fields an analysis key must keep when it is replaced (``--replace``)."""
    return {
        "run_name": args.run_name,
        "stage": args.stage,
        "data_source": DATA_SOURCE_LABELS[args.data_source],
        "ags": _single(_ags_values(args)),
    }


def check_replacement(
    analysis_key: str,
    existing: Mapping[str, Any] | None,
    identity: Mapping[str, object],
    *,
    replace: bool,
) -> bool:
    """Decide whether an existing analysis row may be replaced.

    Returns:
        True if a row exists and is to be deleted, False if there is none.

    Raises:
        RuntimeError: a row exists and ``replace`` is False, or it belongs to a
            different run name, stage, data source or AGS (``--replace`` never
            deletes another analysis).
    """
    if existing is None:
        return False
    if not replace:
        raise RuntimeError(
            f"analysis_key={analysis_key!r} exists. Use --replace to overwrite an existing analysis."
        )
    mismatched = {
        field: (existing.get(field), value)
        for field, value in identity.items()
        if existing.get(field) != value
    }
    if mismatched:
        details = ", ".join(f"{field}: stored {old!r}, requested {new!r}" for field, (old, new) in mismatched.items())
        raise RuntimeError(
            f"--replace refuses to overwrite analysis_key={analysis_key!r} of another analysis ({details}). "
            "Use a distinct --analysis-key."
        )
    return True


def resolve_scenario_id(requested: int | None, selected: set[int]) -> int | None:
    """Scenario of the analysis: the requested one, else the single scenario of the selected runs."""
    if requested is not None:
        return int(requested)
    if len(selected) == 1:
        return int(next(iter(selected)))
    if selected:
        print(
            f"Warning: the selected runs belong to {len(selected)} scenarios {sorted(selected)}; "
            "expansion_analysis_run.scenario_id stays NULL."
        )
    return None


def _create_analysis_run(
    conn: Connection,
    *,
    analysis_key: str,
    args: argparse.Namespace,
    scenario_id: int | None,
) -> int:
    existing = conn.execute(
        text(
            """
            SELECT run_name, stage, data_source, ags
            FROM surrogrid.expansion_analysis_run
            WHERE analysis_key = :analysis_key
            """
        ),
        {"analysis_key": analysis_key},
    ).mappings().first()
    identity = analysis_identity(args)
    if check_replacement(analysis_key, existing, identity, replace=args.replace):
        conn.execute(
            text("DELETE FROM surrogrid.expansion_analysis_run WHERE analysis_key = :analysis_key"),
            {"analysis_key": analysis_key},
        )
    return int(
        conn.execute(
            text(
                """
                INSERT INTO surrogrid.expansion_analysis_run (
                    analysis_key, assumption_key, run_name, stage,
                    scenario_id, ags, plz, note, data_source
                )
                VALUES (
                    :analysis_key, :assumption_key, :run_name, :stage,
                    :scenario_id, :ags, :plz, :note, :data_source
                )
                RETURNING expansion_analysis_run_id
                """
            ),
            {
                **identity,
                "analysis_key": analysis_key,
                "assumption_key": args.assumption_key,
                "scenario_id": scenario_id,
                "plz": _single(_plz_values(args)),
                "note": args.note,
            },
        ).scalar_one()
    )


def check_component_audit(row: Mapping[str, Any]) -> str | None:
    """Evaluate the unmapped-component audit; return a warning or None.

    Raises:
        RuntimeError: no selected run, no component loading, or an overloaded
            component without a visible geometry (its cost would be hidden).
    """
    selected_runs = int(row["selected_runs"] or 0)
    active_components = int(row["active_components"] or 0)
    unmapped = int(row["unmapped_components"] or 0)
    overloaded = int(row["overloaded_unmapped_components"] or 0)
    root_like = int(row["root_connector_like_components"] or 0)
    overloaded_root_like = int(row["overloaded_root_connector_like_components"] or 0)
    max_loading = float(row["max_unmapped_loading_percent"] or 0.0)
    if selected_runs == 0:
        raise RuntimeError("No power-flow runs match the requested expansion scope.")
    if active_components == 0:
        raise RuntimeError(
            "No power-flow cable summaries (surrogrid.powerflow_cable_summary) match the requested "
            "expansion scope and stage. Run Step 4 with --outputs summary (or raw,summary) first."
        )
    if overloaded > 0:
        raise RuntimeError(
            f"Found {overloaded} overloaded line component(s) without a visible pylovo geometry "
            f"(max unmapped loading {max_loading:.2f}%). Refusing to hide expansion needs."
        )
    if unmapped > 0:
        return (
            "Warning: ignored "
            f"{unmapped} active unmapped line component(s) "
            f"({root_like} root-connector-like, {overloaded_root_like} overloaded root-connector-like; "
            f"max loading {max_loading:.2f}%)."
        )
    return None


def _prepare_synthetic_scope(conn: Connection, args: argparse.Namespace) -> set[int]:
    """Create the temp tables of the selected runs and their component loading; return their scenarios."""
    params = _synthetic_params(args)
    conn.execute(text(sql_text("selected_runs.sql")), params)
    conn.execute(text(sql_text("component_loading.sql")), params)
    conn.execute(text("ANALYZE expansion_selected_run"))
    conn.execute(text("ANALYZE expansion_component_loading"))
    warning = check_component_audit(conn.execute(text(sql_text("component_audit.sql"))).mappings().one())
    if warning:
        print(warning)
    return {
        int(value)
        for value in conn.execute(text("SELECT DISTINCT scenario_id FROM expansion_selected_run")).scalars()
    }


def _materialize_synthetic(conn: Connection, run_id: int, args: argparse.Namespace) -> dict[str, int]:
    params = {**_synthetic_params(args), "expansion_analysis_run_id": run_id}
    lines = conn.execute(text(sql_text("line_insert.sql")), params).rowcount
    transformers = conn.execute(text(sql_text("transformer_insert.sql")), params).rowcount
    return {"line_rows": int(lines or 0), "transformer_rows": int(transformers or 0)}


def _print_summary(db: SurroGridDatabase, analysis_key: str) -> None:
    query = text(
        """
        WITH selected AS (
            SELECT expansion_analysis_run_id, analysis_key, data_source
            FROM surrogrid.expansion_analysis_run
            WHERE analysis_key = :analysis_key
        ), synthetic AS (
            SELECT
                COUNT(DISTINCT elr.grid_case_id) AS grids_with_line_rows,
                COUNT(*) FILTER (WHERE elr.requires_expansion) AS cable_expansion_segments,
                COALESCE(SUM(elr.estimated_cost_eur), 0.0) AS cable_cost_eur,
                (SELECT COUNT(*) FROM surrogrid.expansion_transformer_result etr
                 WHERE etr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)
                   AND etr.requires_expansion) AS transformer_expansion_count,
                (SELECT COALESCE(SUM(etr.estimated_cost_eur), 0.0)
                 FROM surrogrid.expansion_transformer_result etr
                 WHERE etr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)) AS transformer_cost_eur
            FROM surrogrid.expansion_line_result elr
            WHERE elr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)
        ), real AS (
            SELECT
                COUNT(DISTINCT erlr.real_grid_case_id) AS grids_with_line_rows,
                COUNT(*) FILTER (WHERE erlr.requires_expansion) AS cable_expansion_segments,
                COALESCE(SUM(erlr.estimated_cost_eur), 0.0) AS cable_cost_eur,
                (SELECT COUNT(*) FROM surrogrid.expansion_real_transformer_result ertr
                 WHERE ertr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)
                   AND ertr.requires_expansion) AS transformer_expansion_count,
                (SELECT COALESCE(SUM(ertr.estimated_cost_eur), 0.0)
                 FROM surrogrid.expansion_real_transformer_result ertr
                 WHERE ertr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)) AS transformer_cost_eur
            FROM surrogrid.expansion_real_line_result erlr
            WHERE erlr.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM selected)
        )
        SELECT
            selected.analysis_key,
            selected.data_source,
            CASE WHEN selected.data_source <> 'Synthetic' THEN real.grids_with_line_rows ELSE synthetic.grids_with_line_rows END AS grids_with_line_rows,
            CASE WHEN selected.data_source <> 'Synthetic' THEN real.cable_expansion_segments ELSE synthetic.cable_expansion_segments END AS cable_expansion_segments,
            CASE WHEN selected.data_source <> 'Synthetic' THEN real.cable_cost_eur ELSE synthetic.cable_cost_eur END AS cable_cost_eur,
            CASE WHEN selected.data_source <> 'Synthetic' THEN real.transformer_expansion_count ELSE synthetic.transformer_expansion_count END AS transformer_expansion_count,
            CASE WHEN selected.data_source <> 'Synthetic' THEN real.transformer_cost_eur ELSE synthetic.transformer_cost_eur END AS transformer_cost_eur,
            (SELECT COUNT(*) FROM surrogrid.expansion_real_grid_status s
             WHERE s.expansion_analysis_run_id = selected.expansion_analysis_run_id
               AND s.cost_status = 'incomplete') AS incomplete_grids,
            (SELECT COUNT(*) FROM surrogrid.expansion_real_grid_status s
             WHERE s.expansion_analysis_run_id = selected.expansion_analysis_run_id
               AND s.cost_status = 'excluded') AS excluded_grids
        FROM selected CROSS JOIN synthetic CROSS JOIN real
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(query, {"analysis_key": analysis_key}).mappings().one()
    total = float(row["cable_cost_eur"]) + float(row["transformer_cost_eur"])
    print(f"analysis_key: {row['analysis_key']}")
    print(f"data_source: {row['data_source']}")
    print(f"grids_with_line_rows: {row['grids_with_line_rows']}")
    print(f"incomplete_grids: {row['incomplete_grids']}")
    print(f"excluded_grids: {row['excluded_grids']}")
    print(f"cable_expansion_segments: {row['cable_expansion_segments']}")
    print(f"transformer_expansion_count: {row['transformer_expansion_count']}")
    print(f"cable_cost_eur: {float(row['cable_cost_eur']):.2f}")
    print(f"transformer_cost_eur: {float(row['transformer_cost_eur']):.2f}")
    print(f"total_cost_eur: {total:.2f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize overload-based cable and transformer expansion estimates."
    )
    parser.add_argument(
        "--run-name",
        help="Power-flow run name to analyze (the summary run of Step 4); required unless --schema-only/--refresh-only.",
    )
    parser.add_argument(
        "--data-source",
        choices=tuple(DATA_SOURCE_LABELS),
        default="synthetic",
        help="Network model whose compact power-flow summaries are materialized.",
    )
    parser.add_argument(
        "--stage",
        default="post",
        choices=("pre", "post"),
        help="Power-flow stage to analyze.",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        help=(
            "Optional scenario_id filter. The analysis records it, or else the single "
            "scenario of the selected runs."
        ),
    )
    parser.add_argument(
        "--ags",
        action="append",
        default=[],
        help="Optional synthetic AGS filter, for example 09162000 for Munich; repeatable.",
    )
    parser.add_argument(
        "--plz",
        action="append",
        type=int,
        default=[],
        help="Optional PLZ filter (synthetic grid PLZ or real majority PLZ); repeatable.",
    )
    parser.add_argument(
        "--pylovo-version-id",
        help=(
            "pylovo version of the run. Filters synthetic grid cases and selects the "
            "real-grid settlement type; real runs otherwise use the version recorded "
            "in the power-flow run assumptions."
        ),
    )
    parser.add_argument(
        "--exclude-real-lv-id",
        action="append",
        default=[],
        help="Real grid id (SWF LV or ÜZW area) to retain in coverage reporting but exclude from costing; repeatable.",
    )
    parser.add_argument(
        "--assumption-key",
        default="de_lv_heuristic_2026",
        help="Cost/planning assumption row to use.",
    )
    parser.add_argument(
        "--line-existing-duct-share",
        type=float,
        help=(
            "Optional share of reinforced LV routes that can use existing ducts/empty pipes. "
            "If omitted, the value from the selected cost assumption is used."
        ),
    )
    parser.add_argument("--analysis-key", help="Readable key for this materialized result.")
    parser.add_argument("--note", default="", help="Free-text note stored with the analysis run.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace an existing analysis with the same key. Refused if that analysis has "
            "another run name, stage, data source or AGS."
        ),
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help=(
            "Do not refresh the QGIS materialized views (batch runners refresh once at the end "
            "with gridexpand.db.refresh_qgis_views())."
        ),
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Only refresh the QGIS materialized views and exit.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only initialise a fresh database (tables and views) and exit.",
    )
    return parser


def materialize(
    db: SurroGridDatabase, args: argparse.Namespace, *, refresh_views: bool = False
) -> str:
    """Materialize one expansion analysis in one transaction and return its key.

    Args:
        db: database facade.
        args: parsed ``grid_expansion`` arguments (``_build_parser``).
        refresh_views: refresh the QGIS materialized views after the commit.
    """
    analysis_key = _analysis_key(args)
    if args.data_source == "synthetic":
        with db.engine.begin() as conn:
            scenario_ids = _prepare_synthetic_scope(conn, args)
            run_id = _create_analysis_run(
                conn,
                analysis_key=analysis_key,
                args=args,
                scenario_id=resolve_scenario_id(args.scenario_id, scenario_ids),
            )
            counts = _materialize_synthetic(conn, run_id, args)
    else:
        # Reads and grid files first; the transaction only writes.
        results = prepare_real_results(db, args)
        with db.engine.begin() as conn:
            run_id = _create_analysis_run(
                conn,
                analysis_key=analysis_key,
                args=args,
                scenario_id=resolve_scenario_id(args.scenario_id, results.scenario_ids),
            )
            counts = insert_real_results(conn, run_id, results)
        print(f"grid status rows inserted: {counts['grid_status_rows']}")
    print(f"line rows inserted: {counts['line_rows']}")
    print(f"transformer rows inserted: {counts['transformer_rows']}")
    if refresh_views:
        refresh_qgis_views(db.engine)
        print("QGIS materialized views refreshed.")
    _print_summary(db, analysis_key)
    return analysis_key


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.run_name is None and not (args.schema_only or args.refresh_only):
        parser.error("--run-name is required")

    db = SurroGridDatabase()
    db.ensure_schema()
    if args.schema_only:
        print("Expansion schema and QGIS views are ready.")
        return
    if args.refresh_only:
        refresh_qgis_views(db.engine)
        print("QGIS materialized views refreshed.")
        return
    materialize(db, args, refresh_views=not args.no_refresh)


if __name__ == "__main__":
    main()
