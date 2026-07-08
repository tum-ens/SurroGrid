"""Read-only helpers for inspecting materialized expansion analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase


def latest_expansion_analysis_key(
    db: SurroGridDatabase | None = None,
    *,
    run_name: str | None = None,
    stage: str | None = None,
) -> str:
    """Return the newest expansion analysis key matching optional filters."""
    db = db or SurroGridDatabase()
    filters = []
    params: dict[str, object] = {}
    if run_name is not None:
        filters.append("run_name = :run_name")
        params["run_name"] = run_name
    if stage is not None:
        filters.append("stage = :stage")
        params["stage"] = stage
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = text(
        f"""
        SELECT analysis_key
        FROM surrogrid.expansion_analysis_run
        {where}
        ORDER BY created_at DESC, expansion_analysis_run_id DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        key = conn.execute(query, params).scalar_one_or_none()
    if key is None:
        raise ValueError("No expansion analysis run found for the selected filters.")
    return str(key)

def _resolve_analysis_key(db: SurroGridDatabase, analysis_key: str | None) -> str:
    return analysis_key or latest_expansion_analysis_key(db)

def load_expansion_overview(
    *,
    analysis_key: str | None = None,
    db: SurroGridDatabase | None = None,
) -> dict[str, object]:
    """Load notebook-friendly summary tables for one expansion analysis."""
    db = db or SurroGridDatabase()
    analysis_key = _resolve_analysis_key(db, analysis_key)
    queries = {
        "analysis_run": text(
            """
            SELECT
                ar.expansion_analysis_run_id,
                ar.analysis_key,
                ar.assumption_key,
                ar.run_name,
                ar.stage,
                ar.scenario_id,
                ar.ags,
                ar.plz,
                ar.created_at,
                ar.note
            FROM surrogrid.expansion_analysis_run ar
            WHERE ar.analysis_key = :analysis_key
            """
        ),
        "cost_summary": text(
            """
            WITH ar AS (
                SELECT expansion_analysis_run_id
                FROM surrogrid.expansion_analysis_run
                WHERE analysis_key = :analysis_key
            ), line_summary AS (
                SELECT
                    COUNT(DISTINCT grid_case_id) AS grids_with_line_rows,
                    COUNT(*) AS cable_segments,
                    COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                    COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS cable_segments_overloaded,
                    COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur
                FROM surrogrid.expansion_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), trafo_summary AS (
                SELECT
                    COUNT(*) AS transformers,
                    COUNT(*) FILTER (WHERE requires_expansion) AS transformers_requiring_expansion,
                    COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS transformers_overloaded,
                    COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur
                FROM surrogrid.expansion_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            )
            SELECT
                grids_with_line_rows,
                cable_segments,
                cable_segments_requiring_expansion,
                cable_segments_overloaded,
                cable_cost_eur,
                transformers,
                transformers_requiring_expansion,
                transformers_overloaded,
                transformer_cost_eur,
                cable_cost_eur + transformer_cost_eur AS total_cost_eur
            FROM line_summary CROSS JOIN trafo_summary
            """
        ),
        "grid_cost_summary": text(
            """
            WITH ar AS (
                SELECT expansion_analysis_run_id
                FROM surrogrid.expansion_analysis_run
                WHERE analysis_key = :analysis_key
            ), lines AS (
                SELECT
                    grid_case_id,
                    MAX(plz) AS plz,
                    MAX(kcid) AS kcid,
                    MAX(bcid) AS bcid,
                    COUNT(*) AS cable_segments,
                    COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                    COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur,
                    MAX(loading_percent) AS max_cable_loading_percent
                FROM surrogrid.expansion_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY grid_case_id
            ), trafos AS (
                SELECT
                    grid_case_id,
                    MAX(transformer_rated_power_kva) AS transformer_rated_power_kva,
                    MAX(required_transformer_kva) AS required_transformer_kva,
                    MAX(additional_transformer_kva) AS additional_transformer_kva,
                    BOOL_OR(requires_expansion) AS transformer_requires_expansion,
                    COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur,
                    MAX(loading_percent) AS transformer_loading_percent
                FROM surrogrid.expansion_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY grid_case_id
            )
            SELECT
                COALESCE(lines.grid_case_id, trafos.grid_case_id) AS grid_case_id,
                lines.plz,
                lines.kcid,
                lines.bcid,
                cable_segments,
                cable_segments_requiring_expansion,
                cable_cost_eur,
                max_cable_loading_percent,
                transformer_rated_power_kva,
                required_transformer_kva,
                additional_transformer_kva,
                transformer_requires_expansion,
                transformer_cost_eur,
                transformer_loading_percent,
                COALESCE(cable_cost_eur, 0.0) + COALESCE(transformer_cost_eur, 0.0) AS total_cost_eur
            FROM lines
            FULL OUTER JOIN trafos USING (grid_case_id)
            ORDER BY total_cost_eur DESC, grid_case_id
            """
        ),
        "cable_summary_by_type": text(
            """
            SELECT
                visible_std_type,
                COUNT(*) AS cable_segments,
                COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                COALESCE(SUM(length_km), 0.0) AS length_km,
                COALESCE(SUM(estimated_cost_eur), 0.0) AS estimated_cost_eur,
                MAX(loading_percent) AS max_loading_percent
            FROM surrogrid.expansion_line_result elr
            JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
            WHERE ar.analysis_key = :analysis_key
            GROUP BY visible_std_type
            ORDER BY estimated_cost_eur DESC, cable_segments_requiring_expansion DESC, visible_std_type
            """
        ),
        "transformer_summary_by_basis": text(
            """
            SELECT
                transformer_cost_basis,
                COUNT(*) AS transformers,
                COUNT(*) FILTER (WHERE requires_expansion) AS transformers_requiring_expansion,
                COALESCE(SUM(additional_transformer_kva), 0.0) AS additional_transformer_kva,
                COALESCE(SUM(estimated_cost_eur), 0.0) AS estimated_cost_eur,
                MAX(loading_percent) AS max_loading_percent
            FROM surrogrid.expansion_transformer_result etr
            JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
            WHERE ar.analysis_key = :analysis_key
            GROUP BY transformer_cost_basis
            ORDER BY estimated_cost_eur DESC, transformers_requiring_expansion DESC, transformer_cost_basis
            """
        ),
    }
    with db.engine.connect() as conn:
        tables = {
            name: pd.read_sql_query(query, conn, params={"analysis_key": analysis_key})
            for name, query in queries.items()
        }
    tables["analysis_key"] = analysis_key
    return tables
