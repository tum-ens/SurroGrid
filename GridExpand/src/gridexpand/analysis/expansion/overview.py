"""Read-only helpers for synthetic and real expansion analyses."""

from __future__ import annotations


import pandas as pd
from sqlalchemy import text

from gridexpand.db.database import SurroGridDatabase


def latest_expansion_analysis_key(
    db: SurroGridDatabase | None = None,
    *,
    run_name: str | None = None,
    stage: str | None = None,
    data_source: str | None = None,
) -> str:
    """Return the newest expansion analysis key matching optional filters."""
    db = db or SurroGridDatabase()
    filters = []
    params: dict[str, object] = {}
    for column, value in (("run_name", run_name), ("stage", stage), ("data_source", data_source)):
        if value is not None:
            filters.append(f"{column} = :{column}")
            params[column] = value
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
    """Load source-neutral summary tables for one materialized analysis."""
    db = db or SurroGridDatabase()
    analysis_key = _resolve_analysis_key(db, analysis_key)
    queries = {
        "analysis_run": text(
            """
            SELECT expansion_analysis_run_id, analysis_key, assumption_key,
                   run_name, stage, scenario_id, ags, plz, data_source,
                   created_at, note
            FROM surrogrid.expansion_analysis_run
            WHERE analysis_key = :analysis_key
            """
        ),
        "cost_summary": text(
            """
            WITH ar AS (
                SELECT expansion_analysis_run_id, data_source
                FROM surrogrid.expansion_analysis_run
                WHERE analysis_key = :analysis_key
            ), synthetic_lines AS (
                SELECT COUNT(DISTINCT grid_case_id) AS grids_with_line_rows,
                       COUNT(*) AS cable_segments,
                       COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                       COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS cable_segments_overloaded,
                       COALESCE(SUM(reinforcement_150_count), 0) AS reinforcement_150_count,
                       COALESCE(SUM(reinforcement_185_count), 0) AS reinforcement_185_count,
                       COALESCE(SUM(reinforcement_240_count), 0) AS reinforcement_240_count,
                       COALESCE(SUM(reinforcement_added_capacity_ka), 0.0) AS reinforcement_added_capacity_ka,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur
                FROM surrogrid.expansion_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), synthetic_trafos AS (
                SELECT COUNT(*) AS transformers,
                       COUNT(*) FILTER (WHERE requires_expansion) AS transformers_requiring_expansion,
                       COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS transformers_overloaded,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur
                FROM surrogrid.expansion_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), real_lines AS (
                SELECT COUNT(DISTINCT real_grid_case_id) AS grids_with_line_rows,
                       COUNT(*) AS cable_segments,
                       COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                       COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS cable_segments_overloaded,
                       COALESCE(SUM(reinforcement_150_count), 0) AS reinforcement_150_count,
                       COALESCE(SUM(reinforcement_185_count), 0) AS reinforcement_185_count,
                       COALESCE(SUM(reinforcement_240_count), 0) AS reinforcement_240_count,
                       COALESCE(SUM(reinforcement_added_capacity_ka), 0.0) AS reinforcement_added_capacity_ka,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur
                FROM surrogrid.expansion_real_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), real_trafos AS (
                SELECT COUNT(*) AS transformers,
                       COUNT(*) FILTER (WHERE requires_expansion) AS transformers_requiring_expansion,
                       COUNT(*) FILTER (WHERE overloaded_at_100_percent) AS transformers_overloaded,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur
                FROM surrogrid.expansion_real_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), real_status AS (
                SELECT COUNT(*) AS grids_total,
                       COUNT(*) FILTER (WHERE cost_status = 'complete') AS grids_complete,
                       COUNT(*) FILTER (WHERE cost_status = 'incomplete') AS grids_incomplete,
                       COUNT(*) FILTER (WHERE cost_status = 'excluded') AS grids_excluded
                FROM surrogrid.expansion_real_grid_status
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), grid_parts AS (
                SELECT MAX(rule_set) AS rule_set,
                       COALESCE(SUM(service_cost_eur), 0.0) AS service_cost_eur,
                       COALESCE(SUM(transformer_exchange_cost_eur), 0.0) AS transformer_exchange_cost_eur,
                       COALESCE(SUM(load_transfer_cost_eur), 0.0) AS load_transfer_cost_eur,
                       COALESCE(SUM(new_station_cost_eur), 0.0) AS new_station_cost_eur,
                       COALESCE(SUM(voltage_cost_eur), 0.0) AS voltage_cost_eur,
                       COUNT(*) FILTER (WHERE station_measure = 'transfer') AS load_transfers,
                       COALESCE(SUM(new_stations), 0.0) AS new_stations,
                       COUNT(*) FILTER (WHERE voltage_measure IN ('ront', 'split')) AS voltage_measures
                FROM surrogrid.expansion_grid_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            )
            SELECT ar.data_source,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.grids_with_line_rows ELSE sl.grids_with_line_rows END AS grids_with_line_rows,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.cable_segments ELSE sl.cable_segments END AS cable_segments,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.cable_segments_requiring_expansion ELSE sl.cable_segments_requiring_expansion END AS cable_segments_requiring_expansion,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.cable_segments_overloaded ELSE sl.cable_segments_overloaded END AS cable_segments_overloaded,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.reinforcement_150_count ELSE sl.reinforcement_150_count END AS reinforcement_150_count,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.reinforcement_185_count ELSE sl.reinforcement_185_count END AS reinforcement_185_count,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.reinforcement_240_count ELSE sl.reinforcement_240_count END AS reinforcement_240_count,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.reinforcement_added_capacity_ka ELSE sl.reinforcement_added_capacity_ka END AS reinforcement_added_capacity_ka,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rl.cable_cost_eur ELSE sl.cable_cost_eur END AS cable_cost_eur,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rt.transformers ELSE st.transformers END AS transformers,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rt.transformers_requiring_expansion ELSE st.transformers_requiring_expansion END AS transformers_requiring_expansion,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rt.transformers_overloaded ELSE st.transformers_overloaded END AS transformers_overloaded,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rt.transformer_cost_eur ELSE st.transformer_cost_eur END AS transformer_cost_eur,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rs.grids_total ELSE sl.grids_with_line_rows END AS grids_total,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rs.grids_complete ELSE sl.grids_with_line_rows END AS grids_complete,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rs.grids_incomplete ELSE 0 END AS grids_incomplete,
                   CASE WHEN ar.data_source <> 'Synthetic' THEN rs.grids_excluded ELSE 0 END AS grids_excluded,
                   (CASE WHEN ar.data_source <> 'Synthetic' THEN rl.cable_cost_eur ELSE sl.cable_cost_eur END)
                   + (CASE WHEN ar.data_source <> 'Synthetic' THEN rt.transformer_cost_eur ELSE st.transformer_cost_eur END) AS total_cost_eur,
                   gp.rule_set, gp.service_cost_eur, gp.transformer_exchange_cost_eur,
                   gp.load_transfer_cost_eur, gp.new_station_cost_eur, gp.voltage_cost_eur,
                   gp.load_transfers, gp.new_stations, gp.voltage_measures
            FROM ar
            CROSS JOIN synthetic_lines sl CROSS JOIN synthetic_trafos st
            CROSS JOIN real_lines rl CROSS JOIN real_trafos rt CROSS JOIN real_status rs
            CROSS JOIN grid_parts gp
            """
        ),
        "grid_cost_summary": text(
            """
            WITH ar AS (
                SELECT expansion_analysis_run_id, data_source
                FROM surrogrid.expansion_analysis_run
                WHERE analysis_key = :analysis_key
            ), synthetic_lines AS (
                SELECT grid_case_id, MAX(plz) AS plz, MAX(kcid) AS kcid, MAX(bcid) AS bcid,
                       COUNT(*) AS cable_segments,
                       COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                       COALESCE(SUM(reinforcement_150_count), 0) AS reinforcement_150_count,
                       COALESCE(SUM(reinforcement_185_count), 0) AS reinforcement_185_count,
                       COALESCE(SUM(reinforcement_240_count), 0) AS reinforcement_240_count,
                       COALESCE(SUM(reinforcement_added_capacity_ka), 0.0) AS reinforcement_added_capacity_ka,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur,
                       MAX(loading_percent) AS max_cable_loading_percent
                FROM surrogrid.expansion_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY grid_case_id
            ), synthetic_trafos AS (
                SELECT grid_case_id,
                       MAX(transformer_rated_power_kva) AS transformer_rated_power_kva,
                       MAX(required_transformer_kva) AS required_transformer_kva,
                       MAX(additional_transformer_kva) AS additional_transformer_kva,
                       BOOL_OR(requires_expansion) AS transformer_requires_expansion,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur,
                       MAX(loading_percent) AS transformer_loading_percent
                FROM surrogrid.expansion_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY grid_case_id
            ), grid_parts AS (
                SELECT grid_case_id, real_grid_case_id, station_measure, escalation_reason,
                       new_stations, voltage_measure, service_cost_eur, transformer_exchange_cost_eur,
                       load_transfer_cost_eur, new_station_cost_eur, voltage_cost_eur
                FROM surrogrid.expansion_grid_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            ), synthetic_rows AS (
                SELECT 'Synthetic'::TEXT AS data_source,
                       COALESCE(l.grid_case_id, t.grid_case_id)::TEXT AS source_grid_id,
                       CONCAT(l.plz, '-', l.kcid, '-', l.bcid) AS grid_label,
                       COALESCE(l.grid_case_id, t.grid_case_id) AS grid_case_id,
                       NULL::BIGINT AS real_grid_case_id, l.plz, l.kcid, l.bcid,
                       l.cable_segments, l.cable_segments_requiring_expansion,
                       l.reinforcement_150_count, l.reinforcement_185_count,
                       l.reinforcement_240_count, l.reinforcement_added_capacity_ka,
                       COALESCE(l.cable_cost_eur, 0.0) AS cable_cost_eur,
                       l.max_cable_loading_percent,
                       t.transformer_rated_power_kva, t.required_transformer_kva,
                       t.additional_transformer_kva, t.transformer_requires_expansion,
                       COALESCE(t.transformer_cost_eur, 0.0) AS transformer_cost_eur,
                       t.transformer_loading_percent, 'complete'::TEXT AS cost_status,
                       0 AS n_failed_timesteps, NULL::TEXT AS status_reason,
                       COALESCE(l.cable_cost_eur, 0.0) + COALESCE(t.transformer_cost_eur, 0.0) AS total_cost_eur,
                       NULL::TEXT AS real_source, NULL::TEXT AS lv_id,
                       gp.station_measure, gp.escalation_reason, gp.new_stations, gp.voltage_measure,
                       gp.service_cost_eur, gp.transformer_exchange_cost_eur, gp.load_transfer_cost_eur,
                       gp.new_station_cost_eur, gp.voltage_cost_eur
                FROM synthetic_lines l FULL OUTER JOIN synthetic_trafos t USING (grid_case_id)
                LEFT JOIN grid_parts gp ON gp.grid_case_id = COALESCE(l.grid_case_id, t.grid_case_id)
            ), real_lines AS (
                SELECT real_grid_case_id, MAX(plz) AS plz, MAX(lv_id) AS lv_id,
                       COUNT(*) AS cable_segments,
                       COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                       COALESCE(SUM(reinforcement_150_count), 0) AS reinforcement_150_count,
                       COALESCE(SUM(reinforcement_185_count), 0) AS reinforcement_185_count,
                       COALESCE(SUM(reinforcement_240_count), 0) AS reinforcement_240_count,
                       COALESCE(SUM(reinforcement_added_capacity_ka), 0.0) AS reinforcement_added_capacity_ka,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS cable_cost_eur,
                       MAX(loading_percent) AS max_cable_loading_percent
                FROM surrogrid.expansion_real_line_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY real_grid_case_id
            ), real_trafos AS (
                SELECT real_grid_case_id, MAX(lv_id) AS lv_id,
                       MAX(transformer_rated_power_kva) AS transformer_rated_power_kva,
                       MAX(required_transformer_kva) AS required_transformer_kva,
                       MAX(additional_transformer_kva) AS additional_transformer_kva,
                       BOOL_OR(requires_expansion) AS transformer_requires_expansion,
                       COALESCE(SUM(estimated_cost_eur), 0.0) AS transformer_cost_eur,
                       MAX(loading_percent) AS transformer_loading_percent
                FROM surrogrid.expansion_real_transformer_result
                WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
                GROUP BY real_grid_case_id
            ), real_rows AS (
                SELECT (SELECT data_source FROM ar) AS data_source, s.real_grid_case_id::TEXT AS source_grid_id,
                       CASE WHEN rgc.source = 'dso_b' THEN CONCAT('DSO B area-', LPAD(s.lv_id, 4, '0'))
                            ELSE CONCAT(UPPER(rgc.source), ' LV_', LPAD(s.lv_id, 3, '0'))
                       END AS grid_label,
                       NULL::BIGINT AS grid_case_id, s.real_grid_case_id,
                       s.plz, NULL::INTEGER AS kcid, NULL::INTEGER AS bcid,
                       l.cable_segments, l.cable_segments_requiring_expansion,
                       l.reinforcement_150_count, l.reinforcement_185_count,
                       l.reinforcement_240_count, l.reinforcement_added_capacity_ka,
                       CASE WHEN s.cost_status = 'complete' THEN COALESCE(l.cable_cost_eur, 0.0) END AS cable_cost_eur,
                       l.max_cable_loading_percent,
                       t.transformer_rated_power_kva, t.required_transformer_kva,
                       t.additional_transformer_kva, t.transformer_requires_expansion,
                       CASE WHEN s.cost_status = 'complete' THEN COALESCE(t.transformer_cost_eur, 0.0) END AS transformer_cost_eur,
                       t.transformer_loading_percent, s.cost_status, s.n_failed_timesteps,
                       s.status_reason,
                       CASE WHEN s.cost_status = 'complete' THEN COALESCE(l.cable_cost_eur, 0.0) + COALESCE(t.transformer_cost_eur, 0.0) END AS total_cost_eur,
                       rgc.source AS real_source, s.lv_id,
                       gp.station_measure, gp.escalation_reason, gp.new_stations, gp.voltage_measure,
                       gp.service_cost_eur, gp.transformer_exchange_cost_eur, gp.load_transfer_cost_eur,
                       gp.new_station_cost_eur, gp.voltage_cost_eur
                FROM surrogrid.expansion_real_grid_status s
                JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
                LEFT JOIN real_lines l USING (real_grid_case_id)
                LEFT JOIN real_trafos t USING (real_grid_case_id)
                LEFT JOIN grid_parts gp USING (real_grid_case_id)
                WHERE s.expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar)
            )
            SELECT * FROM synthetic_rows WHERE (SELECT data_source FROM ar) = 'Synthetic'
            UNION ALL
            SELECT * FROM real_rows WHERE (SELECT data_source FROM ar) <> 'Synthetic'
            ORDER BY total_cost_eur DESC NULLS LAST, grid_label
            """
        ),
        "cable_summary_by_type": text(
            """
            WITH ar AS (SELECT expansion_analysis_run_id, data_source FROM surrogrid.expansion_analysis_run WHERE analysis_key = :analysis_key), rows AS (
                SELECT visible_std_type AS cable_type, requires_expansion, length_km,
                       reinforcement_150_count, reinforcement_185_count,
                       reinforcement_240_count, reinforcement_added_capacity_ka,
                       estimated_cost_eur, loading_percent
                FROM surrogrid.expansion_line_result WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar) AND (SELECT data_source FROM ar) = 'Synthetic'
                UNION ALL
                SELECT std_type, requires_expansion, length_km,
                       reinforcement_150_count, reinforcement_185_count,
                       reinforcement_240_count, reinforcement_added_capacity_ka,
                       estimated_cost_eur, loading_percent
                FROM surrogrid.expansion_real_line_result WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar) AND (SELECT data_source FROM ar) <> 'Synthetic'
            )
            SELECT cable_type AS visible_std_type, COUNT(*) AS cable_segments,
                   COUNT(*) FILTER (WHERE requires_expansion) AS cable_segments_requiring_expansion,
                   COALESCE(SUM(length_km), 0.0) AS length_km,
                   COALESCE(SUM(reinforcement_150_count), 0) AS reinforcement_150_count,
                   COALESCE(SUM(reinforcement_185_count), 0) AS reinforcement_185_count,
                   COALESCE(SUM(reinforcement_240_count), 0) AS reinforcement_240_count,
                   COALESCE(SUM(reinforcement_added_capacity_ka), 0.0) AS reinforcement_added_capacity_ka,
                   COALESCE(SUM(estimated_cost_eur), 0.0) AS estimated_cost_eur,
                   MAX(loading_percent) AS max_loading_percent
            FROM rows GROUP BY cable_type
            ORDER BY estimated_cost_eur DESC, cable_segments_requiring_expansion DESC, cable_type
            """
        ),
        "transformer_summary_by_basis": text(
            """
            WITH ar AS (SELECT expansion_analysis_run_id, data_source FROM surrogrid.expansion_analysis_run WHERE analysis_key = :analysis_key), rows AS (
                SELECT transformer_cost_basis, requires_expansion, additional_transformer_kva, estimated_cost_eur, loading_percent
                FROM surrogrid.expansion_transformer_result WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar) AND (SELECT data_source FROM ar) = 'Synthetic'
                UNION ALL
                SELECT transformer_cost_basis, requires_expansion, additional_transformer_kva, estimated_cost_eur, loading_percent
                FROM surrogrid.expansion_real_transformer_result WHERE expansion_analysis_run_id = (SELECT expansion_analysis_run_id FROM ar) AND (SELECT data_source FROM ar) <> 'Synthetic'
            )
            SELECT transformer_cost_basis, COUNT(*) AS transformers,
                   COUNT(*) FILTER (WHERE requires_expansion) AS transformers_requiring_expansion,
                   COALESCE(SUM(additional_transformer_kva), 0.0) AS additional_transformer_kva,
                   COALESCE(SUM(estimated_cost_eur), 0.0) AS estimated_cost_eur,
                   MAX(loading_percent) AS max_loading_percent
            FROM rows GROUP BY transformer_cost_basis
            ORDER BY estimated_cost_eur DESC, transformers_requiring_expansion DESC, transformer_cost_basis
            """
        ),
        "real_grid_status": text(
            """
            SELECT s.* FROM surrogrid.expansion_real_grid_status s
            JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
            WHERE ar.analysis_key = :analysis_key
            ORDER BY s.cost_status, LENGTH(s.lv_id), s.lv_id
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


# Indicators per analysis (after Lee 2026, Sections 4.1.3-4.1.5) ------------------------------------------------------

# Every indicator is a count or a sum, so the PLZ rows add up to the analysis total.
INDICATOR_QUERY = text(
    """
    WITH ar AS (
        SELECT expansion_analysis_run_id AS id FROM surrogrid.expansion_analysis_run
        WHERE analysis_key = :analysis_key
    ), grids AS (
        SELECT plz,
               COUNT(*) AS grids,
               COUNT(*) FILTER (WHERE total_cost_eur > 0) AS grids_with_measures,
               COUNT(*) FILTER (WHERE routes_reinforced > 0) AS grids_with_cable_measures,
               COUNT(*) FILTER (WHERE station_measure IN ('exchange', 'over_limit', 'transfer', 'new_station'))
                   AS grids_with_station_measures,
               COUNT(*) FILTER (WHERE voltage_measure IN ('ront', 'split')) AS grids_with_voltage_measures,
               COUNT(*) FILTER (WHERE station_measure = 'transfer') AS load_transfers,
               COALESCE(SUM(new_stations), 0.0) AS new_stations,
               COALESCE(SUM(routes_reinforced), 0) AS routes_reinforced,
               COALESCE(SUM(unresolved_routes), 0) AS unresolved_routes,
               COALESCE(SUM(cable_cost_eur), 0.0) AS cable_cost_eur,
               COALESCE(SUM(transformer_exchange_cost_eur), 0.0) AS transformer_exchange_cost_eur,
               COALESCE(SUM(load_transfer_cost_eur), 0.0) AS load_transfer_cost_eur,
               COALESCE(SUM(new_station_cost_eur), 0.0) AS new_station_cost_eur,
               COALESCE(SUM(voltage_cost_eur), 0.0) AS voltage_cost_eur,
               COALESCE(SUM(total_cost_eur), 0.0) AS total_cost_eur,
               COALESCE(SUM(service_cost_eur), 0.0) AS service_cost_eur
        FROM surrogrid.expansion_grid_result
        WHERE expansion_analysis_run_id = (SELECT id FROM ar)
        GROUP BY plz
    ), lines AS (
        SELECT plz, length_km, additional_parallel, reinforcement_150_count, reinforcement_185_count,
               reinforcement_240_count, measure
        FROM surrogrid.expansion_line_result WHERE expansion_analysis_run_id = (SELECT id FROM ar)
        UNION ALL
        SELECT plz, length_km, additional_parallel, reinforcement_150_count, reinforcement_185_count,
               reinforcement_240_count, measure
        FROM surrogrid.expansion_real_line_result WHERE expansion_analysis_run_id = (SELECT id FROM ar)
    ), line_parts AS (
        SELECT plz,
               COALESCE(SUM(length_km * additional_parallel) FILTER (WHERE measure IN ('local', 'outlet')), 0.0)
                   AS added_cable_km,
               COALESCE(SUM(length_km * reinforcement_150_count) FILTER (WHERE measure IN ('local', 'outlet')), 0.0)
                   AS added_cable_km_150,
               COALESCE(SUM(length_km * reinforcement_185_count) FILTER (WHERE measure IN ('local', 'outlet')), 0.0)
                   AS added_cable_km_185,
               COALESCE(SUM(length_km * reinforcement_240_count) FILTER (WHERE measure IN ('local', 'outlet')), 0.0)
                   AS added_cable_km_240,
               COALESCE(SUM(length_km * additional_parallel) FILTER (WHERE measure = 'service'), 0.0)
                   AS service_cable_km
        FROM lines GROUP BY plz
    ), trafos AS (
        SELECT plz, transformer_rated_power_kva, required_transformer_kva, transformer_cost_basis
        FROM surrogrid.expansion_transformer_result WHERE expansion_analysis_run_id = (SELECT id FROM ar)
        UNION ALL
        SELECT plz, transformer_rated_power_kva, required_transformer_kva, transformer_cost_basis
        FROM surrogrid.expansion_real_transformer_result WHERE expansion_analysis_run_id = (SELECT id FROM ar)
    ), trafo_parts AS (
        SELECT plz,
               COALESCE(SUM(transformer_rated_power_kva), 0.0) AS transformer_kva_before,
               COUNT(*) FILTER (WHERE required_transformer_kva > transformer_rated_power_kva) AS transformer_exchanges,
               COALESCE(SUM(GREATEST(required_transformer_kva - transformer_rated_power_kva, 0.0)), 0.0)
                   AS transformer_kva_added,
               COUNT(*) FILTER (WHERE transformer_cost_basis LIKE '%station_replaced%') AS station_replacements
        FROM trafos GROUP BY plz
    )
    SELECT g.plz,
           g.grids, g.grids_with_measures, g.grids_with_cable_measures, g.grids_with_station_measures,
           g.grids_with_voltage_measures,
           g.routes_reinforced, g.unresolved_routes,
           COALESCE(l.added_cable_km, 0.0) AS added_cable_km,
           COALESCE(l.added_cable_km_150, 0.0) AS added_cable_km_150,
           COALESCE(l.added_cable_km_185, 0.0) AS added_cable_km_185,
           COALESCE(l.added_cable_km_240, 0.0) AS added_cable_km_240,
           COALESCE(t.transformer_kva_before, 0.0) AS transformer_kva_before,
           COALESCE(t.transformer_exchanges, 0) AS transformer_exchanges,
           COALESCE(t.transformer_kva_added, 0.0) AS transformer_kva_added,
           COALESCE(t.station_replacements, 0) AS station_replacements,
           g.load_transfers, g.new_stations,
           g.cable_cost_eur, g.transformer_exchange_cost_eur, g.load_transfer_cost_eur, g.new_station_cost_eur,
           g.voltage_cost_eur, g.total_cost_eur,
           g.service_cost_eur, COALESCE(l.service_cable_km, 0.0) AS service_cable_km
    FROM grids g
    LEFT JOIN line_parts l ON l.plz IS NOT DISTINCT FROM g.plz
    LEFT JOIN trafo_parts t ON t.plz IS NOT DISTINCT FROM g.plz
    ORDER BY g.plz
    """
)

POPULATION_QUERY = text(
    """
    SELECT plz, MAX(pop) AS population
    FROM pylovo.municipal_register
    WHERE plz = ANY(:plz)
    GROUP BY plz
    """
)


def expansion_indicators(
    analysis_key: str | None = None,
    *,
    by_plz: bool = False,
    db: SurroGridDatabase | None = None,
) -> pd.DataFrame:
    """Reinforcement need, measures and cost of one analysis, as a total row or one row per PLZ.

    The indicators follow Lee (2026): grids with measures (any, cable, station, voltage), reinforced and
    unresolved routes, added cable km (per cross-section), transformer exchanges and added kVA, station
    replacements, load transfers, new substations and the cost breakdown. Service lines are reported
    separately, as in the total (decision D9). Per PLZ, ``population`` comes from ``pylovo.municipal_register``
    and ``cost_per_inhabitant_eur`` assumes that the analysis covers every grid of the PLZ.
    """
    db = db or SurroGridDatabase()
    analysis_key = _resolve_analysis_key(db, analysis_key)
    with db.engine.connect() as conn:
        frame = pd.read_sql(INDICATOR_QUERY, conn, params={"analysis_key": analysis_key})
        if frame.empty:
            raise ValueError(f"Expansion analysis {analysis_key!r} has no grid results.")
        if by_plz:
            plz = [int(value) for value in frame["plz"].dropna()]
            population = pd.read_sql(POPULATION_QUERY, conn, params={"plz": plz})
    if not by_plz:
        values = frame.drop(columns="plz")
        total = values.sum().to_frame().T.astype(values.dtypes.to_dict())
        total.insert(0, "analysis_key", analysis_key)
        return total
    frame = frame.merge(population, on="plz", how="left")
    frame["cost_per_inhabitant_eur"] = frame["total_cost_eur"] / frame["population"]
    frame.insert(0, "analysis_key", analysis_key)
    return frame
