"""Materialize DB-backed grid expansion heuristics for QGIS.

The heavy hourly power-flow tables stay inside PostgreSQL. This script reduces
them to peak loading and expansion-cost estimates per visible pylovo cable and
per transformer position.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text


GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
SCHEMA_SQL_PATH = Path(__file__).with_name("schema.sql")
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase, normalize_ags


def _execute_sql_file(db: SurroGridDatabase, path: Path) -> None:
    statements = [
        statement.strip()
        for statement in path.read_text(encoding="utf-8").split(";")
        if statement.strip()
    ]
    with db.engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def _refresh_qgis_materialized_views(db: SurroGridDatabase) -> None:
    with db.engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW surrogrid.expansion_line_qgis_mv"))
        conn.execute(text("REFRESH MATERIALIZED VIEW surrogrid.expansion_transformer_qgis_mv"))


def _optional_ags(value: str | int | None) -> int | None:
    if value is None:
        return None
    return normalize_ags(value)


def _analysis_key(args: argparse.Namespace) -> str:
    if args.analysis_key:
        return args.analysis_key
    scope = str(args.ags).zfill(8) if args.ags is not None else "all"
    run = args.run_name.replace("baseline_static_", "").replace("_powerflow", "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scope}_{run}_{args.stage}_{stamp}"


def _create_analysis_run(
    db: SurroGridDatabase,
    *,
    analysis_key: str,
    args: argparse.Namespace,
) -> int:
    if args.replace:
        with db.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM surrogrid.expansion_analysis_run
                    WHERE analysis_key = :analysis_key
                    """
                ),
                {"analysis_key": analysis_key},
            )

    query = text(
        """
        INSERT INTO surrogrid.expansion_analysis_run (
            analysis_key, assumption_key, run_name, stage,
            scenario_id, ags, plz, note
        )
        VALUES (
            :analysis_key, :assumption_key, :run_name, :stage,
            :scenario_id, :ags, :plz, :note
        )
        RETURNING expansion_analysis_run_id
        """
    )
    try:
        with db.engine.begin() as conn:
            return int(
                conn.execute(
                    query,
                    {
                        "analysis_key": analysis_key,
                        "assumption_key": args.assumption_key,
                        "run_name": args.run_name,
                        "stage": args.stage,
                        "scenario_id": args.scenario_id,
                        "ags": _optional_ags(args.ags),
                        "plz": args.plz,
                        "note": args.note,
                    },
                ).scalar_one()
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not create analysis_key={analysis_key!r}. "
            "Use --replace to overwrite an existing analysis."
        ) from exc


def _audit_unmapped_line_components(db: SurroGridDatabase, *, args: argparse.Namespace) -> None:
    """Report active electrical line components that cannot be attached to QGIS geometries.

    Root-adjacent connector lines can be legitimate non-asset artefacts, but
    overloaded unmapped components would hide expansion needs. Treat those as a
    hard failure before materializing costs.
    """
    query = text(
        """
        WITH selected_runs AS (
            SELECT
                pr.powerflow_run_id,
                pr.grid_case_id,
                gc.ags,
                gc.plz,
                gc.kcid,
                gc.bcid,
                gc.pylovo_grid_result_id,
                gc.pylovo_version_id
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:plz IS NULL OR gc.plz = :plz)
        ),
        pp_source AS (
            SELECT
                sr.*,
                pl.pp_index AS line,
                pl.name AS component_line_name,
                pl.length_km AS component_length_km,
                pl.max_i_ka,
                pl.parallel AS component_parallel,
                pl.from_bus,
                pl.to_bus,
                regexp_replace(pl.name, '^Line to ', 'L') AS pylovo_line_name
            FROM selected_runs sr
            JOIN pylovo.pandapower_line pl
              ON pl.grid_result_id = sr.pylovo_grid_result_id
        ),
        source_lines AS (
            SELECT
                pp.*,
                lr.geom AS source_geom,
                lr.line_name AS source_line_name
            FROM pp_source pp
            LEFT JOIN pylovo.lines_result lr
              ON lr.grid_result_id = pp.pylovo_grid_result_id
             AND lr.line_name = pp.pylovo_line_name
        ),
        visible_map AS (
            SELECT
                src.*,
                COALESCE(direct.id, spatial.id) AS visible_line_id
            FROM source_lines src
            LEFT JOIN LATERAL (
                SELECT v.id
                FROM pylovo.lines_result_view v
                WHERE v.grid_result_id = src.pylovo_grid_result_id
                  AND v.version_id = src.pylovo_version_id
                  AND v.plz = src.plz
                  AND v.kcid = src.kcid
                  AND v.bcid = src.bcid
                  AND v.line_name = src.source_line_name
                LIMIT 1
            ) direct ON TRUE
            LEFT JOIN LATERAL (
                SELECT v.id
                FROM pylovo.lines_result_view v
                WHERE direct.id IS NULL
                  AND src.source_geom IS NOT NULL
                  AND v.grid_result_id = src.pylovo_grid_result_id
                  AND v.version_id = src.pylovo_version_id
                  AND v.plz = src.plz
                  AND v.kcid = src.kcid
                  AND v.bcid = src.bcid
                  AND v.line_name <> src.source_line_name
                  AND ST_DWithin(v.geom, src.source_geom, 0.05)
                ORDER BY
                    ST_Length(ST_Intersection(v.geom, src.source_geom)) DESC,
                    ST_Distance(v.geom, src.source_geom) ASC
                LIMIT 1
            ) spatial ON direct.id IS NULL
        ),
        peak_line AS (
            SELECT DISTINCT ON (plr.powerflow_run_id, plr.line)
                plr.powerflow_run_id,
                plr.line,
                ABS(plr.i_from_ka) AS max_i_from_ka
            FROM surrogrid.powerflow_line_result plr
            JOIN selected_runs sr USING (powerflow_run_id)
            WHERE plr.stage = :stage
            ORDER BY plr.powerflow_run_id, plr.line, ABS(plr.i_from_ka) DESC
        ),
        active_components AS (
            SELECT
                vm.*,
                peak.max_i_from_ka,
                CASE
                    WHEN vm.max_i_ka IS NULL OR vm.max_i_ka = 0.0 THEN NULL
                    ELSE peak.max_i_from_ka
                        / (vm.max_i_ka * COALESCE(vm.component_parallel, 1))
                        * 100.0
                END AS loading_percent
            FROM visible_map vm
            JOIN peak_line peak
              ON peak.powerflow_run_id = vm.powerflow_run_id
             AND peak.line = vm.line
        ),
        unmapped AS (
            SELECT *
            FROM active_components
            WHERE visible_line_id IS NULL
        )
        SELECT
            (SELECT COUNT(*) FROM selected_runs) AS selected_runs,
            (SELECT COUNT(*) FROM active_components) AS active_components,
            (SELECT COUNT(*) FROM unmapped) AS unmapped_components,
            COUNT(*) FILTER (WHERE COALESCE(loading_percent, 0.0) > 100.0) AS overloaded_unmapped_components,
            COALESCE(MAX(loading_percent), 0.0) AS max_unmapped_loading_percent,
            COUNT(*) FILTER (
                WHERE source_line_name IS NULL
                  AND source_geom IS NULL
                  AND component_length_km <= 0.005
            ) AS root_connector_like_components
        FROM unmapped
        """
    )
    params = {
        "run_name": args.run_name,
        "stage": args.stage,
        "scenario_id": args.scenario_id,
        "ags": _optional_ags(args.ags),
        "plz": args.plz,
    }
    with db.engine.connect() as conn:
        row = conn.execute(query, params).mappings().one()

    selected_runs = int(row["selected_runs"] or 0)
    active_components = int(row["active_components"] or 0)
    unmapped = int(row["unmapped_components"] or 0)
    overloaded = int(row["overloaded_unmapped_components"] or 0)
    root_like = int(row["root_connector_like_components"] or 0)
    max_loading = float(row["max_unmapped_loading_percent"] or 0.0)

    if selected_runs == 0:
        raise RuntimeError("No power-flow runs match the requested expansion scope.")
    if active_components == 0:
        raise RuntimeError(
            "No raw power-flow line-result rows match the requested expansion scope. "
            "Run Step 4 with raw DB storage before materializing expansion costs."
        )
    if overloaded > 0:
        raise RuntimeError(
            f"Found {overloaded} overloaded line component(s) without a visible pylovo geometry "
            f"(max unmapped loading {max_loading:.2f}%). Refusing to hide expansion needs."
        )
    if unmapped > 0:
        print(
            "Warning: ignored "
            f"{unmapped} active unmapped line component(s) "
            f"({root_like} root-connector-like; max loading {max_loading:.2f}%)."
        )


def _materialize_line_results(
    db: SurroGridDatabase,
    *,
    expansion_analysis_run_id: int,
    args: argparse.Namespace,
) -> int:
    query = text(
        """
        WITH assumption AS (
            SELECT *
            FROM surrogrid.expansion_cost_assumption
            WHERE assumption_key = :assumption_key
        ),
        selected_runs AS (
            SELECT
                pr.powerflow_run_id,
                pr.grid_case_id,
                pr.scenario_id,
                gc.ags,
                gc.plz,
                gc.kcid,
                gc.bcid,
                gc.pylovo_grid_result_id,
                gc.pylovo_version_id
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:plz IS NULL OR gc.plz = :plz)
        ),
        pp_source AS (
            SELECT
                sr.*,
                pl.pp_index AS line,
                pl.name AS component_line_name,
                pl.std_type AS component_std_type,
                pl.length_km AS component_length_km,
                pl.max_i_ka,
                pl.parallel AS component_parallel,
                regexp_replace(pl.name, '^Line to ', 'L') AS pylovo_line_name
            FROM selected_runs sr
            JOIN pylovo.pandapower_line pl
              ON pl.grid_result_id = sr.pylovo_grid_result_id
        ),
        source_lines AS (
            SELECT
                pp.*,
                lr.lines_result_id,
                lr.geom AS source_geom,
                lr.line_name AS source_line_name
            FROM pp_source pp
            LEFT JOIN pylovo.lines_result lr
              ON lr.grid_result_id = pp.pylovo_grid_result_id
             AND lr.line_name = pp.pylovo_line_name
        ),
        visible_map AS (
            SELECT
                src.*,
                COALESCE(direct.id, spatial.id) AS visible_line_id
            FROM source_lines src
            LEFT JOIN LATERAL (
                SELECT v.id
                FROM pylovo.lines_result_view v
                WHERE v.grid_result_id = src.pylovo_grid_result_id
                  AND v.version_id = src.pylovo_version_id
                  AND v.plz = src.plz
                  AND v.kcid = src.kcid
                  AND v.bcid = src.bcid
                  AND v.line_name = src.source_line_name
                LIMIT 1
            ) direct ON TRUE
            LEFT JOIN LATERAL (
                SELECT v.id
                FROM pylovo.lines_result_view v
                WHERE direct.id IS NULL
                  AND src.source_geom IS NOT NULL
                  AND v.grid_result_id = src.pylovo_grid_result_id
                  AND v.version_id = src.pylovo_version_id
                  AND v.plz = src.plz
                  AND v.kcid = src.kcid
                  AND v.bcid = src.bcid
                  AND v.line_name <> src.source_line_name
                  AND ST_DWithin(v.geom, src.source_geom, 0.05)
                ORDER BY
                    ST_Length(ST_Intersection(v.geom, src.source_geom)) DESC,
                    ST_Distance(v.geom, src.source_geom) ASC
                LIMIT 1
            ) spatial ON direct.id IS NULL
        ),
        peak_line AS (
            SELECT DISTINCT ON (plr.powerflow_run_id, plr.line)
                plr.powerflow_run_id,
                plr.line,
                ABS(plr.i_from_ka) AS max_i_from_ka,
                plr.t_index AS critical_t_index,
                plr.ts AS critical_ts
            FROM surrogrid.powerflow_line_result plr
            JOIN selected_runs sr USING (powerflow_run_id)
            WHERE plr.stage = :stage
            ORDER BY plr.powerflow_run_id, plr.line, ABS(plr.i_from_ka) DESC
        ),
        component_loading AS (
            SELECT
                vm.powerflow_run_id,
                vm.grid_case_id,
                vm.scenario_id,
                vm.ags,
                vm.plz,
                vm.kcid,
                vm.bcid,
                vm.pylovo_grid_result_id,
                vm.pylovo_version_id,
                vm.visible_line_id,
                vm.line AS component_line,
                vm.component_line_name,
                vm.component_std_type,
                vm.component_length_km,
                COALESCE(vm.component_parallel, 1) AS component_parallel,
                peak.max_i_from_ka,
                NULLIF(vm.max_i_ka, 0.0) AS max_i_ka,
                peak.critical_t_index,
                peak.critical_ts,
                CASE
                    WHEN vm.max_i_ka IS NULL OR vm.max_i_ka = 0.0 THEN NULL
                    ELSE peak.max_i_from_ka
                        / (vm.max_i_ka * COALESCE(vm.component_parallel, 1))
                        * 100.0
                END AS loading_percent,
                CASE
                    WHEN vm.max_i_ka IS NULL OR vm.max_i_ka = 0.0 THEN COALESCE(vm.component_parallel, 1)
                    ELSE GREATEST(
                        CEIL(
                            peak.max_i_from_ka
                            / NULLIF(vm.max_i_ka, 0.0)
                        )::INTEGER,
                        COALESCE(vm.component_parallel, 1)
                    )
                END AS required_parallel
            FROM visible_map vm
            JOIN peak_line peak
              ON peak.powerflow_run_id = vm.powerflow_run_id
             AND peak.line = vm.line
            WHERE vm.visible_line_id IS NOT NULL
        ),
        component_cost AS (
            SELECT
                cl.*,
                GREATEST(cl.required_parallel - cl.component_parallel, 0) AS additional_parallel,
                line_cost.line_cost_eur_per_km,
                line_cost.line_cost_basis,
                GREATEST(cl.required_parallel - cl.component_parallel, 0)
                    * COALESCE(cl.component_length_km, 0.0)
                    * line_cost.line_cost_eur_per_km AS estimated_component_cost_eur
            FROM component_loading cl
            CROSS JOIN assumption
            CROSS JOIN LATERAL (
                SELECT
                    CASE
                        WHEN cl.component_std_type ~ '240' THEN assumption.line_parallel_240_eur_per_km
                        WHEN cl.component_std_type ~ '185' THEN assumption.line_parallel_185_eur_per_km
                        WHEN cl.component_std_type ~ '150' THEN assumption.line_parallel_150_eur_per_km
                        WHEN cl.component_std_type ~ '120|95|70|50|35' THEN assumption.line_parallel_150_eur_per_km
                        ELSE assumption.line_parallel_default_eur_per_km
                    END AS line_cost_eur_per_km,
                    CASE
                        WHEN cl.component_std_type ~ '240' THEN 'parallel_existing_route_240mm2'
                        WHEN cl.component_std_type ~ '185' THEN 'parallel_existing_route_185mm2'
                        WHEN cl.component_std_type ~ '150' THEN 'parallel_existing_route_150mm2'
                        WHEN cl.component_std_type ~ '120|95|70|50|35' THEN 'parallel_existing_route_le_150mm2'
                        ELSE 'parallel_existing_route_default_240mm2'
                    END AS line_cost_basis
            ) line_cost
        ),
        visible_counts AS (
            SELECT powerflow_run_id, visible_line_id, COUNT(*) AS mapped_component_lines
            FROM component_cost
            GROUP BY powerflow_run_id, visible_line_id
        ),
        visible_aggregate AS (
            SELECT
                powerflow_run_id,
                visible_line_id,
                MAX(required_parallel) AS required_parallel,
                SUM(additional_parallel)::INTEGER AS additional_parallel,
                BOOL_OR(additional_parallel > 0) AS requires_expansion,
                BOOL_OR(COALESCE(loading_percent, 0.0) > 100.0) AS overloaded_at_100_percent,
                COALESCE(SUM(estimated_component_cost_eur), 0.0) AS estimated_cost_eur,
                COUNT(DISTINCT line_cost_basis) AS component_cost_basis_count,
                COUNT(DISTINCT component_std_type) AS component_std_type_count
            FROM component_cost
            GROUP BY powerflow_run_id, visible_line_id
        ),
        critical_component AS (
            SELECT DISTINCT ON (powerflow_run_id, visible_line_id)
                *
            FROM component_cost
            ORDER BY powerflow_run_id, visible_line_id, loading_percent DESC NULLS LAST
        )
        INSERT INTO surrogrid.expansion_line_result (
            expansion_analysis_run_id,
            powerflow_run_id,
            grid_case_id,
            scenario_id,
            ags,
            plz,
            kcid,
            bcid,
            pylovo_grid_result_id,
            pylovo_version_id,
            visible_line_id,
            visible_line_name,
            visible_std_type,
            is_helper,
            helper_type,
            from_bus,
            to_bus,
            length_km,
            critical_component_parallel,
            max_component_line,
            max_component_line_name,
            max_i_from_ka,
            max_i_ka,
            loading_percent,
            required_parallel,
            additional_parallel,
            requires_expansion,
            overloaded_at_100_percent,
            estimated_cost_eur,
            critical_component_cost_eur_per_km,
            critical_component_cost_basis,
            critical_t_index,
            critical_ts,
            mapped_component_lines,
            component_cost_basis_count,
            component_std_type_count
        )
        SELECT
            :expansion_analysis_run_id,
            cc.powerflow_run_id,
            cc.grid_case_id,
            cc.scenario_id,
            cc.ags,
            cc.plz,
            cc.kcid,
            cc.bcid,
            cc.pylovo_grid_result_id,
            cc.pylovo_version_id,
            lv.id,
            lv.line_name,
            lv.std_type,
            lv.is_helper,
            lv.helper_type,
            lv.from_bus,
            lv.to_bus,
            lv.length_km,
            cc.component_parallel,
            cc.component_line,
            cc.component_line_name,
            cc.max_i_from_ka,
            cc.max_i_ka,
            cc.loading_percent,
            va.required_parallel,
            va.additional_parallel,
            va.requires_expansion,
            va.overloaded_at_100_percent,
            va.estimated_cost_eur,
            cc.line_cost_eur_per_km,
            cc.line_cost_basis,
            cc.critical_t_index,
            cc.critical_ts,
            vc.mapped_component_lines,
            va.component_cost_basis_count,
            va.component_std_type_count
        FROM critical_component cc
        JOIN visible_aggregate va
          ON va.powerflow_run_id = cc.powerflow_run_id
         AND va.visible_line_id = cc.visible_line_id
        JOIN visible_counts vc
          ON vc.powerflow_run_id = cc.powerflow_run_id
         AND vc.visible_line_id = cc.visible_line_id
        JOIN pylovo.lines_result_view lv
          ON lv.grid_result_id = cc.pylovo_grid_result_id
         AND lv.version_id = cc.pylovo_version_id
         AND lv.plz = cc.plz
         AND lv.kcid = cc.kcid
         AND lv.bcid = cc.bcid
         AND lv.id = cc.visible_line_id
        """
    )
    with db.engine.begin() as conn:
        result = conn.execute(
            query,
            {
                "expansion_analysis_run_id": expansion_analysis_run_id,
                "assumption_key": args.assumption_key,
                "run_name": args.run_name,
                "stage": args.stage,
                "scenario_id": args.scenario_id,
                "ags": _optional_ags(args.ags),
                "plz": args.plz,
            },
        )
        return int(result.rowcount or 0)


def _materialize_transformer_results(
    db: SurroGridDatabase,
    *,
    expansion_analysis_run_id: int,
    args: argparse.Namespace,
) -> int:
    query = text(
        """
        WITH assumption AS (
            SELECT *
            FROM surrogrid.expansion_cost_assumption
            WHERE assumption_key = :assumption_key
        ),
        selected_runs AS (
            SELECT
                pr.powerflow_run_id,
                pr.grid_case_id,
                pr.scenario_id,
                gc.ags,
                gc.plz,
                gc.kcid,
                gc.bcid,
                gc.pylovo_grid_result_id,
                gc.pylovo_version_id
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:plz IS NULL OR gc.plz = :plz)
        ),
        peak_import AS (
            SELECT DISTINCT ON (pi.powerflow_run_id)
                pi.powerflow_run_id,
                pi.ts AS critical_ts,
                pi.t_index AS critical_t_index,
                pi.p_mw,
                pi.q_mvar,
                SQRT(POWER(pi.p_mw, 2) + POWER(pi.q_mvar, 2)) AS s_mva
            FROM surrogrid.powerflow_import pi
            JOIN selected_runs sr USING (powerflow_run_id)
            WHERE pi.stage = :stage
            ORDER BY pi.powerflow_run_id, SQRT(POWER(pi.p_mw, 2) + POWER(pi.q_mvar, 2)) DESC
        ),
        transformer_base AS (
            SELECT
                sr.*,
                gr.transformer_equipment_name,
                COALESCE(tpwg.s_max_kva, gr.transformer_rated_power::DOUBLE PRECISION) AS rated_kva
            FROM selected_runs sr
            JOIN pylovo.grid_result gr
              ON gr.grid_result_id = sr.pylovo_grid_result_id
            LEFT JOIN pylovo.transformer_positions_with_grid tpwg
              ON tpwg.grid_result_id = sr.pylovo_grid_result_id
             AND tpwg.version_id = sr.pylovo_version_id
             AND tpwg.plz = sr.plz
             AND tpwg.kcid = sr.kcid
             AND tpwg.bcid = sr.bcid
        ),
        estimated AS (
            SELECT
                tb.*,
                peak.critical_ts,
                peak.critical_t_index,
                peak.p_mw,
                peak.q_mvar,
                peak.s_mva,
                peak.s_mva * 1000.0 / NULLIF(tb.rated_kva, 0.0) * 100.0 AS loading_percent,
                CEIL(
                    (peak.s_mva * 1000.0)
                    / assumption.transformer_capacity_step_kva
                ) * assumption.transformer_capacity_step_kva AS required_kva
            FROM transformer_base tb
            JOIN peak_import peak USING (powerflow_run_id)
            CROSS JOIN assumption
            WHERE tb.rated_kva IS NOT NULL
              AND tb.rated_kva > 0.0
        )
        INSERT INTO surrogrid.expansion_transformer_result (
            expansion_analysis_run_id,
            powerflow_run_id,
            grid_case_id,
            scenario_id,
            ags,
            plz,
            kcid,
            bcid,
            pylovo_grid_result_id,
            pylovo_version_id,
            transformer_rated_power_kva,
            transformer_equipment_name,
            max_s_mva,
            max_p_mw,
            max_q_mvar,
            loading_percent,
            required_transformer_kva,
            additional_transformer_kva,
            requires_expansion,
            overloaded_at_100_percent,
            estimated_cost_eur,
            transformer_cost_basis,
            critical_t_index,
            critical_ts
        )
        SELECT
            :expansion_analysis_run_id,
            estimated.powerflow_run_id,
            estimated.grid_case_id,
            estimated.scenario_id,
            estimated.ags,
            estimated.plz,
            estimated.kcid,
            estimated.bcid,
            estimated.pylovo_grid_result_id,
            estimated.pylovo_version_id,
            estimated.rated_kva,
            estimated.transformer_equipment_name,
            estimated.s_mva,
            estimated.p_mw,
            estimated.q_mvar,
            estimated.loading_percent,
            GREATEST(estimated.required_kva, estimated.rated_kva),
            GREATEST(estimated.required_kva - estimated.rated_kva, 0.0),
            estimated.required_kva > estimated.rated_kva,
            estimated.loading_percent > 100.0,
            CASE
                WHEN estimated.required_kva <= estimated.rated_kva THEN 0.0
                WHEN estimated.required_kva <= 400.0 THEN assumption.transformer_replace_400_eur
                WHEN estimated.required_kva <= 630.0 THEN assumption.transformer_replace_630_eur
                WHEN estimated.required_kva <= 800.0 THEN assumption.transformer_replace_800_eur
                WHEN estimated.required_kva <= 1000.0 THEN assumption.transformer_replace_1000_eur
                ELSE assumption.transformer_station_rebuild_boundary_eur
            END,
            CASE
                WHEN estimated.required_kva <= estimated.rated_kva THEN 'none_existing_capacity_sufficient'
                WHEN estimated.required_kva <= 400.0 THEN 'all_in_replacement_to_400kva'
                WHEN estimated.required_kva <= 630.0 THEN 'all_in_replacement_to_630kva'
                WHEN estimated.required_kva <= 800.0 THEN 'all_in_replacement_to_800kva'
                WHEN estimated.required_kva <= 1000.0 THEN 'all_in_replacement_to_1000kva'
                ELSE 'station_rebuild_boundary_case_gt_1000kva'
            END,
            estimated.critical_t_index,
            estimated.critical_ts
        FROM estimated
        CROSS JOIN assumption
        """
    )
    with db.engine.begin() as conn:
        result = conn.execute(
            query,
            {
                "expansion_analysis_run_id": expansion_analysis_run_id,
                "assumption_key": args.assumption_key,
                "run_name": args.run_name,
                "stage": args.stage,
                "scenario_id": args.scenario_id,
                "ags": _optional_ags(args.ags),
                "plz": args.plz,
            },
        )
        return int(result.rowcount or 0)


def _print_summary(db: SurroGridDatabase, analysis_key: str) -> None:
    query = text(
        """
        SELECT
            ar.analysis_key,
            COUNT(DISTINCT elr.grid_case_id) AS grids_with_line_rows,
            COUNT(*) FILTER (WHERE elr.requires_expansion) AS cable_expansion_segments,
            COALESCE(SUM(elr.estimated_cost_eur), 0.0) AS cable_cost_eur,
            (
                SELECT COUNT(*)
                FROM surrogrid.expansion_transformer_result etr
                WHERE etr.expansion_analysis_run_id = ar.expansion_analysis_run_id
                  AND etr.requires_expansion
            ) AS transformer_expansion_count,
            (
                SELECT COALESCE(SUM(etr.estimated_cost_eur), 0.0)
                FROM surrogrid.expansion_transformer_result etr
                WHERE etr.expansion_analysis_run_id = ar.expansion_analysis_run_id
            ) AS transformer_cost_eur
        FROM surrogrid.expansion_analysis_run ar
        LEFT JOIN surrogrid.expansion_line_result elr USING (expansion_analysis_run_id)
        WHERE ar.analysis_key = :analysis_key
        GROUP BY ar.expansion_analysis_run_id, ar.analysis_key
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(query, {"analysis_key": analysis_key}).mappings().one()
    total = float(row["cable_cost_eur"]) + float(row["transformer_cost_eur"])
    print(f"analysis_key: {row['analysis_key']}")
    print(f"grids_with_line_rows: {row['grids_with_line_rows']}")
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
        default="baseline_static_full_powerflow",
        help="Power-flow run name to analyze.",
    )
    parser.add_argument(
        "--stage",
        default="post",
        choices=("pre", "post"),
        help="Power-flow stage to analyze.",
    )
    parser.add_argument("--scenario-id", type=int, help="Optional scenario_id filter.")
    parser.add_argument("--ags", help="Optional AGS filter, for example 09162000 for Munich.")
    parser.add_argument("--plz", type=int, help="Optional PLZ filter.")
    parser.add_argument(
        "--assumption-key",
        default="de_lv_heuristic_2026",
        help="Cost/planning assumption row to use.",
    )
    parser.add_argument("--analysis-key", help="Readable key for this materialized result.")
    parser.add_argument("--note", default="", help="Free-text note stored with the analysis run.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete an existing analysis with the same key before materializing.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only create/update expansion tables and QGIS views.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    db = SurroGridDatabase()
    _execute_sql_file(db, SCHEMA_SQL_PATH)
    if args.schema_only:
        print("Expansion schema and QGIS views are ready.")
        return

    analysis_key = _analysis_key(args)
    _audit_unmapped_line_components(db, args=args)
    run_id = _create_analysis_run(db, analysis_key=analysis_key, args=args)
    line_rows = _materialize_line_results(db, expansion_analysis_run_id=run_id, args=args)
    transformer_rows = _materialize_transformer_results(
        db,
        expansion_analysis_run_id=run_id,
        args=args,
    )
    _refresh_qgis_materialized_views(db)
    print(f"line rows inserted: {line_rows}")
    print(f"transformer rows inserted: {transformer_rows}")
    print("QGIS materialized views refreshed.")
    _print_summary(db, analysis_key)


if __name__ == "__main__":
    main()
