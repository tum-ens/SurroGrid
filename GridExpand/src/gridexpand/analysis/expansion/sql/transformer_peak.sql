-- Peak station load and rating of every selected synthetic run (temp table of the materialization
-- transaction; read by the rules in Python and by transformer_insert.sql).
-- Rating: the station rating of the pylovo grid (grid_result.transformer_rated_power). The equipment
-- columns of transformer_positions_with_grid describe one unit (an 800 kVA station has two 400 kVA units).
CREATE TEMP TABLE expansion_transformer_peak ON COMMIT DROP AS
WITH peak_import AS (
    SELECT
        pfs.powerflow_run_id,
        pfs.trafo_critical_ts AS critical_ts,
        pfs.trafo_critical_t_index AS critical_t_index,
        pfs.trafo_max_p_mw AS p_mw,
        pfs.trafo_max_q_mvar AS q_mvar,
        pfs.trafo_max_s_mva AS s_mva
    FROM surrogrid.powerflow_summary pfs
    JOIN expansion_selected_run sr USING (powerflow_run_id)
    WHERE pfs.stage = :stage
      AND pfs.trafo_max_s_mva IS NOT NULL
),
transformer_base AS (
    SELECT
        sr.powerflow_run_id,
        sr.grid_case_id,
        sr.scenario_id,
        sr.ags,
        sr.plz,
        sr.kcid,
        sr.bcid,
        sr.pylovo_grid_result_id,
        sr.pylovo_version_id,
        gr.transformer_equipment_name,
        gr.transformer_rated_power::DOUBLE PRECISION AS rated_kva
    FROM expansion_selected_run sr
    JOIN pylovo.grid_result gr
      ON gr.grid_result_id = sr.pylovo_grid_result_id
)
SELECT
    tb.*,
    peak.critical_ts,
    peak.critical_t_index,
    peak.p_mw,
    peak.q_mvar,
    peak.s_mva,
    peak.s_mva * 1000.0 / NULLIF(tb.rated_kva, 0.0) * 100.0 AS loading_percent
FROM transformer_base tb
JOIN peak_import peak USING (powerflow_run_id)
WHERE tb.rated_kva IS NOT NULL
  AND tb.rated_kva > 0.0
