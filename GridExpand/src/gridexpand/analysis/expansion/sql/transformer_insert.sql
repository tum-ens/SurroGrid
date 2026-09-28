-- Transformer expansion rows of one synthetic analysis (one per selected power-flow run).
-- Rating: the station rating of the pylovo grid (grid_result.transformer_rated_power). The equipment
-- columns of transformer_positions_with_grid describe one unit (an 800 kVA station has two 400 kVA units).
WITH assumption AS (
    SELECT *
    FROM surrogrid.expansion_cost_assumption
    WHERE assumption_key = :assumption_key
),
peak_import AS (
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
),
transformer_peak AS (
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
),
estimated AS (
/*TRANSFORMER_COST*/
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
    estimated.estimated_cost_eur,
    estimated.transformer_cost_basis,
    estimated.critical_t_index,
    estimated.critical_ts
FROM estimated
