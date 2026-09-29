-- Cable expansion rows of one synthetic analysis: component costs summed per visible pylovo
-- line; the most loaded component describes the line (ties: lowest pandapower line index).
-- The decision per component comes from the rules in Python (temp table
-- expansion_component_selection, written by synthetic_materialization).
WITH component_loading AS (
    SELECT
        ecl.powerflow_run_id,
        ecl.grid_case_id,
        ecl.scenario_id,
        ecl.ags,
        ecl.plz,
        ecl.kcid,
        ecl.bcid,
        ecl.pylovo_grid_result_id,
        ecl.pylovo_version_id,
        ecl.settlement_type,
        ecl.visible_line_id,
        ecl.line AS component_line,
        ecl.component_line_name,
        ecl.component_std_type,
        ecl.component_length_km,
        COALESCE(ecl.component_parallel, 1) AS component_parallel,
        ecl.max_i_from_ka,
        NULLIF(ecl.max_i_ka, 0.0) AS max_i_ka,
        NULLIF(ecl.max_i_ka, 0.0) * COALESCE(ecl.component_parallel, 1)
            AS installed_capacity_ka,
        GREATEST(
            ecl.max_i_from_ka
                - NULLIF(ecl.max_i_ka, 0.0) * COALESCE(ecl.component_parallel, 1),
            0.0
        ) AS required_added_capacity_ka,
        ecl.critical_t_index,
        ecl.critical_ts,
        CASE
            WHEN ecl.max_i_ka IS NULL OR ecl.max_i_ka = 0.0 THEN NULL
            ELSE ecl.max_i_from_ka
                / (ecl.max_i_ka * COALESCE(ecl.component_parallel, 1))
                * 100.0
        END AS loading_percent
    FROM expansion_component_loading ecl
    WHERE ecl.visible_line_id IS NOT NULL
),
component_cost AS (
    SELECT
        cl.*,
        sel.required_parallel,
        sel.additional_parallel,
        sel.reinforcement_150_count,
        sel.reinforcement_185_count,
        sel.reinforcement_240_count,
        sel.reinforcement_added_capacity_ka,
        sel.reinforcement_catalog,
        sel.line_cost_eur_per_km,
        sel.line_cost_basis,
        sel.duct_cost_eur_per_km,
        sel.reopen_cost_eur_per_km,
        sel.existing_duct_share,
        sel.trenching_share,
        sel.estimated_component_cost_eur,
        sel.measure,
        sel.is_station_outlet,
        sel.is_service_line,
        sel.route_cable_count,
        sel.service_cost_eur
    FROM component_loading cl
    JOIN expansion_component_selection sel
      ON sel.powerflow_run_id = cl.powerflow_run_id
     AND sel.component_line = cl.component_line
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
        SUM(reinforcement_150_count)::INTEGER AS reinforcement_150_count,
        SUM(reinforcement_185_count)::INTEGER AS reinforcement_185_count,
        SUM(reinforcement_240_count)::INTEGER AS reinforcement_240_count,
        SUM(reinforcement_added_capacity_ka) AS reinforcement_added_capacity_ka,
        BOOL_OR(additional_parallel > 0) AS requires_expansion,
        BOOL_OR(COALESCE(loading_percent, 0.0) > 100.0) AS overloaded_at_100_percent,
        COALESCE(SUM(estimated_component_cost_eur), 0.0) AS estimated_cost_eur,
        COALESCE(SUM(service_cost_eur), 0.0) AS service_cost_eur,
        BOOL_OR(is_station_outlet) AS is_station_outlet,
        COUNT(DISTINCT line_cost_basis) AS component_cost_basis_count,
        COUNT(DISTINCT component_std_type) AS component_std_type_count
    FROM component_cost
    GROUP BY powerflow_run_id, visible_line_id
),
critical_component AS (
    SELECT DISTINCT ON (powerflow_run_id, visible_line_id)
        *
    FROM component_cost
    ORDER BY powerflow_run_id, visible_line_id, loading_percent DESC NULLS LAST, component_line
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
    settlement_type,
    line_existing_duct_share,
    line_trenching_share,
    critical_component_parallel,
    max_component_line,
    max_component_line_name,
    max_i_from_ka,
    max_i_ka,
    loading_percent,
    required_parallel,
    additional_parallel,
    reinforcement_150_count,
    reinforcement_185_count,
    reinforcement_240_count,
    reinforcement_added_capacity_ka,
    reinforcement_catalog,
    requires_expansion,
    overloaded_at_100_percent,
    estimated_cost_eur,
    critical_component_cost_eur_per_km,
    critical_component_cost_basis,
    critical_component_duct_cost_eur_per_km,
    critical_component_reopen_cost_eur_per_km,
    critical_t_index,
    critical_ts,
    mapped_component_lines,
    component_cost_basis_count,
    component_std_type_count,
    measure,
    is_station_outlet,
    is_service_line,
    route_cable_count,
    service_cost_eur
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
    cc.settlement_type,
    cc.existing_duct_share,
    cc.trenching_share,
    cc.component_parallel,
    cc.component_line,
    cc.component_line_name,
    cc.max_i_from_ka,
    cc.max_i_ka,
    cc.loading_percent,
    va.required_parallel,
    va.additional_parallel,
    va.reinforcement_150_count,
    va.reinforcement_185_count,
    va.reinforcement_240_count,
    va.reinforcement_added_capacity_ka,
    cc.reinforcement_catalog,
    va.requires_expansion,
    va.overloaded_at_100_percent,
    va.estimated_cost_eur,
    cc.line_cost_eur_per_km,
    cc.line_cost_basis,
    cc.duct_cost_eur_per_km,
    cc.reopen_cost_eur_per_km,
    cc.critical_t_index,
    cc.critical_ts,
    vc.mapped_component_lines,
    va.component_cost_basis_count,
    va.component_std_type_count,
    cc.measure,
    va.is_station_outlet,
    cc.is_service_line,
    cc.route_cable_count,
    va.service_cost_eur
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
