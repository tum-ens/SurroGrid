-- Peak loading of every active pandapower line component of the selected runs, mapped to the
-- visible pylovo line (QGIS geometry): by name first, else by the best-overlapping line within
-- 5 cm. Unmapped components keep visible_line_id NULL (audited before costing).
CREATE TEMP TABLE expansion_component_loading ON COMMIT DROP AS
WITH pp_source AS MATERIALIZED (
    -- Materialized so the join on the derived pylovo line name can hash (E8).
    SELECT
        sr.*,
        pl.pp_index AS line,
        pl.name AS component_line_name,
        pl.std_type AS component_std_type,
        pl.length_km AS component_length_km,
        pl.max_i_ka,
        pl.parallel AS component_parallel,
        regexp_replace(pl.name, '^Line to ', 'L') AS pylovo_line_name
    FROM expansion_selected_run sr
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
    SELECT
        pcs.powerflow_run_id,
        pcs.cable AS line,
        pcs.cable_loading_max_time_percent
            / 100.0
            * pcs.cable_installed_capacity_ka AS max_i_from_ka,
        pcs.cable_loading_max_t_index AS critical_t_index,
        -- Same rule as the raw tables' ts (db.writers.RunTimestamps).
        sr.timeframe_start
            + pcs.cable_loading_max_t_index * INTERVAL '1 hour' AS critical_ts
    FROM surrogrid.powerflow_cable_summary pcs
    JOIN expansion_selected_run sr USING (powerflow_run_id)
    WHERE pcs.stage = :stage
)
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
    vm.settlement_type,
    vm.visible_line_id,
    vm.line,
    vm.component_line_name,
    vm.component_std_type,
    vm.component_length_km,
    vm.max_i_ka,
    vm.component_parallel,
    vm.source_line_name,
    vm.source_geom IS NULL AS source_geom_missing,
    peak.max_i_from_ka,
    peak.critical_t_index,
    peak.critical_ts
FROM visible_map vm
JOIN peak_line peak
  ON peak.powerflow_run_id = vm.powerflow_run_id
 AND peak.line = vm.line
