-- Views over surrogrid and pylovo; re-runnable (applied by ensure_views() when a
-- view is missing and by `gridexpand db migrate --apply`).
--
-- They join grid_case to pylovo by pylovo_grid_result_id, so they are created
-- only while fk_grid_case_pylovo_grid_result exists and is validated: after a
-- pylovo re-generation stale ids would silently join the wrong grids (run
-- `gridexpand db relink-pylovo` first).

-- One row per physical building and grid case, with its pandapower bus.
-- The pandas twin is SurroGridDatabase.read_buildings; read_building_components
-- asserts that both agree. The loads are aggregated per building bus of this
-- grid only (LATERAL), not for every pylovo grid.
CREATE OR REPLACE VIEW surrogrid.grid_building_bus AS
SELECT
    gc.grid_case_id,
    gc.cell_id,
    gc.ags,
    gc.plz,
    gc.kcid,
    gc.bcid,
    gc.pylovo_grid_result_id,
    gc.pylovo_version_id,
    br.objectid,
    br.id AS building_id,
    br.feature_id,
    br.vertice_id,
    br.connection_point,
    COALESCE(pb_name.pp_index, pb_connection.pp_index, br.connection_point) AS bus,
    COALESCE(pb_name.name, pb_connection.name) AS bus_name,
    loads.load_index,
    br.building_use,
    br.building_type,
    br.type,
    br.occupants,
    br.households,
    br.floor_area,
    br.floor_number,
    br.construction_year,
    br.postcode,
    br.street,
    br.house_number,
    br.gemeindeschluessel,
    br.assigned_way_id,
    br.peak_load_in_kw,
    ST_Transform(br.centroid, 4326) AS centroid,
    ST_Y(ST_Transform(br.centroid, 4326)) AS lat,
    ST_X(ST_Transform(br.centroid, 4326)) AS lon,
    loads.load_indices,
    loads.load_count,
    br.vertice_id AS consumer_vertex,
    br.building_use_id,
    br.residential_floor_area,
    br.nonresidential_floor_area,
    br.nonresidential_use,
    br.mix_score,
    br.mix_rule,
    br.mix_confidence,
    br.residential_peak_load_in_kw,
    br.nonresidential_peak_load_in_kw,
    br.nonresidential_mv_direct
FROM surrogrid.grid_case gc
JOIN pylovo.buildings_result br
  ON br.grid_result_id = gc.pylovo_grid_result_id
 AND br.version_id = gc.pylovo_version_id
LEFT JOIN pylovo.pandapower_bus pb_name
  ON pb_name.grid_result_id = gc.pylovo_grid_result_id
 AND pb_name.name = CONCAT('Consumer Nodebus ', br.vertice_id)
LEFT JOIN pylovo.pandapower_bus pb_connection
  ON pb_connection.grid_result_id = gc.pylovo_grid_result_id
 AND pb_connection.pp_index = br.connection_point
LEFT JOIN LATERAL (
    SELECT
        CASE WHEN COUNT(*) = 1 THEN MIN(pl.pp_index) END AS load_index,
        ARRAY_AGG(pl.pp_index ORDER BY pl.pp_index) AS load_indices,
        COUNT(*)::INTEGER AS load_count
    FROM pylovo.pandapower_load pl
    WHERE pl.grid_result_id = gc.pylovo_grid_result_id
      AND pl.bus = COALESCE(pb_name.pp_index, pb_connection.pp_index, br.connection_point)
    -- Grouping keeps "no load on the bus" a missing row (NULL columns), as in
    -- a join against loads aggregated per (grid, bus).
    GROUP BY pl.bus
) loads ON TRUE;

-- One row per positive Residential or Commercial/Public component of a
-- physical building (MV-direct components with included_in_lv = false). An
-- Unknown non-residential part is non-demand and has no row; the pandas twin is
-- gridexpand.common.building_components.build_building_components.
CREATE OR REPLACE VIEW surrogrid.grid_building_component AS
SELECT
    CONCAT(p.objectid, '::residential') AS component_id,
    p.grid_case_id,
    p.objectid,
    p.pylovo_grid_result_id,
    p.pylovo_version_id,
    'Residential'::TEXT AS component_category,
    p.residential_floor_area AS effective_floor_area_m2,
    p.floor_area * p.floor_number AS gross_floor_area_m2,
    p.households,
    p.occupants,
    p.residential_peak_load_in_kw AS installed_peak_kw,
    p.households::DOUBLE PRECISION AS load_units,
    p.consumer_vertex,
    p.bus,
    TRUE AS included_in_lv,
    FALSE AS mv_direct,
    p.mix_score,
    p.mix_rule,
    p.mix_confidence,
    p.building_use AS source_building_use,
    p.building_use_id AS source_building_use_id,
    p.building_type AS source_building_type
FROM surrogrid.grid_building_bus p
WHERE p.residential_floor_area > 0

UNION ALL

SELECT
    CONCAT(p.objectid, '::', LOWER(p.nonresidential_use)) AS component_id,
    p.grid_case_id,
    p.objectid,
    p.pylovo_grid_result_id,
    p.pylovo_version_id,
    p.nonresidential_use AS component_category,
    p.nonresidential_floor_area AS effective_floor_area_m2,
    p.floor_area * p.floor_number AS gross_floor_area_m2,
    NULL::INTEGER AS households,
    NULL::INTEGER AS occupants,
    p.nonresidential_peak_load_in_kw AS installed_peak_kw,
    1.0::DOUBLE PRECISION AS load_units,
    p.consumer_vertex,
    p.bus,
    NOT COALESCE(p.nonresidential_mv_direct, FALSE) AS included_in_lv,
    COALESCE(p.nonresidential_mv_direct, FALSE) AS mv_direct,
    p.mix_score,
    p.mix_rule,
    p.mix_confidence,
    p.building_use AS source_building_use,
    p.building_use_id AS source_building_use_id,
    p.building_type AS source_building_type
FROM surrogrid.grid_building_bus p
WHERE p.nonresidential_floor_area > 0
  AND p.nonresidential_use IN ('Commercial', 'Public');

-- QGIS layers of the synthetic expansion results. Created once (never dropped
-- by schema code); refreshed by refresh_qgis_views() after materializations.
CREATE MATERIALIZED VIEW IF NOT EXISTS surrogrid.expansion_line_qgis_mv AS
SELECT
    ROW_NUMBER() OVER (
        ORDER BY
            ar.analysis_key,
            elr.powerflow_run_id,
            elr.visible_line_id
    )::BIGINT AS qgis_id,
    ar.analysis_key,
    ar.assumption_key,
    elr.*,
    lv.geom::geometry(LineString, 25832) AS geom
FROM surrogrid.expansion_line_result elr
JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
JOIN pylovo.lines_result_view lv
  ON lv.grid_result_id = elr.pylovo_grid_result_id
 AND lv.version_id = elr.pylovo_version_id
 AND lv.plz = elr.plz
 AND lv.kcid = elr.kcid
 AND lv.bcid = elr.bcid
 AND lv.id = elr.visible_line_id
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_qgis_id
    ON surrogrid.expansion_line_qgis_mv (qgis_id);
CREATE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_analysis
    ON surrogrid.expansion_line_qgis_mv (analysis_key, requires_expansion, overloaded_at_100_percent);
CREATE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_geom
    ON surrogrid.expansion_line_qgis_mv USING GIST (geom);

CREATE MATERIALIZED VIEW IF NOT EXISTS surrogrid.expansion_transformer_qgis_mv AS
SELECT
    ROW_NUMBER() OVER (
        ORDER BY
            ar.analysis_key,
            etr.powerflow_run_id
    )::BIGINT AS qgis_id,
    ar.analysis_key,
    ar.assumption_key,
    etr.*,
    tpwg.osm_id,
    tpwg.comment,
    tpwg.s_max_kva AS equipment_s_max_kva,
    tpwg.cost_eur AS equipment_cost_eur,
    tpwg.equipment_type,
    tpwg.geom::geometry(Point, 25832) AS geom
FROM surrogrid.expansion_transformer_result etr
JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
JOIN pylovo.transformer_positions_with_grid tpwg
  ON tpwg.grid_result_id = etr.pylovo_grid_result_id
 AND tpwg.version_id = etr.pylovo_version_id
 AND tpwg.plz = etr.plz
 AND tpwg.kcid = etr.kcid
 AND tpwg.bcid = etr.bcid
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_qgis_id
    ON surrogrid.expansion_transformer_qgis_mv (qgis_id);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_analysis
    ON surrogrid.expansion_transformer_qgis_mv (analysis_key, requires_expansion, overloaded_at_100_percent);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_geom
    ON surrogrid.expansion_transformer_qgis_mv USING GIST (geom);
