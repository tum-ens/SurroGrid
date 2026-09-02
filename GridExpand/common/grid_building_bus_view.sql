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
LEFT JOIN (
    SELECT
        pl.grid_result_id,
        pl.bus,
        CASE WHEN COUNT(*) = 1 THEN MIN(pl.pp_index) END AS load_index,
        ARRAY_AGG(pl.pp_index ORDER BY pl.pp_index) AS load_indices,
        COUNT(*)::INTEGER AS load_count
    FROM pylovo.pandapower_load pl
    GROUP BY pl.grid_result_id, pl.bus
) loads
  ON loads.grid_result_id = gc.pylovo_grid_result_id
 AND loads.bus = COALESCE(
     pb_name.pp_index,
     pb_connection.pp_index,
     br.connection_point
 );

CREATE OR REPLACE VIEW surrogrid.grid_building_component AS
WITH physical AS (
    SELECT *
    FROM surrogrid.grid_building_bus
)
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
FROM physical p
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
FROM physical p
WHERE p.nonresidential_floor_area > 0;
