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
    loads.load_count
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
