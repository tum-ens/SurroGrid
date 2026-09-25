-- Deterministic synthetic LoD2 roof sections for the regression sandbox (NOT real data).
-- Every pylovo version-1 building whose hashtext(objectid) % 10 <> 0 gets two roof sections
-- (55 % / 45 % of its footprint, tilt 0 or 20-40 degrees, two orientations); the others have no
-- roof, which exercises the PV fallback/ineligibility path. Only the columns GridExpand's
-- ROOF_SURFACE_QUERY reads exist.
CREATE SCHEMA IF NOT EXISTS citydb;
CREATE TABLE IF NOT EXISTS citydb.feature (id BIGINT PRIMARY KEY, objectid TEXT);
CREATE TABLE IF NOT EXISTS citydb.property (
    id BIGSERIAL PRIMARY KEY, feature_id BIGINT, name TEXT, val_feature_id BIGINT, val_string TEXT);
TRUNCATE citydb.feature, citydb.property;
WITH b AS (
  SELECT DISTINCT ON (objectid) objectid, geom, abs(hashtext(objectid)) AS h
  FROM pylovo.buildings_result WHERE version_id = '1' ORDER BY objectid
), sel AS (
  SELECT row_number() OVER (ORDER BY objectid) AS n, objectid, geom, h FROM b WHERE h % 10 <> 0
), ins_b AS (
  INSERT INTO citydb.feature (id, objectid) SELECT n, objectid FROM sel RETURNING id
), roofs AS (
  SELECT n, objectid, h, 1000000 + n * 10 + k AS roof_id, k,
         greatest(ST_Area(geom), 20.0) * CASE WHEN k = 0 THEN 0.55 ELSE 0.45 END AS area
  FROM sel CROSS JOIN generate_series(0, 1) AS k
), ins_r AS (
  INSERT INTO citydb.feature (id, objectid) SELECT roof_id, NULL FROM roofs RETURNING id
), ins_boundary AS (
  INSERT INTO citydb.property (feature_id, name, val_feature_id)
  SELECT n, 'boundary', roof_id FROM roofs RETURNING id
)
INSERT INTO citydb.property (feature_id, name, val_string)
SELECT roof_id, 'Dachneigung', CASE WHEN h % 7 = 0 THEN '0' ELSE (20 + (h % 5) * 5)::text END FROM roofs
UNION ALL
SELECT roof_id, 'Dachorientierung', (CASE WHEN k = 0 THEN 90 + (h % 180) ELSE (270 + (h % 180)) % 360 END)::text FROM roofs
UNION ALL
SELECT roof_id, 'Flaeche', round(area::numeric, 2)::text FROM roofs;
