-- 0008 anonymised provider names. The real-grid providers are dso_a (station
-- workbooks) and dso_b (area deliveries) instead of the DSO abbreviations; the
-- stored values follow. Run names and analysis keys of earlier runs keep their
-- old provider strings (they are history, not looked up by the new code).

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'surrogrid' AND table_name = 'demand_component_audit'
          AND column_name = 'matched_swf_asset_count'
    ) THEN
        ALTER TABLE surrogrid.demand_component_audit RENAME COLUMN matched_swf_asset_count TO matched_dso_asset_count;
    END IF;
END $$;

UPDATE surrogrid.real_grid_case SET source = 'dso_a' WHERE source = 'swf';
UPDATE surrogrid.real_grid_case SET source = 'dso_b' WHERE source = 'uzw';

ALTER TABLE surrogrid.expansion_analysis_run DROP CONSTRAINT IF EXISTS ck_expansion_analysis_run_data_source;
UPDATE surrogrid.expansion_analysis_run SET data_source = 'Real DSO A' WHERE data_source = 'Real SWF';
UPDATE surrogrid.expansion_analysis_run SET data_source = 'Real DSO B' WHERE data_source = 'Real ÜZW';

CREATE OR REPLACE FUNCTION pg_temp.add_constraint_if_missing(tbl regclass, name text, definition text)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = tbl AND conname = name) THEN
        EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I %s', tbl, name, definition);
    END IF;
END $$;

SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_analysis_run', 'ck_expansion_analysis_run_data_source',
    'CHECK (data_source IN (''Synthetic'', ''Real DSO A'', ''Real DSO B''))');
