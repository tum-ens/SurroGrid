-- 0009 own grids (pipeline: real): expansion analyses of given grid files carry the data source 'Real'
-- next to 'Synthetic' and the validation's 'Real DSO A' / 'Real DSO B'. real_grid_case.source is 'real'.

ALTER TABLE surrogrid.expansion_analysis_run DROP CONSTRAINT IF EXISTS ck_expansion_analysis_run_data_source;

CREATE OR REPLACE FUNCTION pg_temp.add_constraint_if_missing(tbl regclass, name text, definition text)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = tbl AND conname = name) THEN
        EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I %s', tbl, name, definition);
    END IF;
END $$;

SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_analysis_run', 'ck_expansion_analysis_run_data_source',
    'CHECK (data_source IN (''Synthetic'', ''Real'', ''Real DSO A'', ''Real DSO B''))');
