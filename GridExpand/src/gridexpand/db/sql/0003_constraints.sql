-- 0003 constraints for databases created by the pre-migration code.
-- No-op on a database created from 0001 (every change is guarded).
--
-- New foreign keys and CHECKs are added NOT VALID here (new rows are checked at
-- once, existing rows are not scanned); 0004 validates them in a separate
-- transaction that does not block reads or writes, and only then drops the
-- single-column foreign keys they replace.

CREATE OR REPLACE FUNCTION pg_temp.add_constraint_if_missing(tbl regclass, name text, definition text)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = tbl AND conname = name) THEN
        EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I %s', tbl, name, definition);
    END IF;
END $$;

-- (a) Exact duplicates of fk_powerflow_run_pipeline / fk_demand_allocation_run_pipeline
-- (the inline REFERENCES of the old DDL plus its later ALTER TABLE).
ALTER TABLE surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS powerflow_run_pipeline_run_id_fkey;
ALTER TABLE surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS demand_allocation_run_pipeline_run_id_fkey;

-- (b) Unique keys referenced by the composite foreign keys; unique indexes of
-- the old DDL become named constraints (instant, the index is reused).
SELECT pg_temp.add_constraint_if_missing('surrogrid.pipeline_run', 'uq_pipeline_run_identity',
    'UNIQUE (pipeline_run_id, grid_case_id, scenario_id)');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_run', 'uq_powerflow_run_identity',
    'UNIQUE (powerflow_run_id, grid_case_id, scenario_id)');
SELECT pg_temp.add_constraint_if_missing('surrogrid.real_powerflow_run', 'uq_real_powerflow_run_identity',
    'UNIQUE (real_powerflow_run_id, real_grid_case_id, scenario_id)');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_run', 'uq_powerflow_run_grid_scenario_name',
    'UNIQUE USING INDEX uq_powerflow_run_grid_scenario_name');
SELECT pg_temp.add_constraint_if_missing('surrogrid.demand_allocation_run', 'uq_demand_allocation_run_grid_scenario_name',
    'UNIQUE USING INDEX uq_demand_allocation_run_grid_scenario_name');

-- (c) Composite foreign keys: the repeated grid_case_id/scenario_id of a run or
-- result row must be those of its parent run. They replace the separate keys
-- to grid_case, scenario and the parent run (dropped in 0004).
SELECT pg_temp.add_constraint_if_missing('surrogrid.demand_allocation_run', 'fk_demand_allocation_run_pipeline_run',
    'FOREIGN KEY (pipeline_run_id, grid_case_id, scenario_id)
     REFERENCES surrogrid.pipeline_run (pipeline_run_id, grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_run', 'fk_powerflow_run_pipeline_run',
    'FOREIGN KEY (pipeline_run_id, grid_case_id, scenario_id)
     REFERENCES surrogrid.pipeline_run (pipeline_run_id, grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_line_result', 'fk_expansion_line_result_powerflow_run',
    'FOREIGN KEY (powerflow_run_id, grid_case_id, scenario_id)
     REFERENCES surrogrid.powerflow_run (powerflow_run_id, grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_transformer_result', 'fk_expansion_transformer_result_powerflow_run',
    'FOREIGN KEY (powerflow_run_id, grid_case_id, scenario_id)
     REFERENCES surrogrid.powerflow_run (powerflow_run_id, grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_real_grid_status', 'fk_expansion_real_grid_status_real_powerflow_run',
    'FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
     REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_real_line_result', 'fk_expansion_real_line_result_real_powerflow_run',
    'FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
     REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_real_transformer_result', 'fk_expansion_real_transformer_result_real_powerflow_run',
    'FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
     REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE NOT VALID');

-- (d) Closed vocabularies (the raw hypertables get no CHECK: validating it
-- would scan the largest tables for a column only the writers fill).
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_summary', 'ck_powerflow_summary_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_cable_summary', 'ck_powerflow_cable_summary_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_bus_voltage_summary', 'ck_powerflow_bus_voltage_summary_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_transformer_diagnostic', 'ck_powerflow_transformer_diagnostic_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.powerflow_tail_value', 'ck_powerflow_tail_value_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_analysis_run', 'ck_expansion_analysis_run_stage',
    'CHECK (stage IN (''pre'', ''post'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_analysis_run', 'ck_expansion_analysis_run_data_source',
    'CHECK (data_source IN (''Synthetic'', ''Real SWF'', ''Real ÜZW'')) NOT VALID');
SELECT pg_temp.add_constraint_if_missing('surrogrid.expansion_real_grid_status', 'ck_expansion_real_grid_status_cost_status',
    'CHECK (cost_status IN (''complete'', ''incomplete'', ''excluded'')) NOT VALID');

-- (e) grid_case -> pylovo.grid_result: RESTRICT instead of CASCADE (a pylovo
-- deletion must not silently delete SurroGrid results). Only where the old
-- foreign key still exists; `gridexpand db relink-pylovo` adds it elsewhere.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'surrogrid.grid_case'::regclass
          AND conname = 'grid_case_pylovo_grid_result_id_fkey'
    ) THEN
        PERFORM pg_temp.add_constraint_if_missing('surrogrid.grid_case', 'fk_grid_case_pylovo_grid_result',
            'FOREIGN KEY (pylovo_grid_result_id) REFERENCES pylovo.grid_result (grid_result_id)
             ON DELETE RESTRICT NOT VALID');
    END IF;
END $$;

-- (f) The old DDL seeded a 'baseline_static' scenario on every call. Remove it
-- where nothing refers to it (ensure_scenario creates it again when needed).
DELETE FROM surrogrid.scenario s
WHERE s.scenario_key = 'baseline_static'
  AND NOT EXISTS (SELECT 1 FROM surrogrid.pipeline_run r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.demand_allocation_run r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.powerflow_run r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.real_powerflow_run r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_analysis_run r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_line_result r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_transformer_result r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_real_grid_status r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_real_line_result r WHERE r.scenario_id = s.scenario_id)
  AND NOT EXISTS (SELECT 1 FROM surrogrid.expansion_real_transformer_result r WHERE r.scenario_id = s.scenario_id);

-- (g) expansion_analysis_run.scenario_id was never set: fill it where the
-- analysis's result rows all belong to one scenario.
UPDATE surrogrid.expansion_analysis_run ar
SET scenario_id = s.scenario_id
FROM (
    SELECT expansion_analysis_run_id, min(scenario_id) AS scenario_id
    FROM (
        SELECT DISTINCT expansion_analysis_run_id, scenario_id FROM surrogrid.expansion_line_result
        UNION SELECT DISTINCT expansion_analysis_run_id, scenario_id FROM surrogrid.expansion_transformer_result
        UNION SELECT DISTINCT expansion_analysis_run_id, scenario_id FROM surrogrid.expansion_real_grid_status
    ) results
    GROUP BY expansion_analysis_run_id
    HAVING count(DISTINCT scenario_id) = 1
) s
WHERE ar.expansion_analysis_run_id = s.expansion_analysis_run_id
  AND ar.scenario_id IS NULL;

-- (h) The content-hash marker of the old schema code.
DROP TABLE IF EXISTS surrogrid.schema_marker;

DROP FUNCTION pg_temp.add_constraint_if_missing(regclass, text, text);
