-- 0004 validate the constraints of 0003 and drop the foreign keys they replace.
-- No-op on a database created from 0001.
--
-- VALIDATE CONSTRAINT scans the table under a SHARE UPDATE EXCLUSIVE lock
-- (reads and writes continue). A failure means that existing rows break the
-- rule: the error names the constraint; fix the rows and re-run
-- `gridexpand db migrate --apply` (0003 stays applied, the old keys stay).

DO $$
DECLARE
    item record;
BEGIN
    FOR item IN
        SELECT c.conrelid::regclass AS tbl, c.conname
        FROM pg_constraint c
        WHERE c.connamespace = 'surrogrid'::regnamespace
          AND NOT c.convalidated
          AND c.conname IN (
              'fk_demand_allocation_run_pipeline_run',
              'fk_powerflow_run_pipeline_run',
              'fk_expansion_line_result_powerflow_run',
              'fk_expansion_transformer_result_powerflow_run',
              'fk_expansion_real_grid_status_real_powerflow_run',
              'fk_expansion_real_line_result_real_powerflow_run',
              'fk_expansion_real_transformer_result_real_powerflow_run',
              'fk_grid_case_pylovo_grid_result',
              'ck_powerflow_summary_stage',
              'ck_powerflow_cable_summary_stage',
              'ck_powerflow_bus_voltage_summary_stage',
              'ck_powerflow_transformer_diagnostic_stage',
              'ck_powerflow_tail_value_stage',
              'ck_expansion_analysis_run_stage',
              'ck_expansion_analysis_run_data_source',
              'ck_expansion_real_grid_status_cost_status'
          )
        ORDER BY 1::text, 2
    LOOP
        EXECUTE format('ALTER TABLE %s VALIDATE CONSTRAINT %I', item.tbl, item.conname);
    END LOOP;
END $$;

-- Replaced by the composite keys to the parent run (one cascade path per row).
ALTER TABLE surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS fk_demand_allocation_run_pipeline;
ALTER TABLE surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS demand_allocation_run_grid_case_id_fkey;
ALTER TABLE surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS fk_demand_allocation_run_scenario;
ALTER TABLE surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_pipeline;
ALTER TABLE surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS powerflow_run_grid_case_id_fkey;
ALTER TABLE surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_scenario;
ALTER TABLE surrogrid.expansion_line_result DROP CONSTRAINT IF EXISTS expansion_line_result_powerflow_run_id_fkey;
ALTER TABLE surrogrid.expansion_line_result DROP CONSTRAINT IF EXISTS expansion_line_result_grid_case_id_fkey;
ALTER TABLE surrogrid.expansion_line_result DROP CONSTRAINT IF EXISTS fk_expansion_line_result_scenario;
ALTER TABLE surrogrid.expansion_transformer_result DROP CONSTRAINT IF EXISTS expansion_transformer_result_powerflow_run_id_fkey;
ALTER TABLE surrogrid.expansion_transformer_result DROP CONSTRAINT IF EXISTS expansion_transformer_result_grid_case_id_fkey;
ALTER TABLE surrogrid.expansion_transformer_result DROP CONSTRAINT IF EXISTS fk_expansion_transformer_result_scenario;
ALTER TABLE surrogrid.expansion_real_grid_status DROP CONSTRAINT IF EXISTS expansion_real_grid_status_real_powerflow_run_id_fkey;
ALTER TABLE surrogrid.expansion_real_grid_status DROP CONSTRAINT IF EXISTS expansion_real_grid_status_real_grid_case_id_fkey;
ALTER TABLE surrogrid.expansion_real_grid_status DROP CONSTRAINT IF EXISTS expansion_real_grid_status_scenario_id_fkey;
ALTER TABLE surrogrid.expansion_real_line_result DROP CONSTRAINT IF EXISTS expansion_real_line_result_real_powerflow_run_id_fkey;
ALTER TABLE surrogrid.expansion_real_line_result DROP CONSTRAINT IF EXISTS expansion_real_line_result_real_grid_case_id_fkey;
ALTER TABLE surrogrid.expansion_real_line_result DROP CONSTRAINT IF EXISTS expansion_real_line_result_scenario_id_fkey;
ALTER TABLE surrogrid.expansion_real_transformer_result DROP CONSTRAINT IF EXISTS expansion_real_transformer_result_real_powerflow_run_id_fkey;
ALTER TABLE surrogrid.expansion_real_transformer_result DROP CONSTRAINT IF EXISTS expansion_real_transformer_result_real_grid_case_id_fkey;
ALTER TABLE surrogrid.expansion_real_transformer_result DROP CONSTRAINT IF EXISTS expansion_real_transformer_result_scenario_id_fkey;

-- The CASCADE key to pylovo, replaced by fk_grid_case_pylovo_grid_result (RESTRICT).
ALTER TABLE surrogrid.grid_case DROP CONSTRAINT IF EXISTS grid_case_pylovo_grid_result_id_fkey;
