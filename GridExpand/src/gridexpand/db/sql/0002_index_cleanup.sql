-- 0002 index cleanup for databases created by the pre-migration code.
-- No-op on a database created from 0001 (every statement is IF [NOT] EXISTS).
--
-- Dropping an index of a hypertable also drops it on every chunk. Rollback: the
-- dropped definitions are listed in docs/database.md ("Migrations").

-- Default ts indexes of create_hypertable: no query filters on ts alone.
DROP INDEX IF EXISTS surrogrid.allocated_demand_ts_idx;
DROP INDEX IF EXISTS surrogrid.allocated_eff_factor_ts_idx;
DROP INDEX IF EXISTS surrogrid.powerflow_demand_ts_idx;
DROP INDEX IF EXISTS surrogrid.powerflow_import_ts_idx;
DROP INDEX IF EXISTS surrogrid.powerflow_bus_voltage_ts_idx;
DROP INDEX IF EXISTS surrogrid.powerflow_line_result_ts_idx;
DROP INDEX IF EXISTS surrogrid.powerflow_reactive_component_ts_idx;

-- Bus/line indexes of the raw series: never used alone, and the line index
-- misleads the planner into a BitmapAnd over the run index.
DROP INDEX IF EXISTS surrogrid.idx_powerflow_bus_voltage_bus;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_line_result_line;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_demand_bus;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_reactive_bus;
DROP INDEX IF EXISTS surrogrid.idx_allocated_demand_bus;
DROP INDEX IF EXISTS surrogrid.idx_allocated_eff_factor_bus;

-- Duplicates of a unique index with the same leading columns.
DROP INDEX IF EXISTS surrogrid.uq_scenario_key;
DROP INDEX IF EXISTS surrogrid.idx_pipeline_run_grid_case;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_run_grid_case;
DROP INDEX IF EXISTS surrogrid.idx_demand_allocation_run_grid_case;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_cable_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_bus_voltage_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_tail_value_run_stage_metric;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_transformer_diagnostic_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_real_powerflow_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_real_powerflow_cable_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_real_powerflow_bus_voltage_summary_run_stage;
DROP INDEX IF EXISTS surrogrid.idx_real_powerflow_tail_value_run_stage_metric;

-- Indexes without a reader (write-only audit tables, lookups by asset or
-- component alone); superseded by (scenario_id, run_name).
DROP INDEX IF EXISTS surrogrid.idx_powerflow_tail_value_asset;
DROP INDEX IF EXISTS surrogrid.idx_electrification_assignment_building;
DROP INDEX IF EXISTS surrogrid.idx_electrification_assignment_run_technology;
DROP INDEX IF EXISTS surrogrid.idx_allocated_vehicle_run_model;
DROP INDEX IF EXISTS surrogrid.idx_demand_component_audit_component;
DROP INDEX IF EXISTS surrogrid.idx_powerflow_run_scenario;

-- New indexes: run-name lookups and the foreign keys used by cascading deletes.
CREATE INDEX IF NOT EXISTS idx_powerflow_run_scenario_name ON surrogrid.powerflow_run (scenario_id, run_name);
CREATE INDEX IF NOT EXISTS idx_expansion_analysis_run_scenario ON surrogrid.expansion_analysis_run (scenario_id);
CREATE INDEX IF NOT EXISTS idx_expansion_line_result_powerflow_run ON surrogrid.expansion_line_result (powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_result_powerflow_run ON surrogrid.expansion_transformer_result (powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_real_grid_status_real_powerflow_run ON surrogrid.expansion_real_grid_status (real_powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_real_line_result_real_powerflow_run ON surrogrid.expansion_real_line_result (real_powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_real_transformer_result_real_powerflow_run ON surrogrid.expansion_real_transformer_result (real_powerflow_run_id);
