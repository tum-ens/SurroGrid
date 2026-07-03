CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE SCHEMA IF NOT EXISTS surrogrid;

CREATE TABLE IF NOT EXISTS surrogrid.grid_case (
    grid_case_id BIGSERIAL PRIMARY KEY,
    ags BIGINT NOT NULL,
    plz INTEGER NOT NULL,
    kcid INTEGER NOT NULL,
    bcid INTEGER NOT NULL,
    pylovo_grid_result_id BIGINT NOT NULL REFERENCES pylovo.grid_result(grid_result_id) ON DELETE CASCADE,
    pylovo_version_id VARCHAR(10) NOT NULL,
    cell_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_grid_case UNIQUE (ags, plz, kcid, bcid, pylovo_grid_result_id)
);

ALTER TABLE IF EXISTS surrogrid.grid_case DROP COLUMN IF EXISTS source_filename;

CREATE INDEX IF NOT EXISTS idx_grid_case_ags ON surrogrid.grid_case (ags);
CREATE INDEX IF NOT EXISTS idx_grid_case_pylovo_grid_result ON surrogrid.grid_case (pylovo_grid_result_id);

CREATE TABLE IF NOT EXISTS surrogrid.scenario (
    scenario_id BIGSERIAL PRIMARY KEY,
    scenario_key TEXT NOT NULL UNIQUE,
    scenario_label TEXT NOT NULL,
    description TEXT,
    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS surrogrid.scenario ADD COLUMN IF NOT EXISTS scenario_key TEXT;
ALTER TABLE IF EXISTS surrogrid.scenario ADD COLUMN IF NOT EXISTS scenario_label TEXT;
ALTER TABLE IF EXISTS surrogrid.scenario ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE IF EXISTS surrogrid.scenario ADD COLUMN IF NOT EXISTS assumptions JSONB NOT NULL DEFAULT '{}'::JSONB;
ALTER TABLE IF EXISTS surrogrid.scenario ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
CREATE UNIQUE INDEX IF NOT EXISTS uq_scenario_key ON surrogrid.scenario (scenario_key);

ALTER TABLE IF EXISTS surrogrid.scenario DROP COLUMN IF EXISTS timeframe_mode;

INSERT INTO surrogrid.scenario (
    scenario_key, scenario_label, description, assumptions
)
VALUES (
    'baseline_static',
    'Baseline static assumptions',
    'Initial static full-pipeline scenario. Explicit scenario dimensions will be added once scenario variation is introduced.',
    '{"pipeline": "GridExpand", "variant": "static", "timeframe_mode": "full_year", "horizon_hours": 8760, "timeframe_start": "2009-01-01T00:00:00+00:00", "timeframe_end": "2009-12-31T23:00:00+00:00", "source_year_or_reference_year": 2009, "timeframe_kind": "full_year", "methodological_note": "Full 8760-hour reference-year run. Cost and investment outputs keep their existing annual interpretation.", "cost_investment_interpretation": "annual_valid", "annual_valid": true}'::JSONB
)
ON CONFLICT (scenario_key) DO UPDATE SET
    scenario_label = EXCLUDED.scenario_label,
    description = EXCLUDED.description,
    assumptions = EXCLUDED.assumptions,
    updated_at = NOW();

CREATE TABLE IF NOT EXISTS surrogrid.pipeline_run (
    pipeline_run_id BIGSERIAL PRIMARY KEY,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    run_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_pipeline_run_grid_scenario_name UNIQUE (grid_case_id, scenario_id, run_name)
);

ALTER TABLE IF EXISTS surrogrid.pipeline_run DROP CONSTRAINT IF EXISTS pipeline_run_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.pipeline_run DROP CONSTRAINT IF EXISTS fk_pipeline_run_scenario;
ALTER TABLE IF EXISTS surrogrid.pipeline_run
    ADD CONSTRAINT fk_pipeline_run_scenario
    FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_pipeline_run_grid_case ON surrogrid.pipeline_run (grid_case_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_run_scenario ON surrogrid.pipeline_run (scenario_id);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_run (
    powerflow_run_id BIGSERIAL PRIMARY KEY,
    pipeline_run_id BIGINT NOT NULL REFERENCES surrogrid.pipeline_run(pipeline_run_id) ON DELETE CASCADE,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    run_name TEXT NOT NULL,
    urbs_input_file TEXT NOT NULL DEFAULT '',
    storage_mode TEXT NOT NULL DEFAULT 'db',
    pre_only BOOLEAN NOT NULL DEFAULT FALSE,
    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS pipeline_run_id BIGINT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS scenario_id BIGINT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS run_name TEXT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS urbs_input_file TEXT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS assumptions JSONB NOT NULL DEFAULT '{}'::JSONB;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
UPDATE surrogrid.powerflow_run pr
SET scenario_id = sc.scenario_id
FROM surrogrid.scenario sc
WHERE pr.scenario_id IS NULL AND sc.scenario_key = 'baseline_static';
UPDATE surrogrid.powerflow_run
SET run_name = CASE WHEN pre_only THEN 'baseline_static_pre_powerflow' ELSE 'baseline_static_full_powerflow' END
WHERE run_name IS NULL
   OR run_name IN (
       'baseline_static_pre_powerflow',
       'baseline_static_full_powerflow',
       'baseline_static_full_year_pre_powerflow',
       'baseline_static_full_year_full_powerflow'
   );
UPDATE surrogrid.powerflow_run SET urbs_input_file = COALESCE(urbs_input_file, '') WHERE urbs_input_file IS NULL;
INSERT INTO surrogrid.pipeline_run (grid_case_id, scenario_id, run_name)
SELECT DISTINCT grid_case_id, scenario_id, 'baseline_static_pipeline'
FROM surrogrid.powerflow_run
WHERE pipeline_run_id IS NULL
ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET updated_at = NOW();
UPDATE surrogrid.powerflow_run pr
SET pipeline_run_id = pipe.pipeline_run_id
FROM surrogrid.pipeline_run pipe
WHERE pr.pipeline_run_id IS NULL
  AND pipe.grid_case_id = pr.grid_case_id
  AND pipe.scenario_id = pr.scenario_id
  AND pipe.run_name = 'baseline_static_pipeline';
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN pipeline_run_id SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN scenario_id SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN run_name SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN urbs_input_file SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS scenario_label;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS scenario_key;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS source_input_file;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS powerflow_run_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_scenario;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD CONSTRAINT fk_powerflow_run_scenario FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_pipeline;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD CONSTRAINT fk_powerflow_run_pipeline FOREIGN KEY (pipeline_run_id) REFERENCES surrogrid.pipeline_run(pipeline_run_id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_powerflow_run_pipeline ON surrogrid.powerflow_run (pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_powerflow_run_grid_case ON surrogrid.powerflow_run (grid_case_id);
CREATE INDEX IF NOT EXISTS idx_powerflow_run_scenario ON surrogrid.powerflow_run (scenario_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_powerflow_run_grid_scenario_name
    ON surrogrid.powerflow_run (grid_case_id, scenario_id, run_name);

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
    pl.pp_index AS load_index,
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
    ST_X(ST_Transform(br.centroid, 4326)) AS lon
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
LEFT JOIN pylovo.pandapower_load pl
  ON pl.grid_result_id = gc.pylovo_grid_result_id
 AND pl.bus = COALESCE(pb_name.pp_index, pb_connection.pp_index, br.connection_point);

DROP TABLE IF EXISTS surrogrid.mobility_profile_availability;
DROP TABLE IF EXISTS surrogrid.mobility_profile_demand;
DROP TABLE IF EXISTS surrogrid.mobility_profile_pool;

CREATE TABLE IF NOT EXISTS surrogrid.demand_allocation_run (
    demand_allocation_run_id BIGSERIAL PRIMARY KEY,
    pipeline_run_id BIGINT NOT NULL REFERENCES surrogrid.pipeline_run(pipeline_run_id) ON DELETE CASCADE,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    run_name TEXT NOT NULL,
    bridge_filename TEXT NOT NULL DEFAULT '',
    storage_mode TEXT NOT NULL DEFAULT 'db',
    profiles TEXT NOT NULL DEFAULT 'all',
    mobility_source TEXT NOT NULL DEFAULT 'emobpy',
    assumptions JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS surrogrid.demand_allocation_run ADD COLUMN IF NOT EXISTS pipeline_run_id BIGINT;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run ADD COLUMN IF NOT EXISTS assumptions JSONB NOT NULL DEFAULT '{}'::JSONB;
INSERT INTO surrogrid.pipeline_run (grid_case_id, scenario_id, run_name)
SELECT DISTINCT grid_case_id, scenario_id, 'baseline_static_pipeline'
FROM surrogrid.demand_allocation_run
WHERE pipeline_run_id IS NULL
ON CONFLICT (grid_case_id, scenario_id, run_name) DO UPDATE SET updated_at = NOW();
UPDATE surrogrid.demand_allocation_run dar
SET pipeline_run_id = pipe.pipeline_run_id
FROM surrogrid.pipeline_run pipe
WHERE dar.pipeline_run_id IS NULL
  AND pipe.grid_case_id = dar.grid_case_id
  AND pipe.scenario_id = dar.scenario_id
  AND pipe.run_name = 'baseline_static_pipeline';
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run ALTER COLUMN pipeline_run_id SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS fk_demand_allocation_run_pipeline;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run ADD CONSTRAINT fk_demand_allocation_run_pipeline FOREIGN KEY (pipeline_run_id) REFERENCES surrogrid.pipeline_run(pipeline_run_id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS demand_allocation_run_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run DROP CONSTRAINT IF EXISTS fk_demand_allocation_run_scenario;
ALTER TABLE IF EXISTS surrogrid.demand_allocation_run
    ADD CONSTRAINT fk_demand_allocation_run_scenario
    FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;

CREATE UNIQUE INDEX IF NOT EXISTS uq_demand_allocation_run_grid_scenario_name
    ON surrogrid.demand_allocation_run (grid_case_id, scenario_id, run_name);
CREATE INDEX IF NOT EXISTS idx_demand_allocation_run_pipeline
    ON surrogrid.demand_allocation_run (pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_demand_allocation_run_grid_case
    ON surrogrid.demand_allocation_run (grid_case_id);

CREATE TABLE IF NOT EXISTS surrogrid.allocated_demand (
    demand_allocation_run_id BIGINT NOT NULL REFERENCES surrogrid.demand_allocation_run(demand_allocation_run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    bus INTEGER NOT NULL,
    commodity TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_allocated_demand_run_commodity
    ON surrogrid.allocated_demand (demand_allocation_run_id, commodity, t_index);
CREATE INDEX IF NOT EXISTS idx_allocated_demand_bus
    ON surrogrid.allocated_demand (bus);
SELECT create_hypertable('surrogrid.allocated_demand', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.allocated_eff_factor (
    demand_allocation_run_id BIGINT NOT NULL REFERENCES surrogrid.demand_allocation_run(demand_allocation_run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    bus INTEGER NOT NULL,
    component TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_allocated_eff_factor_run_component
    ON surrogrid.allocated_eff_factor (demand_allocation_run_id, component, t_index);
CREATE INDEX IF NOT EXISTS idx_allocated_eff_factor_bus
    ON surrogrid.allocated_eff_factor (bus);
SELECT create_hypertable('surrogrid.allocated_eff_factor', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.allocated_vehicle (
    demand_allocation_run_id BIGINT NOT NULL REFERENCES surrogrid.demand_allocation_run(demand_allocation_run_id) ON DELETE CASCADE,
    bus INTEGER NOT NULL,
    vehicle_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    schedule TEXT NOT NULL,
    seed BIGINT NOT NULL,
    profile_id TEXT,
    battery_cap_kwh DOUBLE PRECISION,
    PRIMARY KEY (demand_allocation_run_id, bus, vehicle_id)
);

CREATE INDEX IF NOT EXISTS idx_allocated_vehicle_run_model
    ON surrogrid.allocated_vehicle (demand_allocation_run_id, model, schedule);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_demand (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    bus INTEGER NOT NULL,
    p_kw DOUBLE PRECISION,
    q_kvar DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_powerflow_demand_run_stage ON surrogrid.powerflow_demand (powerflow_run_id, stage, t_index);
CREATE INDEX IF NOT EXISTS idx_powerflow_demand_bus ON surrogrid.powerflow_demand (bus);
SELECT create_hypertable('surrogrid.powerflow_demand', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_import (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    p_mw DOUBLE PRECISION,
    q_mvar DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_powerflow_import_run_stage ON surrogrid.powerflow_import (powerflow_run_id, stage, t_index);
SELECT create_hypertable('surrogrid.powerflow_import', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_bus_voltage (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    bus INTEGER NOT NULL,
    vm_pu DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_powerflow_bus_voltage_run_stage ON surrogrid.powerflow_bus_voltage (powerflow_run_id, stage, t_index);
CREATE INDEX IF NOT EXISTS idx_powerflow_bus_voltage_bus ON surrogrid.powerflow_bus_voltage (bus);
SELECT create_hypertable('surrogrid.powerflow_bus_voltage', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_line_result (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    line INTEGER NOT NULL,
    p_from_mw DOUBLE PRECISION,
    q_from_mvar DOUBLE PRECISION,
    i_from_ka DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_powerflow_line_result_run_stage ON surrogrid.powerflow_line_result (powerflow_run_id, stage, t_index);
CREATE INDEX IF NOT EXISTS idx_powerflow_line_result_line ON surrogrid.powerflow_line_result (line);
SELECT create_hypertable('surrogrid.powerflow_line_result', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_reactive_component (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    t_index INTEGER NOT NULL,
    bus INTEGER NOT NULL,
    component TEXT NOT NULL,
    source TEXT NOT NULL,
    q_kvar DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_powerflow_reactive_run ON surrogrid.powerflow_reactive_component (powerflow_run_id, t_index);
CREATE INDEX IF NOT EXISTS idx_powerflow_reactive_bus ON surrogrid.powerflow_reactive_component (bus);
SELECT create_hypertable('surrogrid.powerflow_reactive_component', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_summary (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    n_timesteps INTEGER NOT NULL,
    n_voltage_buses INTEGER NOT NULL,
    n_cables INTEGER NOT NULL,
    transformer_s_rated_mva DOUBLE PRECISION,
    trafo_loading_p50_time_percent DOUBLE PRECISION,
    trafo_loading_p90_time_percent DOUBLE PRECISION,
    trafo_loading_p95_time_percent DOUBLE PRECISION,
    trafo_loading_p99_time_percent DOUBLE PRECISION,
    trafo_loading_max_time_percent DOUBLE PRECISION,
    trafo_loading_hours_above_100 INTEGER,
    cable_loading_p95_asset_percent DOUBLE PRECISION,
    cable_hours_above_100_p95_asset DOUBLE PRECISION,
    voltage_p05_load_bus_hour_pu DOUBLE PRECISION,
    voltage_hours_below_0_90_p95_asset DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_powerflow_summary_run_stage UNIQUE (powerflow_run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_powerflow_summary_run_stage
    ON surrogrid.powerflow_summary (powerflow_run_id, stage);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_cable_summary (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    cable INTEGER NOT NULL,
    cable_loading_p50_time_percent DOUBLE PRECISION,
    cable_loading_p90_time_percent DOUBLE PRECISION,
    cable_loading_p95_time_percent DOUBLE PRECISION,
    cable_loading_p99_time_percent DOUBLE PRECISION,
    cable_loading_max_time_percent DOUBLE PRECISION,
    cable_loading_hours_above_100 INTEGER,
    cable_max_i_ka DOUBLE PRECISION,
    cable_parallel DOUBLE PRECISION,
    cable_installed_capacity_ka DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_powerflow_cable_summary_run_stage_cable UNIQUE (powerflow_run_id, stage, cable)
);

CREATE INDEX IF NOT EXISTS idx_powerflow_cable_summary_run_stage
    ON surrogrid.powerflow_cable_summary (powerflow_run_id, stage);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_bus_voltage_summary (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    bus INTEGER NOT NULL,
    voltage_p50_time_pu DOUBLE PRECISION,
    voltage_p10_time_pu DOUBLE PRECISION,
    voltage_p05_time_pu DOUBLE PRECISION,
    voltage_p01_time_pu DOUBLE PRECISION,
    voltage_min_time_pu DOUBLE PRECISION,
    voltage_hours_below_0_90 INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_powerflow_bus_voltage_summary_run_stage_bus UNIQUE (powerflow_run_id, stage, bus)
);

CREATE INDEX IF NOT EXISTS idx_powerflow_bus_voltage_summary_run_stage
    ON surrogrid.powerflow_bus_voltage_summary (powerflow_run_id, stage);

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_tail_value (
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    metric TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    asset_id INTEGER NOT NULL,
    tail TEXT NOT NULL,
    threshold_value DOUBLE PRECISION,
    t_index INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_powerflow_tail_value UNIQUE (
        powerflow_run_id, stage, metric, asset_type, asset_id, tail, t_index
    )
);

CREATE INDEX IF NOT EXISTS idx_powerflow_tail_value_run_stage_metric
    ON surrogrid.powerflow_tail_value (powerflow_run_id, stage, metric);

CREATE INDEX IF NOT EXISTS idx_powerflow_tail_value_asset
    ON surrogrid.powerflow_tail_value (metric, asset_type, asset_id);
