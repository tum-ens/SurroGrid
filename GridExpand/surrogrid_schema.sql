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

UPDATE surrogrid.scenario
SET assumptions = assumptions - 'timeframe_mode'
WHERE assumptions ? 'timeframe_mode';

INSERT INTO surrogrid.scenario (
    scenario_key, scenario_label, description, assumptions
)
VALUES (
    'baseline_static',
    'Baseline static assumptions',
    'Initial static full-pipeline scenario. Explicit scenario dimensions will be added once scenario variation is introduced.',
    '{"pipeline": "GridExpand", "variant": "static"}'::JSONB
)
ON CONFLICT (scenario_key) DO UPDATE SET
    scenario_label = EXCLUDED.scenario_label,
    description = EXCLUDED.description,
    assumptions = EXCLUDED.assumptions,
    updated_at = NOW();

CREATE TABLE IF NOT EXISTS surrogrid.powerflow_run (
    powerflow_run_id BIGSERIAL PRIMARY KEY,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id),
    run_name TEXT NOT NULL,
    urbs_input_file TEXT NOT NULL DEFAULT '',
    storage_mode TEXT NOT NULL DEFAULT 'db',
    pre_only BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS scenario_id BIGINT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS run_name TEXT;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD COLUMN IF NOT EXISTS urbs_input_file TEXT;
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
DELETE FROM surrogrid.powerflow_run
WHERE run_name LIKE '%max_electricity_demand_week%'
   OR run_name LIKE '%min_temperature_week%'
   OR run_name LIKE '%max_solar_generation_week%'
   OR run_name LIKE '%max_mobility_demand_week%'
   OR run_name LIKE '%max_reverse_power_flow_week%'
   OR run_name LIKE '%max_net_load_week%';
UPDATE surrogrid.powerflow_run SET urbs_input_file = COALESCE(urbs_input_file, '') WHERE urbs_input_file IS NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN scenario_id SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN run_name SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ALTER COLUMN urbs_input_file SET NOT NULL;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS scenario_label;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS scenario_key;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP COLUMN IF EXISTS source_input_file;
ALTER TABLE IF EXISTS surrogrid.powerflow_run DROP CONSTRAINT IF EXISTS fk_powerflow_run_scenario;
ALTER TABLE IF EXISTS surrogrid.powerflow_run ADD CONSTRAINT fk_powerflow_run_scenario FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id);

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

CREATE TABLE IF NOT EXISTS surrogrid.mobility_profile_pool (
    profile_id TEXT PRIMARY KEY,
    schedule TEXT NOT NULL,
    model TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    pool_seed BIGINT NOT NULL,
    weather_key TEXT NOT NULL,
    weather_source TEXT NOT NULL,
    battery_cap_kwh DOUBLE PRECISION NOT NULL,
    total_hours INTEGER NOT NULL,
    emobpy_timestep_h DOUBLE PRECISION NOT NULL,
    output_timestep_h DOUBLE PRECISION NOT NULL,
    ref_year INTEGER NOT NULL,
    demand_sum_kwh DOUBLE PRECISION NOT NULL,
    availability_hours DOUBLE PRECISION NOT NULL,
    generation_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mobility_profile_pool_stratum UNIQUE (schedule, model, sample_index, weather_key)
);

CREATE INDEX IF NOT EXISTS idx_mobility_profile_pool_stratum
    ON surrogrid.mobility_profile_pool (schedule, model, weather_key);

CREATE TABLE IF NOT EXISTS surrogrid.mobility_profile_demand (
    profile_id TEXT NOT NULL REFERENCES surrogrid.mobility_profile_pool(profile_id) ON DELETE CASCADE,
    t_index INTEGER NOT NULL,
    demand_kwh DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (profile_id, t_index)
);

CREATE TABLE IF NOT EXISTS surrogrid.mobility_profile_availability (
    profile_id TEXT NOT NULL REFERENCES surrogrid.mobility_profile_pool(profile_id) ON DELETE CASCADE,
    t_index INTEGER NOT NULL,
    availability DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (profile_id, t_index)
);

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
