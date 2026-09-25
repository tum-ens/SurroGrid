-- 0001 baseline: the consolidated surrogrid schema.
--
-- Fresh databases execute this file (then 0002.., which are no-ops on it).
-- Databases created by the pre-migration code are *stamped* with version 1
-- after `gridexpand db migrate` has checked that they have every table and
-- column below; 0002.. then bring their indexes and constraints to this state.
-- Column order and constraint names follow the pre-migration schema, so both
-- paths end with the same catalog. Never edit this file after it has been
-- applied somewhere: add a new numbered migration instead.
--
-- Format rules (parsed by gridexpand.db.schema.baseline_tables for the shape
-- check): one column per line, `name type ...`; table constraints start with
-- CONSTRAINT or PRIMARY KEY.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE SCHEMA IF NOT EXISTS surrogrid;

-- Identity ------------------------------------------------------------------

CREATE TABLE surrogrid.grid_case (
    grid_case_id bigserial PRIMARY KEY,
    ags bigint NOT NULL,
    plz integer NOT NULL,
    kcid integer NOT NULL,
    bcid integer NOT NULL,
    pylovo_grid_result_id bigint NOT NULL,
    pylovo_version_id varchar(10) NOT NULL,
    cell_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_grid_case UNIQUE (ags, plz, kcid, bcid, pylovo_grid_result_id),
    -- Survives a pylovo re-generation (new grid_result ids); added to
    -- pre-migration databases by `gridexpand db relink-pylovo`.
    CONSTRAINT uq_grid_case_natural UNIQUE (ags, pylovo_version_id, plz, kcid, bcid),
    -- RESTRICT: deleting a pylovo grid fails while SurroGrid results use it.
    CONSTRAINT fk_grid_case_pylovo_grid_result FOREIGN KEY (pylovo_grid_result_id)
        REFERENCES pylovo.grid_result (grid_result_id) ON DELETE RESTRICT
);
CREATE INDEX idx_grid_case_ags ON surrogrid.grid_case (ags);
CREATE INDEX idx_grid_case_pylovo_grid_result ON surrogrid.grid_case (pylovo_grid_result_id);

CREATE TABLE surrogrid.scenario (
    scenario_id bigserial PRIMARY KEY,
    scenario_key text NOT NULL UNIQUE,
    scenario_label text NOT NULL,
    description text,
    assumptions jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- One row per (grid case, scenario): the parent of every Step 2 and Step 4 run.
CREATE TABLE surrogrid.pipeline_run (
    pipeline_run_id bigserial PRIMARY KEY,
    grid_case_id bigint NOT NULL REFERENCES surrogrid.grid_case (grid_case_id) ON DELETE CASCADE,
    scenario_id bigint NOT NULL,
    run_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_pipeline_run_grid_scenario_name UNIQUE (grid_case_id, scenario_id, run_name),
    CONSTRAINT uq_pipeline_run_identity UNIQUE (pipeline_run_id, grid_case_id, scenario_id),
    CONSTRAINT fk_pipeline_run_scenario FOREIGN KEY (scenario_id)
        REFERENCES surrogrid.scenario (scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_pipeline_run_scenario ON surrogrid.pipeline_run (scenario_id);

-- The run tables repeat grid_case_id and scenario_id of their pipeline run; the
-- composite foreign keys keep the copies consistent and give every child
-- exactly one cascade path (scenario/grid_case -> pipeline_run -> run).
CREATE TABLE surrogrid.demand_allocation_run (
    demand_allocation_run_id bigserial PRIMARY KEY,
    pipeline_run_id bigint NOT NULL,
    grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    run_name text NOT NULL,
    bridge_filename text NOT NULL DEFAULT '',
    storage_mode text NOT NULL DEFAULT 'db',
    profiles text NOT NULL DEFAULT 'all',
    mobility_source text NOT NULL DEFAULT 'emobpy',
    assumptions jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_demand_allocation_run_grid_scenario_name UNIQUE (grid_case_id, scenario_id, run_name),
    CONSTRAINT fk_demand_allocation_run_pipeline_run FOREIGN KEY (pipeline_run_id, grid_case_id, scenario_id)
        REFERENCES surrogrid.pipeline_run (pipeline_run_id, grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_demand_allocation_run_pipeline ON surrogrid.demand_allocation_run (pipeline_run_id);

CREATE TABLE surrogrid.powerflow_run (
    powerflow_run_id bigserial PRIMARY KEY,
    pipeline_run_id bigint NOT NULL,
    grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    run_name text NOT NULL,
    urbs_input_file text NOT NULL DEFAULT '',
    storage_mode text NOT NULL DEFAULT 'db',
    pre_only boolean NOT NULL DEFAULT false,
    assumptions jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_powerflow_run_grid_scenario_name UNIQUE (grid_case_id, scenario_id, run_name),
    CONSTRAINT uq_powerflow_run_identity UNIQUE (powerflow_run_id, grid_case_id, scenario_id),
    CONSTRAINT fk_powerflow_run_pipeline_run FOREIGN KEY (pipeline_run_id, grid_case_id, scenario_id)
        REFERENCES surrogrid.pipeline_run (pipeline_run_id, grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_powerflow_run_pipeline ON surrogrid.powerflow_run (pipeline_run_id);
CREATE INDEX idx_powerflow_run_scenario_name ON surrogrid.powerflow_run (scenario_id, run_name);

-- Real (DSO) grids and their power-flow runs.
CREATE TABLE surrogrid.real_grid_case (
    real_grid_case_id bigserial PRIMARY KEY,
    source text NOT NULL,
    plz integer,
    lv_id text NOT NULL,
    variant text,
    category text,
    load_status text,
    status text,
    source_file text NOT NULL,
    bus_count integer,
    line_count integer,
    load_count integer,
    assumptions jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_real_grid_case_source_file UNIQUE (source, source_file)
);
CREATE INDEX idx_real_grid_case_plz ON surrogrid.real_grid_case (plz);

CREATE TABLE surrogrid.real_powerflow_run (
    real_powerflow_run_id bigserial PRIMARY KEY,
    real_grid_case_id bigint NOT NULL REFERENCES surrogrid.real_grid_case (real_grid_case_id) ON DELETE CASCADE,
    scenario_id bigint NOT NULL REFERENCES surrogrid.scenario (scenario_id) ON DELETE CASCADE,
    run_name text NOT NULL,
    storage_mode text NOT NULL DEFAULT 'db',
    pre_only boolean NOT NULL DEFAULT true,
    assumptions jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_real_powerflow_run_case_scenario_name UNIQUE (real_grid_case_id, scenario_id, run_name),
    CONSTRAINT uq_real_powerflow_run_identity UNIQUE (real_powerflow_run_id, real_grid_case_id, scenario_id)
);
CREATE INDEX idx_real_powerflow_run_scenario ON surrogrid.real_powerflow_run (scenario_id);

-- Step 2 demand allocation ---------------------------------------------------

-- Compact per-component evidence; hourly component series are not stored.
CREATE TABLE surrogrid.demand_component_audit (
    demand_allocation_run_id bigint NOT NULL REFERENCES surrogrid.demand_allocation_run (demand_allocation_run_id) ON DELETE CASCADE,
    component_id text NOT NULL,
    objectid text NOT NULL,
    scenario_unit_id text,
    bus integer,
    category text NOT NULL,
    commodity text NOT NULL,
    annual_energy_kwh double precision,
    max_profile_value double precision,
    profile_hash text,
    profile_method text NOT NULL,
    stable_seed bigint,
    source_asset_count integer,
    matched_swf_asset_count integer,
    included_in_lv boolean NOT NULL,
    suppression_reason text,
    pylovo_version_id varchar(10) NOT NULL,
    mix_score double precision,
    mix_rule text,
    mix_confidence text,
    mv_direct boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_demand_component_audit UNIQUE (demand_allocation_run_id, component_id, commodity),
    CONSTRAINT ck_demand_component_audit_suppression CHECK (included_in_lv OR suppression_reason IS NOT NULL)
);
CREATE INDEX idx_demand_component_audit_run ON surrogrid.demand_component_audit (demand_allocation_run_id, category, commodity);

CREATE TABLE surrogrid.electrification_assignment (
    demand_allocation_run_id bigint NOT NULL REFERENCES surrogrid.demand_allocation_run (demand_allocation_run_id) ON DELETE CASCADE,
    building_objectid text NOT NULL,
    technology text NOT NULL,
    selection_scope_id text NOT NULL,
    adoption_mode text NOT NULL,
    configured_share double precision,
    eligible boolean NOT NULL,
    selection_score double precision,
    selection_rank integer,
    selected boolean NOT NULL,
    exclusion_reason text,
    source_evidence text,
    profile_seed bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_electrification_assignment UNIQUE (demand_allocation_run_id, building_objectid, technology),
    CONSTRAINT ck_electrification_assignment_technology CHECK (technology IN ('heat', 'mobility', 'pv_battery')),
    CONSTRAINT ck_electrification_assignment_selection CHECK (NOT selected OR eligible)
);

-- Hourly series (written only with --step2-timeseries-storage db|both).
-- Hypertables on ts (= run timeframe_start + t_index hours); the only index is
-- (run, label, t_index): no query filters on ts, bus or line alone.
CREATE TABLE surrogrid.allocated_demand (
    demand_allocation_run_id bigint NOT NULL REFERENCES surrogrid.demand_allocation_run (demand_allocation_run_id) ON DELETE CASCADE,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    bus integer NOT NULL,
    commodity text NOT NULL,
    value double precision NOT NULL
);
SELECT create_hypertable('surrogrid.allocated_demand', 'ts', create_default_indexes => false);
CREATE INDEX idx_allocated_demand_run_commodity ON surrogrid.allocated_demand (demand_allocation_run_id, commodity, t_index);

CREATE TABLE surrogrid.allocated_eff_factor (
    demand_allocation_run_id bigint NOT NULL REFERENCES surrogrid.demand_allocation_run (demand_allocation_run_id) ON DELETE CASCADE,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    bus integer NOT NULL,
    component text NOT NULL,
    value double precision NOT NULL
);
SELECT create_hypertable('surrogrid.allocated_eff_factor', 'ts', create_default_indexes => false);
CREATE INDEX idx_allocated_eff_factor_run_component ON surrogrid.allocated_eff_factor (demand_allocation_run_id, component, t_index);

CREATE TABLE surrogrid.allocated_vehicle (
    demand_allocation_run_id bigint NOT NULL REFERENCES surrogrid.demand_allocation_run (demand_allocation_run_id) ON DELETE CASCADE,
    bus integer NOT NULL,
    vehicle_id integer NOT NULL,
    model text NOT NULL,
    schedule text NOT NULL,
    seed bigint NOT NULL,
    profile_id text,
    battery_cap_kwh double precision,
    PRIMARY KEY (demand_allocation_run_id, bus, vehicle_id)
);

-- Step 4 raw hourly results (--powerflow-output raw|both) ---------------------

CREATE TABLE surrogrid.powerflow_demand (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    bus integer NOT NULL,
    p_kw double precision,
    q_kvar double precision
);
SELECT create_hypertable('surrogrid.powerflow_demand', 'ts', create_default_indexes => false);
CREATE INDEX idx_powerflow_demand_run_stage ON surrogrid.powerflow_demand (powerflow_run_id, stage, t_index);

CREATE TABLE surrogrid.powerflow_import (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    p_mw double precision,
    q_mvar double precision
);
SELECT create_hypertable('surrogrid.powerflow_import', 'ts', create_default_indexes => false);
CREATE INDEX idx_powerflow_import_run_stage ON surrogrid.powerflow_import (powerflow_run_id, stage, t_index);

CREATE TABLE surrogrid.powerflow_bus_voltage (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    bus integer NOT NULL,
    vm_pu double precision
);
SELECT create_hypertable('surrogrid.powerflow_bus_voltage', 'ts', create_default_indexes => false);
CREATE INDEX idx_powerflow_bus_voltage_run_stage ON surrogrid.powerflow_bus_voltage (powerflow_run_id, stage, t_index);

CREATE TABLE surrogrid.powerflow_line_result (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    line integer NOT NULL,
    p_from_mw double precision,
    q_from_mvar double precision,
    i_from_ka double precision
);
SELECT create_hypertable('surrogrid.powerflow_line_result', 'ts', create_default_indexes => false);
CREATE INDEX idx_powerflow_line_result_run_stage ON surrogrid.powerflow_line_result (powerflow_run_id, stage, t_index);

CREATE TABLE surrogrid.powerflow_reactive_component (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    ts timestamptz NOT NULL,
    t_index integer NOT NULL,
    bus integer NOT NULL,
    component text NOT NULL,
    source text NOT NULL,
    q_kvar double precision
);
SELECT create_hypertable('surrogrid.powerflow_reactive_component', 'ts', create_default_indexes => false);
CREATE INDEX idx_powerflow_reactive_run ON surrogrid.powerflow_reactive_component (powerflow_run_id, t_index);

-- Step 4 compact summaries (plain tables; the UNIQUE key is the only index) ----

CREATE TABLE surrogrid.powerflow_summary (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    n_timesteps integer NOT NULL,
    n_voltage_buses integer NOT NULL,
    n_cables integer NOT NULL,
    transformer_s_rated_mva double precision,
    trafo_mean_s_mva double precision,
    trafo_max_s_mva double precision,
    trafo_max_p_mw double precision,
    trafo_max_q_mvar double precision,
    trafo_critical_t_index integer,
    trafo_critical_ts timestamptz,
    trafo_loading_p50_time_percent double precision,
    trafo_loading_p90_time_percent double precision,
    trafo_loading_p95_time_percent double precision,
    trafo_loading_p99_time_percent double precision,
    trafo_loading_max_time_percent double precision,
    trafo_loading_hours_above_100 integer,
    cable_loading_p95_asset_percent double precision,
    cable_hours_above_100_p95_asset double precision,
    voltage_p05_load_bus_hour_pu double precision,
    voltage_hours_below_0_90_p95_asset double precision,
    voltage_hours_above_1_03_p95_asset double precision,
    voltage_hours_above_1_10_p95_asset double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    -- Annual-boundary sensitivity of the peak: a cyclic annual boundary can
    -- concentrate flexible load in the first and last hours of the modeled year.
    boundary_first_24h_max_percent double precision,
    boundary_last_24h_max_percent double precision,
    boundary_outside_24h_max_percent double precision,
    boundary_24h_excess_percent double precision,
    boundary_first_168h_max_percent double precision,
    boundary_last_168h_max_percent double precision,
    boundary_outside_168h_max_percent double precision,
    boundary_168h_excess_percent double precision,
    boundary_overall_max_percent double precision,
    boundary_peak_t_index integer,
    boundary_peak_in_first_24h boolean,
    boundary_peak_in_last_24h boolean,
    boundary_peak_in_first_168h boolean,
    boundary_peak_in_last_168h boolean,
    n_converged_timesteps integer,
    n_failed_timesteps integer,
    CONSTRAINT uq_powerflow_summary_run_stage UNIQUE (powerflow_run_id, stage),
    CONSTRAINT ck_powerflow_summary_stage CHECK (stage IN ('pre', 'post'))
);

CREATE TABLE surrogrid.powerflow_cable_summary (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    cable integer NOT NULL,
    cable_loading_p50_time_percent double precision,
    cable_loading_p90_time_percent double precision,
    cable_loading_p95_time_percent double precision,
    cable_loading_p99_time_percent double precision,
    cable_loading_max_time_percent double precision,
    cable_loading_max_t_index integer,
    cable_loading_hours_above_100 integer,
    cable_max_i_ka double precision,
    cable_parallel double precision,
    cable_installed_capacity_ka double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_powerflow_cable_summary_run_stage_cable UNIQUE (powerflow_run_id, stage, cable),
    CONSTRAINT ck_powerflow_cable_summary_stage CHECK (stage IN ('pre', 'post'))
);

CREATE TABLE surrogrid.powerflow_bus_voltage_summary (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    bus integer NOT NULL,
    voltage_p50_time_pu double precision,
    voltage_p10_time_pu double precision,
    voltage_p05_time_pu double precision,
    voltage_p01_time_pu double precision,
    voltage_min_time_pu double precision,
    voltage_max_time_pu double precision,
    voltage_hours_below_0_90 integer,
    voltage_hours_above_1_03 integer,
    voltage_hours_above_1_10 integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_powerflow_bus_voltage_summary_run_stage_bus UNIQUE (powerflow_run_id, stage, bus),
    CONSTRAINT ck_powerflow_bus_voltage_summary_stage CHECK (stage IN ('pre', 'post'))
);

CREATE TABLE surrogrid.powerflow_transformer_diagnostic (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    diagnostic text NOT NULL,
    point_index integer NOT NULL,
    x_value double precision NOT NULL,
    t_index integer,
    ts timestamptz,
    p_mw double precision,
    q_mvar double precision,
    q_abs_mvar double precision,
    s_mva double precision,
    mean_s_mva double precision,
    max_s_mva double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_powerflow_transformer_diagnostic UNIQUE (powerflow_run_id, stage, diagnostic, point_index),
    CONSTRAINT ck_powerflow_transformer_diagnostic_stage CHECK (stage IN ('pre', 'post'))
);

CREATE TABLE surrogrid.powerflow_tail_value (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    metric text NOT NULL,
    asset_type text NOT NULL,
    asset_id integer NOT NULL,
    tail text NOT NULL,
    threshold_value double precision,
    t_index integer NOT NULL,
    value double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_powerflow_tail_value UNIQUE (powerflow_run_id, stage, metric, asset_type, asset_id, tail, t_index),
    CONSTRAINT ck_powerflow_tail_value_stage CHECK (stage IN ('pre', 'post'))
);

-- Real-grid summaries: no stage CHECK (stages include 'base_electricity' and
-- scenario stages).
CREATE TABLE surrogrid.real_powerflow_summary (
    real_powerflow_run_id bigint NOT NULL REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    n_timesteps integer NOT NULL,
    n_voltage_buses integer NOT NULL,
    n_cables integer NOT NULL,
    n_converged_timesteps integer,
    n_failed_timesteps integer,
    transformer_s_rated_mva double precision,
    trafo_loading_p50_time_percent double precision,
    trafo_loading_p90_time_percent double precision,
    trafo_loading_p95_time_percent double precision,
    trafo_loading_p99_time_percent double precision,
    trafo_loading_max_time_percent double precision,
    trafo_loading_hours_above_100 integer,
    cable_loading_p95_asset_percent double precision,
    cable_hours_above_100_p95_asset double precision,
    voltage_p05_load_bus_hour_pu double precision,
    voltage_hours_below_0_90_p95_asset double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    boundary_first_24h_max_percent double precision,
    boundary_last_24h_max_percent double precision,
    boundary_outside_24h_max_percent double precision,
    boundary_24h_excess_percent double precision,
    boundary_first_168h_max_percent double precision,
    boundary_last_168h_max_percent double precision,
    boundary_outside_168h_max_percent double precision,
    boundary_168h_excess_percent double precision,
    boundary_overall_max_percent double precision,
    boundary_peak_t_index integer,
    boundary_peak_in_first_24h boolean,
    boundary_peak_in_last_24h boolean,
    boundary_peak_in_first_168h boolean,
    boundary_peak_in_last_168h boolean,
    CONSTRAINT uq_real_powerflow_summary_run_stage UNIQUE (real_powerflow_run_id, stage)
);

CREATE TABLE surrogrid.real_powerflow_cable_summary (
    real_powerflow_run_id bigint NOT NULL REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    cable integer NOT NULL,
    cable_loading_p50_time_percent double precision,
    cable_loading_p90_time_percent double precision,
    cable_loading_p95_time_percent double precision,
    cable_loading_p99_time_percent double precision,
    cable_loading_max_time_percent double precision,
    cable_loading_hours_above_100 integer,
    cable_max_i_ka double precision,
    cable_parallel double precision,
    cable_installed_capacity_ka double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_real_powerflow_cable_summary_run_stage_cable UNIQUE (real_powerflow_run_id, stage, cable)
);

CREATE TABLE surrogrid.real_powerflow_bus_voltage_summary (
    real_powerflow_run_id bigint NOT NULL REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    bus integer NOT NULL,
    voltage_p50_time_pu double precision,
    voltage_p10_time_pu double precision,
    voltage_p05_time_pu double precision,
    voltage_p01_time_pu double precision,
    voltage_min_time_pu double precision,
    voltage_hours_below_0_90 integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_real_powerflow_bus_voltage_summary_run_stage_bus UNIQUE (real_powerflow_run_id, stage, bus)
);

CREATE TABLE surrogrid.real_powerflow_tail_value (
    real_powerflow_run_id bigint NOT NULL REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id) ON DELETE CASCADE,
    stage text NOT NULL,
    metric text NOT NULL,
    asset_type text NOT NULL,
    asset_id integer NOT NULL,
    tail text NOT NULL,
    threshold_value double precision,
    t_index integer NOT NULL,
    value double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_real_powerflow_tail_value UNIQUE (real_powerflow_run_id, stage, metric, asset_type, asset_id, tail, t_index)
);

-- Step 5 grid expansion --------------------------------------------------------

-- Cost and planning assumptions (scientific inputs; see docs/expansion).
CREATE TABLE surrogrid.expansion_cost_assumption (
    assumption_key text PRIMARY KEY,
    description text NOT NULL,
    line_parallel_150_eur_per_km double precision NOT NULL DEFAULT 25000.0,
    line_parallel_185_eur_per_km double precision NOT NULL DEFAULT 45000.0,
    line_parallel_240_eur_per_km double precision NOT NULL DEFAULT 70000.0,
    line_reinforcement_150_max_i_ka double precision NOT NULL DEFAULT 0.270,
    line_reinforcement_185_max_i_ka double precision NOT NULL DEFAULT 0.313,
    line_reinforcement_240_max_i_ka double precision NOT NULL DEFAULT 0.357,
    line_existing_duct_share double precision NOT NULL DEFAULT 0.20,
    line_reopen_rural_eur_per_km double precision NOT NULL DEFAULT 90000.0,
    line_reopen_suburban_eur_per_km double precision NOT NULL DEFAULT 100000.0,
    line_reopen_urban_eur_per_km double precision NOT NULL DEFAULT 165000.0,
    transformer_replace_100_eur double precision NOT NULL DEFAULT 28000.0,
    transformer_replace_160_eur double precision NOT NULL DEFAULT 28800.0,
    transformer_replace_250_eur double precision NOT NULL DEFAULT 30000.0,
    transformer_replace_400_eur double precision NOT NULL DEFAULT 33000.0,
    transformer_replace_630_eur double precision NOT NULL DEFAULT 38000.0,
    transformer_replace_800_eur double precision NOT NULL DEFAULT 42000.0,
    transformer_replace_1000_eur double precision NOT NULL DEFAULT 48000.0,
    transformer_station_rebuild_boundary_eur double precision NOT NULL DEFAULT 100000.0,
    transformer_capacity_step_kva integer NOT NULL DEFAULT 50,
    source_note text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO surrogrid.expansion_cost_assumption (
    assumption_key,
    description,
    line_parallel_150_eur_per_km,
    line_parallel_185_eur_per_km,
    line_parallel_240_eur_per_km,
    line_reinforcement_150_max_i_ka,
    line_reinforcement_185_max_i_ka,
    line_reinforcement_240_max_i_ka,
    line_existing_duct_share,
    line_reopen_rural_eur_per_km,
    line_reopen_suburban_eur_per_km,
    line_reopen_urban_eur_per_km,
    transformer_replace_100_eur,
    transformer_replace_160_eur,
    transformer_replace_250_eur,
    transformer_replace_400_eur,
    transformer_replace_630_eur,
    transformer_replace_800_eur,
    transformer_replace_1000_eur,
    transformer_station_rebuild_boundary_eur,
    transformer_capacity_step_kva,
    source_note
)
VALUES (
    'de_lv_heuristic_2026',
    'Simple German brownfield LV expansion screening assumptions based on nominal overloads.',
    25000.0,
    45000.0,
    70000.0,
    0.270,
    0.313,
    0.357,
    0.20,
    90000.0,
    100000.0,
    165000.0,
    28000.0,
    28800.0,
    30000.0,
    33000.0,
    38000.0,
    42000.0,
    48000.0,
    100000.0,
    50,
    'Added LV capacity is selected from NAYY_4_150 (270 A), NAYY_4_185 (313 A), and NAYY_4_240 (357 A). Route costs blend 20% existing-duct cable cost with 80% reopened-route/trenching cost selected by pylovo settlement_type. Transformer costs use all-in replacement bins for 100/160/250/400/630/800/1000 kVA and a 100k EUR station-rebuild boundary case, where the 100/160/250 kVA bins are pylovo equipment costs (3.0k/3.8k/5.0k EUR) plus the 25k EUR installation share implied by the 400/630 kVA all-in bins.'
)
ON CONFLICT (assumption_key) DO NOTHING;

CREATE TABLE surrogrid.expansion_analysis_run (
    expansion_analysis_run_id bigserial PRIMARY KEY,
    analysis_key text NOT NULL UNIQUE,
    assumption_key text NOT NULL REFERENCES surrogrid.expansion_cost_assumption (assumption_key),
    run_name text NOT NULL,
    stage text NOT NULL,
    scenario_id bigint,
    ags bigint,
    plz integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    note text NOT NULL DEFAULT '',
    data_source text NOT NULL DEFAULT 'Synthetic',
    CONSTRAINT fk_expansion_analysis_run_scenario FOREIGN KEY (scenario_id)
        REFERENCES surrogrid.scenario (scenario_id) ON DELETE CASCADE,
    CONSTRAINT ck_expansion_analysis_run_stage CHECK (stage IN ('pre', 'post')),
    CONSTRAINT ck_expansion_analysis_run_data_source CHECK (data_source IN ('Synthetic', 'Real SWF', 'Real ÜZW'))
);
CREATE INDEX idx_expansion_analysis_run_scenario ON surrogrid.expansion_analysis_run (scenario_id);

CREATE TABLE surrogrid.expansion_line_result (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    powerflow_run_id bigint NOT NULL,
    grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    ags bigint NOT NULL,
    plz integer NOT NULL,
    kcid integer NOT NULL,
    bcid integer NOT NULL,
    pylovo_grid_result_id bigint NOT NULL,
    pylovo_version_id varchar(10) NOT NULL,
    visible_line_id bigint NOT NULL,
    visible_line_name text,
    visible_std_type text,
    is_helper boolean,
    helper_type text,
    from_bus integer,
    to_bus integer,
    length_km double precision,
    settlement_type integer,
    line_existing_duct_share double precision,
    line_trenching_share double precision,
    critical_component_parallel integer,
    max_component_line integer,
    max_component_line_name text,
    max_i_from_ka double precision,
    max_i_ka double precision,
    loading_percent double precision,
    required_parallel integer NOT NULL,
    additional_parallel integer NOT NULL,
    reinforcement_150_count integer NOT NULL DEFAULT 0,
    reinforcement_185_count integer NOT NULL DEFAULT 0,
    reinforcement_240_count integer NOT NULL DEFAULT 0,
    reinforcement_added_capacity_ka double precision NOT NULL DEFAULT 0.0,
    reinforcement_catalog text NOT NULL DEFAULT 'NAYY_4_150|NAYY_4_185|NAYY_4_240',
    requires_expansion boolean NOT NULL,
    overloaded_at_100_percent boolean NOT NULL,
    estimated_cost_eur double precision NOT NULL,
    critical_component_cost_eur_per_km double precision,
    critical_component_cost_basis text,
    critical_component_duct_cost_eur_per_km double precision,
    critical_component_reopen_cost_eur_per_km double precision,
    critical_t_index integer,
    critical_ts timestamptz,
    mapped_component_lines integer NOT NULL,
    component_cost_basis_count integer NOT NULL DEFAULT 1,
    component_std_type_count integer NOT NULL DEFAULT 1,
    PRIMARY KEY (expansion_analysis_run_id, powerflow_run_id, visible_line_id),
    CONSTRAINT fk_expansion_line_result_powerflow_run FOREIGN KEY (powerflow_run_id, grid_case_id, scenario_id)
        REFERENCES surrogrid.powerflow_run (powerflow_run_id, grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_expansion_line_result_grid ON surrogrid.expansion_line_result (grid_case_id, powerflow_run_id);
CREATE INDEX idx_expansion_line_result_need ON surrogrid.expansion_line_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);
CREATE INDEX idx_expansion_line_result_powerflow_run ON surrogrid.expansion_line_result (powerflow_run_id);

CREATE TABLE surrogrid.expansion_transformer_result (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    powerflow_run_id bigint NOT NULL,
    grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    ags bigint NOT NULL,
    plz integer NOT NULL,
    kcid integer NOT NULL,
    bcid integer NOT NULL,
    pylovo_grid_result_id bigint NOT NULL,
    pylovo_version_id varchar(10) NOT NULL,
    transformer_rated_power_kva double precision NOT NULL,
    transformer_equipment_name text,
    max_s_mva double precision NOT NULL,
    max_p_mw double precision NOT NULL,
    max_q_mvar double precision NOT NULL,
    loading_percent double precision NOT NULL,
    required_transformer_kva double precision NOT NULL,
    additional_transformer_kva double precision NOT NULL,
    requires_expansion boolean NOT NULL,
    overloaded_at_100_percent boolean NOT NULL,
    estimated_cost_eur double precision NOT NULL,
    transformer_cost_basis text,
    critical_t_index integer NOT NULL,
    critical_ts timestamptz NOT NULL,
    PRIMARY KEY (expansion_analysis_run_id, powerflow_run_id),
    CONSTRAINT fk_expansion_transformer_result_powerflow_run FOREIGN KEY (powerflow_run_id, grid_case_id, scenario_id)
        REFERENCES surrogrid.powerflow_run (powerflow_run_id, grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_expansion_transformer_result_grid ON surrogrid.expansion_transformer_result (grid_case_id, powerflow_run_id);
CREATE INDEX idx_expansion_transformer_result_need ON surrogrid.expansion_transformer_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);
CREATE INDEX idx_expansion_transformer_result_powerflow_run ON surrogrid.expansion_transformer_result (powerflow_run_id);

CREATE TABLE surrogrid.expansion_real_grid_status (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    real_powerflow_run_id bigint NOT NULL,
    real_grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    plz integer,
    lv_id text NOT NULL,
    n_timesteps integer NOT NULL,
    n_failed_timesteps integer NOT NULL,
    cost_status text NOT NULL,
    status_reason text,
    PRIMARY KEY (expansion_analysis_run_id, real_powerflow_run_id),
    CONSTRAINT fk_expansion_real_grid_status_real_powerflow_run FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
        REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE,
    CONSTRAINT ck_expansion_real_grid_status_cost_status CHECK (cost_status IN ('complete', 'incomplete', 'excluded'))
);
CREATE INDEX idx_expansion_real_grid_status_analysis ON surrogrid.expansion_real_grid_status (expansion_analysis_run_id, cost_status, lv_id);
CREATE INDEX idx_expansion_real_grid_status_real_powerflow_run ON surrogrid.expansion_real_grid_status (real_powerflow_run_id);

CREATE TABLE surrogrid.expansion_real_line_result (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    real_powerflow_run_id bigint NOT NULL,
    real_grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    plz integer,
    lv_id text NOT NULL,
    cable integer NOT NULL,
    cable_name text,
    std_type text,
    corridor_cable_ids text NOT NULL,
    corridor_line_count integer NOT NULL DEFAULT 1,
    corridor_grouping_method text NOT NULL DEFAULT 'single_line',
    from_bus integer,
    to_bus integer,
    length_km double precision,
    settlement_type integer,
    line_existing_duct_share double precision,
    line_trenching_share double precision,
    existing_parallel integer NOT NULL,
    max_i_from_ka double precision,
    max_i_ka double precision,
    installed_capacity_ka double precision,
    loading_percent double precision,
    required_parallel integer NOT NULL,
    additional_parallel integer NOT NULL,
    reinforcement_150_count integer NOT NULL DEFAULT 0,
    reinforcement_185_count integer NOT NULL DEFAULT 0,
    reinforcement_240_count integer NOT NULL DEFAULT 0,
    reinforcement_added_capacity_ka double precision NOT NULL DEFAULT 0.0,
    reinforcement_catalog text NOT NULL DEFAULT 'NAYY_4_150|NAYY_4_185|NAYY_4_240',
    requires_expansion boolean NOT NULL,
    overloaded_at_100_percent boolean NOT NULL,
    estimated_cost_eur double precision NOT NULL,
    cost_eur_per_km double precision,
    cost_basis text,
    duct_cost_eur_per_km double precision,
    reopen_cost_eur_per_km double precision,
    critical_t_index integer,
    geom geometry(LineString, 25832),
    PRIMARY KEY (expansion_analysis_run_id, real_powerflow_run_id, cable),
    CONSTRAINT fk_expansion_real_line_result_real_powerflow_run FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
        REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_expansion_real_line_result_need ON surrogrid.expansion_real_line_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);
CREATE INDEX idx_expansion_real_line_result_geom ON surrogrid.expansion_real_line_result USING GIST (geom);
CREATE INDEX idx_expansion_real_line_result_real_powerflow_run ON surrogrid.expansion_real_line_result (real_powerflow_run_id);

CREATE TABLE surrogrid.expansion_real_transformer_result (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    real_powerflow_run_id bigint NOT NULL,
    real_grid_case_id bigint NOT NULL,
    scenario_id bigint NOT NULL,
    plz integer,
    lv_id text NOT NULL,
    transformer_rated_power_kva double precision NOT NULL,
    transformer_equipment_name text,
    max_s_mva double precision NOT NULL,
    loading_percent double precision NOT NULL,
    required_transformer_kva double precision NOT NULL,
    additional_transformer_kva double precision NOT NULL,
    requires_expansion boolean NOT NULL,
    overloaded_at_100_percent boolean NOT NULL,
    estimated_cost_eur double precision NOT NULL,
    transformer_cost_basis text,
    critical_t_index integer,
    geom geometry(Point, 25832),
    PRIMARY KEY (expansion_analysis_run_id, real_powerflow_run_id),
    CONSTRAINT fk_expansion_real_transformer_result_real_powerflow_run FOREIGN KEY (real_powerflow_run_id, real_grid_case_id, scenario_id)
        REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id, real_grid_case_id, scenario_id) ON DELETE CASCADE
);
CREATE INDEX idx_expansion_real_transformer_result_need ON surrogrid.expansion_real_transformer_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);
CREATE INDEX idx_expansion_real_transformer_result_geom ON surrogrid.expansion_real_transformer_result USING GIST (geom);
CREATE INDEX idx_expansion_real_transformer_result_real_powerflow_run ON surrogrid.expansion_real_transformer_result (real_powerflow_run_id);
