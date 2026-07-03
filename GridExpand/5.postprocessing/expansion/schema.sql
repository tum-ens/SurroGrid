CREATE TABLE IF NOT EXISTS surrogrid.expansion_cost_assumption (
    assumption_key TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    line_parallel_150_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 25000.0,
    line_parallel_185_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 45000.0,
    line_parallel_240_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 70000.0,
    line_parallel_default_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 70000.0,
    line_reopen_rural_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 90000.0,
    line_reopen_suburban_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 95000.0,
    line_reopen_urban_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 165000.0,
    transformer_replace_400_eur DOUBLE PRECISION NOT NULL DEFAULT 33000.0,
    transformer_replace_630_eur DOUBLE PRECISION NOT NULL DEFAULT 38000.0,
    transformer_replace_800_eur DOUBLE PRECISION NOT NULL DEFAULT 42000.0,
    transformer_replace_1000_eur DOUBLE PRECISION NOT NULL DEFAULT 48000.0,
    transformer_station_rebuild_boundary_eur DOUBLE PRECISION NOT NULL DEFAULT 100000.0,
    transformer_capacity_step_kva INTEGER NOT NULL DEFAULT 50,
    source_note TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_parallel_150_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 25000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_parallel_185_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 45000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_parallel_240_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 70000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_parallel_default_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 70000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_reopen_rural_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 90000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_reopen_suburban_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 95000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS line_reopen_urban_eur_per_km DOUBLE PRECISION NOT NULL DEFAULT 165000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_replace_400_eur DOUBLE PRECISION NOT NULL DEFAULT 33000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_replace_630_eur DOUBLE PRECISION NOT NULL DEFAULT 38000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_replace_800_eur DOUBLE PRECISION NOT NULL DEFAULT 42000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_replace_1000_eur DOUBLE PRECISION NOT NULL DEFAULT 48000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_station_rebuild_boundary_eur DOUBLE PRECISION NOT NULL DEFAULT 100000.0;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption ADD COLUMN IF NOT EXISTS transformer_capacity_step_kva INTEGER NOT NULL DEFAULT 50;

INSERT INTO surrogrid.expansion_cost_assumption (
    assumption_key,
    description,
    line_parallel_150_eur_per_km,
    line_parallel_185_eur_per_km,
    line_parallel_240_eur_per_km,
    line_parallel_default_eur_per_km,
    line_reopen_rural_eur_per_km,
    line_reopen_suburban_eur_per_km,
    line_reopen_urban_eur_per_km,
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
    70000.0,
    90000.0,
    95000.0,
    165000.0,
    33000.0,
    38000.0,
    42000.0,
    48000.0,
    100000.0,
    50,
    'Default line costs use direct brownfield parallel-cable values in existing route/duct by cable size. Reopened-route values are stored as sensitivity/context. Transformer costs use all-in replacement bins for 400/630/800/1000 kVA and a 100k EUR station-rebuild boundary case.'
)
ON CONFLICT (assumption_key) DO UPDATE SET
    description = EXCLUDED.description,
    line_parallel_150_eur_per_km = EXCLUDED.line_parallel_150_eur_per_km,
    line_parallel_185_eur_per_km = EXCLUDED.line_parallel_185_eur_per_km,
    line_parallel_240_eur_per_km = EXCLUDED.line_parallel_240_eur_per_km,
    line_parallel_default_eur_per_km = EXCLUDED.line_parallel_default_eur_per_km,
    line_reopen_rural_eur_per_km = EXCLUDED.line_reopen_rural_eur_per_km,
    line_reopen_suburban_eur_per_km = EXCLUDED.line_reopen_suburban_eur_per_km,
    line_reopen_urban_eur_per_km = EXCLUDED.line_reopen_urban_eur_per_km,
    transformer_replace_400_eur = EXCLUDED.transformer_replace_400_eur,
    transformer_replace_630_eur = EXCLUDED.transformer_replace_630_eur,
    transformer_replace_800_eur = EXCLUDED.transformer_replace_800_eur,
    transformer_replace_1000_eur = EXCLUDED.transformer_replace_1000_eur,
    transformer_station_rebuild_boundary_eur = EXCLUDED.transformer_station_rebuild_boundary_eur,
    transformer_capacity_step_kva = EXCLUDED.transformer_capacity_step_kva,
    source_note = EXCLUDED.source_note,
    updated_at = NOW();

ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption DROP COLUMN IF EXISTS line_target_loading_percent;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption DROP COLUMN IF EXISTS transformer_target_loading_percent;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption DROP COLUMN IF EXISTS line_cost_eur_per_km;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption DROP COLUMN IF EXISTS transformer_cost_eur_per_kva;
ALTER TABLE IF EXISTS surrogrid.expansion_cost_assumption DROP COLUMN IF EXISTS transformer_fixed_replacement_eur;

CREATE TABLE IF NOT EXISTS surrogrid.expansion_analysis_run (
    expansion_analysis_run_id BIGSERIAL PRIMARY KEY,
    analysis_key TEXT NOT NULL UNIQUE,
    assumption_key TEXT NOT NULL REFERENCES surrogrid.expansion_cost_assumption(assumption_key),
    run_name TEXT NOT NULL,
    stage TEXT NOT NULL,
    scenario_id BIGINT REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    ags BIGINT,
    plz INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    note TEXT NOT NULL DEFAULT ''
);

DROP MATERIALIZED VIEW IF EXISTS surrogrid.expansion_line_qgis_mv;
DROP MATERIALIZED VIEW IF EXISTS surrogrid.expansion_transformer_qgis_mv;
DROP VIEW IF EXISTS surrogrid.expansion_line_qgis;
DROP VIEW IF EXISTS surrogrid.expansion_transformer_qgis;

CREATE TABLE IF NOT EXISTS surrogrid.expansion_line_result (
    expansion_analysis_run_id BIGINT NOT NULL REFERENCES surrogrid.expansion_analysis_run(expansion_analysis_run_id) ON DELETE CASCADE,
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    ags BIGINT NOT NULL,
    plz INTEGER NOT NULL,
    kcid INTEGER NOT NULL,
    bcid INTEGER NOT NULL,
    pylovo_grid_result_id BIGINT NOT NULL,
    pylovo_version_id VARCHAR(10) NOT NULL,
    visible_line_id BIGINT NOT NULL,
    visible_line_name TEXT,
    visible_std_type TEXT,
    is_helper BOOLEAN,
    helper_type TEXT,
    from_bus INTEGER,
    to_bus INTEGER,
    length_km DOUBLE PRECISION,
    existing_parallel INTEGER,
    max_component_line INTEGER,
    max_component_line_name TEXT,
    max_i_from_ka DOUBLE PRECISION,
    max_i_ka DOUBLE PRECISION,
    loading_percent DOUBLE PRECISION,
    required_parallel INTEGER NOT NULL,
    additional_parallel INTEGER NOT NULL,
    requires_expansion BOOLEAN NOT NULL,
    overloaded_at_100_percent BOOLEAN NOT NULL,
    estimated_cost_eur DOUBLE PRECISION NOT NULL,
    line_cost_eur_per_km DOUBLE PRECISION,
    line_cost_basis TEXT,
    critical_t_index INTEGER,
    critical_ts TIMESTAMPTZ,
    mapped_component_lines INTEGER NOT NULL,
    PRIMARY KEY (expansion_analysis_run_id, powerflow_run_id, visible_line_id)
);

ALTER TABLE IF EXISTS surrogrid.expansion_line_result DROP COLUMN IF EXISTS target_loading_percent;
ALTER TABLE IF EXISTS surrogrid.expansion_line_result ADD COLUMN IF NOT EXISTS line_cost_eur_per_km DOUBLE PRECISION;
ALTER TABLE IF EXISTS surrogrid.expansion_line_result ADD COLUMN IF NOT EXISTS line_cost_basis TEXT;

CREATE INDEX IF NOT EXISTS idx_expansion_line_result_grid
    ON surrogrid.expansion_line_result (grid_case_id, powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_line_result_need
    ON surrogrid.expansion_line_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);

CREATE TABLE IF NOT EXISTS surrogrid.expansion_transformer_result (
    expansion_analysis_run_id BIGINT NOT NULL REFERENCES surrogrid.expansion_analysis_run(expansion_analysis_run_id) ON DELETE CASCADE,
    powerflow_run_id BIGINT NOT NULL REFERENCES surrogrid.powerflow_run(powerflow_run_id) ON DELETE CASCADE,
    grid_case_id BIGINT NOT NULL REFERENCES surrogrid.grid_case(grid_case_id) ON DELETE CASCADE,
    scenario_id BIGINT NOT NULL REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE,
    ags BIGINT NOT NULL,
    plz INTEGER NOT NULL,
    kcid INTEGER NOT NULL,
    bcid INTEGER NOT NULL,
    pylovo_grid_result_id BIGINT NOT NULL,
    pylovo_version_id VARCHAR(10) NOT NULL,
    transformer_rated_power_kva DOUBLE PRECISION NOT NULL,
    transformer_equipment_name TEXT,
    max_s_mva DOUBLE PRECISION NOT NULL,
    max_p_mw DOUBLE PRECISION NOT NULL,
    max_q_mvar DOUBLE PRECISION NOT NULL,
    loading_percent DOUBLE PRECISION NOT NULL,
    required_transformer_kva DOUBLE PRECISION NOT NULL,
    additional_transformer_kva DOUBLE PRECISION NOT NULL,
    requires_expansion BOOLEAN NOT NULL,
    overloaded_at_100_percent BOOLEAN NOT NULL,
    estimated_cost_eur DOUBLE PRECISION NOT NULL,
    transformer_cost_basis TEXT,
    critical_t_index INTEGER NOT NULL,
    critical_ts TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (expansion_analysis_run_id, powerflow_run_id)
);

ALTER TABLE IF EXISTS surrogrid.expansion_transformer_result DROP COLUMN IF EXISTS target_loading_percent;
ALTER TABLE IF EXISTS surrogrid.expansion_transformer_result ADD COLUMN IF NOT EXISTS transformer_cost_basis TEXT;

CREATE INDEX IF NOT EXISTS idx_expansion_transformer_result_grid
    ON surrogrid.expansion_transformer_result (grid_case_id, powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_result_need
    ON surrogrid.expansion_transformer_result (expansion_analysis_run_id, requires_expansion, overloaded_at_100_percent);

ALTER TABLE IF EXISTS surrogrid.expansion_analysis_run DROP CONSTRAINT IF EXISTS expansion_analysis_run_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.expansion_analysis_run DROP CONSTRAINT IF EXISTS fk_expansion_analysis_run_scenario;
ALTER TABLE IF EXISTS surrogrid.expansion_analysis_run
    ADD CONSTRAINT fk_expansion_analysis_run_scenario
    FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS surrogrid.expansion_line_result DROP CONSTRAINT IF EXISTS expansion_line_result_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.expansion_line_result DROP CONSTRAINT IF EXISTS fk_expansion_line_result_scenario;
ALTER TABLE IF EXISTS surrogrid.expansion_line_result
    ADD CONSTRAINT fk_expansion_line_result_scenario
    FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS surrogrid.expansion_transformer_result DROP CONSTRAINT IF EXISTS expansion_transformer_result_scenario_id_fkey;
ALTER TABLE IF EXISTS surrogrid.expansion_transformer_result DROP CONSTRAINT IF EXISTS fk_expansion_transformer_result_scenario;
ALTER TABLE IF EXISTS surrogrid.expansion_transformer_result
    ADD CONSTRAINT fk_expansion_transformer_result_scenario
    FOREIGN KEY (scenario_id) REFERENCES surrogrid.scenario(scenario_id) ON DELETE CASCADE;

CREATE MATERIALIZED VIEW IF NOT EXISTS surrogrid.expansion_line_qgis_mv AS
SELECT
    ROW_NUMBER() OVER (
        ORDER BY
            ar.analysis_key,
            elr.powerflow_run_id,
            elr.visible_line_id
    )::BIGINT AS qgis_id,
    ar.analysis_key,
    ar.assumption_key,
    elr.*,
    lv.geom::geometry(LineString, 25832) AS geom
FROM surrogrid.expansion_line_result elr
JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
JOIN pylovo.lines_result_view lv
  ON lv.grid_result_id = elr.pylovo_grid_result_id
 AND lv.version_id = elr.pylovo_version_id
 AND lv.plz = elr.plz
 AND lv.kcid = elr.kcid
 AND lv.bcid = elr.bcid
 AND lv.id = elr.visible_line_id
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_qgis_id
    ON surrogrid.expansion_line_qgis_mv (qgis_id);
CREATE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_analysis
    ON surrogrid.expansion_line_qgis_mv (analysis_key, requires_expansion, overloaded_at_100_percent);
CREATE INDEX IF NOT EXISTS idx_expansion_line_qgis_mv_geom
    ON surrogrid.expansion_line_qgis_mv USING GIST (geom);

CREATE MATERIALIZED VIEW IF NOT EXISTS surrogrid.expansion_transformer_qgis_mv AS
SELECT
    ROW_NUMBER() OVER (
        ORDER BY
            ar.analysis_key,
            etr.powerflow_run_id
    )::BIGINT AS qgis_id,
    ar.analysis_key,
    ar.assumption_key,
    etr.*,
    tpwg.osm_id,
    tpwg.comment,
    tpwg.s_max_kva AS equipment_s_max_kva,
    tpwg.cost_eur AS equipment_cost_eur,
    tpwg.equipment_type,
    tpwg.geom::geometry(Point, 25832) AS geom
FROM surrogrid.expansion_transformer_result etr
JOIN surrogrid.expansion_analysis_run ar USING (expansion_analysis_run_id)
JOIN pylovo.transformer_positions_with_grid tpwg
  ON tpwg.grid_result_id = etr.pylovo_grid_result_id
 AND tpwg.version_id = etr.pylovo_version_id
 AND tpwg.plz = etr.plz
 AND tpwg.kcid = etr.kcid
 AND tpwg.bcid = etr.bcid
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_qgis_id
    ON surrogrid.expansion_transformer_qgis_mv (qgis_id);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_analysis
    ON surrogrid.expansion_transformer_qgis_mv (analysis_key, requires_expansion, overloaded_at_100_percent);
CREATE INDEX IF NOT EXISTS idx_expansion_transformer_qgis_mv_geom
    ON surrogrid.expansion_transformer_qgis_mv USING GIST (geom);

