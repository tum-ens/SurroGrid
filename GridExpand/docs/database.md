# SurroGrid database (`surrogrid` schema)

DB-backed GridExpand runs store grid identity, scenarios, run metadata, Step 2
audits, Step 4 power-flow results and Step 5 expansion results in the PostgreSQL
schema `surrogrid` (PostGIS + TimescaleDB), next to the `pylovo` schema it reads
grids and buildings from. Credentials come from `GridExpand/.env` (or
`$GRIDEXPAND_ENV_FILE`); the file wins over environment variables.

Code: `src/gridexpand/db/` — `engine` (connection, one cached engine per URL),
`schema` (migrations, views), `grids` (pylovo grids, grid cases), `runs`
(scenario and run rows), `writers` (COPY writers), `maintenance`
(`gridexpand db …`), and the facade `SurroGridDatabase` used by the steps and
notebooks.

The mobility profile pool is not stored in the database (CSV under
`data/statistics/general/mobility_profile_pool/`); the database stores the
vehicles allocated per run.

## Schema definition and migrations

The schema is defined by numbered SQL files in `src/gridexpand/db/sql/`, applied
in order and recorded in `surrogrid.schema_migration`, plus the re-runnable
`views.sql`:

| file | content |
|---|---|
| `0001_baseline.sql` | all tables, keys, indexes, hypertables, the `de_lv_heuristic_2026` cost-assumption row |
| `0002_index_cleanup.sql` | drops indexes of the pre-migration schema that no query uses; adds the foreign-key indexes |
| `0003_constraints.sql` | composite run keys, CHECKs and the RESTRICT key to pylovo, all `NOT VALID`; removes the unused `baseline_static` seed row; fills `expansion_analysis_run.scenario_id` |
| `0004_validate_constraints.sql` | validates the 0003 constraints (no write lock), then drops the single-column keys they replace |
| `views.sql` | `grid_building_bus`, `grid_building_component`, the two QGIS materialized views |

0002–0004 only change databases created by the pre-migration code; on a
database created from 0001 they are no-ops. Both paths end with the same
catalog (checked on the sandbox: identical columns, constraints, indexes,
hypertables and views, except `uq_grid_case_natural`, which
`relink-pylovo` adds to old databases). Never edit an applied migration; add a
new numbered file.

What happens on first use (`ensure_schema()`, once per process):

- **no `surrogrid` tables** (fresh database, Docker): all migrations and the
  views are applied automatically;
- **current `schema_migration`**: missing views are created (only while
  `fk_grid_case_pylovo_grid_result` exists and is validated, see below); done;
- **pre-migration schema** (no `schema_migration`) **or pending migrations**:
  the process stops with `SchemaMigrationRequired` and the command to run.
  Pipeline runs never migrate an existing database.

### Migrating an existing database

1. Stop runners, notebooks and QGIS; take a physical backup (hypertables:
   `pg_dump` needs `timescaledb_pre_restore()` on restore).
2. `gridexpand db migrate --plan` — read-only: shape check of a pre-migration
   schema against `0001_baseline` (every table and column with type and
   `NOT NULL`, leftover columns, the unique keys used by `ON CONFLICT`,
   hypertables), then the plan and the SQL of every pending migration.
3. `gridexpand db migrate --apply` — records version 1 without DDL if the shape
   check passes ("stamp"), then applies each pending migration in its own
   transaction with `lock_timeout` (`--lock-timeout 5s`), then creates or
   updates the views if the pylovo key allows it.
4. If `grid_case` has no validated key to `pylovo.grid_result` (for example
   after the pylovo schema was dropped and regenerated), run
   `gridexpand db relink-pylovo --plan` and `--apply` (below).

Rollback: 0002 dropped these index definitions (recreate them to undo):
`<hypertable>_ts_idx ON (ts DESC)` for the 7 raw tables;
`idx_powerflow_bus_voltage_bus`, `idx_powerflow_demand_bus`,
`idx_powerflow_reactive_bus`, `idx_allocated_demand_bus`,
`idx_allocated_eff_factor_bus` `(bus)`; `idx_powerflow_line_result_line (line)`;
`uq_scenario_key` (unique, `scenario_key`); `idx_{pipeline_run,powerflow_run,
demand_allocation_run}_grid_case (grid_case_id)`; `idx_powerflow_run_scenario
(scenario_id)`; the `(run, stage[, metric|diagnostic])` twins of the summary
unique keys; `idx_powerflow_tail_value_asset (metric, asset_type, asset_id)`;
`idx_electrification_assignment_building (building_objectid)`,
`…_run_technology (run, technology, selected)`;
`idx_allocated_vehicle_run_model (run, model, schedule)`;
`idx_demand_component_audit_component (component_id)`.

## Maintenance commands (never run automatically; dry run by default)

```bash
uv run gridexpand db init-schema                  # create the schema on an empty database
uv run gridexpand db migrate --plan|--apply        # see above
uv run gridexpand db compress --plan|--apply [--table powerflow_bus_voltage ...]
uv run gridexpand db relink-pylovo --plan|--apply [--accept-unverified]
uv run gridexpand db delete-scenario <scenario_key> [--execute] [--keep-demands]
```

- **compress**: enables TimescaleDB compression on the raw hourly tables
  (segment by run and stage, ordered by asset and `t_index`) and compresses
  every chunk in its own transaction; recompresses chunks written after an
  earlier compression. Needs `timescaledb.license = timescale`. Sandbox, one
  week, one grid: bus voltage 8.4 → 1.1 MB, line results 9.9 → 2.9 MB, demand
  4.7 → 1.1 MB, reactive 5.3 → 0.6 MB; results read back identical, and
  re-writing and deleting runs in compressed chunks works. No time-based policy:
  every `ts` lies in 2009, so a policy would compress chunks still being
  written. Run it at the end of a batch.
- **relink-pylovo**: after a pylovo re-generation, `grid_case.pylovo_grid_result_id`
  may point to other grids (new ids). For every grid case the command finds the
  current grid with the same version, plz, kcid and bcid and accepts it only if
  its buildings equal the building set of the grid case's newest Step 2
  component audit (grid cases without runs need no evidence;
  `--accept-unverified` accepts runs without audit). `--apply` updates the
  accepted rows and the copies in `expansion_*_result` in one transaction
  (old ids in `surrogrid.grid_case_relink_backup`), adds
  `uq_grid_case_natural` and `fk_grid_case_pylovo_grid_result` if missing,
  validates the key when no grid case was rejected, then recreates the views
  and refreshes the QGIS views. Rejected grid cases (grid gone, building set
  changed, duplicates) are listed and left unchanged; they belong to a pylovo
  state that no longer exists — delete them (`DELETE FROM surrogrid.grid_case
  WHERE grid_case_id = …` cascades to their runs) or restore that pylovo state.
- **delete-scenario**: counts every table of the scenario tree (Step 2, Step 4,
  real-grid runs, expansion analyses including those without `scenario_id`
  whose results all belong to the scenario, QGIS views) and deletes each run in
  its own transaction. `--keep-demands` keeps the scenario, pipeline and Step 2
  rows and deletes all power-flow (synthetic and real) and expansion results.
  Step 3 files of the scenario are deleted too.

## Run identity

```text
pylovo.grid_result ◄─RESTRICT─ grid_case     scenario
                                  │              │
                                  └──► pipeline_run ◄─┘      (one per grid case and scenario)
                                  ┌────────┴─────────┐
                     demand_allocation_run      powerflow_run ──► expansion_line_result
                          │                         │            expansion_transformer_result
                    Step 2 audits            raw + summary tables      ▲
                                                               expansion_analysis_run
real_grid_case ──► real_powerflow_run (scenario) ──► real summaries, expansion_real_*
```

Run and result rows repeat `grid_case_id` and `scenario_id`; composite foreign
keys to their parent run keep the copies consistent, and every row has exactly
one cascade path (deleting a scenario or grid case deletes its pipeline runs,
their runs and results). Deleting a pylovo grid that SurroGrid results use
fails (`ON DELETE RESTRICT`).

Time stamps: `ts = timeframe_start(run) + t_index hours`, where
`timeframe_start` is taken from the run's `assumptions` (or its scenario's if the
run has none; default 2009-01-01 00:00 UTC). This holds for every raw table,
`powerflow_summary.trafo_critical_ts`, `powerflow_transformer_diagnostic.ts`
and the expansion `critical_ts` columns.

## Tables

### Identity

- `grid_case`: one pylovo LV grid (`ags, plz, kcid, bcid`, `pylovo_grid_result_id`,
  `pylovo_version_id`, readable `cell_id` such as `9278140-00`). Unique by
  `(ags, plz, kcid, bcid, pylovo_grid_result_id)` (the upsert key) and by the
  natural key `(ags, pylovo_version_id, plz, kcid, bcid)`.
- `scenario`: one scenario key (`scenario_key` unique) with scenario-level
  `assumptions` only (timeframe mode, horizon, scenario hash); per-grid facts
  such as the selected week live on the run rows.
- `pipeline_run`: parent of all Step 2 and Step 4 runs of one grid case and
  scenario (`<scenario_key>_pipeline`).
- `real_grid_case`, `real_powerflow_run`: real (DSO) grids by `(source,
  source_file)` and their power-flow runs.

### Step 2 demand allocation

- `demand_allocation_run`: one Step 2 run (`run_name`, `bridge_filename`,
  `profiles`, `mobility_source`, `assumptions`).
- `demand_component_audit`: compact evidence per residential/non-residential
  component (annual energy, profile hash and method, seed, MV-direct and
  suppression reason); hourly component series are not stored.
- `electrification_assignment`: the heat/mobility/PV-battery assignment per
  physical building (eligibility, rank, selection, exclusion reason, seed).
- `allocated_vehicle`: the vehicle (pool profile or emobpy vehicle) per bus.
- `allocated_demand`, `allocated_eff_factor`: hourly `urbs_in/demand` and
  `urbs_in/eff_factor` per bus, written only with
  `--step2-timeseries-storage db|both` (default `temp` writes none).

### Step 4 power flow

- `powerflow_run`: one Step 4 run (`run_name`, `urbs_input_file`, `pre_only`,
  `assumptions`). Raw and summary passes are separate runs.
- Raw hourly results (`--powerflow-output raw|both`; TimescaleDB hypertables on
  `ts`, index `(run, stage, t_index)`): `powerflow_demand` (p_kw, q_kvar per bus),
  `powerflow_import` (p_mw, q_mvar), `powerflow_bus_voltage` (vm_pu per bus),
  `powerflow_line_result` (p_from_mw, q_from_mvar, i_from_ka per line),
  `powerflow_reactive_component` (q_kvar per bus, component, source).
- Summaries (plain tables, one unique key each, `stage` is `pre` or `post`):
  `powerflow_summary` (one row per run and stage: transformer loading
  percentiles, cable/voltage tails, annual-boundary diagnostics),
  `powerflow_cable_summary`, `powerflow_bus_voltage_summary`,
  `powerflow_tail_value`, `powerflow_transformer_diagnostic`; the `real_*`
  summary tables hold the same for real grids (their stages include
  `base_electricity` and scenario stages).

### Step 5 expansion

- `expansion_cost_assumption`: cost and catalogue assumptions
  (`de_lv_heuristic_2026`, see `docs/expansion/assumptions_costs.md`).
- `expansion_analysis_run`: one materialization (`analysis_key` unique,
  `run_name`, `stage`, `data_source` Synthetic / Real SWF / Real ÜZW).
- `expansion_line_result`, `expansion_transformer_result`: per visible pylovo
  line / transformer of each synthetic power-flow run: peak loading, required
  parallel cables or transformer size, estimated cost, critical time step.
- `expansion_real_grid_status`, `expansion_real_line_result`,
  `expansion_real_transformer_result`: the same for real grids (with geometry).

## Views

- `grid_building_bus`: one row per physical building and grid case with its
  pandapower bus (`Consumer Nodebus <vertice_id>`, else the connection point)
  and the loads on that bus (`load_indices`, `load_count`; `load_index` only
  when exactly one load). `SurroGridDatabase.read_buildings` is its pandas twin;
  `read_building_components` asserts that both agree.
- `grid_building_component`: one row per positive Residential or
  Commercial/Public component (`<objectid>::residential`,
  `::<commercial|public>`), MV-direct components with `included_in_lv = false`.
- `expansion_line_qgis_mv`, `expansion_transformer_qgis_mv`: QGIS layers of the
  synthetic expansion results with pylovo geometry; refreshed by
  `refresh_qgis_views()` (after materializations, `delete-scenario`, relink),
  never dropped by schema code.

## Example queries

Pipeline runs and their child runs:

```sql
SELECT pipe.pipeline_run_id, gc.cell_id, gc.plz, sc.scenario_key,
       dar.demand_allocation_run_id, dar.run_name AS demand_allocation_run_name,
       pr.powerflow_run_id, pr.run_name AS powerflow_run_name
FROM surrogrid.pipeline_run pipe
JOIN surrogrid.grid_case gc USING (grid_case_id)
JOIN surrogrid.scenario sc USING (scenario_id)
LEFT JOIN surrogrid.demand_allocation_run dar USING (pipeline_run_id)
LEFT JOIN surrogrid.powerflow_run pr USING (pipeline_run_id)
ORDER BY pipe.created_at DESC;
```

Selected mobility profiles of a Step 2 run:

```sql
SELECT av.bus, av.vehicle_id, av.model, av.schedule, av.profile_id, av.battery_cap_kwh
FROM surrogrid.allocated_vehicle av
JOIN surrogrid.demand_allocation_run dar USING (demand_allocation_run_id)
WHERE dar.run_name = '<scenario_key>_all_pool_demand_allocation'
ORDER BY av.bus, av.vehicle_id;
```

Maximum post-expansion line current of a raw power-flow run:

```sql
SELECT line, MAX(i_from_ka) AS max_i_from_ka
FROM surrogrid.powerflow_line_result plr
JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
WHERE pr.run_name = '<scenario_key>_<profiles>_raw_powerflow' AND plr.stage = 'post'
GROUP BY line
ORDER BY max_i_from_ka DESC;
```
