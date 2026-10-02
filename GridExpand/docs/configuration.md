# Configuration reference

GridExpand reads two kinds of YAML files, both strictly validated (unknown keys, wrong types, invalid ranges,
unknown model cases and missing required fields fail while loading):

| file | directory | owns | fingerprint |
|---|---|---|---|
| scenario YAML | `config/scenarios/` | scientific and policy assumptions that change results | `scenario_hash` |
| run YAML | `config/runs/` | the input (region, pylovo version, prepared datasets), model cases, seed, resources | `run_hash` |

Both hashes are the SHA-256 of the parsed YAML (sorted-key JSON): every key and value counts, comments and
formatting do not. The scenario identity (scenario key) is `scenario_<scenario.id>_<scenario_hash[:12]>`, with
`_<timeframe_mode>` appended for the one-week timeframes. It names the Step 2/3 result directories
(`work/allocation/results/<key>/`, `work/optimization/result/<key>/`), the database `scenario` row and the
power-flow and expansion run names. Step 3 refuses a Step 2 file of another scenario hash. The profile seed is a
run-YAML value: it is recorded in every run (and guarded on resume) but is not part of the scenario key.

```bash
uv run gridexpand config check config/scenarios config/runs     # no database: validity, hashes, keys, groups
```

The scientific reasoning behind the values is in [method.md](method.md). A short checklist for new files is in
[config/README.md](../config/README.md). The scenario editor of the GridPlanner UI ([api.md](api.md#scenario-editor))
saves edited copies of these files, with their own `scenario.id`, into its user scenario directory; it never
changes the shipped files.

## Scenario YAML

Top-level blocks: `scenario`, `economics`, `asset_sizing`, `electrification`, `mobility`, `technologies`,
`time_aggregation`; all are required, and every listed key is required unless marked optional.

| key | type / allowed values | meaning |
|---|---|---|
| `scenario.id` | string | stable scientific-case identifier (part of the scenario key) |
| `scenario.milestone_year` | positive integer | modelled year (paired preparation `--final-year`) |
| `economics.electricity.import_price_eur_per_kwh` | ≥ 0 | electricity import price |
| `economics.electricity.pv_feed_in_tariff_eur_per_kwh` | ≥ 0 | PV feed-in tariff (0 keeps unremunerated export) |
| `asset_sizing.pv.heuristic_method` | `annual_electricity_rule`, `optimization` | PV sizing of the heuristic cases |
| `asset_sizing.pv.optimized_method` | same | PV sizing of `post-hems-optimized` |
| `asset_sizing.pv.demand_multiplier` | > 0 | kWp per MWh/a of base electricity (heuristic rule) |
| `asset_sizing.pv.fallback_capacity_kwp` | > 0 | fallback section for buildings without usable LoD2 roof |
| `asset_sizing.pv.maximum_fallback_share` | 0–1 | largest allowed share of fallback buildings |
| `asset_sizing.pv.module_capacity_kw_per_m2` | > 0 | module peak power per roof area |
| `asset_sizing.pv.flat_roof_utilization`, `slanted_roof_utilization` | (0, 1] | usable roof-area fractions |
| `asset_sizing.pv.tilt_bin_degrees`, `azimuth_bin_degrees` | > 0 | pvlib profile bins |
| `asset_sizing.battery.heuristic_method` | `htw_2025_scaled_rule` | only allowed value |
| `asset_sizing.battery.optimized_method` | `optimization` | only allowed value |
| `asset_sizing.battery.minimum_pv_kwp_per_annual_mwh` | > 0 | eligibility threshold of the battery rule |
| `asset_sizing.battery.heuristic_usable_kwh_per_pv_kwp`, `heuristic_usable_kwh_per_annual_mwh` | (0, 1.5] | heuristic coefficients |
| `asset_sizing.battery.optimized_upper_kwh_per_pv_kwp`, `optimized_upper_kwh_per_annual_mwh` | (0, 1.5] | upper bounds of the optimized case |
| `asset_sizing.battery.energy_to_power_hours` | > 0 | E/P ratio |
| `asset_sizing.heat.space_heat_source` | `teaser`, `infdb_ro_heat`, `internal` | fixed space-heat source or internal temperature model |
| `asset_sizing.heat.heated_area_fraction` | 0.8 for `internal` | effective thermal zone; already embedded in imported RC |
| `asset_sizing.heat.internal.minimum_temperature_c` | finite (default 20) | thermostat and HEMS comfort minimum |
| `asset_sizing.heat.internal.hems_preheat_uplift_k` | ≥ 0 (default 2) | permitted active preheating above the minimum |
| `asset_sizing.heat.indoor_design_temperature_c` | > heating limit | degree-day indoor temperature |
| `asset_sizing.heat.heating_limit_temperature_c` | > 0 | degree-day heating limit |
| `asset_sizing.heat.heat_pump_design_share` | (0, 1] | heat-pump share of the design load |
| `asset_sizing.heat.buffer_volume_l_per_kw_th` | > 0 | buffer litres per kW<sub>th</sub> |
| `asset_sizing.heat.buffer_usable_temperature_spread_k` | > 0 | usable buffer spread |
| `asset_sizing.heat.teaser_retrofit_level` | optional, 0/1/2 (default 0) | TABULA variant of the `teaser` source |
| `electrification.<heat\|mobility\|pv_battery>.adoption_mode` | `deterministic_share`, `source_inventory` | selection rule per technology |
| `electrification.<…>.building_share` | 0–1; required for `deterministic_share`, forbidden otherwise | selected share of eligible buildings |
| `mobility.commuting_probability` | 0–1 | emobpy commuter share |
| `mobility.emobpy_timestep_hours` | > 0 | emobpy resolution |
| `mobility.reference_year` | must be 2009 | the fixed calendar year of every time axis |
| `mobility.passenger_mass_kg`, `passenger_sensible_heat_w`, `passengers_per_vehicle` | > 0 (heat ≥ 0) | emobpy consumption model |
| `mobility.cabin_heat_transfer_coefficient`, `cabin_air_flow_m3_per_s` | > 0 (air flow ≥ 0) | emobpy cabin model |
| `mobility.driving_cycle_type` | `WLTC`, `EPA` | emobpy driving cycle |
| `mobility.road_type` | ≥ 0 integer | emobpy road type |
| `mobility.road_slope` | number | emobpy road slope |
| `technologies.processes.<name>.*` | numbers or null | urbs process parameters, see below |
| `technologies.storages.<name>.*` | numbers or null | urbs storage parameters, see below |
| `time_aggregation.enabled` | boolean | TSAM on/off (Step 3; `--tsam` overrides to on) |
| `time_aggregation.number_of_typical_periods`, `hours_per_period` | positive integers | TSAM periods |
| `time_aggregation.extreme_period_method` | `append`, `new_cluster_center`, `replace_cluster_center` | extreme-period handling |
| `time_aggregation.clustering_method`, `cluster_representation` | strings (e.g. `hierarchical`, `medoid`) | passed to tsam |
| `time_aggregation.segmentation`, `rescale_cluster_periods` | booleans | passed to tsam |
| `time_aggregation.feature_weights` | non-empty mapping of positive numbers | TSAM features (`Tamb`, `Irradiation`) |
| `time_aggregation.extreme_features` | list of strings | e.g. `minimum_mean_temperature`, `maximum_mean_irradiation` |

The TSAM fields are required even when `enabled: false` (then they are inactive).

**`technologies.processes`** must define exactly `rooftop_pv`, `heatpump_air`, `heatpump_booster`,
`heat_dummy`, `home_charger`, `grid_connection`, each with all of: `installed_capacity_kw` (urbs `inst-cap`),
`capacity_upper_kw` (`cap-up`), `fixed_investment_cost_eur` (`inv-cost-fix`), `investment_cost_eur_per_kw`
(`inv-cost`), `fixed_cost_eur_per_hour` (`fix-cost`), `variable_cost_eur_per_kwh` (`var-cost`), `wacc`,
`depreciation_years` (`depreciation`), `minimum_power_factor` (`pf-min`).

**`technologies.storages`** must define exactly `stationary_battery`, `thermal_storage`, `mobility_storage`,
each with all of: `installed_energy_kwh`, `capacity_upper_kwh`, `installed_power_kw`, `power_upper_kw`,
`energy_to_power_hours`, `charge_efficiency` (`eff-in`), `discharge_efficiency` (`eff-out`),
`self_discharge_per_timestep` (`discharge`), `investment_cost_eur_per_kw` (`inv-cost-p`),
`investment_cost_eur_per_kwh` (`inv-cost-c`), `fixed_investment_cost_power_eur` (`fix-cost-p`),
`fixed_investment_cost_energy_eur` (`fix-cost-c`), `variable_cost_eur_per_kwh` (`var-cost-p`), `wacc`,
`depreciation_years`. Fixed (heuristic) assets are written without investment costs whatever these values are.

Repository scenarios: `forchheim_2045_synthetic.yaml` and `schweinfurt_2045.yaml` (synthetic runs),
`forchheim_2045_full_year.yaml` (paired SWF runs), `joint_2045_full_year.yaml` (aligned SWF + ÜZW runs),
`00_scenario_template.yaml`; their differences are tabulated at the top of [method.md](method.md).

## Run YAML

Three blocks, all required: `run`, `resources`, `execution`.

| key | meaning |
|---|---|
| `run.id` | directory-safe identifier (`[A-Za-z0-9][A-Za-z0-9._-]*`); default run directory `work/runs/<run.id>/` |
| `run.scenario` | scenario YAML, relative to the run YAML's directory |
| `run.pipeline` | `synthetic`, `paired_validation` or `paired_aligned` (the old name `scenario` is refused with a hint) |

Relative paths resolve against the YAML's directory; booleans must be YAML booleans (`"false"` is an error).
`execution.profile_seed` (all pipelines, default 481527) is the realization seed, and
`execution.powerflow_grid_scope` (`full` default, or `backbone`) selects the assets of the power-flow summary
statistics (`backbone` drops terminal service lines and maps terminal voltages one bus upstream).

### `pipeline: synthetic`

Steps 2–4 and the expansion analyses for the synthetic pylovo grids of one region, one `gridexpand synthetic`
batch per execution group (see [method.md](method.md#model-cases)). Outputs are always case-qualified.

| `resources` key | default | meaning |
|---|---|---|
| `pylovo_version_id` | required | exact pylovo topology version (never taken from `.env`) |
| `ags` | required | municipality key, e.g. `9184137` |
| `plz` | null | only the grids of this postcode |
| `kcid`, `bcid` | null | one exact grid (both together, and `plz`) |
| `min_buildings` | 5 | minimum buildings per candidate grid (changes the candidate numbering) |
| `start_index`, `limit` | null | a range of candidate numbers |
| `storage` | `db` | must be `db` |
| `output_directory` | null | must be null |

`null`, `""` and `"-"` mean "any" for `plz`, `kcid`, `bcid`, `start_index`.

| `execution` key | default | meaning (`gridexpand synthetic` flag) |
|---|---|---|
| `model_cases` | required | `pre`, `post-hems-heuristic`, `post-hems-optimized` (`post-inflex-heuristic` is refused) |
| `timeframe_mode` | `full_year` | `full_year`, `min_temperature_week`, `max_solar_radiation_week`, `max_base_electricity_demand_week` |
| `demand_scope` | `all` | `all` or `residential` (household-only Step 2–4, `_hh_only` keys) |
| `mobility_source` | `pool` | must be `pool` |
| `step2_cpus` (alias `n_cpu`) | 4 | Step 2 processes (`--step2-cpus`) |
| `step2_timeseries_storage` | `temp` | `db`, `temp` or `both`: also store the Step 2 hourly series in the database |
| `workers` | 1 | grids in parallel |
| `step3_cpus` | 16 | minimum number of Step 3 building clusters (partitions; changes results) |
| `step3_max_cpus` | 32 | maximum number of clusters |
| `step3_target_columns` | 35 | demand/efficiency columns per cluster that the dynamic choice aims at |
| `dynamic_step3` | true | choose the cluster count per grid from 4/8/12/16/24/32 (at least `step3_cpus`, at most `step3_max_cpus`); false: always `step3_cpus` |
| `step3_cluster_concurrency` | null | Step 3 models solved at the same time; null: 1 for urbs, automatic for pypsa (CPUs / 4, at most one per 2 GB of available memory: 8 on the development VM) |
| `step4_cpus` | 4 | Step 4 time chunks in parallel |
| `solver` | null | `gurobi` or `appsi_highs` (null: `$GRIDEXPAND_SOLVER`, else `gurobi`) |
| `optimizer` | null | Step 3 optimizer `urbs` or `pypsa` (null: `$GRIDEXPAND_OPTIMIZER`, else `urbs`); `pypsa` ignores the cluster keys above (one model per building), see [Step 3](steps/3_urbs.md#pypsa-optimizer) |
| `powerflow_output` | `summary` | `summary`, `raw` or `both` |
| `cleanup_intermediates` | `never` | `success`: delete a grid's hand-off HDF5 files after its Step 4 validation |
| `materialize_expansion` | true | expansion analyses at the end of each batch |
| `pilot_gate`, `pilot_index` | true, 0 | stop the batch if the pilot grid fails |
| `resume`, `rerun_failed` | false | skip grids already done; with resume, retry failed ones |

### `pipeline: paired_validation`

SWF real grids versus the synthetic grids of one prepared paired dataset ([paired_validation.md](paired_validation.md)).

| key | default | meaning |
|---|---|---|
| `resources.ags`, `resources.plz` | required | region (preparation registers every eligible grid of that pylovo version) |
| `resources.pylovo_version_id` | required | must equal the version recorded in the prepared dataset |
| `resources.min_buildings` | 5 | symmetric minimum retained buildings per grid |
| `resources.paired_dataset_id` | required | dataset directory `work/allocation/outputs/scenario_calibration/<id>/` |
| `resources.heat_profile_set_id` | required | heat library `…/scenario_calibration/profile_libraries/<id>.h5` |
| `resources.weather_source_hdf` | required | weather file name in `work/allocation/results/` |
| `resources.excluded_real_lv_ids` | empty | real grids kept in coverage but excluded from costing |
| `resources.target_network` | `both` | `both`, `real_swf` or `synthetic` |
| `resources.target_grid_id` | null | diagnostic single-grid filter |
| `execution.model_cases` | required | post cases only (`pre` is added to the first group) |
| `execution.workers`, `step3_cpus`, `step4_cpus` | 1 each | resources |
| `execution.step3_cluster_concurrency` | null | Step 3 models solved at the same time; null: 1 for urbs, automatic for pypsa |
| `execution.cleanup_intermediates`, `resume` | false | lifecycle |
| `execution.materialize_expansion` | true | expansion analyses `<run.id>[_real]_<case suffix>` after success |
| `execution.optimizer` | null | Step 3 optimizer `urbs` or `pypsa` for all jobs of the run (null: `$GRIDEXPAND_OPTIMIZER`, else `urbs`) |

The SWF export root is the `.env` variable `GRID_DATA_PATH` (a deployment setting, not a run choice); a station
dataset must contain `station_split_manifest.csv` and `station_radialization_manifest.csv`.

### `pipeline: paired_aligned`

Every provider (SWF, ÜZW) of a pylovo alignment bundle, each against its synthetic counterpart.

| key | default | meaning |
|---|---|---|
| `resources.pylovo_version_id` | required | pylovo version of the alignment bundle |
| `resources.alignment_dir` | required | pylovo alignment bundle |
| `resources.population` | required | pylovo comparison JSON (metric cohort) |
| `resources.uzw_grids_dir` | required with `uzw` | ÜZW grid delivery (fingerprint checked against the bundle) |
| `resources.providers.<swf\|uzw>.paired_dataset_id` | required | dataset directory per provider |
| `resources.providers.<…>.weather_source_hdf` | required | weather file name in `work/allocation/results/` |
| `resources.providers.<…>.workers` | `execution.workers` | grids in parallel for this provider |
| `resources.target_network` | `both` | `both`, `real` or `synthetic` |
| `execution.model_cases` | required | post cases only |
| `execution.workers`, `step3_cpus`, `step4_cpus` | 1 each | resources |
| `execution.step3_cluster_concurrency` | null | Step 3 models solved at the same time; null: 1 for urbs, automatic for pypsa |
| `execution.heat_workers` | 4 | parallel TEASER heat regeneration jobs |
| `execution.powerflow_max_timesteps` | null | smoke-test cap (not for publication) |
| `execution.cleanup_intermediates`, `resume`, `parallel_providers` | false | lifecycle; providers in parallel |
| `execution.materialize_expansion` | false | expansion analyses of all provider groups after success |
| `execution.grid_subset` | null | `{method: components\|islands, seed, real_grids_per_provider}` |
| `execution.optimizer` | null | Step 3 optimizer `urbs` or `pypsa` for all jobs of the run (null: `$GRIDEXPAND_OPTIMIZER`, else `urbs`) |

The heat library of a provider is derived: `<paired_dataset_id>_teaser_heat_<scenario_hash[:12]>`.
`grid_subset` picks whole pylovo overlap components (`components`, in seeded random order up to
`real_grids_per_provider` real grids) or only one-real-one-synthetic components (`islands`).

## Grid selection

Synthetic runs take one region selector and optional filters; the user never enters a file name or a
candidate index directly:

- `ags` only: all eligible grids of the AGS;
- `ags` + `plz`: all eligible grids of that postcode;
- `ags` + `plz` + `kcid` + `bcid`: one exact grid.

"Eligible" means at least `min_buildings` buildings (with `demand_scope: residential`, residential buildings)
in pylovo version `pylovo_version_id`. The selected region is also the population of the regional
electrification assignment. Candidate numbers stay those of the whole AGS (`uv run gridexpand grids --ags
<AGS> --pylovo-version-id <V>` lists them); `start_index`/`limit` select a range of them without changing the
assignment population. Internally a grid is named `<AGS>-<candidate index>_<PLZ>_<KCID>_<BCID>.h5`.

## Running a run YAML

```bash
uv run gridexpand config check config/runs/<run>.yaml                # validity, hashes, groups; no database
uv run gridexpand run config/runs/<run>.yaml --dry-run               # plan and commands; no database, no files
uv run gridexpand run config/runs/<run>.yaml                         # prepare -> execute -> postprocess
uv run gridexpand status <run.id>                                    # reads work/runs/<run.id>/state.json
uv run gridexpand run config/runs/<run>.yaml --resume                # skip jobs recorded as done
```

`--until prepare|execute|postprocess` (`--prepare-only` = `--until prepare`) stops early, `--model-case` (repeatable)
runs a subset of the YAML's cases, and `--run-dir` overrides the run directory. Paired pipelines add
`--skip-prepare` (use the prepared datasets as they are, validated) and `--target-grid-id` (diagnostic single
grid, own sub-directory); aligned runs add `--provider swf|uzw` and `--pre-only`. `gridexpand run-aligned` is the
older name of the same command. Exit codes: 0 done, 1 failed, 2 finished with failed jobs, 3 invalid
configuration, 143 cancelled (SIGTERM or Ctrl-C stops the running steps and marks their jobs cancelled).

The run directory holds `inputs/run.yaml` and `inputs/scenario.yaml` (frozen copies), `identity.json` (what a
resumed run must share: pipeline, run id, pylovo version, region or datasets, scope, timeframe, seed, output
settings, scenario hash; a mismatch is refused), `state.json` (atomic snapshot: status, stage, jobs),
`events.jsonl`, `plan.json`, `summary.json` and one directory per job group. Synthetic runs add `candidates.json`
and `prepare/` (the regional electrification assignment `electrification_assignment.csv` with its sidecar and
log); each synthetic group contains the batch's `status.tsv`, `events.jsonl`, `logs/candidate_<index>_<file>.log`
and `summary.json`.

## Environment and `.env`

`GridExpand/.env` (template `.env.example`) holds `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and
optionally `GRID_DATA_PATH` (SWF exports), `PYLOVO_VERSION_ID` (only for the Step 1 notebooks and single-step
commands without `--pylovo-version-id`; run YAMLs never read it), `GUROBI_HOME` and `GRB_LICENSE_FILE`. The file
is loaded with override: its values win over variables of the same name in the process environment.

| variable (process environment) | effect |
|---|---|
| `GRIDEXPAND_ENV_FILE` | use another credentials file instead of `GridExpand/.env` |
| `GRIDEXPAND_WORK_DIR` | relocate `work/` (runtime artifacts) |
| `GRIDEXPAND_DATA_DIR` | relocate `data/` (static inputs) |
| `GRIDEXPAND_SOLVER` | default Step 3 solver (`gurobi` or `appsi_highs`) |
| `GRIDEXPAND_OPTIMIZER` | default Step 3 optimizer (`urbs` or `pypsa`); `gridexpand run` sets it from `execution.optimizer` |
| `URBS_CLUSTER_CONCURRENCY` | default of `gridexpand optimize --cluster-concurrency` |
| `GRIDEXPAND_API_SCENARIO_DIRS`, `GRIDEXPAND_API_USER_SCENARIO_DIR`, `GRIDEXPAND_API_CORS_ORIGINS` | API, see [api.md](api.md) |

The `GRIDEXPAND_*_DIR`/`_FILE` variables are read from the process environment when `gridexpand.paths` is first
imported (not from `.env`); child processes of the orchestrators inherit them.

Internal heat is configured under `asset_sizing.heat`. Select
`space_heat_source: internal`, `heated_area_fraction: 0.8` and an `internal` block.
Its other keys (ventilation, zone height, gains, glazing, blinds) default to the
values in [method.md](method.md#internal-space-heat-and-building-preheating).
`parameter_source: infdb_rc` and `refurbishment_state: stored_ro_heat` are the
supported coefficient contract. Missing coefficients, uncorrected unit provenance,
invalid areas and TSAM fail explicitly. Each scenario has its own hash; physical
references and fixed assets are shared across preheating sensitivities.

A paired run with internal heat prepares its own paired dataset. Its preparation
checks RC coverage and builds provider weather/PV inputs; it does not regenerate
TEASER heat. Check worker memory on a pilot grid before starting the full batch.
