# Step 2: demand allocation (urbs inputs)

Step 2 turns one grid into building- and bus-resolved hourly time series (weather, base electricity, rooftop PV,
stationary batteries, space heat and hot water with heat-pump COP, EV charging and availability) and writes the
urbs input tables of Step 3 (`urbs_in/*`) plus the asset plans and audits into one HDF5 file.

Entry point: `gridexpand allocate` = `src/gridexpand/allocation/main.py`. `run_allocation(settings)` runs the
stages of `STAGES[profile]` on a `Grid` (`classes/grid.py`) and writes the keys of `OUTPUT_KEYS[profile]`.

| module | content |
|---|---|
| `classes/grid.py`, `classes/save_grid.py` | stage methods; HDF5/database input and output |
| `functions/electricity.py` | household occupancy and annual electricity, residential load profiles (`elec_lps.h5`), GHD per-m² profiles, component aggregation |
| `functions/heat.py`, `functions/infdb_ro_heat.py` | TEASER / INFDB `ro_heat` space heat, OpenDHW hot water, air-source COP |
| `functions/mobility.py` | vehicles per household, pool or emobpy profiles, EV sessions |
| `functions/dst.py`, `functions/partition.py` | daylight-saving alignment, workload partition for parallel heat generation |
| `assets/pv/`, `assets/battery/`, `assets/heat/`, `assets/urbs_rows.py` | LoD2 roof catalog, sizing and urbs rows of PV, batteries, heat pumps and buffers |
| `electrification.py`, `electrification_preparation.py` | electrification inventory and the regional assignment manifest |
| `generate_mobility_profile_pool.py` | builds the mobility profile pools |
| `scenario_calibration/` | paired (real vs synthetic) preparation, see [paired_validation.md](../paired_validation.md) |
| `external/` | vendored districtgenerator and emobpy (licences in `external/third_party_licenses/`) |

## Inputs

- **Grid.** `--storage db` (used by all orchestrators): the grid, buildings and components are read from pylovo
  through `gridexpand.db`; weather is downloaded from PVGIS. `--storage h5`: a Step 1 file in
  `work/allocation/grids/` that must already contain `raw_data/weather` (see [Step 1](1_grid_sampling.md)).
- **Scenario YAML** (`--scenario-config`, required): all scientific values ([configuration.md](../configuration.md)).
- **Electrification assignment** (`--electrification-assignment`, optional CSV or HDF5 manifest): the regional
  selection of heat, mobility and PV+battery buildings. Required for `source_inventory` scenarios without source
  evidence in the input; the synthetic runner always prepares one per region (below).
- **Static data** in `data/statistics/` (tracked) and the large untracked inputs
  `data/statistics/inhabited_buildings/elec_lps.h5` and the mobility pools (see the
  [README](../../README.md#large-data-assets)).
- **LoD2 roofs** from the database schema `citydb` (buildings without usable roof sections get the fallback).

## Input selection

`inputfile_id` is interpreted by `--storage`:

- `db`: an AGS (`09184137` or `9184137`, optionally `-<candidate index>`), with `--candidate-index`,
  `--min-buildings` and `--demand-scope` choosing among its candidate grids (same numbering as `gridexpand
  grids`), or `--plz --kcid --bcid` for one exact grid, or a bridge file name such as
  `9184137-00_85653_1_3.h5`. `--pylovo-version-id` pins the topology version (default `PYLOVO_VERSION_ID` of
  `.env`).
- `h5`: the first `*.h5` in `work/allocation/grids/` whose name before the first underscore equals
  `inputfile_id` (keep prefixes unique).

## Command line

```bash
uv run gridexpand allocate 9184137 --storage db --pylovo-version-id 1 --candidate-index 0 \
  --scenario-config config/scenarios/forchheim_2045_synthetic.yaml \
  --profiles status_quo --model-case pre --mobility-source pool
```

| option | default | meaning |
|---|---|---|
| `--scenario-config` | required | scenario YAML |
| `--storage` | `h5` | `h5` or `db` (see above) |
| `--profiles` | `all` | `status_quo`, `heat_library`, `electricity_heat`, `electricity_mobility`, `electricity_heat_mobility` (`all` is an alias) |
| `--model-case` | `post-hems-heuristic` | `pre` requires `--profiles status_quo`; post cases require an electrification profile |
| `--timeframe-mode` | `full_year` | or a one-week mode (`min_temperature_week`, `max_solar_radiation_week`, `max_base_electricity_demand_week`); weeks require `--mobility-source pool` |
| `--mobility-source` | `emobpy` | `emobpy` (simulate) or `pool` (pregenerated profiles) |
| `--demand-scope` | `all` | `residential`: household components only, for all later steps |
| `--profile-seed` | 481527 | realization seed ([method.md](../method.md#reproducible-profile-realization)) |
| `--electrification-assignment` | none | regional assignment manifest |
| `--timeseries-storage` | `db` | DB mode: `db`/`both` also store `urbs_in/demand` and `urbs_in/eff_factor` in the database, `temp` only in HDF5 |
| `--case-qualified-output` | off | append `_<model case>` to the output file name |
| `--output-directory` | `work/allocation/results` | parent of the scenario-key directory |
| `--n_cpu` | 1 | processes for heat generation and emobpy |
| `--candidate-index`, `--plz`, `--kcid`, `--bcid`, `--pylovo-version-id`, `--min-buildings` | | DB-mode grid selection |

Profiles and stage order (the order is part of the method):

| `--profiles` | stages | use |
|---|---|---|
| `status_quo` | base electricity, timeframe, `urbs_in/demand` | `pre` case, Step 4 `--pre-only` (no Step 3) |
| `heat_library` | weather, base electricity, heat, demand and COP (no PV, battery, mobility) | regeneration of paired heat-profile libraries |
| `electricity_heat`, `electricity_mobility`, `electricity_heat_mobility`/`all` | weather, base electricity, PV, battery, heat, mobility, urbs tables | post cases |

## Outputs

The file is `<output directory>/<scenario key>/<input or bridge file name>[_<timeframe>][_<model case>].h5`
(scenario key: [configuration.md](../configuration.md)). It is a copy of the HDF5 input (HDF5 mode) or a new
file (DB mode) with these keys:

| key | written for | content |
|---|---|---|
| `metadata/timeframe` | all | run metadata: scenario id, hash and key, model case, timeframe (mode, start, horizon), profile seed, realization id and profile fingerprints, assignment hashes, sizing methods |
| `raw_data/buildings`, `raw_data/weather` | all (HDF5 mode) | physical buildings with sampled attributes; the weather used |
| `raw_data/building_components`, `raw_data/demand_component_audit` | all | validated component manifest; per-component annual energy, profile method and hash, suppression reason |
| `raw_data/heat_asset_plan`, `raw_data/heat_asset_audit` | heat profiles | heat pump, auxiliary heater and buffer per building; sizing audit |
| `raw_data/electrification_assignment`, `raw_data/electrification_assignment_summary` | post profiles | selected buildings per technology with eligibility, rank and exclusion reason |
| `raw_data/pv_roof_sections`, `raw_data/pv_selected_sections` | post profiles (HDF5 mode) | LoD2 roof catalog; selected sections |
| `raw_data/asset_plan`, `raw_data/pv_asset_audit` | post profiles | PV plan and audit |
| `raw_data/battery_asset_plan`, `raw_data/battery_asset_audit` | post profiles | battery plan and audit |
| `urbs_in/demand` | all | demand per bus: `(bus, electricity)`, `(bus, space_heat)`, `(bus, water_heat)`, `(bus, mobility<id>)` |
| `urbs_in/eff_factor` | `heat_library`, post profiles | `(bus, heatpump_air)` COP, `(bus, charging_station<id>)` availability 0/1 |
| `urbs_in/supim` | post profiles | normalized PV supply per bus and roof label |
| `urbs_in/weather` | post profiles | `(ambient, Tamb)`, `(ambient, Irradiation)` (TSAM features) |
| `urbs_in/buy_sell_price` | post profiles | `electricity_import`, `electricity_feed_in` from the scenario YAML |
| `urbs_in/process`, `commodity`, `process_commodity`, `storage` | post profiles | urbs tables (sites are bus ids; per-vehicle `charging_station<id>`, `mobility<id>`, `mobility_storage<id>`) |

Paired preparation additionally writes `raw_data/allocation_plan` (scenario units) and `urbs_in/ev_sessions` /
`ev_session_hours`; the synthetic Step 2 writes no EV sessions.

In DB mode the bulky copies of the database inputs (`raw_data/buildings`, `raw_data/weather`,
`raw_data/pv_roof_sections`, `raw_data/pv_selected_sections`) are not written to the HDF5 file; the compact plans
and audits are. The database receives one `demand_allocation_run` with the final run metadata, the component
audit, the electrification assignment and the allocated vehicles, plus `allocated_demand` and
`allocated_eff_factor` with `--timeseries-storage db|both` ([database.md](../database.md)).

## Regional electrification assignment

The orchestrators prepare one assignment per region before the grid jobs start:

```bash
uv run python -m gridexpand.allocation.electrification_preparation --ags 9184137 --pylovo-version-id 1 \
  --scenario-config config/scenarios/forchheim_2045_synthetic.yaml --mobility-source pool \
  --output work/runs/<run>/electrification_assignment.csv
```

Options: `--plz` (only that postcode's grids), `--kcid --bcid` (with `--plz`: one grid), `--min-buildings`,
`--demand-scope`, `--profile-seed`, `--source-evidence` (evidence file for `source_inventory`). It writes the
manifest, a `.json` sidecar (hashes, candidate grids) and `electrification_candidate_grids.json`; the runner
refuses a manifest whose sidecar belongs to another run or candidate set.

## Mobility profile pools

Synthetic runs (`--mobility-source pool`) use the legacy pool `emobpy_pool_v1` in
`data/statistics/general/mobility_profile_pool_old/` (tracked metadata and weather, untracked demand and
availability CSVs). Paired runs use the session pool in `data/statistics/general/mobility_profile_pool/`
(`emobpy_pool_v2_sessions`, with `mobility_pool_manifest.json`). To build a session pool:

```bash
uv run python -m gridexpand.allocation.generate_mobility_profile_pool --mode session \
  --scenario-config config/scenarios/joint_2045_full_year.yaml --n_cpu 8 --dry-run
uv run python -m gridexpand.allocation.generate_mobility_profile_pool --mode session --freeze-manifest \
  --scenario-config config/scenarios/joint_2045_full_year.yaml
```

(`--mode deadline` reproduces the legacy clipped v1 pool; `--append`, `--profiles-per-stratum`, `--models`,
`--schedules` control the size; consumers require the frozen manifest.)

## Conventions

- Time: 8,760 hourly steps of the reference year 2009 in fixed UTC+1, or 168 hours for a one-week timeframe;
  civil-time profiles are shifted around the 2009 DST hours ([method.md](../method.md#time-axis)).
- Units: demands are energies per hourly step in kWh (equal to mean kW); PV supply (`supim`) is normalized per kWp.
- Reproducibility: all sampling uses stable seeds from `--profile-seed` and the physical building id; `--n_cpu` does
  not change results.
- emobpy (`--mobility-source emobpy`) makes up to three attempts per failing vehicle, bumping the seed each time.
