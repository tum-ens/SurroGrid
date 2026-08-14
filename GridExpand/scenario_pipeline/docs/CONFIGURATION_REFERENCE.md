# Configuration reference

The pipeline uses two YAML files.

- A **scenario YAML** contains assumptions that change scientific results and
  is fingerprinted as `scenario_hash`.
- A **run YAML** selects grids, storage, paths, and execution resources and is
  fingerprinted as `run_hash`.

Unknown keys, incorrect types, invalid ranges, unsupported case names, and
missing required fields fail during loading. This validation prevents a typo
from silently falling back to a different assumption.

## Scenario YAML

- `scenario.id`: stable scenario identifier.
- `scenario.milestone_year`: modeled milestone year.
- `scenario.model_cases`: subset of the four documented model cases.
- `economics.electricity.import_price_eur_per_kwh`: grid purchase price.
- `economics.electricity.pv_feed_in_tariff_eur_per_kwh`: PV export revenue;
  zero disables remuneration without removing physical exports.
- `asset_sizing.pv.heuristic_method`: upstream sizing rule.
- `asset_sizing.pv.optimized_method`: endogenous urbs sizing mode.
- `asset_sizing.pv.location_mode`: predefined source locations or all buildings.
- `asset_sizing.pv.demand_multiplier`: multiplier in the PV sizing equation.
- `asset_sizing.pv.fallback_capacity_kwp`: missing-LoD2 fallback.
- `asset_sizing.pv.maximum_fallback_share`: accepted fallback fraction, 0--1.
- `asset_sizing.pv.module_capacity_kw_per_m2`: module peak density.
- `asset_sizing.pv.*_roof_utilization`: usable surface fraction, 0--1.
- `asset_sizing.pv.*_bin_degrees`: shared pvlib profile resolution.
- `asset_sizing.battery.*_method`: heuristic and optimized sizing modes.
- `asset_sizing.battery.location_mode`: all PV buildings are candidates.
- `asset_sizing.battery.predefined_locations_when_available`: use source
  inventory rows as location evidence when supplied.
- `asset_sizing.battery.minimum_pv_kwp_per_annual_mwh`: HTW surplus threshold.
- `asset_sizing.battery.maximum_usable_*`: the two HTW usable-energy bounds.
- `asset_sizing.battery.energy_to_power_hours`: fixed E/P ratio in hours.
- `asset_sizing.heat.indoor_design_temperature_c`: indoor reference for degree-day normalization.
- `asset_sizing.heat.heating_limit_temperature_c`: daily-mean threshold selecting heating days.
- `asset_sizing.heat.heat_pump_design_share`: HP thermal output at norm outside temperature as a fraction of calculated design load.
- `asset_sizing.heat.buffer_volume_l_per_kw_th`: space-heating buffer volume per thermal HP kW.
- `asset_sizing.heat.buffer_usable_temperature_spread_k`: usable buffer temperature range used to convert litres to kWhth.
- `mobility.commuting_probability`: commuter-schedule probability.
- `mobility.emobpy_timestep_hours` and `reference_year`: emobpy temporal basis.
- `mobility.passenger_*`, `cabin_*`, `driving_cycle_type`, `road_type`, and
  `road_slope`: vehicle-energy-model assumptions.
- `technologies.processes.*`: capacities, costs, WACC, lifetime, and power
  factor written to urbs process rows. Asset plans replace generic PV
  capacity, while source inventory may replace paired charger capacity.
- `technologies.storages.*`: capacities, E/P ratio, efficiencies, costs, WACC,
  and lifetime written to urbs storage rows. `null` capacity fields mean that
  capacity comes from a PV/battery asset plan or an individual EV profile.
- `time_aggregation.*`: complete TSAM methodology; see comments in the example.

## Run YAML

- `run.id`: execution identifier.
- `run.scenario`: scenario YAML path, relative to the run YAML.
- `run.pipeline`: `scenario` or `paired_validation`.
- `resources.inputfile_id`: DB/grid/HDF identifier.
- `resources.storage`: `db` or `h5`.
- `resources.output_directory`: optional output override.
- `resources.target_*`, `paired_directory`, `weather_source_hdf`: paired-only
  resources; null for a normal scenario run.
- `execution.n_cpu`: machine concurrency.
- `execution.mobility_source`: current mobility generator/pool selection.
- `execution.demand_scope`: current building demand scope.
- `execution.timeframe_mode`: full year or supported operational slice.

Implementation references such as database schema names, API endpoints, and
static dataset paths remain in the Python config. Scientific mobility and urbs
numbers do not: `config.py` only maps validated scenario values onto legacy
helper attribute names where those helpers have not yet been refactored.
