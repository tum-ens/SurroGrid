# Configuration reference

The pipeline uses two YAML files with separate responsibilities.

Copy-ready templates and the checklist for fields that must be changed for a
new scenario/run are in `config/templates/`.

- A **scenario YAML** contains scientific and policy assumptions that may change
  results. It is fingerprinted as scenario_hash.
- A **run YAML** identifies the input dataset and selects execution resources.
  It is fingerprinted as run_hash.

Unknown keys, incorrect types, invalid ranges, unsupported case names, and
missing required fields fail during loading. This prevents misspelled options
from silently selecting another assumption.

## Scenario YAML

- scenario.id: stable scenario identifier.
- scenario.milestone_year: modeled milestone year.
- economics.electricity: import price and configurable PV feed-in tariff; a
  zero tariff retains physical export without remuneration.
- asset_sizing.pv: LoD2 potential, heuristic demand multiplier, fallback, roof
  utilization, and pvlib angle-bin choices.
- asset_sizing.battery: HTW sizing coefficients and energy-to-power ratio.
- electrification.heat, electrification.mobility, electrification.pv_battery:
  strict adoption mode and eligibility policy; deterministic_share requires
  building_share, while source_inventory requires explicit evidence.
- asset_sizing.heat: regional heat-pump, auxiliary-heater, and space-heating
  buffer sizing parameters.
- mobility: emobpy temporal, behavioral, and vehicle-energy assumptions.
- technologies.processes and technologies.storages: values written into urbs.
- time_aggregation: complete TSAM method, including typical/extreme periods.

The example scenario YAML comments describe each adjustable value. The
scientific reasoning and equations are centralized in SCENARIO_METHOD.md.

## Ordinary scenario run YAML

For run.pipeline: scenario:

- resources.ags: PyLoVo municipality/region identity.
- resources.plz: optional postcode filter. Null or `"-"` selects all PLZs in
  the AGS.
- resources.kcid and resources.bcid: optional exact-grid filters. They must be
  supplied together; null or `"-"` selects all grids matching broader filters.
- With only `ags`, the ordinary DB-backed runner enumerates all eligible grids;
  adding `plz` narrows that set, and adding `kcid`/`bcid` selects one exact grid.
  The filename form `<AGS>-<candidate_index>_<PLZ>_<KCID>_<BCID>.h5` is an
  internal/export representation; candidate index is not user input.
- resources.pylovo_version_id: exact topology version used for DB-backed
  selection. It is passed explicitly and is never inferred from the process
  environment by the scenario launcher.
- resources.storage: db or h5.
- resources.output_directory: optional output override.
- execution.model_cases: requested cases for this run. Names are validated
  against the global model-case registry in scenario_pipeline/model_cases.py.
- execution.n_cpu, mobility_source, demand_scope, and timeframe_mode:
  machine/runtime choices that do not define scientific assumptions.

For `resources.storage: h5`, use `resources.pylovo_grid_id` as the local HDF5
filename prefix instead of the DB selector fields; HDF5 runs remain single-grid
runs.

The launcher's optional --model-case overrides this list for a one-case run.

## Paired-validation run YAML

For run.pipeline: paired_validation:

- resources.ags, resources.plz, and resources.pylovo_version_id: explicit
  regional and topology identity. Preparation registers every eligible grid in
  that PyLoVo version before the paired allocation is built, so coverage never
  depends on grids touched by earlier runs.
- resources.min_buildings: symmetric minimum retained building count per grid.
- resources.paired_dataset_id: stable paired artifact-directory name.
- resources.heat_profile_set_id and resources.weather_source_hdf: names of the
  shared heat library and weather artifact. Their directories are fixed by
  repository conventions rather than repeated in YAML.
- resources.excluded_real_lv_ids: real-grid IDs retained in coverage reporting
  but excluded from expansion costing.
- resources.target_network: both, real_swf, or synthetic.
- resources.target_grid_id: optional adapter-local grid filter.
- execution.model_cases: requested post-cases. Heuristic HEMS and INFLEX share
  one capacity-plan solve; optimized HEMS is solved separately.
- The paired runner adds the pre case implicitly to the first requested
  post-case group; pre should therefore not be listed for paired runs.
- execution workers/CPU values: concurrency limits.
- execution.powerflow_grid_scope: `full` includes terminal load buses and service
  lines in voltage/loading statistics; `backbone` maps terminal observations one
  bus upstream and removes terminal service edges from those statistics.
- execution.profile_seed: arbitrary fixed seed defining the reproducible input
  realization; it has no geographic or scenario meaning.
- execution.cleanup_intermediates and execution.resume: execution-artifact
  lifecycle controls. Cleanup removes completed Step-2/3 HDFs and copied Step-4
  inputs, not compact expansion materializations.
- execution.materialize_expansion: automatically initialize the expansion
  schema and materialize analysis-ready results after every requested model case
  succeeds.

Internal directories are deliberately absent from this YAML. Outputs follow:

    GridExpand/run_logs/<run.id>/
    ├── run_manifest.json
    ├── heuristic-assets/
    └── post-hems-optimized/

Only requested groups are created. When both groups run, pre is emitted by the
first group only. The external SWF export root remains an environment/deployment
setting (GRID_DATA_PATH in GridExpand/.env), because it describes where data is
installed rather than a scenario or run choice. A station-based dataset must
contain both `station_split_manifest.csv` and
`station_radialization_manifest.csv`; their selected files must exist below
the configured root.

PYLOVO_VERSION_ID in GridExpand/.env is not authoritative for YAML-launched
scenario runs or paired preparation. The paired-allocation CLI requires the
version explicitly, and prepared datasets record it for later validation.

Implementation references such as database schemas, API endpoints, and stable
repository artifact conventions remain in Python. Adjustable scientific values
belong in the scenario YAML; dataset identity and runtime resources belong in
the run YAML.
