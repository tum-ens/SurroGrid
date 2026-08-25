# Configuration reference

The pipeline uses two YAML files with separate responsibilities.

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
- scenario.model_cases: model cases permitted for this scenario.
- economics.electricity: import price and configurable PV feed-in tariff; a
  zero tariff retains physical export without remuneration.
- asset_sizing.pv: LoD2 potential, heuristic demand multiplier, fallback, roof
  utilization, and pvlib angle-bin choices.
- asset_sizing.battery: HTW sizing coefficients, adoption/location policy, and
  energy-to-power ratio.
- asset_sizing.heat: regional heat-pump, auxiliary-heater, and space-heating
  buffer sizing parameters.
- mobility: emobpy temporal, behavioral, and vehicle-energy assumptions.
- technologies.processes and technologies.storages: values written into urbs.
- time_aggregation: complete TSAM method, including typical/extreme periods.

The example scenario YAML comments describe each adjustable value. The
scientific reasoning and equations are centralized in SCENARIO_METHOD.md.

## Ordinary scenario run YAML

For run.pipeline: scenario:

- resources.inputfile_id: AGS, bridge prefix, or DB/HDF grid identifier.
- resources.pylovo_version_id: exact topology version used for DB-backed
  selection. It is passed explicitly and is never inferred from the process
  environment by the scenario launcher.
- resources.storage: db or h5.
- resources.output_directory: optional output override.
- execution.n_cpu, mobility_source, demand_scope, and timeframe_mode:
  machine/runtime choices that do not define scientific assumptions.

One model case is selected with the launcher's optional --model-case; it
defaults to post-hems-heuristic.

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
- execution workers/CPU values: concurrency limits.
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
installed rather than a scenario or run choice.

PYLOVO_VERSION_ID in GridExpand/.env is not authoritative for YAML-launched
scenario runs or paired preparation. The paired-allocation CLI requires the
version explicitly, and prepared datasets record it for later validation.

Implementation references such as database schemas, API endpoints, and stable
repository artifact conventions remain in Python. Adjustable scientific values
belong in the scenario YAML; dataset identity and runtime resources belong in
the run YAML.
