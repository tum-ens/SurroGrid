# GridExpand validation: real vs synthetic grids

This directory holds the paired validation of GridExpand: one post-electrification scenario runs on the real DSO
grids of a pylovo alignment bundle (DSO A and DSO B) and on the synthetic pylovo grids of the same buildings, and
the power flows and expansion needs are compared. It is the case study of the pylovo validation paper. The
scientific contract (scenario units, shared layers, status-quo test, publication criteria) is in
[METHOD.md](METHOD.md).

The validation is not part of the `gridexpand` package: it is not installed, not in the Docker image, and
`gridexpand` never imports it (a test checks this). It imports `gridexpand` and uses its per-grid runner, profile
libraries and real-grid power flow. Run it from `GridExpand/` with the GridExpand environment (`uv sync`).

| path | content |
|---|---|
| `run.py`, `commands.py`, `__main__.py` | `python -m validation run`: the `paired_aligned` run YAML, its stages and commands |
| `aligned_allocation.py` | one provider's paired dataset from its alignment bundle (Step 2 preparation) |
| `grid_subsets.py` | grid subsets of aligned runs (`components`, `islands`) |
| `status_quo.py` | `python -m validation status-quo-test`: the real grids at alignment, exclusion list per provider |
| `aligned_expansion.py` | expansion analyses of every provider group of a run |
| `notebook_workflow.py`, `comparison_data.py` | loaders of the comparison notebooks (four groups: real/synthetic × DSO A/DSO B) |
| `powerflow_asset_plots.py`, `cutoff_data.py`, `geoplotting.py`, `feeder_structure.py`, `feeder_structure_plots.py` | comparison plots and the feeder-structure audit |
| `config/` | run YAMLs of the joint DSO A + DSO B comparison (scenario: `../config/scenarios/joint_2045_full_year.yaml`) |
| `notebooks/` | `analysis_powerflow.ipynb`, `analysis_expansion.ipynb`, `grid_area_envelope_comparison.ipynb`, `joint_setup.py` |
| `tests/` | unit tests (part of `uv run pytest`) |

## Data

The real-grid data are confidential and not part of the repository: the pylovo alignment bundle
(`resources.alignment_dir`), the pylovo comparison JSON with the metric cohort (`resources.population`) and the
DSO B delivery (`resources.dso_b_grids_dir`) named in the run YAML. The envelope notebook also reads the DSO A grid
exports below `GRID_DATA_PATH` of `GridExpand/.env`.

## Running

```bash
uv run python -m validation run validation/config/joint_2045_v1_smoke24.yaml --dry-run         # plan and commands
uv run python -m validation run validation/config/joint_2045_v1_smoke24.yaml --prepare-only    # build and validate datasets
uv run python -m validation run validation/config/joint_2045_v1_smoke24.yaml --skip-prepare    # execute + postprocess
uv run gridexpand status joint_2045_v1_smoke24
```

`--dry-run` also validates the YAML (`gridexpand config check` accepts only `pipeline: synthetic`). The options of
`gridexpand run` apply (`--resume`, `--until`, `--prepare-only`, `--model-case`, `--run-dir`), plus
`--skip-prepare` (use the prepared datasets as they are, validated), `--provider dso_a|dso_b`, `--pre-only` (Step 2
and the Step-4 pre stage, no urbs) and `--target-grid-id` (one real grid for diagnosis: own run sub-directory, no
expansion analyses). Run directories, `state.json`, exit codes and cancellation are those of `gridexpand run`
([configuration.md](../docs/configuration.md#running-a-run-yaml)).

Stages:

1. **prepare** (skipped with `execution.resume: true` or `--skip-prepare`, which validate the prepared datasets
   instead), per provider: `validation.aligned_allocation`, `profiles.aligned_weather`,
   `profiles.paired_heat_profile_regeneration` (TEASER heat of every source grid, `execution.heat_workers`
   parallel jobs), readiness, `profiles.physical_heat_profile_library`, readiness against the library,
   `profiles.pv_profile_library` (the profile modules are below `gridexpand.allocation.scenario_calibration`);
   with `execution.grid_subset` a subset of overlap components.
2. **execute**: one `gridexpand.runner.grid_runner` process per execution group and provider: `heuristic-assets`
   (materializes `post-hems-heuristic`, emits the requested heuristic cases and the `pre` stage) and
   `post-hems-optimized`.
3. **postprocess** (only when every job succeeded and `execution.materialize_expansion`):
   `validation.aligned_expansion` (all provider groups, then one QGIS refresh).

## Run YAML (`pipeline: paired_aligned`)

The `run` block is that of every GridExpand run YAML (`id`, `scenario`, `pipeline`).

| key | default | meaning |
|---|---|---|
| `resources.pylovo_version_id` | required | pylovo version of the alignment bundle |
| `resources.alignment_dir` | required | pylovo alignment bundle |
| `resources.population` | required | pylovo comparison JSON (metric cohort) |
| `resources.dso_b_grids_dir` | required with `dso_b` | DSO B grid delivery (fingerprint checked against the bundle) |
| `resources.providers.<dso_a\|dso_b>.paired_dataset_id` | required | dataset directory per provider |
| `resources.providers.<…>.weather_source_hdf` | required | weather file name in `work/allocation/results/` |
| `resources.providers.<…>.bundle_name` | the key | the provider's name inside the alignment bundle (`<alignment_dir>/<name>/`) and the comparison JSON |
| `resources.weather_year` | null | null: PVGIS TMY per provider; a year (2005–2023): that real year for every provider ([method](../docs/method.md#time-axis)) |
| `resources.providers.<…>.workers` | `execution.workers` | grids in parallel for this provider |
| `resources.target_network` | `both` | `both`, `real` or `synthetic` |
| `execution.model_cases` | required | post cases only (`pre` is added to the first group) |
| `execution.workers`, `step3_cpus`, `step4_cpus` | 1 each | resources |
| `execution.step3_cluster_concurrency` | null | Step 3 models solved at the same time; null: 1 for urbs, automatic for pypsa |
| `execution.heat_workers` | 4 | parallel TEASER heat regeneration jobs |
| `execution.powerflow_max_timesteps` | null | smoke-test cap (not for publication) |
| `execution.cleanup_intermediates`, `resume`, `parallel_providers` | false | lifecycle; providers in parallel |
| `execution.materialize_expansion` | false | expansion analyses of all provider groups after success |
| `execution.grid_subset` | null | `{method: components\|islands, seed, real_grids_per_provider}` |
| `execution.optimizer` | null | Step 3 optimizer `urbs` or `pypsa` for all jobs of the run (null: `$GRIDEXPAND_OPTIMIZER`, else `urbs`) |
| `execution.profile_seed` | 481527 | realization seed |

The heat library of a provider is derived: `<paired_dataset_id>_teaser_heat_<scenario_hash[:12]>`.
`grid_subset` picks whole pylovo overlap components (`components`, in seeded random order up to
`real_grids_per_provider` real grids) or only one-real-one-synthetic components (`islands`).

## Status-quo test

`python -m validation status-quo-test` checks the real grids of one provider under status-quo base electricity
before pylovo generation ([METHOD.md](METHOD.md#status-quo-test)); read-only on the database.

```bash
uv run python -m validation status-quo-test --provider dso_a --alignment-dir <bundle root> \
    --scenario-config config/scenarios/joint_2045_full_year.yaml --output-dir <dir> --jobs 4
uv run python -m validation status-quo-test --provider dso_b --alignment-dir <bundle root> --dso-b-grids-dir <delivery> \
    --scenario-config <scenario.yaml> --db-env-prefix INFDB_ --output-dir <dir>
uv run python -m validation status-quo-test --provider dso_a --dataset work/allocation/outputs/scenario_calibration/<dataset> \
    --output-dir <dir>                                        # regression: demand of a prepared paired dataset
```

| option | default | meaning |
|---|---|---|
| `--provider` | required | `dso_a` or `dso_b` |
| `--alignment-dir` / `--dataset` | one required | bundle root (`<dir>/<provider>/building-mapping.json`) or a prepared paired dataset |
| `--dso-b-grids-dir` | – | frozen DSO B delivery (bundle input) |
| `--bundle-name` | `--provider` | the provider's name inside the alignment bundle |
| `--scenario-config` | required with `--alignment-dir` | its `ghd:` block applies |
| `--db-env-prefix` | `DB_` | InfDB connection from `<prefix>HOST`, `PORT`, `NAME`, `USER`, `PASSWORD` (read-only session) |
| `--basedata-schema` | `basedata` | schema of `buildings` and `classify_building_use` |
| `--profile-seed` | 481527 | realization seed (a dataset must match its own) |
| `--peak-hours`, `--all-hours` | 60, off | evaluated hours per grid |
| `--grids` | all | comma-separated real grid numbers |
| `--jobs` | 1 | grids in parallel |

The exclusion list `status_quo_<provider>_exclusions.csv` has exactly the columns `provider` (`dso_a`, `dso_b`),
`real_grid_id` (the bundle's id: DSO A `station_id`, e.g. `LV_059`; DSO B `area_id`, e.g. `113`) and `reason`
(`transformer_overload`, `cable_overload`, `undervoltage`, `nonconvergence`, joined by `;`), one row per excluded
grid; a pylovo alignment option consumes it to drop these grids and their buildings.

## GridExpand code the validation runs

| module | content |
|---|---|
| `gridexpand.runner.grid_runner` | the per-grid runner: scenario-unit materialization, urbs, canonical temporal mapping, Step 4 per target; status keyed `<target>:<grid id>` for resume |
| `gridexpand.runner.targets.{real,synthetic}` | network targets: allocation-plan discovery, HDF naming, Step 4 commands; `real` takes the network label (`real_dso_a`, `real_dso_b`) and provider from the dataset (real grids: `powerflow.run_real_scenario_powerflow --provider`) |
| `gridexpand.runner.result_checks`, `gridexpand.runner.datasets` | run-wide result checks (shared TSAM mapping, full-year results); resolving and validating a prepared dataset |
| `gridexpand.allocation.scenario_calibration.profiles`, `.pipeline` | shared electricity, PV, mobility and heat profiles, readiness, libraries, aligned weather; paired urbs-input materialization |
| `gridexpand.analysis.expansion.real_materialization` | expansion rows of the real grids |

Scenario logic, prices, asset sizing and model cases stay in `gridexpand.scenario` and the Step 2 asset modules;
the paired layer projects them onto buses and checks equivalence.

## Files

| path | content |
|---|---|
| `work/allocation/outputs/scenario_calibration/<paired_dataset_id>/` | prepared dataset: `paired_scenario_metadata.json` (pylovo version, region, hashes), the plans and audits (`paired_real_bus_allocation_plan.csv`, `paired_synthetic_bus_allocation_plan.csv`, `paired_roof_sections.csv`, `paired_electrification_assignment.csv`, ...), `paired_heat_profile_catalog.csv`, `paired_pv_profile_library.h5`, `paired_registered_synthetic_grids.csv` and `heat_sources/<scenario hash 12>/` |
| `work/allocation/outputs/scenario_calibration/profile_libraries/<heat_profile_set_id>.h5` | physical heat library |
| `work/allocation/results/<weather_source_hdf>` | weather source of the PV library |
| `work/runs/<run.id>/<provider>/<group>[-grid<id>]/` | paired runner directories (`status.tsv`, logs, canonical temporal mapping) |

Power-flow run names: `<run.id>_<provider>_<real_<provider>|synthetic>_<case>`. The `_pre` run holds the pre stage
of a grid; the result-case runs hold only their post stage (Step 4 `--post-only`), since the pre stage is the same
for every case. Analysis keys: `<run.id>_<provider>_<real|synthetic>_<pre|post_inflex|post|post_hems_optimized>`
([Step 5](../docs/steps/5_postprocessing.md#expansion-materialization)).

The runner (`uv run python -m gridexpand.runner.grid_runner --help`) is started by `python -m validation run`; direct use
is for diagnostics. Important options: `--paired-dataset-id`, `--pylovo-version-id` (must match the dataset),
`--scenario-config`, `--target real_dso_a|real_dso_b|synthetic|both` with `--provider dso_a|dso_b`, `--model-case`
(materialized case) and `--result-cases`, `--skip-pre`, `--pre-only`, `--job-subset` (JSON `{target: [grid ids]}`),
`--target-grid-id`, `--max-timesteps` (smoke tests, not for publication), `--resume`, `--run-dir`, and
`--allow-diagnostic-heat-fallback` (area-scaled heat profiles; never for publication runs).

## Notebooks

`notebooks/analysis_powerflow.ipynb` (status-quo power flow, synthetic vs real) and
`notebooks/analysis_expansion.ipynb` (power-flow stress and expansion costs of the status quo, INFLEX and HEMS)
analyse one aligned run: set `RUN_ID` in `notebooks/joint_setup.py`. Both apply the same exclusion rules (grids with
non-converged hours in at least 1 % of the hours of a case; synthetic fragments under five buildings) and label the
pooled figures *Real* and *Synthetic*, so no DSO is named. `grid_area_envelope_comparison.ipynb` compares the
supplied-area envelopes of the synthetic grids with the DSO A exports (set `PLZ`). The first cell of each notebook
puts `GridExpand/` on the import path; start Jupyter with `uv run --extra notebooks jupyter lab` from `GridExpand/`.
