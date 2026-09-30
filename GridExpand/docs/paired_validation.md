# Paired validation (real vs synthetic grids)

The paired pipelines map one scenario onto real DSO grids and onto the synthetic pylovo grids of the same
buildings and compare power flows and expansion needs. The scientific contract (scenario units, shared layers,
evidence rules, publication criteria) is in [method.md](method.md#paired-validation-contract).

| `run.pipeline` | real grids | example run YAML |
|---|---|---|
| `paired_validation` | SWF (Stadtwerke Forchheim) station exports | `config/runs/forchheim_2045_paired_full_year.yaml` |
| `paired_aligned` | SWF and ÜZW of a pylovo alignment bundle | `config/runs/joint_2045_v1_full_year.yaml` (smoke: `joint_2045_v1_smoke24.yaml`) |

The real-grid data are confidential and not part of the repository: SWF exports below `GRID_DATA_PATH` of `.env`;
for aligned runs the alignment bundle, the pylovo comparison JSON and the ÜZW delivery named in the run YAML
(`resources.alignment_dir`, `population`, `uzw_grids_dir`).

## Running

```bash
uv run gridexpand config check config/runs/joint_2045_v1_smoke24.yaml
uv run gridexpand run config/runs/joint_2045_v1_smoke24.yaml --dry-run          # plan and commands
uv run gridexpand run config/runs/joint_2045_v1_smoke24.yaml --prepare-only     # build and validate datasets
uv run gridexpand run config/runs/joint_2045_v1_smoke24.yaml --skip-prepare     # execute + postprocess
uv run gridexpand status joint_2045_v1_smoke24
```

`gridexpand run` stages ([configuration.md](configuration.md#running-a-run-yaml)):

1. **prepare** (skipped with `execution.resume: true` or `--skip-prepare`, which validate the prepared datasets
   instead):
   - `paired_validation`: `allocation.scenario_calibration.allocation.paired_allocation` (registers every eligible
     grid of the region and pylovo version, matches SWF rows to physical buildings, writes the paired plans),
     `profiles.paired_profile_readiness` (heat library coverage), `profiles.pv_profile_library`;
   - `paired_aligned`, per provider: `allocation.aligned_allocation`, `profiles.aligned_weather`,
     `profiles.paired_heat_profile_regeneration` (TEASER heat of every source grid, `execution.heat_workers`
     parallel jobs), readiness, `profiles.physical_heat_profile_library`, readiness against the library,
     `profiles.pv_profile_library`; with `execution.grid_subset` a subset of overlap components.
2. **execute**: one `gridexpand.paired.runner` process per execution group (and provider): `heuristic-assets`
   (materializes `post-hems-heuristic`, emits the requested heuristic cases and the `pre` stage) and
   `post-hems-optimized`.
3. **postprocess** (only when every job succeeded and `execution.materialize_expansion`): expansion analyses of every
   target and case, then one QGIS refresh ([Step 5](steps/5_postprocessing.md)).

## Code

| module | content |
|---|---|
| `gridexpand.paired.runner` | the paired runner: scenario-unit materialization, urbs, canonical temporal mapping, Step 4 per target; status keyed `<target>:<grid id>` for resume |
| `gridexpand.paired.sources.{swf,uzw,synthetic}` | network adapters: allocation-plan discovery, HDF naming, Step 4 commands (real grids: `powerflow.run_real_swf_scenario_powerflow --provider`) |
| `gridexpand.paired.comparison` | network-independent equivalence checks |
| `gridexpand.paired.datasets` | resolves and validates a prepared dataset from `paired_dataset_id` |
| `gridexpand.paired.aligned` | grid subsets of aligned runs (`components`, `islands`) |
| `gridexpand.allocation.scenario_calibration.allocation` | SWF building matching, GHD calibration, paired and aligned allocation plans |
| `gridexpand.allocation.scenario_calibration.profiles` | shared electricity, PV, mobility and heat profiles, readiness, libraries, aligned weather |
| `gridexpand.allocation.scenario_calibration.pipeline` | paired urbs-input materialization |

Scenario logic, prices, asset sizing and model cases stay in `gridexpand.scenario` and the Step 2 asset modules;
the paired layer projects them onto buses and checks equivalence. `gridexpand.scenario.run` imports the paired
modules only on the paired paths. A new DSO needs an allocation path that writes the same paired HDF contract, a
source adapter in `paired/sources/` registered in `paired/sources/__init__.py`, and its Step 4 support.

## Files

| path | content |
|---|---|
| `work/allocation/outputs/scenario_calibration/<paired_dataset_id>/` | prepared dataset: `paired_scenario_metadata.json` (pylovo version, region, hashes), the plans and audits (`paired_real_bus_allocation_plan.csv`, `paired_synthetic_bus_allocation_plan.csv`, `paired_scope_audit.csv`, `paired_roof_sections.csv`, `paired_electrification_assignment.csv`, ...), `paired_heat_profile_catalog.csv`, `paired_pv_profile_library.h5`; aligned datasets also `paired_registered_synthetic_grids.csv` and `heat_sources/<scenario hash 12>/` |
| `work/allocation/outputs/scenario_calibration/profile_libraries/<heat_profile_set_id>.h5` | physical heat library |
| `work/allocation/results/<weather_source_hdf>` | weather source of the PV library |
| `work/runs/<run.id>/[<provider>/]<group>[-grid<id>]/` | paired runner directories (`status.tsv`, logs, canonical temporal mapping) |

Power-flow run names: `<run.id>_<target>_<case>` (paired, target `synthetic` or `real_swf`) and
`<run.id>_<provider>_<real_<provider>|synthetic>_<case>` (aligned). The `_pre` run holds the pre stage of a grid;
the result-case runs hold only their post stage (Step 4 `--post-only`), since the pre stage is the same for every case. Analysis keys: see
[Step 5](steps/5_postprocessing.md#expansion-materialization).

## Runner options

`gridexpand run` builds the runner command; direct use (`uv run python -m gridexpand.paired.runner --help`) is for
diagnostics. Important options: `--paired-dataset-id`, `--pylovo-version-id` (must match the dataset),
`--scenario-config`, `--target real_swf|real_uzw|synthetic|both` with `--provider swf|uzw`, `--model-case`
(materialized case) and `--result-cases`, `--skip-pre`, `--pre-only`, `--job-subset` (JSON `{target: [grid ids]}`),
`--target-grid-id`, `--max-timesteps` (smoke tests, not for publication), `--resume`, `--run-dir`, and
`--allow-diagnostic-heat-fallback` (area-scaled heat profiles; never for publication runs).

A single real grid for diagnosis: `uv run gridexpand run <run.yaml> --target-grid-id <id>` (own run
sub-directory, no expansion analyses). Earlier regional audits and forensic notes of the Forchheim runs are in
[research/](research/).
