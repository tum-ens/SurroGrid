# Scenario-pipeline templates

Copy the template that matches the pipeline into the corresponding config
directory, rename it, and replace every `CHANGE_ME` value before launching.
The templates intentionally keep the complete schema so that the copied file
can be validated immediately.

## Required edits for every new scenario/run

Scenario YAML:

- `scenario.id`: unique, stable identifier for the scientific assumptions.
- `scenario.milestone_year`: modeled year, if it differs from the template.
- Review every scientific assumption in the file. Change the assumptions that
  define the scenario; do not use the template values without checking them.

Ordinary scenario run YAML:

- `run.id`: unique directory-safe run identifier.
- `run.scenario`: path to the scenario YAML copied for this run.
- `resources.pylovo_version_id`: exact topology version used for selection.
- `resources.ags`: PyLoVo municipality/region identity.
- `resources.plz`: optional postcode filter; null or `"-"` means all PLZs.
- `resources.kcid` and `resources.bcid`: optional exact-grid filters; set both
  or neither. Null or `"-"` means all grids matching the broader filters.
- `execution.model_cases`: cases to execute for this run.

Paired-validation run YAML:

- `run.id` and `run.scenario`.
- `resources.ags` and `resources.plz`: region identity.
- `resources.pylovo_version_id`: exact topology version used for preparation.
- `resources.heat_profile_set_id` and `resources.weather_source_hdf`: shared
  artifact identities.
- `resources.paired_dataset_id`: unique prepared paired-artifact directory.
- `execution.model_cases`: post-cases to execute; do not add `pre`.

## Fields that are normally reviewed, not blindly changed

- `execution.profile_seed` controls the reproducible stochastic realization.
  Keep it when comparing cases; change it only for a deliberately independent
  realization.
- CPU, worker, cleanup, resume, target-network, and grid-scope settings are
  run/deployment choices. Adjust them when the execution environment or target
  subset changes.
- `resources.excluded_real_lv_ids` and `resources.target_grid_id` are optional
  filters. Leave them empty/null unless the run needs an explicit exclusion or
  diagnostic grid filter.

## PyLoVo grid selection

The ordinary DB-backed run uses one region selector and optional filters; the
user does not enter a compound filename or candidate index:

```yaml
resources:
  pylovo_version_id: 3
  ags: 9662000
  plz: 97422       # null or "-" = all PLZs in the AGS
  kcid: 1          # set together with bcid for one exact grid
  bcid: 1
```

Selection is hierarchical:

- `ags` only selects all eligible grids for that AGS.
- `ags + plz` selects all eligible grids in that postcode.
- `ags + plz + kcid + bcid` selects one exact grid.

`pylovo_version_id` selects the topology version and is always separate from
the grid filters. The old filename convention
`<AGS>-<candidate_index>_<PLZ>_<KCID>_<BCID>.h5` remains an internal/export
format. Its candidate index is not a user-facing input.

For `storage: h5`, use the existing local HDF5 filename-prefix field instead;
remove the DB selector fields and set
`pylovo_grid_id` to the prefix before the first underscore in the local HDF5
filename. HDF5 runs do not support the DB wildcard selectors.

Launch from the repository root, for example:

```bash
uv run --project GridExpand/2.demand_allocation \
  python GridExpand/scenario_pipeline/run_scenario.py \
  --run-config GridExpand/scenario_pipeline/config/runs/<copied-run>.yaml
```
