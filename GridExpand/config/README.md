# Run and scenario templates

Copy the template that matches the pipeline into the corresponding config
directory, rename it, and replace every `CHANGE_ME` value before launching.
The templates intentionally keep the complete schema so that the copied file
can be validated immediately with `uv run gridexpand config check <file>`
(no database; prints the run and scenario hashes and the scenario key).

Run YAMLs (`config/runs/`) name one pipeline in `run.pipeline`:

| pipeline | template | what `gridexpand run` does |
|---|---|---|
| `synthetic` | `00_run_scenario_template.yaml` | Steps 2-4 and the expansion analyses for the synthetic grids of one region (formerly `scenario`, which ran Step 2 only) |
| `paired_validation` | `00_run_paired_validation_template.yaml` | SWF real vs synthetic grids of one prepared paired dataset |
| `paired_aligned` | see `joint_2045_v1_*.yaml` | every provider (SWF, ÜZW) of a pylovo alignment bundle |

## Required edits for every new scenario/run

Scenario YAML:

- `scenario.id`: unique, stable identifier for the scientific assumptions.
- `scenario.milestone_year`: modeled year, if it differs from the template.
- Review every scientific assumption in the file. Change the assumptions that
  define the scenario; do not use the template values without checking them.

Synthetic run YAML:

- `run.id`: unique directory-safe run identifier.
- `run.scenario`: path to the scenario YAML copied for this run.
- `resources.pylovo_version_id`: exact topology version used for selection.
- `resources.ags`: PyLoVo municipality/region identity.
- `resources.plz`: optional postcode filter; null or `"-"` means all PLZs.
- `resources.kcid` and `resources.bcid`: optional exact-grid filters; set both
  or neither. Null or `"-"` means all grids matching the broader filters.
- `execution.model_cases`: cases to execute for this run. `post-inflex-heuristic`
  is refused for now: the synthetic Step 2 writes no EV sessions for the INFLEX
  power flow.
- Optional `resources.min_buildings` (default 5), `start_index`, `limit`, and the
  `gridexpand synthetic` flags as `execution` keys (`workers`, `step2_cpus` or its
  alias `n_cpu`, `step3_cpus`, `step3_max_cpus`, `step3_cluster_concurrency`,
  `step4_cpus`, `solver`, `powerflow_output` (default `summary`),
  `powerflow_grid_scope`, `cleanup_intermediates`, `materialize_expansion`,
  `pilot_gate`, `resume`). Outputs are always model-case qualified.

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

The synthetic run uses one region selector and optional filters; the user
does not enter a compound filename or candidate index:

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

The selected region is also the population of the regional electrification
assignment (which buildings get heat pumps, EVs, PV). Candidate numbers stay
those of the whole AGS (`gridexpand grids --ags ... --pylovo-version-id ...`
lists them); `start_index`/`limit` select a range of them without changing the
assignment population. `pylovo_version_id` selects the topology version and is
always separate from the grid filters. The filename convention
`<AGS>-<candidate_index>_<PLZ>_<KCID>_<BCID>.h5` remains an internal/export
format. The synthetic pipeline reads grids from the database (`storage: db`).

## Launching and following a run

```bash
uv run gridexpand config check config/runs/<copied-run>.yaml
uv run gridexpand run config/runs/<copied-run>.yaml --dry-run   # plan, no database
uv run gridexpand run config/runs/<copied-run>.yaml
uv run gridexpand status <run id>                               # reads work/runs/<run id>/state.json
uv run gridexpand run config/runs/<copied-run>.yaml --resume    # skip jobs already done
```

The run directory `work/runs/<run.id>/` keeps frozen copies of both YAMLs,
`identity.json`, `state.json`, `events.jsonl`, `plan.json`, `summary.json` and one
directory per job group. SIGTERM or Ctrl-C stops the running steps and records
the jobs as cancelled; `--resume` continues. `config/runs/sandbox_example.yaml`
is the regression harness run (sandbox database only).
