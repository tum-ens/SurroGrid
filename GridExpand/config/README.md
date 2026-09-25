# Scenario and run YAMLs

`scenarios/` holds scenario YAMLs (scientific assumptions), `runs/` holds run YAMLs (input, model cases,
resources). Every key, default and the grid selection are documented in
[docs/configuration.md](../docs/configuration.md); the reasoning behind the values in
[docs/method.md](../docs/method.md).

| `run.pipeline` | template | what `gridexpand run` does |
|---|---|---|
| `synthetic` | `runs/00_run_scenario_template.yaml` | Steps 2-4 and the expansion analyses for the synthetic grids of one region |
| `paired_validation` | `runs/00_run_paired_validation_template.yaml` | SWF real vs synthetic grids of one prepared paired dataset |
| `paired_aligned` | see `runs/joint_2045_v1_*.yaml` | every provider (SWF, ÜZW) of a pylovo alignment bundle |

`runs/sandbox_example.yaml` is the regression-harness run (sandbox database only).

## Checklist for a new scenario or run

1. Copy `scenarios/00_scenario_template.yaml` and the run template of the pipeline; replace every `CHANGE_ME`.
2. Scenario: set a unique, stable `scenario.id` and review every value; changing any value changes the scenario
   hash and therefore the scenario key of all results.
3. Run: set a unique directory-safe `run.id`, `run.scenario` (relative to the run YAML) and the exact
   `resources.pylovo_version_id` (never taken from `.env`).
4. Synthetic runs: `resources.ags`, optionally `plz` or `plz` + `kcid` + `bcid`; choose `execution.model_cases`
   (remove `post-inflex-heuristic` from the template: the synthetic pipeline refuses it).
5. Paired runs: `ags`, `plz`, `paired_dataset_id`, `heat_profile_set_id`, `weather_source_hdf`; post cases only
   (`pre` is added automatically).
6. Keep `execution.profile_seed` when cases or runs must be comparable; change it only for a deliberately
   independent realization.
7. Check and launch:

```bash
uv run gridexpand config check config/runs/<run>.yaml
uv run gridexpand run config/runs/<run>.yaml --dry-run
uv run gridexpand run config/runs/<run>.yaml
uv run gridexpand status <run.id>
```
