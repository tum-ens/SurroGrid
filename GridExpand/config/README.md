# Scenario and run YAMLs

- `scenarios/` holds the scenario library: the scientific assumptions, shared by many runs. `default_2045.yaml`
  is the shipped default.
- `runs/` holds run YAMLs: one short file per run that names a scenario, the grids, the model cases and the
  machine settings. `00_run_template.yaml` is the template.

Every key, default and the grid selection are documented in [docs/configuration.md](../docs/configuration.md); the
reasoning behind the scenario values in [docs/method.md](../docs/method.md). Test fixtures (scenarios and the
regression-sandbox run) live in `tests/fixtures/`, the validation's method scenario in
`validation/config/scenarios/`.

## A new run

1. Copy `runs/00_run_template.yaml` to `runs/<name>.yaml`; set `run` (directory-safe id), `grids.pylovo.version`
   (the exact pylovo version, never taken from `.env`) and `grids.pylovo.ags`, optionally `plz` or
   `plz` + `kcid` + `bcid`.
2. Scenario: keep `scenario: default_2045`, change single values with `scenario_overrides`, or copy
   `scenarios/default_2045.yaml` to `scenarios/<name>.yaml` with a new `scenario.id` and name it. Any changed value
   changes the scenario hash and therefore the scenario key of all results.
3. Optional: `model_cases` (default `post-inflex-heuristic`, `post-hems-heuristic`, `post-hems-optimized`; `pre`
   always runs) and `execution`. Keep
   `execution.profile_seed` when cases or runs must be comparable.
4. Check and launch:

```bash
uv run gridexpand config check config/runs/<name>.yaml
uv run gridexpand run config/runs/<name>.yaml --dry-run
uv run gridexpand run config/runs/<name>.yaml
uv run gridexpand status <run>
```
