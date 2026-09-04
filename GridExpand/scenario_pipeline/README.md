# Scenario pipeline

The scenario pipeline owns scientific configuration, physical-building asset
plans, and stable model-case semantics. It must run without importing
`paired_validation`. See `docs/SCENARIO_METHOD.md` for the method and
`docs/CONFIGURATION_REFERENCE.md` for YAML options.

The package layout is deliberately flat: Python configuration models and
loaders live beside the runner, while user-edited YAML files live under
`config/scenarios` and `config/runs`. Generated manifests are written to
`GridExpand/run_logs/scenario_manifests`, never into this source directory.

## Starting a new scenario or run

Copy `config/scenarios/00_scenario_template.yaml` and either
`config/runs/00_run_scenario_template.yaml` or
`config/runs/00_run_paired_validation_template.yaml`. Replace all `CHANGE_ME`
values before running. `config/README.md` documents the required identifiers
and the structured PyLoVo grid-selection filters.

## Staged entry point

`run_scenario.py` is the single user-facing launcher. For paired validation it
runs `prepare -> validate -> execute -> postprocess`: it registers the complete
PyLoVo region, rebuilds the paired allocation and shared heat/PV artifacts,
checks paired equivalence and heat readiness, executes the requested cases, and
materializes expansion results. Expansion schema initialization is part of the
postprocessing command, so no separate `--schema-only` setup is required.

```bash
uv run --project GridExpand/2.demand_allocation \
  python GridExpand/scenario_pipeline/run_scenario.py \
  --run-config GridExpand/scenario_pipeline/config/runs/forchheim_2045_paired.yaml
```

Use `--prepare-only` to rebuild and validate shared inputs without starting
urbs or power flow. With `resume: true`, preparation is deliberately reused and
validated rather than rebuilt, preserving the existing job-index ledger.
