# Scenario pipeline

The scenario pipeline owns scientific configuration, physical-building asset
plans, and stable model-case semantics. It must run without importing
`paired_validation`. See `docs/SCENARIO_METHOD.md` for the method and
`docs/CONFIGURATION_REFERENCE.md` for YAML options.

The package layout is deliberately flat: Python configuration models and
loaders live beside the runner, while user-edited YAML files live under
`config/scenarios` and `config/runs`. Generated manifests are written to
`GridExpand/run_logs/scenario_manifests`, never into this source directory.
