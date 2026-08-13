# Paired validation

This layer maps one scenario onto corresponding real and synthetic networks and
checks equivalence. Scenario logic, prices, asset sizing, profiles, and TSAM
assumptions belong to `scenario_pipeline`; dependencies only point from paired
validation to that shared layer.

`runner.py` is intentionally a temporary compatibility entry point. It
delegates to `runme/paired_swf_pipeline_runner.py` until the paired-only
orchestration is moved here; it must not acquire scenario assumptions or asset
sizing logic. Those already come from `scenario_pipeline` and the shared
Step-2 asset modules.
