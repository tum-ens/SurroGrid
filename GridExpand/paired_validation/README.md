# Paired validation

This layer maps one scenario onto corresponding real and synthetic networks and
checks equivalence. Scenario logic, prices, asset sizing, profiles, and TSAM
assumptions belong to `scenario_pipeline`; dependencies only point from paired
validation to that shared layer.

Use `python paired_validation/runner.py ...` as the validation entry point.
It currently delegates execution to the existing `runme` orchestrator while
batch commands migrate. Scientific configuration and asset logic already come
from `scenario_pipeline` and the shared Step-2 asset modules.
