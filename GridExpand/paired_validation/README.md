# Paired validation

This layer maps one scenario onto corresponding real and synthetic networks and
checks equivalence. Scenario logic, prices, asset sizing, profiles, and TSAM
assumptions belong to `scenario_pipeline`; dependencies only point from paired
validation to that shared layer.

`runner.py` is the authoritative comparison entry point for both `real_swf`
and `synthetic` targets. It owns only paired projection, shared temporal
mapping checks, and equivalent execution across the two network models.
Scenario assumptions and asset sizing remain in `scenario_pipeline` and the
shared Step-2 asset modules. The paired runner is launched through the paired
run YAML in `scenario_pipeline/config/runs`. Either heuristic case reuses one
asset-plan solve for separately named HEMS and INFLEX dispatches, while
optimized HEMS never attempts an INFLEX reconstruction.

## Dependencies

```text
scenario_pipeline -------------------\
common/orchestration -----------------+--> paired_validation/runner.py
Step 2 paired materialization -------/              |
                                                    +--> sources/swf.py --> SWF Step 4
                                                    +--> sources/synthetic.py --> synthetic Step 4
```

The paired runner does not call the synthetic AGS runner. Both runners reuse
the same lower-level Step 2--4 programs and `common/orchestration.py`. The AGS
runner discovers arbitrary synthetic grids for a regional study; paired
validation reads an existing allocation pairing and runs only its selected
synthetic counterpart.

## Source adapters

`sources/swf.py` and `sources/synthetic.py` contain only network-specific
allocation-plan discovery, HDF naming, and Step-4 commands. Shared scenario,
TSAM, URBS, concurrency, and result handling remain in `runner.py`, while
network-independent equivalence checks live in `comparison.py`.

A future SWN integration should add `sources/swn.py` and register it in
`sources/__init__.py`. It must also provide an SWN allocation/materialization
path that emits the same canonical paired HDF contract and an SWN Step-4
adapter. The SWN schema is not known yet, so no placeholder implementation is
included.
