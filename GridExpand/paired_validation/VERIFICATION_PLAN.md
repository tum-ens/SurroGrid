# Paired model-case verification gate

This is the gated execution plan for accepting the new model cases. The
complete regional run must not start merely because a smoke command finished.
Each stage records commands, configuration hashes, selected grid IDs, HDF
metadata, audit tables, solver termination, and power-flow convergence.

## Agent execution contract

Process the stages in order. Diagnose and fix a failed stage, then repeat that
stage. Do not continue until every gate passes. Do not launch the complete
paired run without explicit user authorization after reporting the audits.

## Gate 0: configuration and prepared-data readiness

1. Load both YAML files and save their hashes.
2. Run paired heat-profile readiness without diagnostic fallback.
3. Validate identical real/synthetic scenario-unit IDs, building IDs,
   eligibility flags, and annual base-electricity quantities.
4. Build or reuse the PV pool and require exactly one profile for every
   target-local roof bin.
5. Dry-run the authoritative YAML command:

~~~bash
uv run --project GridExpand/2.demand_allocation \
  python GridExpand/scenario_pipeline/run_scenario.py \
  --run-config GridExpand/scenario_pipeline/config/runs/forchheim_2045_paired.yaml \
  --dry-run
~~~

Pass: all inputs exist, no diagnostic heat profiles, no LoD2 fallbacks under
the current zero-fallback policy, and no missing or surplus local PV labels.

## Gate 1: heuristic real/synthetic smoke comparison

Run one small target per adapter with identical scenario YAML, seed, libraries,
TSAM settings, and run-name stem. Target IDs are adapter-specific; the current
pylovo-V1 diagnostics are real SWF 28 and its mapped synthetic target 151.\n
The shared heuristic plan must emit:

- post-hems-heuristic: optimized dispatch of fixed heuristic assets;
- post-inflex-heuristic: rule-based dispatch of identical assets;
- pre: reference electricity demand.

Pass: optimal solvers; no silently skipped power-flow timesteps; installed
capacity equals upper capacity for every heuristic PV, HP, auxiliary, battery,
and buffer row; HEMS and INFLEX asset plans are identical after result labels
are removed; selected-unit annual energy is conserved; provenance distinguishes
the two dispatch cases.

## Gate 2: optimized real/synthetic smoke comparison

Repeat both adapter targets with `post-hems-optimized`. A standalone
optimized smoke emits `pre` and `post-hems-optimized`; when it follows the
heuristic group in one configured invocation, `pre` is intentionally skipped
because it was already emitted. No optimized INFLEX reconstruction is valid.

Pass: optimal solvers; finite non-negative capacities within LoD2, HTW, and heat
bounds; no roof outside the selected target; no fixed-capacity assertion for
endogenous assets; explicit power-flow convergence accounting.

## Gate 3: capacity and energy-conservation audit

Aggregate by physical building, scenario unit, target bus, and whole target:

1. heuristic PV equals min(E_base * 2.5 / 1000, pv_max_kwp);
2. optimized PV lies between zero and pv_max_kwp;
3. heuristic batteries reproduce the HTW equation and 2 h E/P ratio;
4. optimized batteries remain within potential-based bounds;
5. heuristic HP, auxiliary, and buffer reproduce the documented rules;
6. optimized heat assets remain within finite bounds;
7. base electricity, space heat, DHW, and mobility energy are conserved through
   scenario-unit and bus aggregation;
8. PV generation equals normalized profile times selected capacity;
9. shared buildings or buses introduce no duplicated physical asset;
10. real and synthetic contracts are identical before bus projection.

Use absolute tolerance 1e-6 for deterministic capacity and energy
transformations and report a separate relative tolerance for solver results.
Any unexplained mismatch blocks release.

## Gate 4: launch decision

Report smoke evidence, audit results, warnings, expected job count, resource
settings, and output footprint. Launch only after explicit user authorization.
The authoritative paired YAML launches the requested groups sequentially.
Their output directories are derived automatically as `heuristic-assets/` and
`post-hems-optimized/` beneath `GridExpand/run_logs/<run.id>/`.
