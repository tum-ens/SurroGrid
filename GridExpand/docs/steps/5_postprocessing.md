# Step 5: expansion analysis and postprocessing

Step 5 reads power-flow results from the `surrogrid` schema and turns them into expansion measures and costs. It
also provides the loaders and plots of the result analysis. It writes only derived analysis tables, QGIS
views, figures and audit exports.

The rules are the staged rules (`staged_2026`): station limit, capped cables, load transfer to neighbouring
stations, new substations, and residual voltage. The former heuristic (`heuristic_2026`) was retired on
2026-09-29. See [expansion_costs.md](../expansion_costs.md).

Code: `src/gridexpand/analysis/`

| module | content |
|---|---|
| `expansion/grid_expansion.py` | `gridexpand expansion`: one analysis per run name and stage, one transaction |
| `expansion/staged.py` | the rules, rule set `staged_2026` (stages 1–5, cable search, route cost; pure functions) |
| `expansion/topology.py` | station busbar, radial tree, outlets, service lines, bus coordinates (both sources) |
| `expansion/results.py` | row fields: route decisions, station-level cost breakdown, `expansion_grid_result` rows |
| `expansion/synthetic_materialization.py` | synthetic grids: rules in Python; decisions in temp tables for the SQL inserts |
| `expansion/sql/*.sql` | synthetic scope and mapping: selected runs, component loading, transformer peaks, line and transformer inserts |
| `expansion/real_materialization.py` | real DSO A / DSO B grids: cable corridors, grid status |
| `expansion/cases.py` | model case → stage and analysis-key suffix |
| `expansion/overview.py` | read-only loaders of expansion analyses |
| `powerflow/raw.py`, `powerflow/scope.py` | raw-series loaders, run/scope resolution |
| `plotting/*` | figures (voltage, transformer import, heatmaps, expansion costs) |

## Expansion materialization

Pipelines materialize their analyses automatically at the end (`materialize_expansion`), passing `--no-refresh`
and refreshing the QGIS views once per batch. Manual use:

```bash
uv run gridexpand expansion --run-name <power-flow run name> --stage post --ags 9184137 \
  --analysis-key <key> --replace
uv run gridexpand expansion --data-source real_dso_a --run-name <real run name> --stage post --plz 99999 \
  --exclude-real-lv-id 113 --analysis-key <key> --replace
uv run gridexpand expansion --refresh-only        # refresh the QGIS materialized views
```

| option | default | meaning |
|---|---|---|
| `--run-name` | required (except `--schema-only`, `--refresh-only`) | power-flow run whose compact summaries are materialized |
| `--stage` | `post` | `pre` or `post` |
| `--data-source` | `synthetic` | `synthetic`, `real_dso_a`, `real_dso_b` |
| `--ags`, `--plz` | none | filters (repeatable); synthetic AGS or grid PLZ, real majority PLZ |
| `--scenario-id` | the single scenario of the selected runs | scenario recorded with the analysis |
| `--pylovo-version-id` | run assumptions | synthetic grid-case filter; real-grid settlement type |
| `--exclude-real-lv-id` | none | real grid kept in coverage but not costed (repeatable) |
| `--assumption-key` | `de_lv_staged_2026` | cost row of `expansion_cost_assumption`; its `rule_set` picks the rules. Sensitivity rows: `_low`, `_high`, `_trafo_allin`, `_load_factor_1_0`, `_no_transfer` ([expansion_costs.md](../expansion_costs.md)) |
| `--line-existing-duct-share` | from the assumption | existing-duct share override |
| `--analysis-key` | `<ags or all>_<run>_<stage>_<UTC stamp>` | readable key |
| `--note` | empty | free text |
| `--replace` | off | replace an analysis of the same key; refused if it has another run name, stage, data source or AGS |
| `--no-refresh`, `--refresh-only` | | skip the QGIS refresh; only refresh |
| `--schema-only` | | only initialise a fresh database and exit |

Analysis keys written by the pipelines:

| pipeline | key |
|---|---|
| synthetic | `<ags:08d>_<scenario key>_<timeframe>_<profiles>[_hh_only][_tsam][_<model case>]_<pre\|post>` (override the prefix with `gridexpand synthetic --expansion-analysis-prefix`) |
| paired_aligned (validation) | `<run.id>_<provider>_<real\|synthetic>_<pre\|post_inflex\|post\|post_hems_optimized>` |

The validation materializes all provider groups of an aligned run with `python -m validation.aligned_expansion`
([validation/README.md](../../validation/README.md)).

**Method in short** ([expansion_costs.md](../expansion_costs.md)), rule set `staged_2026`:
1. **Station.** Transformer exchange up to 1000 kVA in every settlement type, at literature unit costs.
2. **Routes.** At most 3 added NAYY cables per route, at their nominal ratings.
   Service lines are costed separately and stay out of the total.
3. **Panels.** Reported per grid.
4. **Grids above the station limit or with an unresolvable route** (per analysis). The excess goes to neighbouring
   stations with spare capacity (load transfer), else neighbouring grids share whole new substations. Both relieve
   the routes that need two or more cables.
5. **Residual voltage** after the Step 4 tap: an rONT, or a feeder split at 2/3.

The rules run in Python for both sources and use the same assumption row. Real grid-stages with failed
power-flow timesteps in fewer than 1 % of the timesteps are costed from the converged ones (a lower bound, noted in the
status reason); with 1 % or more they are `incomplete` (no cost rows, not zero cost); explicit exclusions are
`excluded`.

## Outputs

Tables ([database.md](../database.md#step-5-expansion)): `expansion_analysis_run`, `expansion_line_result`,
`expansion_transformer_result`, `expansion_real_grid_status`, `expansion_real_line_result`,
`expansion_real_transformer_result`, `expansion_grid_result` (one row per grid with the decision trail and every
cost component). QGIS views (synthetic grids, pylovo geometry): `expansion_line_qgis_mv`,
`expansion_transformer_qgis_mv`. Useful fields: `analysis_key`, `requires_expansion`, `loading_percent`,
`estimated_cost_eur`, `additional_parallel`, `reinforcement_150_count` / `_185_count` / `_240_count`,
`additional_transformer_kva`, `critical_ts`, the cost-basis columns (`critical_component_cost_basis`,
`transformer_cost_basis`, ...), and the staged columns: `measure`, `service_cost_eur`, `station_measure`, and the
breakdown `transformer_exchange_cost_eur`, `load_transfer_cost_eur`, `new_station_cost_eur`, `voltage_cost_eur`.
The transformer row's `estimated_cost_eur` carries every station-level cost, so line plus transformer rows add up to
the grid total.

Files go below `work/analysis/output/` (plots under `plots/`, audits under `audits/<workflow>/`); callers choose the
destination of plots.

## Notebooks

The analysis notebooks of the real-vs-synthetic comparison (`analysis_powerflow.ipynb`, `analysis_expansion.ipynb`,
`grid_area_envelope_comparison.ipynb`) are in [validation/notebooks](../../validation/notebooks). Step 1's sampling
notebooks are in `notebooks/sampling/`.

## Plots from the command line

`python -m gridexpand.analysis.plotting.powerflow_heatmaps` plots one grid from an HDF5 file or, with
`--storage db --run-name <run>`, from the database.
