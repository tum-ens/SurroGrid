# Step 5: expansion analysis and postprocessing

Step 5 reads power-flow results from the `surrogrid` schema and turns them into expansion measures and costs. It
also provides the loaders, plots and notebooks of the result analysis. It writes only derived analysis tables, QGIS
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
| `expansion/real_materialization.py` | real SWF / ÜZW grids: cable corridors, grid status |
| `expansion/aligned_expansion.py` | all provider groups of an aligned run in one command |
| `expansion/cases.py` | model case → stage and analysis-key suffix |
| `expansion/overview.py`, `expansion/notebook_workflow.py` | read-only loaders for the notebooks |
| `powerflow/comparison_data.py`, `powerflow/raw.py`, `powerflow/scope.py` | summary and raw-series loaders, run/scope resolution |
| `plotting/*` | figures (asset cutoff and percentile plots, voltage, transformer import, heatmaps, expansion costs, geoplots) |
| `audits/topology_bottleneck.py`, `audits/feeder_structure.py` | real-grid voltage-path audit (CLI), feeder-structure comparison (functions) |

## Expansion materialization

Pipelines materialize their analyses automatically at the end (`materialize_expansion`), passing `--no-refresh`
and refreshing the QGIS views once per batch. Manual use:

```bash
uv run gridexpand expansion --run-name <power-flow run name> --stage post --ags 9184137 \
  --analysis-key <key> --replace
uv run gridexpand expansion --data-source real_swf --run-name <real run name> --stage post --plz 91301 \
  --exclude-real-lv-id 113 --analysis-key <key> --replace
uv run gridexpand expansion --refresh-only        # refresh the QGIS materialized views
```

| option | default | meaning |
|---|---|---|
| `--run-name` | required (except `--schema-only`, `--refresh-only`) | power-flow run whose compact summaries are materialized |
| `--stage` | `post` | `pre` or `post` |
| `--data-source` | `synthetic` | `synthetic`, `real_swf`, `real_uzw` |
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
| paired_validation | `<run.id>[_real]_<pre\|post_inflex\|post\|post_hems_optimized>` |
| paired_aligned | `<run.id>_<provider>_<real\|synthetic>_<pre\|post_inflex\|post\|post_hems_optimized>` |

Aligned runs (all four groups SWF real/synthetic, ÜZW real/synthetic, or a subset):

```bash
uv run python -m gridexpand.analysis.expansion.aligned_expansion --run-id joint_2045_v1_smoke24 \
  --providers swf uzw --cases pre post-inflex-heuristic post-hems-heuristic [--dry-run]
```

(`--pylovo-version-id`, `--exclude-real-grid PROVIDER:ID` repeatable; power-flow run names
`<run.id>_<provider>_<real_<provider>|synthetic>_<case>`.)

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
power-flow timesteps are `incomplete` (no cost rows, not zero cost); explicit exclusions are `excluded`.

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

```bash
uv sync --extra notebooks
uv run jupyter lab notebooks/analysis
```

- `analysis_powerflow.ipynb`: paired real/synthetic status-quo power-flow analysis;
- `analysis_expansion.ipynb`: paired pre/post expansion analysis. `prepare_expansion_analysis(scenario_prefix=<run.id>,
  providers=("swf", "uzw"))` (in `expansion.notebook_workflow`) prepares the four groups of an aligned run and
  enforces consistent provenance (temporal method, run readiness);
- `grid_area_envelope_comparison.ipynb`: supplied-area envelope diagnostic (OSM tiles need `contextily` and
  network access).

`notebooks/archive/` keeps run-bound notebooks with their outputs as a historical record; they are not maintained.

## Audits

```bash
uv run python -m gridexpand.analysis.audits.topology_bottleneck --real-run-name <real run> --plz 91301
```

(`--stage` default `pre`, `--voltage-threshold` default 0.90, `--output-dir` default
`work/analysis/output/audits/topology`; writes `critical_grid_summary.csv`, `critical_path_lines.csv` and
`critical_path_alternative_lines.csv`.)

`audits.feeder_structure` (graph-normalized feeder, downstream-demand and path-depth comparison) and the plotting
modules are used from notebooks; `python -m gridexpand.analysis.plotting.powerflow_heatmaps` plots one grid from
an HDF5 file or, with `--storage db --run-name <run>`, from the database. Findings of earlier audits are in [docs/research/](../research/).
