# 5. Postprocessing

This step contains the DB-backed analysis, plotting, and expansion-cost materialization tools for GridExpand results. It reads power-flow outputs from the `surrogrid` PostgreSQL schema and writes only derived analysis tables, figures, or QGIS views.

## Setup

```bash
cd GridExpand
uv sync --extra notebooks
```

Database connection details are read from `GridExpand/.env` (or `$GRIDEXPAND_ENV_FILE`) through `SurroGridDatabase`.

## Module Layout

Step 5 (`gridexpand.analysis`) is split by workflow responsibility:

- `powerflow.comparison_data`: load compact synthetic and real power-flow summaries, build comparison datasets, compute similarity tables, and summarize component-category/mixed-use exposure from the compact Step-2 audit. `demand_component_exposure_summary_db()` reports category energy and effective-area shares, MV-direct shares, mixed-use counts, and low-confidence classification rows.
- `plotting.*`: create figures from prepared data. Notebooks import the concrete plotting modules directly.
- `audits.topology_bottleneck`: reusable diagnostics for critical real-grid voltage paths and bottlenecks.
- `audits.feeder_structure`: graph-normalized feeder, downstream-demand, path-depth, and physical-corridor comparison.
- `expansion.*`: materialize expansion summaries/costs and load expansion overview tables.

Import from the owning module instead of using compatibility facades. For example, use `powerflow.comparison_data` for DB loaders and `plotting.powerflow_asset_plots` for asset stress plots.

## Output Layout

All generated Step 5 files belong below `work/analysis/output/`. Plotting modules do not own an output directory; notebooks and callers provide a destination below `work/analysis/output/plots/`. Audit CLIs default to `work/analysis/output/audits/<workflow>/`. Scenario-calibration exports remain owned by Step 2 under `work/allocation/outputs/` and must not be duplicated in Step 5.

```text
output/
  plots/
    asset_powerflow/
    expansion/<AGS>/<scenario_prefix>/
  audits/
    building_coverage/
    feeder_structure/<AGS>/<scenario_prefix>/
    topology/
```

The entire `work/` tree is generated and ignored by Git. Durable conclusions and methodological decisions belong in the Markdown files under `docs/audits/` or `docs/expansion/`, not only in generated CSV or figure files.

## Main Files

```text
src/gridexpand/analysis/
  powerflow/
    comparison_data.py                    # DB loaders, comparison datasets, and component exposure summaries
  plotting/
    powerflow_heatmaps.py                 # HDF/DB timestep heatmaps and single-grid loading CLI
    powerflow_asset_plots.py              # asset cutoff, percentile, and violin plot functions
    powerflow_voltage.py                  # voltage deviation summaries and plots
    powerflow_transformer.py              # transformer import and stage-comparison plots
    powerflow_io.py                       # shared Plotly export helper
    geoplotting.py                        # envelope and geospatial plotting helpers
  audits/
    feeder_structure.py                  # synthetic/real feeder and physical-corridor audit
    topology_bottleneck.py                # critical voltage path/bottleneck audit
  expansion/
    grid_expansion.py                     # source-neutral CLI/orchestration for expansion costs
    real_materialization.py                # real SWF/ÜZW asset adapter for the shared cost heuristic
    aligned_expansion.py                  # one-command expansion of all provider groups of an aligned run
    overview.py                           # read-only expansion summary loaders for notebooks
    materialize_powerflow_summary.py      # derive compact summaries from stored raw rows
src/gridexpand/db/sql/expansion_schema.sql  # expansion tables, assumptions, QGIS views
notebooks/analysis/
  analysis_powerflow.ipynb               # paired real/synthetic status-quo power-flow analysis
  analysis_expansion.ipynb               # paired pre/post-flex/post-inflex expansion analysis
  grid_area_envelope_comparison.ipynb    # spatial supplied-area diagnostic
docs/audits/feeder_structure_comparison.md  # current paired-scenario findings and interpretation
docs/expansion/assumptions_costs.md      # cost assumptions and source evidence
docs/expansion/assumptions_scenario.md   # authoritative scenario-run summary
```

## Command Summary

Run the full AGS pipeline and store raw pre/post power-flow time series:

```bash
uv run gridexpand synthetic \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output raw \
  --run-dir work/runs/<run_name>
```

Run the pipeline but store only compact summaries, not raw time series:

```bash
uv run gridexpand synthetic \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output summary \
  --run-dir work/runs/<run_name>
```

Run a full-year all-assets scenario with TSAM typical weeks and compact post/pre power-flow summaries:

```bash
uv run gridexpand synthetic \
  --ags <AGS> \
  --profiles all \
  --timeframe-mode full_year \
  --powerflow-output summary \
  --scenario-config config/scenarios/forchheim_2045_full_year.yaml \
  --run-dir work/runs/<run_name> \
  --workers 1 \
  --step2-cpus 4 \
  --step3-cpus 32 \
  --step3-max-cpus 32 \
  --step3-cluster-concurrency 16 \
  --step4-cpus 4
```

Add `--demand-scope residential` to run the same Step 2 to Step 4 pipeline on household/residential buildings only. The runner then passes `--hh-only` to Step 4, uses the `baseline_static_hh_only` scenario key family, and appends `_hh_only` to automatic expansion analysis keys.

With TSAM enabled, Step 4 uses the reduced demand horizon written by Step 3, so the power-flow summaries cover `6 * 168 = 1008` representative hours instead of a reconstructed 8760-hour series. Summary and both-mode pipeline runs automatically materialize expansion analyses at the end; use `--no-materialize-expansion` to disable this.

Use `--powerflow-output both` if both raw time series and compact summaries should be written during one run.

If raw time series already exist, derive compact summaries without rerunning pandapower:

```bash
uv run python -m gridexpand.analysis.expansion.materialize_powerflow_summary \
  --run-name <raw_powerflow_run_name> \
  --stages post \
  --ags <AGS> \
  --plz <PLZ> \
  --replace
```

Create or update the expansion schema and QGIS views:

```bash
uv run gridexpand expansion --schema-only
```

Materialize expansion costs from stored compact power-flow summaries:

```bash
uv run gridexpand expansion \
  --run-name <raw_powerflow_run_name> \
  --stage post \
  --ags <AGS> \
  --plz <PLZ> \
  --analysis-key <analysis_key> \
  --replace
```

Materialize an equivalent real SWF analysis from compact summaries and the exported pandapower assets:

```bash
uv run gridexpand expansion \
  --data-source real_swf \
  --run-name <real_powerflow_run_name> \
  --stage post \
  --plz <PLZ> \
  --analysis-key <analysis_key> \
  --exclude-real-lv-id <LV_ID> \
  --replace
```

`--data-source real_uzw` materializes ÜZW areas (pandapower JSON, `real_grid_case.source = 'uzw'`) the same way. Real selection always filters on the source; `--plz`, `--ags` and `--exclude-real-lv-id` are repeatable, grid ids are text, and `--pylovo-version-id` selects the synthetic grid cases and the real settlement type (real runs otherwise use the version recorded in the run assumptions).

Materialize all four groups (SWF real, SWF synthetic, ÜZW real, ÜZW synthetic) of an aligned run (`src/gridexpand/scenario/run_aligned.py`, run names `{run_id}_{provider}_{real_<provider>|synthetic}_{case}`) with one command. Analysis keys are `{run_id}_{provider}_{real|synthetic}_{pre|post_inflex|post}`; `--dry-run` only lists the groups and their summary counts:

```bash
uv run python -m gridexpand.analysis.expansion.aligned_expansion \
  --run-id <run_id> \
  --providers swf uzw \
  --cases pre post-inflex-heuristic post-hems-heuristic
```

In notebooks, `prepare_expansion_analysis(scenario_prefix=<run_id>, providers=("swf", "uzw"))` returns the four groups in `specs_by_source` and `analysis_keys_by_source`; pass `specs_by_source=...` to the power-flow loaders in `expansion.notebook_workflow`.

Real grids with non-converged timesteps remain in `expansion_real_grid_status` with `cost_status=incomplete`; they are not assigned zero cost. Explicit exclusions remain visible with `cost_status=excluded`. Synthetic and real results use the same row from `expansion_cost_assumption`. Existing cables are retained; added circuits are selected from the shared `NAYY_4_150`, `NAYY_4_185`, and `NAYY_4_240` catalogue.

Which Path Should I Use?

- Use `--powerflow-output summary` when storage should stay small and the notebooks only need compact stress metrics; enable representative periods in the Scenario YAML for faster full-year screening. This now also creates `expansion_analysis_run` rows automatically.
- Use `--powerflow-output raw` for detailed timestep diagnostics or custom postprocessing. Expansion-cost materialization now works from compact summary rows for both synthetic and real SWF runs.
- Use `python -m gridexpand.analysis.expansion.materialize_powerflow_summary` only when raw time series already exist and compact notebook summaries are missing or stale.
- Use `gridexpand expansion` manually only when you need to re-materialize or rename expansion-cost estimates; the normal summary pipeline does this automatically from compact summary rows.

## Expansion Outputs

`grid_expansion.py` writes one source-labelled analysis run plus derived cable and transformer expansion rows. Synthetic rows retain pylovo foreign keys; real rows retain SWF pandapower asset ids and source geometry. It also refreshes the existing synthetic QGIS views:

- `surrogrid.expansion_line_qgis_mv`
- `surrogrid.expansion_transformer_qgis_mv`

Useful fields for QGIS or notebooks:

- `analysis_key`: filter for one materialized analysis.
- `requires_expansion`: true if the heuristic adds capacity.
- `loading_percent`: critical peak loading.
- `estimated_cost_eur`: heuristic expansion cost.
- `additional_parallel`: total selected additional cable circuits.
- `reinforcement_150_count`, `reinforcement_185_count`, and `reinforcement_240_count`: selected standard reinforcement cables.
- `additional_transformer_kva`: additional transformer capacity.

The default cost assumptions are documented in `docs/expansion/assumptions_costs.md`. They are screening assumptions, not construction estimates.

## Notes

- Compact summaries can now represent `post` electrification results. Older summary-only runs may contain only `pre` rows.
- `grid_area_envelope_comparison.ipynb` is the dedicated spatial diagnostic for comparing convex supplied-area envelopes; it is not duplicated in the main power-flow or expansion notebooks.
- Expansion notebook envelope plots use `gridexpand/analysis/plotting/geoplotting.py`; OSM background tiles require `contextily` and network access at plot time.
- Keep raw run names, compact summary run names, and expansion `analysis_key`s explicit in notes or run logs. This is the easiest way to avoid mixing analysis generations.
