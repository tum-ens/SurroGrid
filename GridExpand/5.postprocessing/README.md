# 5. Postprocessing

This step contains the DB-backed analysis, plotting, and expansion-cost materialization tools for GridExpand results. It reads power-flow outputs from the `surrogrid` PostgreSQL schema and writes only derived analysis tables, figures, or QGIS views.

## Setup

```bash
cd GridExpand/5.postprocessing
uv sync
```

Database connection details are read from the repository-level `.env` through `SurroGridDatabase`.

## Module Layout

Step 5 is split by workflow responsibility:

- `powerflow.comparison_data`: load compact synthetic and real power-flow summaries, build comparison datasets, and compute similarity tables.
- `plotting.*`: create figures from prepared data. Notebooks import the concrete plotting modules directly.
- `audits.*`: diagnostic scripts for demand allocation and topology bottlenecks.
- `expansion.*`: materialize expansion summaries/costs and load expansion overview tables.

Import from the owning module instead of using compatibility facades. For example, use `powerflow.comparison_data` for DB loaders and `plotting.powerflow_asset_plots` for asset stress plots.

## Main Files

```text
5.postprocessing/
  pyproject.toml
  powerflow/
    comparison_data.py                    # DB loaders and synthetic/real comparison datasets
  notebooks/
    asset_powerflow_comparison.ipynb      # real/synthetic power-flow comparison notebook
    expansion_analysis.ipynb              # expansion analysis notebook
    grid_area_envelope_comparison.ipynb   # real/synthetic grid-area envelope notebook
    timeseries_plotting.ipynb             # DB-backed time-series diagnostics notebook
  plotting/
    powerflow_heatmaps.py                 # HDF/DB timestep heatmaps and line-loading CLI helpers
    powerflow_asset_plots.py              # asset cutoff, percentile, and violin plot functions
    powerflow_voltage.py                  # voltage deviation summaries and plots
    powerflow_transformer.py              # transformer import and stage-comparison plots
    powerflow_io.py                       # shared Plotly export helper
    geoplotting.py                        # envelope and geospatial plotting helpers
  audits/
    demand.py                             # real/synthetic annual demand diagnostics
    topology_bottleneck.py                # critical voltage path/bottleneck audit
    topology_whatif.py                    # read-only topology what-if checks
  expansion/
    grid_expansion.py                     # CLI/orchestration for expansion-cost materialization
    overview.py                           # read-only expansion summary loaders for notebooks
    materialize_powerflow_summary.py      # derive compact summaries from stored raw rows
    schema.sql                            # expansion tables, assumptions, QGIS views
    cost_assumptions.md                   # cost-assumption notes
```

## Command Summary

Run the full AGS pipeline and store raw pre/post power-flow time series:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/runme/ags_pipeline_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output raw \
  --run-dir GridExpand/run_logs/<run_name>
```

Run the pipeline but store only compact summaries, not raw time series:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/runme/ags_pipeline_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output summary \
  --run-dir GridExpand/run_logs/<run_name>
```

Run a full-year all-assets scenario with TSAM typical weeks and compact post/pre power-flow summaries:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/runme/ags_pipeline_runner.py \
  --repo-root . \
  --ags <AGS> \
  --profiles all \
  --timeframe-mode full_year \
  --powerflow-output summary \
  --tsam \
  --tsam-periods 6 \
  --tsam-hours-per-period 168 \
  --tsam-extreme-method replace_cluster_center \
  --run-dir GridExpand/run_logs/<run_name> \
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
cd GridExpand/5.postprocessing
uv run python -m expansion.materialize_powerflow_summary \
  --run-name <raw_powerflow_run_name> \
  --stages post \
  --ags <AGS> \
  --plz <PLZ> \
  --replace
```

Create or update the expansion schema and QGIS views:

```bash
cd GridExpand/5.postprocessing
uv run python -m expansion.grid_expansion --schema-only
```

Materialize expansion costs from stored raw post power-flow results:

```bash
cd GridExpand/5.postprocessing
uv run python -m expansion.grid_expansion \
  --run-name <raw_powerflow_run_name> \
  --stage post \
  --ags <AGS> \
  --plz <PLZ> \
  --analysis-key <analysis_key> \
  --replace
```

Which Path Should I Use?

- Use `--powerflow-output summary` when storage should stay small and the notebooks only need compact stress metrics; combine it with `--tsam` for faster full-year screening. This now also creates `expansion_analysis_run` rows automatically.
- Use `--powerflow-output raw` when later expansion-cost materialization, detailed diagnostics, or custom postprocessing are needed.
- Use `materialize_powerflow_summary.py` only when raw time series already exist and compact notebook summaries are missing or stale.
- Use `grid_expansion.py` manually only when you need to re-materialize or rename expansion-cost estimates; the normal summary pipeline does this automatically from compact summary rows.

## Expansion Outputs

`grid_expansion.py` writes one analysis run plus derived cable and transformer expansion rows. It also refreshes:

- `surrogrid.expansion_line_qgis_mv`
- `surrogrid.expansion_transformer_qgis_mv`

Useful fields for QGIS or notebooks:

- `analysis_key`: filter for one materialized analysis.
- `requires_expansion`: true if the heuristic adds capacity.
- `loading_percent`: critical peak loading.
- `estimated_cost_eur`: heuristic expansion cost.
- `additional_parallel`: additional cable parallels.
- `additional_transformer_kva`: additional transformer capacity.

The default cost assumptions are documented in `expansion/cost_assumptions.md`. They are screening assumptions, not construction estimates.

## Notes

- Compact summaries can now represent `post` electrification results. Older summary-only runs may contain only `pre` rows.
- Expansion notebook envelope plots use `plotting/geoplotting.py`; OSM background tiles require `contextily` and network access at plot time.
- Keep raw run names, compact summary run names, and expansion `analysis_key`s explicit in notes or run logs. This is the easiest way to avoid mixing analysis generations.
