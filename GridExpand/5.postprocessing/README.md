# 5. Postprocessing

This step contains the DB-backed analysis, plotting, and expansion-cost materialization tools for GridExpand results. It reads power-flow outputs from the `surrogrid` PostgreSQL schema and writes only derived analysis tables, figures, or QGIS views.

## Setup

```bash
cd GridExpand/5.postprocessing
uv sync
```

Database connection details are read from the repository-level `.env` through `SurroGridDatabase`.

## Main Files

```text
5.postprocessing/
  pyproject.toml
  expansion/
    expansion_analysis.ipynb              # expansion analysis notebook
    grid_expansion.py                     # materialize expansion costs from raw power-flow rows
    materialize_powerflow_summary.py      # derive compact summaries from stored raw rows
    schema.sql                            # expansion tables, assumptions, QGIS views
    cost_assumptions.md                   # cost-assumption notes
  plotting/
    powerflow_plotting.py                 # reusable power-flow plotting helpers
    geoplotting.py                        # envelope and geospatial plotting helpers
    asset_powerflow_comparison.ipynb      # real/synthetic power-flow comparison notebook
```

## Command Summary

Run the full AGS pipeline and store raw pre/post power-flow time series:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/ags_pipeline_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output raw \
  --run-dir GridExpand/run_logs/<run_name>
```

Run the pipeline but store only compact summaries, not raw time series:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/ags_pipeline_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles electricity_heat \
  --timeframe-mode min_temperature_week \
  --powerflow-output summary \
  --run-dir GridExpand/run_logs/<run_name>
```

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

- Use `--powerflow-output summary` when storage should stay small and the notebooks only need compact stress metrics.
- Use `--powerflow-output raw` when later expansion-cost materialization, detailed diagnostics, or custom postprocessing are needed.
- Use `materialize_powerflow_summary.py` only when raw time series already exist and compact notebook summaries are missing or stale.
- Use `grid_expansion.py` only for expansion-cost estimates; it requires raw `post` power-flow rows.

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
