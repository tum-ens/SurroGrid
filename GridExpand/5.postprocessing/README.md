# 5. Postprocessing

This step is the analysis and plotting workspace for GridExpand results. It is intentionally separate from Step 4 so the power-flow simulation environment stays focused while postprocessing can carry notebook, plotting, export, and presentation dependencies.

## Folder & file structure

```text
5.postprocessing/
  pyproject.toml              # uv environment for result analysis and plotting
  expansion/
    cost_assumptions.md       # source notes and derivation for cost defaults
    grid_expansion.py         # DB-backed overload/cost materialization for QGIS
    schema.sql                # expansion tables, assumptions, and QGIS materialized views
  plotting/
    plotting_notebook.ipynb   # current DB-backed plotting notebook
    thesis_plots.ipynb        # original reference notebook for transformer distributions
    powerflow_plotting.py     # reusable plotting helpers and CLI
```

## Setup

```bash
cd GridExpand/5.postprocessing
uv sync
```

The notebook can use the kernel from this uv environment through `ipykernel`. Database-backed plots read connection details from the repository `.env` file through `SurroGridDatabase`.

## Data sources

The plotting helpers currently read:

- Step 4 HDF5 files from `../4.powerflow/Output/` for file-backed heatmaps.
- Step 4 PostgreSQL result tables in the `surrogrid` schema for DB-backed heatmaps, voltage deviation histograms, transformer import distributions, and loading comparisons.

Step 5 should not write simulation results back into the pipeline. Use it for notebooks, figures, exports, and result-analysis scripts.

## Expansion Heuristic

`expansion/grid_expansion.py` materializes overload-based grid expansion estimates into the database for large-region analysis. It reduces hourly Step 4 power-flow rows inside PostgreSQL, then stores one row per QGIS-visible cable segment and one row per transformer.

The QGIS-facing materialized views are:

- `surrogrid.expansion_line_qgis_mv`: cable expansion estimates joined to `pylovo.lines_result_with_grid`.
- `surrogrid.expansion_transformer_qgis_mv`: transformer expansion estimates joined to `pylovo.transformer_positions_with_grid`.

The materialized views are refreshed by `expansion/grid_expansion.py` after each materialization run and include a stable `qgis_id` column for QGIS.

Create the schema and default assumptions:

```bash
cd GridExpand/5.postprocessing
uv run python -m expansion.grid_expansion --schema-only
```

Materialize Munich post-expansion estimates from a power-flow run:

```bash
cd GridExpand/5.postprocessing
uv run python -m expansion.grid_expansion \
  --ags 09162000 \
  --run-name baseline_static_min_temperature_week_full_powerflow \
  --stage post \
  --analysis-key munich_min_temperature_week_post \
  --replace
```

Load the two QGIS materialized views from PostgreSQL and filter on `analysis_key = 'munich_min_temperature_week_post'`. Useful styling fields are:

- `requires_expansion`: true when the simulated peak exceeds nominal existing capacity and therefore needs an additional cable parallel or transformer kVA step.
- `overloaded_at_100_percent`: true when the simulated peak exceeds the existing nominal capacity.
- `loading_percent`: peak loading of the critical mapped line or transformer.
- `estimated_cost_eur`: heuristic reinforcement cost.
- `additional_parallel`: additional cable parallels required for the visible cable segment.
- `line_cost_basis` and `line_cost_eur_per_km`: selected literature-backed cable cost tier.
- `additional_transformer_kva`: additional transformer capacity required.
- `transformer_cost_basis`: selected all-in transformer replacement tier.

### Method

Cable results start from `surrogrid.powerflow_line_result` and are aggregated to peak current per pandapower line. They are mapped to original pylovo line rows by line name, then rolled up to the visible geometry in `pylovo.lines_result_with_grid`. This keeps merged feeder sections QGIS-friendly while avoiding Python-side loading of large time series.

Transformer results use `surrogrid.powerflow_import` as the transformer loading proxy because Step 4 replaces the transformer with a switch before running pandapower. Apparent import `sqrt(P^2 + Q^2)` is compared with `pylovo.grid_result.transformer_rated_power`, falling back to `transformer_positions_with_grid.s_max_kva` where available.

The default assumption row is `de_lv_heuristic_2026`. The detailed source/value derivation is documented in `expansion/cost_assumptions.md`.

Current defaults:

- Expansion threshold: 100% nominal loading.
- Parallel cable in existing route/duct: 25,000 EUR/km for <=150 mm2, 45,000 EUR/km for 185 mm2, 70,000 EUR/km for 240 mm2 or unknown size.
- Reopened-route references are stored for sensitivity: 90,000 EUR/km rural, 95,000 EUR/km suburban, 165,000 EUR/km urban.
- All-in transformer replacement bins: 33,000 EUR to 400 kVA, 38,000 EUR to 630 kVA, 42,000 EUR to 800 kVA, 48,000 EUR to 1,000 kVA.
- Full station-rebuild boundary case: 100,000 EUR.
- Transformer capacity is rounded to 50 kVA steps.

These are screening assumptions, not a construction estimate. They intentionally use the simplest transparent capacity rule: reinforce only when peak loading exceeds the nominal rating. Cable costs assume added parallel cable in an existing route/duct; reopened paved routes should be evaluated with the stored sensitivity values. Detailed planning still needs voltage checks, protection checks, route feasibility, soil/installation conditions, simultaneity assumptions, switching state validation, DSO-specific reserve criteria, and DSO-specific unit costs.

### Research Notes

The heuristic follows the current state-of-practice direction of using digital grid models and time-series load-flow results to identify cable and transformer bottlenecks, then assigning conventional reinforcement when equipment ratings or planning reserves are exceeded.

Sources used for the default assumptions and context:

- dena Verteilnetzstudie II, 2025: distribution-grid transformation needs more investment, better planning cooperation, and improved data quality.
- BNetzA-reported German DSO plans summarized by Clean Energy Wire, 2024: about 110 billion EUR distribution-grid expansion need by 2033, plus about 90 billion EUR from 2034 to 2045, excluding additional replacement investments.
- Wintzek 2021, Agora/FfE 2023, and WEI/GridSim 2025 synthesis: direct brownfield parallel-cable and transformer replacement cost tiers.
- dena Verteilnetzstudie II Gutachten, 2025: recent DSO-informed NS-line and station-cost boundary checks.
- VDE study on higher utilization of grid assets, 2024: thermal reserves can exist, but higher loading accelerates equipment aging and needs monitoring, diagnostics, and accepted technical rules.
