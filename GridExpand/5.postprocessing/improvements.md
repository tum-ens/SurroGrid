# Future Improvements

## Expansion Notebook: Derive Stage From Analysis Key

`expansion_analysis.ipynb` currently has room for user-side ambiguity when switching analyses: users can set an `ANALYSIS_KEY` while also keeping a separate `SUMMARY_STAGE` or stage-specific notebook variable. Each `expansion_analysis_run` already stores `stage`, `run_name`, and assumptions, so the notebook should make `analysis_key` the source of truth and derive the stage from `expansion.overview.load_expansion_overview(...)`. Explicit pairs such as `PRE_ANALYSIS_KEY` / `POST_ANALYSIS_KEY` should only be used for side-by-side comparisons.

## Power-Flow Plotting: Legacy Function Cleanup

The retained asset-cutoff comparison notebook superseded the older standalone percentile/violin plotting workflow. The stale public helpers were removed from `plotting.powerflow_asset_plots` and `powerflow.comparison_data`. Keep `real_powerflow_percentile_profile_db` because `load_powerflow_comparison_data(...)` still uses it to build the active comparison dataset.
