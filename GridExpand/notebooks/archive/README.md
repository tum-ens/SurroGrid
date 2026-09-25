# Archived notebooks (historical, not maintained)

These notebooks are kept with their stored outputs as a record of earlier runs. They are
bound to data that no longer exists (deleted pylovo versions 3 and 100, run manifests
and intermediate files under `work/runs/<run>/`, SWF-only runs) and to APIs as they were
when the outputs were produced. They are not updated when the package changes and are
not expected to run.

| notebook | bound to |
|---|---|
| `schweinfurt_2045_synthetic_regional.ipynb` | Schweinfurt synthetic regional run, pylovo version 100, `baseline_static_*` run names |
| `schweinfurt_2045_synthetic_regional_v3.ipynb` | the same run on pylovo version 3 |
| `forchheim_2045_synthetic_smoke_grid20.ipynb` | `work/runs/forchheim_2045_synthetic_smoke_grid20/run_manifest.json` and its intermediate files |
| `schweinfurt_2045_synthetic_smoke_grid55.ipynb` | `work/runs/schweinfurt_2045_synthetic_smoke_grid55/run_manifest.json` |
| `lv080_expansion_cost.ipynb` | one SWF grid (LV 80) of a paired run; imports `grid_expansion.load_expansion_overview` (now `analysis.expansion.overview`) |

The maintained notebooks are `notebooks/analysis/analysis_expansion.ipynb`,
`analysis_powerflow.ipynb`, `grid_area_envelope_comparison.ipynb` and the sampling
notebooks in `notebooks/sampling/`.
