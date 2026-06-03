# 5. Postprocessing

This step is the analysis and plotting workspace for GridExpand results. It is intentionally separate from Step 4 so the power-flow simulation environment stays focused while postprocessing can carry notebook, plotting, export, and presentation dependencies.

## Folder & file structure

```text
5.postprocessing/
  pyproject.toml              # uv environment for result analysis and plotting
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
