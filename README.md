
# SurroGrid

SurroGrid is a research codebase that combines two workflows:

1) **GridExpand**: one Python package (`gridexpand`) that simulates low-voltage (LV) distribution grids before and after electrification in five steps (grid sampling, demand allocation, urbs optimization, power flow, expansion analysis), backed by an InfDB/pylovo PostgreSQL database, plus paired real/synthetic validation and a web service.
2) **GridForecast**: preprocessing + machine learning models (MLP and Transformer) to train **surrogate forecasters** on the GridExpand outputs.

In short: **GridExpand produces grid-level results (in the `surrogrid` database schema and, in HDF5 mode, one `.h5` file per grid/scenario with inputs, optimization and power-flow results), and GridForecast turns those HDF5 files into ML-ready time-series tables and trains forecasting models.**

> Note: Several scripts in this repository are tuned for an HPC environment (Slurm) and may contain absolute filesystem paths (e.g. `/dss/...`). If you run elsewhere, adjust the configs accordingly.

---
## Resources and Detailed Descriptions
- IPCC 2026 Paper (to be published)
- Underlying Master's Thesis:
  _Hanser, Elias (2025). Deep Learning Surrogates for Low-Voltage Grid Planning. Master’s thesis, Technical University of Munich, https://mediatum.ub.tum.de/node?id=1851114._

## What this repo can do

**Data generation / simulation (GridExpand):**

- Sample representative LV distribution grids (pandapower networks) from pylovo and export each grid to an HDF5 file.
- Allocate building- and bus-resolved hourly time series (electricity/heat/mobility/PV) and write **MILP-ready** urbs input tables.
- Run the **urbs** (Pyomo) optimization to simulate DER dispatch and sizing (PV, battery, heat pump, EV) and write the results.
- Run time-series **pandapower** power flow before and after electrification and store voltages, line loadings and external-grid imports (raw series and/or compact summaries).
- Estimate cable and transformer **expansion costs**, compare real DSO grids with synthetic pylovo grids, and run all of it for a region from one run YAML (`gridexpand run`) or from a web UI (`gridexpand serve`).

**Surrogate modeling / forecasting (GridForecast):**

- Convert GridExpand (post-powerflow) HDF5 outputs into compact ML tables (`ts_train.h5`, `ts_test.h5`).
- Train + evaluate:
  - an **MLP baseline** with Ray Tune hyperparameter optimization (Optuna TPE + ASHA)
  - a **Transformer** model (optional VMD feature decomposition) with Ray Tune HPO
- Forecast grid-level net demand targets (commonly active and reactive power).

---

## How the pieces fit together

The intended end-to-end flow is:

```text
GridExpand (simulation)                                           

1) Grid sampling (.h5 per grid, optional)
2) Demand allocation + urbs inputs (writes /urbs_in/*)
3) urbs optimization (writes /urbs_out/*)
4) Power flow (writes /pwrflw/* or the surrogrid database)
5) Expansion analysis (surrogrid database)

GridForecast (ML)
1) 0_preprocessing/ (extract ML features/targets)
2)	a) 2_mlp/ (MLP training + HPO)
	b) 3_transformer/ (Transformer training + HPO)
```

GridExpand Steps 2-4 hand over a **single `.h5` file per grid/scenario**: each step copies its input and appends new groups. GridForecast needs the HDF5 power-flow output (`gridexpand powerflow --storage h5`).

---

## Repository layout

- [GridExpand/](GridExpand): LV grid sampling → demand allocation → optimization → power flow → expansion analysis (one Python package, `gridexpand`; start with [GridExpand/README.md](GridExpand/README.md))
  - [GridExpand/src/gridexpand/sampling/](GridExpand/src/gridexpand/sampling): grid sampling/export (notebooks in [GridExpand/notebooks/sampling/](GridExpand/notebooks/sampling))
  - [GridExpand/src/gridexpand/allocation/](GridExpand/src/gridexpand/allocation): generate demands + write `/urbs_in/*`
  - [GridExpand/src/gridexpand/optimization/](GridExpand/src/gridexpand/optimization): run the urbs optimization + write `/urbs_out/*`
  - [GridExpand/src/gridexpand/powerflow/](GridExpand/src/gridexpand/powerflow): run the pandapower power flow + write `/pwrflw/*`
  - [GridExpand/src/gridexpand/analysis/](GridExpand/src/gridexpand/analysis): expansion costs, loaders, plots (notebooks in [GridExpand/notebooks/analysis/](GridExpand/notebooks/analysis))
  - [GridExpand/src/gridexpand/scenario/](GridExpand/src/gridexpand/scenario), [paired/](GridExpand/src/gridexpand/paired), [service/](GridExpand/src/gridexpand/service): run YAMLs and orchestration, paired validation, web service
  - [GridExpand/docs/](GridExpand/docs): documentation ([index](GridExpand/docs/README.md), one document per step in [GridExpand/docs/steps/](GridExpand/docs/steps))

- [GridForecast/](GridForecast): preprocessing + ML training for forecasting
  - [GridForecast/0_preprocessing/](GridForecast/0_preprocessing): build `ts_train.h5` / `ts_test.h5`
  - [GridForecast/2_mlp/](GridForecast/2_mlp): MLP baseline + HPO scripts
  - [GridForecast/3_transformer/](GridForecast/3_transformer): Transformer + HPO scripts

Each subfolder contains a more detailed README describing its inputs/outputs and run commands.

---

## Quickstart (typical usage)

### A) If you want to run GridExpand end-to-end

From `GridExpand/`: `uv sync`, `cp .env.example .env` (database credentials), then run a region from a run YAML:

```bash
uv run gridexpand config check config/runs/<run>.yaml
uv run gridexpand run config/runs/<run>.yaml          # Steps 2-4 + expansion analysis
```

See [GridExpand/README.md](GridExpand/README.md) for installation, data assets, database setup, the paired
validation and the web service. To produce HDF5 files step by step (for example for GridForecast):

1) **Grids**: Step 1 export ([GridExpand/docs/steps/1_grid_sampling.md](GridExpand/docs/steps/1_grid_sampling.md)) into `GridExpand/work/sampling/results/`, then copy the files to `GridExpand/work/allocation/grids/`.
2) **Demand allocation**: `uv run gridexpand allocate <id> --scenario-config config/scenarios/<scenario>.yaml` ([Step 2](GridExpand/docs/steps/2_demand_allocation.md)); output in `GridExpand/work/allocation/results/<scenario key>/`.
3) **Optimization**: copy the Step 2 file to `GridExpand/work/optimization/input/`, `uv run gridexpand optimize <id> --scenario-config config/scenarios/<scenario>.yaml` ([Step 3](GridExpand/docs/steps/3_urbs.md); Gurobi by default, or `--solver appsi_highs`).
4) **Power flow**: copy the Step 3 result to `GridExpand/work/powerflow/input/`, `uv run gridexpand powerflow <id>` ([Step 4](GridExpand/docs/steps/4_powerflow.md)); output in `GridExpand/work/powerflow/output/`.

At the end, you will have `.h5` files containing raw grid data plus `/urbs_*` and `/pwrflw/*` groups.

### B) If you want to train forecasting models (GridForecast)

1) **Prepare ML tables**
	- Point [GridForecast/0_preprocessing/config.py](GridForecast/0_preprocessing/config.py) to the directory containing your post-powerflow `.h5` files.
	- Run the preprocessing notebook in [GridForecast/0_preprocessing/](GridForecast/0_preprocessing) to generate:
	  - `GridForecast/0_preprocessing/Data/ts_train.h5`
	  - `GridForecast/0_preprocessing/Data/ts_test.h5`

2) **Train models**
	- MLP: see [GridForecast/2_mlp/README context in GridForecast](GridForecast/README.md) and the entry script in `2_mlp/`.
	- Transformer: see [GridForecast/3_transformer/README context in GridForecast](GridForecast/README.md) and the entry script in `3_transformer/`.

GridForecast is script-oriented (Ray Tune experiments, Slurm sbatch templates). Many defaults are HPC-specific.

---

## Data interface (important concept)

GridExpand’s contract between steps is the **HDF5 file structure**. At a high level:

- Step 1 writes `raw_data/*` (pandapower network, buildings, region, weather).
- Step 2 adds `urbs_in/*` (URBS input sheets + time series).
- Step 3 adds `urbs_out/*` (optimization results).
- Step 4 adds `pwrflw/*` (power-flow inputs and outputs).

For the precise keys and expectations, refer to:

- [GridExpand/README.md](GridExpand/README.md) (overview)
- [GridExpand/docs/steps/1_grid_sampling.md](GridExpand/docs/steps/1_grid_sampling.md)
- [GridExpand/docs/steps/2_demand_allocation.md](GridExpand/docs/steps/2_demand_allocation.md)
- [GridExpand/docs/steps/3_urbs.md](GridExpand/docs/steps/3_urbs.md)
- [GridExpand/docs/steps/4_powerflow.md](GridExpand/docs/steps/4_powerflow.md)

---

## Environments / dependencies

- GridExpand is one uv project (`GridExpand/pyproject.toml`, `uv.lock`, Python 3.12): run `uv sync` in `GridExpand/` (extras `notebooks`, `service`).
- GridForecast does not ship a single canonical environment file; the Slurm scripts and training scripts may install packages at runtime.

---

## License

Licensing and third-party notices are documented in:

- [GridExpand/LICENSE](GridExpand/LICENSE)
- [GridExpand/THIRD_PARTY_LICENSES](GridExpand/THIRD_PARTY_LICENSES)
- [GridForecast/LICENSE](GridForecast/LICENSE)

