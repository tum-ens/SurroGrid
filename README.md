
# SurroGrid

SurroGrid is a research codebase that combines two workflows:

1) **GridExpand**: a 4-step pipeline to generate *grid-level simulation data* for low-voltage (LV) distribution grids.
2) **GridForecast**: preprocessing + machine learning models (MLP and Transformer) to train **surrogate forecasters** on the GridExpand outputs.

In short: **GridExpand produces one HDF5 (`.h5`) file per grid/scenario containing inputs + optimization + power-flow results, and GridForecast turns those files into ML-ready time-series tables and trains forecasting models.**

> Note: Several scripts in this repository are tuned for an HPC environment (Slurm) and may contain absolute filesystem paths (e.g. `/dss/...`). If you run elsewhere, adjust the configs accordingly.

---

## What this repo can do

**Data generation / simulation (GridExpand):**

- Sample representative LV distribution grids (pandapower networks) and export each grid to an HDF5 file.
- Allocate building- and bus-resolved hourly time series (electricity/heat/mobility/PV) and write **MILP-ready** URBS input tables.
- Run the **urbs** (Pyomo) optimization to simulate DER adoption/dispatch (e.g., PV/HP/EV) and write results back into the same HDF5.
- Run time-series **pandapower** power-flow “pre” vs “post” expansion and store voltages, line loadings, and external-grid imports.

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

1) Grid sampling (.h5 per grid)
2) Demand allocation + URBS inputs (writes /urbs_in/*)
3) urbs optimization (writes /urbs_out/*)
4) Power flow (writes /pwrflw/*)

GridForecast (ML)
1) 0_preprocessing/ (extract ML features/targets)
2)	a) 2_mlp/ (MLP training + HPO)
	b) 3_transformer/ (Transformer training + HPO)
```

All GridExpand steps communicate through a **single `.h5` file per grid/scenario**. Downstream steps copy the input file into their own output folder and append new groups/datasets.

---

## Repository layout

- [GridExpand/](GridExpand): LV grid sampling → demand allocation → optimization → power flow
  - [GridExpand/1.grid_sampling/](GridExpand/1.grid_sampling): notebook-driven grid sampling/export
  - [GridExpand/2.demand_allocation/](GridExpand/2.demand_allocation): generate demands + write `/urbs_in/*`
  - [GridExpand/3.urbs/](GridExpand/3.urbs): run URBS optimization + write `/urbs_out/*`
  - [GridExpand/4.powerflow/](GridExpand/4.powerflow): run pandapower PF + write `/pwrflw/*`

- [GridForecast/](GridForecast): preprocessing + ML training for forecasting
  - [GridForecast/0_preprocessing/](GridForecast/0_preprocessing): build `ts_train.h5` / `ts_test.h5`
  - [GridForecast/2_mlp/](GridForecast/2_mlp): MLP baseline + HPO scripts
  - [GridForecast/3_transformer/](GridForecast/3_transformer): Transformer + HPO scripts

Each subfolder contains a more detailed README describing its inputs/outputs and run commands.

---

## Quickstart (typical usage)

### A) If you want to run GridExpand end-to-end

1) **Create/obtain input grids**
	- Either run Step 1 sampling in [GridExpand/1.grid_sampling/](GridExpand/1.grid_sampling) (notebook-driven, often requires pylovo DB access),
	- Or start from existing compatible `.h5` grid files.

2) **Demand allocation (Step 2)**
	- Place your Step-1 `.h5` files into `GridExpand/2.demand_allocation/gridalloc/data/grids/`.
	- Run the entrypoint described in [GridExpand/2.demand_allocation/README.md](GridExpand/2.demand_allocation/README.md).

3) **Optimization (Step 3: urbs)**
	- Copy Step-2 result `.h5` files into `GridExpand/3.urbs/Input/`.
	- Run the entrypoint described in [GridExpand/3.urbs/README.md](GridExpand/3.urbs/README.md).
	- This step typically requires a MILP solver (the code is configured for Gurobi by default).

4) **Power flow (Step 4)**
	- Copy Step-3 scenario `.h5` files into `GridExpand/4.powerflow/Input/`.
	- Run the entrypoint described in [GridExpand/4.powerflow/README.md](GridExpand/4.powerflow/README.md).

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

- [GridExpand/README.md](GridExpand/README.md) (overview + interface)
- [GridExpand/2.demand_allocation/README.md](GridExpand/2.demand_allocation/README.md)
- [GridExpand/3.urbs/README.md](GridExpand/3.urbs/README.md)
- [GridExpand/4.powerflow/README.md](GridExpand/4.powerflow/README.md)

---

## Environments / dependencies

- GridExpand provides per-step conda environments (see each step’s `environment.yml` and optional `environment_HPC.yml`).
- GridForecast does not ship a single canonical environment file; the Slurm scripts and training scripts may install packages at runtime.

If conda complains about a `prefix:` entry in an environment file, remove that line (it may point to a different machine).

---

## License

Licensing and third-party notices are documented in:

- [GridExpand/LICENSE](GridExpand/LICENSE)
- [GridExpand/THIRD_PARTY_LICENSES](GridExpand/THIRD_PARTY_LICENSES)
- [GridForecast/LICENSE](GridForecast/LICENSE)

