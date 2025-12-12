
# GridForecast

This folder contains the **GridForecast** step: creating machine-learning ready time series datasets from GridExpand results and training **MLP** and **Transformer** models to forecast grid-level net demand.

The code is organized into three sub-steps:

- `0_preprocessing/`: convert the raw GridExpand/PostPowerflow HDF5 outputs into compact ML tables (`ts_train.h5`, `ts_test.h5`).
- `2_mlp/`: train an MLP baseline with Ray Tune hyperparameter optimization (HPO).
- `3_transformer/`: train a Transformer model (optionally with VMD feature decomposition) with Ray Tune HPO.

> Important: Several scripts contain **absolute paths** to the LRZ DSS filesystem (e.g. `/dss/...`). You must adjust those paths if you run elsewhere.

---

## Capabilities (Exact)

This step is designed to: (1) convert GridExpand/PostPowerflow outputs into ML-ready time series tables, and (2) train + evaluate forecasting models on those tables.

### Data / preprocessing capabilities

- Reads **one HDF5 per grid** from `0_preprocessing/config.py:Config.input_dir` and extracts a consistent set of time series and scalar features.
- Produces ML tables in a format consumable by both models:
  - `X` and `y` stored as `pandas.DataFrame` in HDF5 keys `X` and `y`
  - MultiIndex rows: `(batch, hour)` where `batch` is the grid id
- Creates an intermediate inspection-friendly xarray file (`Data/all_data.h5`) with groups `X_ts`, `X_sclr`, `y_ts`, `y_sclr`.
- Performs a **train/test split by grid id** (not by time) to reduce leakage across grids.
- Optional (thesis-specific): filters “small grids” and writes `ts_*_large_grids.h5`.

Limitations / assumptions:

- Input raw files must contain the expected group structure (e.g. `raw_data/...`, `urbs_out/reduced_data/...`).
- The produced features/targets must match the column names listed in `X_BASE_COLS` / `TARGET_COLS` in the training scripts.

### Model training capabilities

MLP (`2_mlp/`):

- Trains an MLP on per-hour rows (no windowing) with a preprocessing pipeline that can add:
  - `log1p` features (`LOG1P_COLS`)
  - seasonal sin/cos features (`TS_PERIODS`)
  - scaling via `StandardScaler`
- Supports **single-target** or **two-target** training (interpreted as active power P and reactive power Q).
- Implements normalized error training via **MAE/MAEx**.
- Runs hyperparameter optimization (HPO) via **Ray Tune** + **Optuna (TPE)** + **ASHA** and writes results under `2_mlp/ray_tune/`.

Transformer (`3_transformer/`):

- Trains a Transformer encoder model on sliding windows of length `core_len + 2*pad_hours`.
- Supports two aggregation modes to produce aggregated-hour outputs:
  - `aggregation_mode='sum'`: trim pad, then sum blocks of `agg_hours`
  - `aggregation_mode='conv'`: depthwise Conv1D aggregation with `conv_padding` context
- Supports optional feature augmentation via **VMD** (Variational Mode Decomposition) for selected columns:
  - VMD can be computed and cached to disk (`VMD_APPROACH='write'`) or read from cache (`'read'`).
- Supports multiple loss functions (e.g. `mae`, `mse`, `mae_maex`, `alpha_peak`).
- Provides evaluation helpers that can produce metrics and plots, plus integrated gradients explainability.
- Runs HPO via **Ray Tune** + **Optuna (TPE)** + **ASHA** and writes results under `3_transformer/ray_tune/`.

General limitations:

- This repo is not packaged as a pip-installable library; the entrypoints are scripts and notebooks.
- The HPO scripts currently perform runtime `pip install` calls.
- Many defaults are tuned for LRZ GPU partitions and may need adjustment on other systems.

---

## Folder / File Structure

```text
GridForecast/
  0_preprocessing/
    0.extract_ML_data.ipynb
    config.py
    Data/
      all_data.h5
      ts_train.h5
      ts_test.h5
      ts_train_large_grids.h5        # optional (created by the notebook)
      ts_test_large_grids.h5         # optional (created by the notebook)

  2_mlp/
    mlp_model.py
    evaluation.py
    run_mlp_models_HPO.py
    submit_asha_tpe.sbatch
    ray_tune/                        # Ray Tune experiment storage
    logs/
      error/
      out/                           # created by Slurm scripts
    models/                          # checkpoints / exported models (project-specific)

  3_transformer/
    transformer_model.py
    evaluation.py
    resource_report.py
    run_transformer_model_HPO.py
    run_transformer_model.ipynb
    submit_asha_tpe.sbatch
    data/                            # cached VMD results (HDF5) if VMD_APPROACH=read/write
    ray_tune/                        # Ray Tune experiment storage
    logs/
      error/
      out/                           # created by Slurm scripts
    models/
    runs/                            # TensorBoard logs if enabled
```

---

## Required Inputs

### 1) Raw simulation outputs (PostPowerflow HDF5)

The preprocessing notebook reads **one HDF5 file per grid** from `config.input_dir` (see `0_preprocessing/config.py`).

Each file is expected to contain groups similar to:

- `raw_data/` (buildings, consumers, region, weather, net, ...)
- `urbs_in/` and `urbs_out/` (reduced tables such as `demand`, `eff_factor`, `supim`, `process`, `storage`, ...)

These files are produced by the upstream GridExpand pipeline (powerflow + urbs).

### 2) Python environment

There is no single `environment.yml` in this folder. On LRZ/HPC the recommended approach is to use the provided **Slurm container** scripts:

- `2_mlp/submit_asha_tpe.sbatch`
- `3_transformer/submit_asha_tpe.sbatch`

If you run locally, note that the HPO scripts currently install packages at runtime via `pip` (see “Notes / Pitfalls”).

---

## Generated Outputs

### Preprocessing outputs (`0_preprocessing/Data/`)

The notebook `0_preprocessing/0.extract_ML_data.ipynb` generates:

- `Data/all_data.h5`
  - Xarray datasets stored under groups: `X_ts`, `X_sclr`, `y_ts`, `y_sclr`
  - This file is an intermediate “wide” format useful for inspection.

- `Data/ts_train.h5` and `Data/ts_test.h5`
  - Each file stores two Pandas DataFrames in HDF5 keys:
    - `X`: features
    - `y`: targets

Data format expectations (used by both MLP and Transformer):

- `X` and `y` are `pandas.DataFrame` with a `MultiIndex`:
  - level 0: `batch` (grid id)
  - level 1: `hour` (time index)
- Columns in `X` match the lists in each training script (e.g. `X_BASE_COLS`).
- Columns in `y` match `TARGET_COLS`, typically:
  - `demand_net_active_post`
  - `demand_net_reactive_post`

Optional small-grid filtering (thesis-specific) can create:

- `Data/ts_train_large_grids.h5`
- `Data/ts_test_large_grids.h5`

### Training / HPO outputs

Both `2_mlp/` and `3_transformer/` use **Ray Tune** for HPO. The main outputs are:

- `ray_tune/<experiment_name>/...`:
  - trial directories
  - checkpoints (via Ray Tune checkpointing)
  - TensorBoard event files (when enabled)

Additional project-specific outputs may appear in `models/` (e.g. exported `.pt` weights) and under `logs/`.

---

## General Working (How the Code Fits Together)

### A) Preprocessing (`0_preprocessing/`)

1. Selects raw HDF5 files from `config.input_dir`.
2. Extracts time series features (weather, demand components, PV production proxies, mobility features, heat/COP) and scalar grid/building descriptors.
3. Writes an intermediate xarray dataset (`all_data.h5`).
4. Splits the dataset into train/test by grid id.
5. Converts xarray datasets into a single multi-indexed table and saves `ts_train.h5` / `ts_test.h5`.

### B) MLP baseline (`2_mlp/`)

- `run_mlp_models_HPO.py` runs Ray Tune + Optuna (TPE) + ASHA.
- Model/training logic is implemented in `mlp_model.py`:
  - preprocessing: log1p, seasonal sin/cos features, `StandardScaler`
  - dual-target scaling: learns scaler on apparent power $S=\sqrt{P^2+Q^2}$ and applies the same scale to both P and Q
  - loss: MAE/MAEx (normalized by per-grid mean absolute exchange)

### C) Transformer model (`3_transformer/`)

- `run_transformer_model_HPO.py` runs Ray Tune + Optuna (TPE) + ASHA.
- Core code is in `transformer_model.py`:
  - windowing: builds sequences of length `core_len + 2*pad_hours`
  - model: per-timestep MLP → positional encoding → transformer encoder → output head
  - aggregation: sum aggregation (classic) or depthwise Conv1D aggregation (`aggregation_mode='conv'`)
  - optional VMD: if enabled, computes or reads cached VMD modes for selected columns (`VMD_COLS`)

---

## How To Run

### 1) Generate datasets

Open and run the notebook:

- `0_preprocessing/0.extract_ML_data.ipynb`

Before running, update:

- `0_preprocessing/config.py` → `Config.input_dir` to point to your PostPowerflow directory.

The final cells write:

- `0_preprocessing/Data/ts_train.h5`
- `0_preprocessing/Data/ts_test.h5`

### 2) Run MLP HPO

On LRZ/HPC:

```bash
cd 2_mlp
sbatch submit_asha_tpe.sbatch
```

The script runs:

- `python run_mlp_models_HPO.py`

### 3) Run Transformer HPO

On LRZ/HPC:

```bash
cd 3_transformer

# defaults: agg_hours=1, loss_type=alpha_peak
sbatch submit_asha_tpe.sbatch

# override defaults
sbatch submit_asha_tpe.sbatch 4 mae_maex
```

This runs:

- `python run_transformer_model_HPO.py --agg-hours <int> --loss-type <str>`

---

## Container Usage (NVIDIA PyTorch 25.06)

The provided Slurm scripts run the code inside an NVIDIA PyTorch container. The upstream reference image is:

- `nvcr.io/nvidia/pytorch:25.06-py3`

On LRZ, the `submit_asha_tpe.sbatch` scripts use `srun --container-image <path>.sqsh` (a SquashFS image). That `.sqsh` is typically built from the NGC image above.

Practical notes:

- If you already have a local `.sqsh` built from `nvcr.io/nvidia/pytorch:25.06-py3`, point `--container-image` to it.
- Make sure you mount:
  - the code folder into the container (the Slurm scripts mount to `/workspace`)
  - the transformer VMD cache folder `3_transformer/data/` (the transformer Slurm script mounts it to `/data`)
- The scripts assume GPU access; verify the partition + `--gres=gpu:1` matches your cluster.

---

## Notes / Pitfalls (Read This)

- **Hard-coded paths**: HDF5 input paths (train/test) and Ray storage paths are hard-coded in the run scripts. Update them if your directory layout differs.

- **Runtime `pip install`**: `run_mlp_models_HPO.py` and `run_transformer_model_HPO.py` call `pip install ...` at runtime. This is convenient in notebooks/containers but can be undesirable on shared systems.

- **Large batch sizes / memory**: the MLP HPO search uses very large `batch_size` options (up to 524288). On GPUs this can OOM depending on feature count and precision.

- **Index naming assumptions**: evaluation utilities frequently assume the DataFrame index has levels named `batch` and `hour`. If you change index names, metrics that group by `batch` may break or silently behave differently.

- **Transformer VMD cache**: if `VMD_APPROACH='read'`, the transformer preprocessor expects cached VMD files under `3_transformer/data/`. The Slurm script mounts this directory into the container (see `submit_asha_tpe.sbatch`). If you run without that mount, VMD reads will fail and the code may fall back to recomputing.

- **Dual-target interpretation (P/Q)**: both models assume the first target is active power (P) and the second is reactive power (Q) when using two targets.

---

## Where To Change Things

- Change raw input location: `0_preprocessing/config.py`
- Change features/targets: the `X_BASE_COLS` / `TARGET_COLS` lists in:
  - `2_mlp/run_mlp_models_HPO.py`
  - `3_transformer/run_transformer_model_HPO.py`
  - `3_transformer/transformer_model.py` (Ray Tune helpers)
- Change Ray output location: `STORAGE_ROOT` in each `run_*_HPO.py`

