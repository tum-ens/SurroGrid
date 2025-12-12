
# 3. urbs (LVDS energy system optimization)

This folder contains the **urbs** optimization step of the GridExpand pipeline. It solves a linear (M)ILP energy system model for a low-voltage distribution system / building-node representation and writes results back to an HDF5 file.

The main entrypoint for this step is:

- `run_urbs_cluster.py` (local execution or called from SLURM)

The code in `urbs/` is a vendored / adapted urbs variant tailored to this GridExpand workflow.

## What this step does (high-level)

1. **Selects an input HDF5** from `Input/` based on a numeric/prefix ID.
2. **Builds a `global_settings` dict** (electrification shares, TSAM on/off, etc.).
3. **Copies the input file to `result/`** (working copy).
4. **Reads the input datasets** from the HDF5 (`/urbs_in/...`).
5. **Optionally applies Time Series Aggregation (TSAM)** to reduce the time horizon.
6. **Optionally reduces electrification adoption** (PV/HP/EV) by removing technologies at a random subset of nodes.
7. **Solves the optimization**:
   - The building nodes are split into `n_cpu` clusters.
   - Each cluster is solved in a separate Python process.
   - Results are merged and written into the output HDF5.
8. **Renames the output file** to include the scenario label (e.g. `..._PV100_HP100_EV100_VarTar0_CapPr0.h5`).

## Folder / file structure

```text
3.urbs/
  README.md
  environment.yml
  environment_HPC.yml
  run_urbs_cluster.py
  run_cluster_serialstd.sh
  start_batch_jobs_serialstd.sh
  urbs_LICENSE

  Input/
    *.h5                  # required input files for this step

  result/
    *_PV*_HP*_EV*_...h5   # generated result files (HDF5)

  logs/
    normal/               # SLURM stdout logs (if using sbatch)
    errors/               # SLURM stderr logs (if using sbatch)
    gurobi/               # solver logs per parallel sub-model

  urbs/
    runfunctions.py       # run_lvds_opt(), parallelization, solver setup
    input.py              # HDF5 input reader (read_input_h5)
    model.py              # Pyomo model construction
    saveload.py           # HDF5 result writer (save)
    features/             # TSAM, helper logic, technology modifiers, etc.
```

## Required inputs

### 1) Input HDF5 file in `Input/`

The step expects one or more `*.h5` files in `Input/`.

Selection logic (in `run_urbs_cluster.py`):

- The script lists all `.h5` files in `Input/`.
- It matches the file whose **prefix before the first underscore** equals `inputfile_id`.

Example:

- File: `Input/0_N2819500E4261500_86165_2_40.h5`
- Run with: `python3 run_urbs_cluster.py 0`

Important:

- If multiple files share the same prefix, the first match is used.
- If no file matches, the script will crash (index error). Make sure the ID exists.

### 2) Required datasets inside the HDF5

The HDF5 must contain input datasets under the prefix:

- `/urbs_in/...`

At minimum, the workflow assumes the presence of time series + techno-economic tables typically including:

- `/urbs_in/demand` (required; also used to infer the set of sites/buildings)
- `/urbs_in/supim`
- `/urbs_in/buy_sell_price`
- `/urbs_in/eff_factor`
- `/urbs_in/weather`
- `/urbs_in/commodity`
- `/urbs_in/process`
- `/urbs_in/process_commodity`
- `/urbs_in/storage`

Notes on expected structure:

- Time series tables (`demand`, `supim`, `buy_sell_price`, `eff_factor`, `weather`) are expected as pandas tables with (site, signal) multi-indexed columns in the urbs conventions.
- The reader adds an initialization row at the top (t = 0) for time series tables.

## Generated outputs

### 1) Result HDF5 in `result/`

The script copies the selected input file into `result/` and writes results into that copy.

Final naming:

- Base: same as input file name
- After solving: renamed to include scenario string

Example:

- Input: `Input/0_N2819500E4261500_86165_2_40.h5`
- Output: `result/0_N2819500E4261500_86165_2_40_PV100_HP100_EV100_VarTar0_CapPr0.h5`

### 2) HDF5 output groups written

The output file is an HDFStore written with compression (`blosc`). You will typically find:

- `/urbs_out/reduced_data/<table>`: the (possibly modified) input tables used for solving
- `/urbs_out/tsam/<tsam_info>`: TSAM metadata (even when TSAM is off, a `kept_timesteps` vector is stored)
- `/urbs_out/MILP/<entity>`: Pyomo entity dumps (sets, parameters, variables, expressions), merged across parallel sub-models

### 3) Logs

- SLURM logs (only if using `sbatch`):
  - `logs/normal/<jobid>_output.log`
  - `logs/errors/<jobid>_error.log`
- Solver logs (one per parallel sub-model): `logs/gurobi/<input>_<scenario>_<clusterIndex>.log`

## How to run

### Local run (interactive)

From the `GridExpand/3.urbs` directory:

1. Create and activate the environment (example):
   - `conda env create -f environment.yml`
   - `conda activate urbs`
2. Run a case:
   - `python3 run_urbs_cluster.py 0 --n_cpu 8`

`--n_cpu` controls how many **parallel Python worker processes** are spawned (clusters of building nodes).

### HPC / SLURM run

This folder includes a serial partition template:

- `run_cluster_serialstd.sh`: runs one case (one `inputfile_id`) as a single SLURM job
- `start_batch_jobs_serialstd.sh`: submits many jobs for `START..END`

Examples:

- Submit one run:
  - `sbatch run_cluster_serialstd.sh 0`
- Submit a range:
  - `bash start_batch_jobs_serialstd.sh 0 24`

The SLURM script:

- loads `miniconda3` and `gurobi`
- activates `conda env urbs`
- runs: `srun python3 run_urbs_cluster.py <INDEX> --n_cpu $SLURM_CPUS_PER_TASK`

## Configuration knobs (most important)

The primary user-exposed settings are currently set inside `run_urbs_cluster.py`:

- `tsam` (`True/False`): enable/disable time series aggregation
- `noTypicalPeriods` (int): number of typical periods (TSAM)
- `hoursPerPeriod` (int): hours per typical period (TSAM), commonly 168 (1 week)
- `PV_electr`, `HP_electr`, `EV_electr` (0–100): adoption shares per technology
- `n_cpu` (int): number of parallel worker processes

Scenario naming is derived from these values as:

`PV{PV}_HP{HP}_EV{EV}_VarTar{VarTar}_CapPr{CapPr}`

## Details to keep in mind

### Result overwrites / reproducibility

- `urbs.prepare_result_directory()` is currently configured to always return `"result"`.
  - This means runs reuse the same folder.
  - If you rerun the same input/scenario, the output file may be overwritten.

If you need per-run folders (timestamped), you can re-enable the previously commented logic in `urbs/runfunctions.py`.

### Parallelism and CPU usage

Parallelism has two layers:

1. Python multiprocessing: `n_cpu` worker processes
2. Solver threads inside each worker (Gurobi): `Threads=4` (hard-coded in `urbs/runfunctions.py`)

So worst-case thread demand is approximately `n_cpu * 4`.

On HPC, if you set `--n_cpu=$SLURM_CPUS_PER_TASK` and keep `Threads=4`, you can easily oversubscribe.

Recommended practice:

- Either reduce `--n_cpu`, or adjust solver `Threads` to match the allocation strategy you want.

### Solver requirements

- The default solver is forced to `gurobi` inside `urbs/runfunctions.py`.
- A working Gurobi installation + license is required unless you modify the code to use another solver supported by Pyomo.

### TSAM behavior

When TSAM is enabled, typical periods are selected using only weather signals:

- Ambient temperature (`Tamb`)
- Irradiation (`Irradiation`)

TSAM writes metadata to `/urbs_out/tsam/` including kept time steps and cluster weights.

### Non-implemented features

Two global settings exist but are intentionally blocked:

- `vartariff != 0` → raises `NotImplementedError`
- `power_price_kw != 0` → raises `NotImplementedError`

### Input assumptions

- Building/site list is inferred from `demand` columns; if a site has no demand column it may not be included in clustering.
- The reader adds an initialization row at `t=0` for time series tables.

## Inspecting results

To quickly inspect what’s inside an output file, you can do:

```python
import pandas as pd

path = "result/<your_output>.h5"
with pd.HDFStore(path, mode="r") as store:
    print(store.keys())

# Example: load a particular result table
df_costs = pd.read_hdf(path, key="/urbs_out/MILP/costs")
```

## License

See [urbs_LICENSE](urbs_LICENSE) for the license information of the urbs code included in this step.

