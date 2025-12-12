
# 4. Power Flow

This step runs time-series low-voltage power-flow calculations for each scenario/grid file created by the previous steps. It evaluates the grid **before** and **after** DER expansion (PV/HP/etc.) using active and reactive power demand time series, and writes the results back into a copy of the scenario `.h5` file.

At a high level:

- Reads a pandapower network from the input `.h5`.
- Builds *pre-expansion* and *post-expansion* per-bus $P/Q$ demand time series.
- Runs pandapower power flow for each timestep (optionally in parallel).
- Stores voltages, line loadings, and external-grid imports into the output `.h5`.

## Folder & file structure

```text
4.powerflow/
  config.py
  run_pwrflw.py
  run_cluster_serialstd.sh
  start_batch_jobs_serialstd.sh
  environment.yml
  environment_HPC.yml
  Input/
    *.h5
  Output/
    *.h5   (copied from Input/ and augmented with pwrflw/* datasets)
  logs/
    normal/
    errors/
  src/
    demands.py
    powerflow.py
    save_grid.py
    grid_topol.py
    resource_report.py
```

Notes:

- `Input/` and `Output/` are controlled by `config.py` (`Config.DATA_DIR`, `Config.STORAGE_DIR`).
- `run_pwrflw.py` always reads from `Input/` and writes to `Output/`.

## Required inputs

### 1) Scenario `.h5` file in `Input/`

The code expects each input file to contain (HDF5 keys / datasets):

- `raw_data/net`: a pandapower network serialized as a JSON string (loaded via `pandapower.from_json_string`).
- `urbs_in/demand`: raw household electricity demand time series (active power), used for the **pre-expansion** case.
- `urbs_out/MILP/tau_pro`: urbs optimization results, used to reconstruct **post-expansion** net imports, PV generation, and HP demand.

The exact schema of these tables is defined by upstream steps; this step assumes they match what `src/save_grid.py` and `src/demands.py` read.

### 2) Python environment

Dependencies are listed in:

- `environment.yml` (local/dev)
- `environment_HPC.yml` (cluster)

Core runtime packages used in this step:

- `pandapower`
- `pandas`, `numpy`, `h5py`, `tables`
- `multiprocessing` (standard library)

## Generated outputs

For each processed input file, an output file is created in `Output/` with the **same filename**. The output file is a copy of the input `.h5` augmented with additional datasets.

### Written HDF5 keys

`run_pwrflw.py` writes:

- `/pwrflw/input/demand_pre`: per-bus pre-expansion demand time series (active + reactive).
- `/pwrflw/input/demand_post`: per-bus post-expansion demand time series (active + reactive).

- `/pwrflw/output/pre/demand_import`: external grid import (p/q) per timestep.
- `/pwrflw/output/pre/vm`: bus voltage magnitudes `vm_pu` per timestep.
- `/pwrflw/output/pre/line_loads`: line flows and currents per timestep.

- `/pwrflw/output/post/demand_import`: external grid import (p/q) per timestep.
- `/pwrflw/output/post/vm`: bus voltage magnitudes `vm_pu` per timestep.
- `/pwrflw/output/post/line_loads`: line flows and currents per timestep.

`src/demands.py` additionally writes reactive-power components derived from urbs results:

- `pwrflw/urbs_out/MILP/reactive`: concatenated reactive components (household, heat pump, PV) for traceability.

### Logs

On the HPC submission scripts, stdout/stderr are written to:

- `logs/normal/<jobid>_output.log`
- `logs/errors/<jobid>_error.log` (removed automatically if empty)

## How the code works (conceptual)

### Demand reconstruction (`src/demands.py`)

- **Pre-expansion**: household active power is taken from `urbs_in/demand`, and reactive power is synthesized using a fixed power factor (`config.PF_ELC`).
- **Post-expansion**: the urbs results `urbs_out/MILP/tau_pro` are filtered to keep:
	- `import` and `feed_in` (to compute net imports),
	- `heatpump_air` (heat pump load),
	- all rooftop PV technologies (`pro` starting with `Rooftop...`).

Reactive power post-expansion is computed using:

- HP reactive demand from `config.PF_HP`.
- PV reactive capability bounded by `config.PF_PV_MIN`. PV is assumed to produce reactive power to reduce net reactive import as much as possible within its capability.

### Grid preparation (`src/powerflow.py`)

Before running the time-series power flow, the network is “relaxed” to avoid artificial constraint binding:

- Line current limits `max_i_ka` are set very high.
- Load `max_p_mw` limits are set very high.
- Bus voltage bounds are widened.
- The transformer is removed and replaced by a closed bus-bus switch between HV/LV sides.

### Power flow execution (`src/powerflow.py`)

For each timestep:

- Loads are updated in `grid.load` (bus-indexed update).
- `pandapower.runpp(..., algorithm="bfsw")` is executed.
- Results are collected:
	- external-grid import from `res_ext_grid`
	- bus voltages from `res_bus["vm_pu"]`
	- line flows/currents from `res_line`

Optional parallelization uses Python multiprocessing by chunking timesteps across workers.

## Running

### Local run

1) Put one or more scenario files into `Input/`.

2) Run for a specific *file ID prefix*:

```bash
python3 run_pwrflw.py 0 --n_cpu 8
```

The positional argument (`0` above) is matched against the prefix before the first underscore in the filename, e.g.:

- `0_N2819500E4261500_... .h5` matches `inputfile_id=0`.

### Cluster (Slurm)

Submit a single job:

```bash
sbatch run_cluster_serialstd.sh 0
```

Submit a range of jobs (`START`..`END`, inclusive):

```bash
bash start_batch_jobs_serialstd.sh 0 24
```

`run_cluster_serialstd.sh` activates conda env `pwrflw-hpc` and passes `--n_cpu $SLURM_CPUS_PER_TASK`.

## Details to keep in mind

- **File selection logic**: `run_pwrflw.py` chooses the first `.h5` in `Input/` whose prefix matches `inputfile_id`. If multiple files share the same prefix, only the first match will be used.
- **Output overwrites**: if `Output/<filename>` already exists it will be overwritten (because the input file is copied over at start).
- **Units**: loads are converted from kW/kVAr to MW/MVAr by dividing by `1000` before running pandapower.
- **Parallel runs**: each worker receives a deep-copied pandapower net. This avoids shared-state issues but increases memory use.
- **Reactive power sign convention**: the code models inductive/lagging demand as negative Q (see `src/demands.py`). Ensure upstream/downstream steps use consistent conventions.
- **Performance**: runtime scales with `#timesteps × #buses/lines`. Use `--n_cpu` to distribute timesteps.

## Troubleshooting

- If you see `IndexError` or “no matched files”, verify that:
	- the file exists in `Input/`,
	- it ends with `.h5`,
	- its prefix before the first `_` matches the `inputfile_id` you pass.
- If pandapower fails to converge for some timesteps, consider checking the input demands and the integrity of the network (zero-length lines etc.). Helper utilities exist in `src/grid_topol.py`.

