
# GridExpand

GridExpand is a 4-step pipeline to (1) sample representative **low-voltage (LV) distribution grids**, (2) generate **building- and bus-resolved time series** (electricity/heat/mobility/PV) and write **MILP-ready inputs**, (3) run the **MILP** energy system optimization (DER expansion / dispatch) and write results back to the same file, and (4) run **time-series LV power-flow** before/after expansion.

All steps communicate via a single **HDF5 (`.h5`) file per grid/scenario**. Downstream steps always read a `.h5`, copy it to their output folder, and append additional datasets.

If you already have compatible `.h5` files (see “HDF5 interface”), you can start at any step.

---

## Pipeline at a glance

```text
Step 1 (grid sampling)        Step 2 (demand allocation)      Step 3 (urbs optimization)        Step 4 (power flow)
1.grid_sampling/              2.demand_allocation/            3.urbs/                           4.powerflow/
  input: pylovo DB + GIS        input: Step-1 .h5               input: Step-2 .h5                input: Step-3 .h5
  output: raw grid .h5          output: same .h5 + /urbs_in     output: same .h5 + /urbs_out     output: same .h5 + /pwrflw
```

### Typical file naming

The repository uses filenames like:

`0_N2819500E4261500_86165_2_40.h5`

The prefix (`0` above) is used by Steps 3 and 4 to select the file to run.

---

## Repository structure

```text
GridExpand/
  1.grid_sampling/                # Step 1: sample/export LV grids
    environment.yml
    gridreadout/
      1_filter_valid_grids.ipynb
      2_sample_grids.ipynb
      config.py
      input_data/                 # census + shapefiles
      results/                    # sampled raw grid .h5 files
      src/                        # db readout, weather, save helpers

  2.demand_allocation/            # Step 2: allocate demands + build /urbs_in
    environment.yml
    environment_HPC.yml
    gridalloc/
      main.py                     # entrypoint
      config.py                   # paths + constants
      data/
        grids/                    # input .h5 files (copied from Step 1)
        statistics/               # included statistical inputs
      results/                    # output .h5 files (copied+augmented)
      logs/                       # slurm logs
      src/

  3.urbs/                         # Step 3: urbs optimization (DER expansion)
    environment.yml
    environment_HPC.yml
    run_urbs_cluster.py            # entrypoint
    Input/                         # input .h5 files (copied from Step 2)
    result/                        # output .h5 files (copied+augmented)
    logs/                          # slurm + solver logs
    urbs/                          # pyomo model + readers/writers

  4.powerflow/                     # Step 4: time-series pandapower power flow
    environment.yml
    environment_HPC.yml
    run_pwrflw.py                  # entrypoint
    config.py                      # input/output dirs + power factors
    Input/                         # input .h5 files (copied from Step 3)
    Output/                        # output .h5 files (copied+augmented)
    logs/                          # slurm logs
    src/                           # demand reconstruction + pf engine
```

Each step folder has its own `README.md` with more detail:

- `1.grid_sampling/README.md`
- `2.demand_allocation/README.md`
- `3.urbs/README.md`
- `4.powerflow/README.md`

---

## HDF5 interface (inputs & outputs)

All steps read/write using `pandas.HDFStore` and store objects under well-known HDF5 keys.

### Minimum HDF5 keys by step

#### Step 1 output (required by Step 2)

- `/raw_data/net` : pandapower network serialized as a JSON string
- `/raw_data/region` : one-row table with at least `lat`, `lon` (recommended: `plz`, `altitude`, `regio7`, `kcid`, `bcid`)
- `/raw_data/buildings` : one row per building, including at least `bus`, `use`, `type`, `houses_per_building`, `occupants`, `area`, `floors`
- `/raw_data/weather` : hourly weather table (recommended; can be generated in Step 2 if missing)

#### Step 2 output (required by Step 3)

- `/urbs_in/*` : URBS input tables and time series, e.g. `/urbs_in/demand`, `/urbs_in/supim`, `/urbs_in/process`, ...

#### Step 3 output (required by Step 4)

- `/urbs_out/MILP/*` : optimization results (key input for Step 4 is `tau_pro`)

#### Step 4 output (final artifacts)

- `/pwrflw/input/*` : reconstructed per-bus $P/Q$ time series (pre/post expansion)
- `/pwrflw/output/{pre,post}/*` : voltages, line loadings, external grid imports

If you bring your own `.h5` files, make sure the required keys exist for the step you start with.

---

## Setup (environments)

Each step provides its own conda environment definition:

- Step 1: `1.grid_sampling/environment.yml`
- Step 2: `2.demand_allocation/environment.yml` (and `environment_HPC.yml`)
- Step 3: `3.urbs/environment.yml` (and `environment_HPC.yml`)
- Step 4: `4.powerflow/environment.yml` (and `environment_HPC.yml`)

Create environments from within the step directory, e.g.:

```bash
cd GridExpand/2.demand_allocation
conda env create -f environment.yml
conda activate <env-name-from-yml>
```

Note: some `environment.yml` files may contain a `prefix:` entry from another machine/OS. If conda errors, remove the `prefix:` line.

---

## How to run (end-to-end)

The pipeline is designed so each step **copies** its input file into its own output folder and appends results.

### Step 1: Grid sampling / readout

Location: `1.grid_sampling/gridreadout/`

- Primary workflow: notebooks

  - `1_filter_valid_grids.ipynb`
  - `2_sample_grids.ipynb`

#### Step 1: Required inputs

- Access to a pylovo PostgreSQL DB (optional if you already have compatible `.h5` grids)
- Census and shapefile inputs already shipped in `gridreadout/input_data/`

#### Step 1: Outputs

- One `.h5` per sampled grid in `1.grid_sampling/gridreadout/results/`

DB credentials are read from environment variables loaded in `1.grid_sampling/gridreadout/config.py` (commonly via a `.env` file next to it).

### Step 2: Demand allocation (write `/urbs_in/*`)

Location: `2.demand_allocation/gridalloc/`

#### Step 2: Required inputs

- Put one or more Step-1 `.h5` files into: `2.demand_allocation/gridalloc/data/grids/`
- Statistics files are read from: `2.demand_allocation/gridalloc/data/statistics/` (already included)

#### Step 2: Run

```bash
cd GridExpand/2.demand_allocation/gridalloc
python3 main.py <inputfile_id> --n_cpu <N>
```

The script selects the first `.h5` in `data/grids/` whose prefix before the first underscore matches `inputfile_id`.

#### Step 2: Outputs

- A copied/augmented `.h5` in `2.demand_allocation/gridalloc/results/` containing:
  - updated `/raw_data/weather` (always written)
  - updated `/raw_data/buildings` (with sampled attributes)
  - new `/urbs_in/*` URBS input tables

### Step 3: URBS optimization (write `/urbs_out/*`)

Location: `3.urbs/`

#### Step 3: Required inputs

- Copy Step-2 result files into: `3.urbs/Input/`

#### Step 3: Run

```bash
cd GridExpand/3.urbs
python3 run_urbs_cluster.py <inputfile_id> --n_cpu <N>
```

#### Step 3: Outputs

- A copied/augmented result file in `3.urbs/result/` whose filename includes a scenario suffix, e.g. `_PV100_HP100_EV100_VarTar0_CapPr0.h5`.

Solver notes:

- The urbs variant in this repository is configured for **Gurobi** by default; a working installation/license is required unless you adapt the solver settings.

### Step 4: Power flow (write `/pwrflw/*`)

Location: `4.powerflow/`

#### Step 4: Required inputs

- Copy Step-3 result files into: `4.powerflow/Input/`

#### Step 4: Run

```bash
cd GridExpand/4.powerflow
python3 run_pwrflw.py <inputfile_id> --n_cpu <N>
```

#### Step 4: Outputs

- A copied/augmented output file in `4.powerflow/Output/` containing `pwrflw/` inputs + results.

---

## Details to keep in mind

### File selection by prefix

Steps 2–4 select the input file by matching `fname.split('_', 1)[0] == inputfile_id`.

- If multiple files share the same prefix, the first match is used.
- If no match exists, the scripts will error (typically `IndexError`).

Recommendation: keep only one file per prefix in the respective input folder.

### Output overwrites

Downstream steps **copy input → output and then write datasets**. If an output file with the same name already exists, it may be overwritten.

Recommendation: move/rename previous outputs before re-running.

### Weather and API usage

Step 1 and Step 2 can fetch data from PVGIS/Open-Meteo (see the step `config.py`).

- On HPC, fetching is often undesirable (rate limits / no internet). Prefer providing `/raw_data/weather` already in Step 1.
- Both steps assume a **UTC+1** time zone and use a fixed `REF_YEAR` (default 2009) to align “human behavior” profiles.

### Parallelism and memory

- Step 2 parallelizes (parts of) generation (notably heat) using multiprocessing.
- Step 3 parallelizes across building-node clusters and may also use solver-internal threads.
- Step 4 can parallelize time steps but deep-copies the pandapower net per worker, increasing memory use.

Recommendation: scale `--n_cpu` based on available RAM as well as CPU.

### Units and conventions

- Step 4 converts kW/kVAr to MW/MVAr internally (pandapower convention).
- Reactive power sign conventions can differ across toolchains; Step 4 assumes inductive/lagging demand as negative Q (see `4.powerflow/README.md`).

---

## HPC / SLURM usage

Steps 2–4 include helper scripts:

- `run_cluster_serialstd.sh`: run one case
- `start_batch_jobs_serialstd.sh`: submit a range of cases

Logs typically go to `logs/normal/` (stdout) and `logs/errors/` (stderr). Step 3 additionally writes solver logs to `logs/gurobi/`.

---

## Troubleshooting checklist

- “No matched files” / `IndexError`: confirm the `.h5` exists in the step’s input folder and the prefix matches the passed `inputfile_id`.
- Weather-related crashes in Step 2: if `/raw_data/weather` is missing, run Step 2 with the setting that indicates weather must be fetched (see Step-2 README); or pre-populate weather in Step 1.
- URBS solver errors: confirm Gurobi is available and licensed; check `3.urbs/logs/gurobi/`.
- Pandapower convergence issues: validate the input net, check demand magnitudes, and inspect `4.powerflow/src/grid_topol.py` helpers.

---

## Licenses

- Project license: see `LICENSE`
- Third-party notices: see `THIRD_PARTY_LICENSES`
- urbs license: see `3.urbs/urbs_LICENSE`

