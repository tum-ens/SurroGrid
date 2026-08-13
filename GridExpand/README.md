
# GridExpand

GridExpand is a 4-step simulation pipeline plus a dedicated Step 5 postprocessing workspace. Steps 1-4 (1) sample representative **low-voltage (LV) distribution grids**, (2) generate **building- and bus-resolved time series** (electricity/heat/mobility/PV) and write **MILP-ready inputs**, (3) run the **MILP** energy system optimization (DER expansion / dispatch), and (4) run **time-series LV power-flow** before/after expansion. Step 5 collects result analysis, plotting notebooks, and figure-generation tooling.

Steps 1-4 communicate via a single **HDF5 (`.h5`) file per grid/scenario**. Downstream simulation steps read a `.h5`, copy it to their output folder, and append additional datasets. Step 5 reads Step 4 HDF5 or database-backed results and should not mutate simulation artifacts.

If you already have compatible `.h5` files (see “HDF5 interface”), you can start at any step.

---

## Pipeline at a glance

```text
Step 1 (grid sampling)        Step 2 (demand allocation)      Step 3 (urbs optimization)
1.grid_sampling/              2.demand_allocation/            3.urbs/
  input: pylovo DB + GIS        input: Step-1 .h5               input: Step-2 .h5
  output: raw grid .h5          output: same .h5 + /urbs_in     output: same .h5 + /urbs_out

Step 4 (power flow)           Step 5 (postprocessing)
4.powerflow/                  5.postprocessing/
  input: Step-3 .h5             input: Step-4 HDF5 or DB results
  output: same .h5 + /pwrflw    output: notebooks, plots, exports
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
      config.py                   # paths, datasets + scenario adapter
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

  common/                          # Shared DB/schema/timeframe helpers used across steps
    database.py
    orchestration.py               # Shared runner logging and process execution
    timeframe.py
    surrogrid_schema.sql

  scenario_pipeline/              # Network-independent scientific scenario logic
    config/                        # Commented scenario and run YAML files
      scenarios/
      runs/
    docs/                          # Central assumptions and option reference
    run_scenario.py                # Standalone scenario entry point
    synthetic_ags_runner.py        # Synthetic AGS Step 2-4 orchestrator

  paired_validation/              # Real/synthetic projection and equivalence layer
    runner.py                      # Paired real/SWF and synthetic comparison runner
    comparison.py                  # Network-independent equivalence checks
    sources/                       # SWF and paired-synthetic target adapters

  maintenance/                    # Explicit destructive administration commands
    delete_scenario_data.py       # Targeted DB/file cleanup (dry-run by default)

  5.postprocessing/                # Step 5: result analysis + plotting
    pyproject.toml                 # uv environment for plotting notebooks
    notebooks/                     # analysis notebooks
    powerflow/                     # comparison-data preparation
    plotting/                      # figure helpers
    audits/                        # demand/topology diagnostics
    expansion/                     # expansion materialization and summaries
```

Shared implementation helpers live in `common/`; scientific assumptions and normal
orchestration live in `scenario_pipeline/`; paired real/synthetic projection lives in
`paired_validation/`; destructive administration commands live in `maintenance/`.

Each step folder has its own `README.md` with more detail:

- `1.grid_sampling/README.md`
- `2.demand_allocation/README.md`
- `3.urbs/README.md`
- `4.powerflow/README.md`
- `5.postprocessing/README.md`

DB-backed SurroGrid storage is documented in `SURROGRID_SCHEMA.md`.

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

#### Step 4 output (power-flow artifacts consumed by Step 5)

- `/pwrflw/input/*` : reconstructed per-bus $P/Q$ time series (pre/post expansion)
- `/pwrflw/output/{pre,post}/*` : voltages, line loadings, external grid imports

If you bring your own `.h5` files, make sure the required keys exist for the step you start with.

---

## Setup (environments)

GridExpand now supports uv-managed environments per step.

Install uv and Python runtimes once:

```bash
uv python install 3.12
uv python install 3.10
```

Then create each step environment from its folder:

- Step 1: `cd GridExpand/1.grid_sampling && uv sync`
- Step 2: `cd GridExpand/2.demand_allocation && uv sync`
- Step 3: `cd GridExpand/3.urbs && uv sync`
- Step 4: `cd GridExpand/4.powerflow && uv sync`
- Step 5: `cd GridExpand/5.postprocessing && uv sync`

Step-specific dependency manifests are in:

- `1.grid_sampling/pyproject.toml`
- `2.demand_allocation/pyproject.toml`
- `3.urbs/pyproject.toml`
- `4.powerflow/pyproject.toml`
- `5.postprocessing/pyproject.toml`

For detailed commands, see `UV_SETUP.md`.

Legacy conda files are still present (`environment.yml`, `environment_HPC.yml`) for backward compatibility.

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

DB credentials are read from environment variables loaded from the GridExpand-level `.env` file.

For DB-backed synthetic-grid runs, `PYLOVO_VERSION_ID` in `GridExpand/.env` pins the pylovo `grid_result.version_id` used by candidate selection and power-flow input resolution. Leave it empty to keep the previous behavior of selecting the latest available version for each `(plz, kcid, bcid)`.

### Step 2: Demand allocation (write `/urbs_in/*`)

Location: `2.demand_allocation/gridalloc/`

#### Step 2: Required inputs

- Put one or more Step-1 `.h5` files into: `2.demand_allocation/gridalloc/data/grids/`
- Statistics files are read from: `2.demand_allocation/gridalloc/data/statistics/` (already included)

#### Step 2: Run

```bash
cd GridExpand/2.demand_allocation/gridalloc
uv run --project .. python main.py <inputfile_id> --n_cpu <N>
```

The script selects the first `.h5` in `data/grids/` whose prefix before the first underscore matches `inputfile_id`.

Use `--demand-scope residential` for a household-only run. This filters the building table before electricity, PV, heat, mobility, and URBS input sheets are generated. DB-backed residential runs use the `baseline_static_hh_only` scenario key family so they do not overwrite all-demand results.

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
uv run python run_urbs_cluster.py <inputfile_id> --n_cpu <N>
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
uv run python run_pwrflw.py <inputfile_id> --n_cpu <N>
```

#### Step 4: Outputs

- A copied/augmented output file in `4.powerflow/Output/` containing `pwrflw/` inputs + results.

#### Optional no-flex post-electrification power flow

The AGS runner can add a post-no-flex power-flow result after the normal post-flex Step 3 optimization. Use `--include-no-flex-powerflow` to run both post-flex and post-no-flex for each candidate in one pass. No-flex is intentionally dependent on the post-flex result: Step 4 reads `urbs_out/MILP/cap_pro` and uses the optimized `heatpump_air` and `heatpump_booster` capacities to translate fixed heat demand into heat-pump and auxiliary electric demand. Mobility profiles are reused from Step 2 and emobpy is not rerun.

```bash
uv run python GridExpand/scenario_pipeline/synthetic_ags_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles all \
  --powerflow-output summary \
  --scenario-config GridExpand/scenario_pipeline/config/scenarios/forchheim_2045.yaml \
  --include-no-flex-powerflow \
  --run-dir GridExpand/run_logs/<RUN_NAME>
```

Use `--no-flex-only` only when you want to skip the flexible Step 4 power-flow output. It still runs Step 3 optimization first, because the no-flex heat reconstruction needs the optimized post-flex capacities. Use `--no-flex-ev-charger-kw <kW>` to override the default 11 kW home charger cap.

#### Intermediate-file cleanup for large AGS runs

The DB-backed summary pipeline only needs the HDF5 hand-off files while a candidate is actively moving through Steps 2-4. After a candidate has passed Step 4 validation, the later analysis uses the PostgreSQL summary tables and the run logs. Add `--cleanup-intermediates success` to delete successful-candidate hand-off files from:

- `2.demand_allocation/gridalloc/results/`
- `3.urbs/Input/`
- `4.powerflow/Input/`

Failed-candidate files are kept for debugging. For an interrupted run that already contains completed candidates, use the same pipeline arguments and run directory with `--cleanup-completed-only`; this removes intermediates for candidates marked done in `status.tsv` or `events.jsonl` and exits without starting new work.

```bash
uv run --project GridExpand/2.demand_allocation python GridExpand/scenario_pipeline/synthetic_ags_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags <AGS> \
  --profiles all \
  --demand-scope residential \
  --powerflow-output summary \
  --scenario-config GridExpand/scenario_pipeline/config/scenarios/forchheim_2045.yaml \
  --include-no-flex-powerflow \
  --cleanup-completed-only \
  --run-dir GridExpand/run_logs/<EXISTING_RUN_DIR>
```

### Paired SWF real/synthetic scenario

`paired_validation/runner.py` is the comparison runner for the calibrated SWF 2045 scenario. It uses stable scenario units, reuses a network-independent physical-building heat-profile library, projects the same profiles onto real and synthetic buses, and runs pre electricity-only, post-flex, and post-no-flex summaries. Representative-period settings come from the Scenario YAML. The runner records one canonical mapping in `shared_tsam_reference.json` and verifies every real and synthetic URBS result against it before power flow starts. See `2.demand_allocation/gridalloc/src/scenario_calibration/PAIRED_SCENARIO.md` for the publication gate and command.

### Step 5: Postprocessing and plotting

Location: `5.postprocessing/`

#### Step 5: Required inputs

- Step 4 HDF5 outputs in `4.powerflow/Output/`, or
- DB-backed Step 4 results in PostgreSQL under the `surrogrid` schema.

#### Step 5: Run

```bash
cd GridExpand/5.postprocessing
uv sync
```

Open `plotting/plotting_notebook.ipynb` with the Step 5 uv kernel for interactive result analysis.

#### Step 5: Outputs

- Notebook outputs, figures, static exports, and other analysis artifacts.

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

