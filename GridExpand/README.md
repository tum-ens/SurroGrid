
# GridExpand

GridExpand is a 4-step simulation pipeline plus a dedicated Step 5 postprocessing workspace. Steps 1-4 (1) sample representative **low-voltage (LV) distribution grids**, (2) generate **building- and bus-resolved time series** (electricity/heat/mobility/PV) and write **MILP-ready inputs**, (3) run the **MILP** energy system optimization (DER expansion / dispatch), and (4) run **time-series LV power-flow** before/after expansion. Step 5 collects result analysis, plotting notebooks, and figure-generation tooling.

Steps 1-4 communicate via a single **HDF5 (`.h5`) file per grid/scenario**. Downstream simulation steps read a `.h5`, copy it to their output folder, and append additional datasets. Step 5 reads Step 4 HDF5 or database-backed results and should not mutate simulation artifacts.

Everything is one installable Python package (`src/gridexpand/`) with one uv environment and one command-line entry point, `gridexpand`.

If you already have compatible `.h5` files (see “HDF5 interface”), you can start at any step.

---

## Pipeline at a glance

```text
Step 1 (grid sampling)        Step 2 (demand allocation)      Step 3 (urbs optimization)
gridexpand.sampling           gridexpand.allocation           gridexpand.optimization
  input: pylovo DB + GIS        input: Step-1 .h5 or DB         input: Step-2 .h5
  output: raw grid .h5          output: same .h5 + /urbs_in     output: same .h5 + /urbs_out

Step 4 (power flow)           Step 5 (analysis)
gridexpand.powerflow          gridexpand.analysis
  input: Step-3 .h5             input: Step-4 HDF5 or DB results
  output: same .h5 + /pwrflw    output: expansion tables, notebooks, plots
```

### Typical file naming

The repository uses filenames like:

`0_N2819500E4261500_86165_2_40.h5`

The prefix (`0` above) is used by Steps 2-4 to select the file to run. An example
Step-1 grid and its Step-2 result are tracked in `data/sample_grids/`.

---

## Repository structure

```text
GridExpand/
  pyproject.toml  uv.lock          # one environment for all steps (Python 3.12)
  .env.example                     # copy to .env: database credentials
  src/gridexpand/
    cli.py                         # `gridexpand <command>` entry point
    paths.py                       # the only module that knows directories
    common/                        # shared helpers (timeframe, electrification, orchestration, ...)
    db/                            # SurroGridDatabase, maintenance CLI, sql/*.sql schema files
    sampling/                      # Step 1: pylovo readout and HDF5 export
    allocation/                    # Step 2: demand allocation (main.py, config.py, assets/,
                                   #   classes/, functions/, scenario_calibration/, external/)
    optimization/                  # Step 3: run_urbs_cluster.py + vendored urbs/
    powerflow/                     # Step 4: run_pwrflw.py, real-grid runners, demands, powerflow
    analysis/                      # Step 5: expansion/, plotting/, audits/, powerflow/
    scenario/                      # scenario/run YAML loading and the orchestrators
    paired/                        # paired real/synthetic validation runner and adapters
  config/                          # commented scenario and run YAML files (scenarios/, runs/)
  data/                            # static inputs: statistics/, sampling/, sample_grids/
  notebooks/                       # sampling/ (Step 1) and analysis/ (Step 5) notebooks
  docs/                            # step documentation (docs/steps/) and method notes
  scripts/                         # migrate_local_layout.py, hpc/ Slurm templates
  tests/                           # import and CLI smoke tests
  work/                            # (gitignored) runtime artifacts, see below
```

Runtime artifacts live below `work/`:

| Directory | Content |
|---|---|
| `work/sampling/results/` | Step-1 grid exports |
| `work/allocation/grids/` | Step-2 HDF5-mode input grids |
| `work/allocation/results/<scenario-key>/` | Step-2 outputs (urbs inputs), weather cache |
| `work/allocation/outputs/` | paired/aligned datasets and profile libraries |
| `work/optimization/{input,result,logs}/` | Step-3 hand-off inputs, results, solver logs |
| `work/powerflow/{input,output}/` | Step-4 hand-off inputs and HDF5 outputs |
| `work/analysis/output/` | Step-5 plots and audit exports |
| `work/runs/` | run folders of the orchestrators (logs, manifests) |

`GRIDEXPAND_WORK_DIR`, `GRIDEXPAND_DATA_DIR` and `GRIDEXPAND_ENV_FILE` relocate
`work/`, `data/` and `.env` (they are read from the process environment, not from
`.env`). Checkouts that still have files in the old step folders can move them with
`uv run python scripts/migrate_local_layout.py` (dry run; add `--execute`).

The large Step-2 inputs `data/statistics/inhabited_buildings/elec_lps.h5`,
`data/statistics/general/mobility_profile_pool/` and the two CSV pools in
`data/statistics/general/mobility_profile_pool_old/` are not tracked (see the
`*_required.txt` placeholders); copy or symlink them into place.

Step documentation: `docs/steps/1_grid_sampling.md` ... `docs/steps/5_postprocessing.md`.
Scenario configuration: `config/README.md` and `docs/scenario_pipeline/`.
DB-backed SurroGrid storage: `docs/SURROGRID_SCHEMA.md`.

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

## Setup (environment)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from `GridExpand/`:

```bash
uv sync                      # Python 3.12 environment in .venv (uv installs Python if needed)
uv sync --extra notebooks    # additionally JupyterLab, ipykernel, ipywidgets, kaleido
cp .env.example .env         # then enter the database credentials
uv run gridexpand --help     # list the commands; `gridexpand <command> --help` for options
uv run pytest -q             # import and CLI smoke tests (no database needed)
```

Step 3 needs a working Gurobi installation and license. The orchestrators take
`GUROBI_HOME` and `GRB_LICENSE_FILE` from `.env` or the environment and otherwise use
`/opt/gurobi1302/linux64` and `~/gurobi.lic` when those exist. uv handles Python
packages only; solver binaries and licenses are external.

| Command | Runs |
|---|---|
| `gridexpand run --run-config config/runs/<run>.yaml` | one scenario or paired-validation run YAML |
| `gridexpand run-aligned --run-config config/runs/<run>.yaml` | aligned SWF/ÜZW paired comparison |
| `gridexpand synthetic --ags <AGS> --pylovo-version-id <V> --run-dir <DIR> ...` | Steps 2-4 (+ expansion) for the synthetic grids of one AGS |
| `gridexpand allocate <inputfile_id> ...` | Step 2 for one grid |
| `gridexpand optimize <inputfile_id> ...` | Step 3 for one Step-2 file |
| `gridexpand powerflow <inputfile_id> ...` | Step 4 for one scenario file |
| `gridexpand expansion --run-name <RUN> --stage <pre/post> ...` | Step 5 expansion materialization |
| `gridexpand db init-schema` / `gridexpand db delete-scenario <key> [--execute]` | database maintenance |

---

## How to run (end-to-end)

The pipeline is designed so each step **copies** its input file into its own output folder and appends results.

### Step 1: Grid sampling / readout

Location: `src/gridexpand/sampling/`

- Primary workflow: notebooks

  - `notebooks/sampling/1_filter_valid_grids.ipynb`
  - `notebooks/sampling/2_sample_grids.ipynb`
- Single-grid export: `uv run python -m gridexpand.sampling.export_single_grid --plz <PLZ> --list-candidates`

#### Step 1: Required inputs

- Access to a pylovo PostgreSQL DB (optional if you already have compatible `.h5` grids)
- Census and shapefile inputs already shipped in `data/sampling/`

#### Step 1: Outputs

- One `.h5` per sampled grid in `work/sampling/results/`

DB credentials are read from environment variables loaded from the GridExpand-level `.env` file.

YAML-launched DB-backed scenario runs take `pylovo_version_id` from the run
YAML and pass it explicitly through the pipeline. `PYLOVO_VERSION_ID` in
`GridExpand/.env` remains only for manual Step-1 notebooks and preparation
commands that have not yet migrated to a run YAML; it is not authoritative for
scenario runs.

### Step 2: Demand allocation (write `/urbs_in/*`)

Location: `src/gridexpand/allocation/`

#### Step 2: Required inputs

- Put one or more Step-1 `.h5` files into: `work/allocation/grids/` (HDF5 mode), or use `--storage db`
- Statistics files are read from: `data/statistics/` (already included, except the untracked files above)

#### Step 2: Run

```bash
uv run gridexpand allocate <inputfile_id> --n_cpu <N>
```

The script selects the first `.h5` in `work/allocation/grids/` whose prefix before the first underscore matches `inputfile_id`.

Use `--demand-scope residential` for a household-only run. This filters the building table before electricity, PV, heat, mobility, and URBS input sheets are generated. Step-2 outputs are isolated below `work/allocation/results/<scenario-key>/`; the key is derived from the scientific scenario identity and timeframe.

#### Step 2: Outputs

- A copied/augmented `.h5` in `work/allocation/results/<scenario-key>/` containing:
  - updated `/raw_data/weather` (always written)
  - updated `/raw_data/buildings` (with sampled attributes)
  - new `/urbs_in/*` URBS input tables

### Step 3: URBS optimization (write `/urbs_out/*`)

Location: `src/gridexpand/optimization/`

#### Step 3: Required inputs

- Copy Step-2 result files into: `work/optimization/input/`

#### Step 3: Run

```bash
uv run gridexpand optimize <inputfile_id> --n_cpu <N>
```

#### Step 3: Outputs

- A copied/augmented result file in `work/optimization/result/<scenario-key>/` whose metadata carries the canonical scenario identity and assignment hash.

Solver notes:

- The urbs variant in this repository is configured for **Gurobi** by default; a working installation/license is required unless you adapt the solver settings.

### Step 4: Power flow (write `/pwrflw/*`)

Location: `src/gridexpand/powerflow/`

#### Step 4: Required inputs

- Copy Step-3 result files into: `work/powerflow/input/`

#### Step 4: Run

```bash
uv run gridexpand powerflow <inputfile_id> --n_cpu <N>
```

#### Step 4: Outputs

- A copied/augmented output file in `work/powerflow/output/` containing `pwrflw/` inputs + results.

#### Optional inflex post-electrification power flow

The AGS runner can add a post-inflex power-flow result after the normal post-flex Step 3 optimization. Use `--include-inflex-powerflow` to run both post-flex and post-inflex for each candidate in one pass. INFLEX is intentionally dependent on the post-flex result: Step 4 reads `urbs_out/MILP/cap_pro` and uses the optimized `heatpump_air` and `heatpump_booster` capacities to translate fixed heat demand into heat-pump and auxiliary electric demand. Mobility profiles are reused from Step 2 and emobpy is not rerun.

```bash
uv run gridexpand synthetic \
  --ags <AGS> \
  --pylovo-version-id <VERSION> \
  --profiles all \
  --powerflow-output summary \
  --scenario-config config/scenarios/forchheim_2045_synthetic.yaml \
  --include-inflex-powerflow \
  --run-dir work/runs/<RUN_NAME>
```

Use `--inflex-only` only when you want to skip the flexible Step 4 power-flow output. It still runs Step 3 optimization first, because the inflex heat reconstruction needs the optimized post-flex capacities. Use `--inflex-ev-charger-kw <kW>` to override the default 11 kW home charger cap.

#### Intermediate-file cleanup for large AGS runs

The DB-backed summary pipeline only needs the HDF5 hand-off files while a candidate is actively moving through Steps 2-4. After a candidate has passed Step 4 validation, the later analysis uses the PostgreSQL summary tables and the run logs. Add `--cleanup-intermediates success` to delete successful-candidate hand-off files from:

- `work/allocation/results/`
- `work/optimization/input/`
- `work/powerflow/input/`

Failed-candidate files are kept for debugging. For an interrupted run that already contains completed candidates, use the same pipeline arguments and run directory with `--cleanup-completed-only`; this removes intermediates for candidates marked done in `status.tsv` or `events.jsonl` and exits without starting new work.

```bash
uv run gridexpand synthetic \
  --ags <AGS> \
  --pylovo-version-id <VERSION> \
  --profiles all \
  --demand-scope residential \
  --powerflow-output summary \
  --scenario-config config/scenarios/forchheim_2045_synthetic.yaml \
  --include-inflex-powerflow \
  --cleanup-completed-only \
  --run-dir work/runs/<EXISTING_RUN_DIR>
```

### Paired SWF real/synthetic scenario

Launch the calibrated SWF comparison through `gridexpand run --run-config <paired run YAML>`. The staged entry point prepares and validates the complete regional dataset, delegates execution to `gridexpand.paired.runner`, and automatically materializes expansion results after success. The paired runner projects stable physical-building scenario units onto real and synthetic buses and verifies a shared temporal mapping before power flow. See `docs/PAIRED_SCENARIO.md` for the publication gate and command.

### Step 5: Postprocessing and plotting

Location: `src/gridexpand/analysis/`, notebooks in `notebooks/analysis/`

#### Step 5: Required inputs

- Step 4 HDF5 outputs in `work/powerflow/output/`, or
- DB-backed Step 4 results in PostgreSQL under the `surrogrid` schema.

#### Step 5: Run

```bash
uv sync --extra notebooks
uv run gridexpand expansion --help        # materialize expansion results
uv run jupyter lab notebooks/analysis     # interactive analysis
```

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

Step 1 and Step 2 can fetch data from PVGIS/Open-Meteo (see `sampling/config.py` and `allocation/config.py`).

- On HPC, fetching is often undesirable (rate limits / no internet). Prefer providing `/raw_data/weather` already in Step 1.
- Both steps assume a **UTC+1** time zone and use a fixed `REF_YEAR` (default 2009) to align “human behavior” profiles.

### Parallelism and memory

- Step 2 parallelizes (parts of) generation (notably heat) using multiprocessing.
- Step 3 parallelizes across building-node clusters and may also use solver-internal threads.
- Step 4 can parallelize time steps but deep-copies the pandapower net per worker, increasing memory use.

Recommendation: scale `--n_cpu` based on available RAM as well as CPU.

### Units and conventions

- Step 4 converts kW/kVAr to MW/MVAr internally (pandapower convention).
- Reactive power sign conventions can differ across toolchains; Step 4 assumes inductive/lagging demand as negative Q (see `docs/steps/4_powerflow.md`).

---

## HPC / SLURM usage

`scripts/hpc/` holds Slurm templates for Steps 2-4:

- `scripts/hpc/<allocation|optimization|powerflow>/run_cluster_serialstd.sh <inputfile_id>`: run one case
- `scripts/hpc/start_batch_jobs.sh <step> <START> <END>`: submit a range of cases

Slurm logs go to `work/runs/slurm/`. Step 3 writes solver logs to `work/optimization/logs/gurobi/`.

---

## Troubleshooting checklist

- “No matched files” / `IndexError`: confirm the `.h5` exists in the step’s input folder and the prefix matches the passed `inputfile_id`.
- Weather-related crashes in Step 2: if `/raw_data/weather` is missing, run Step 2 with the setting that indicates weather must be fetched (see Step-2 README); or pre-populate weather in Step 1.
- URBS solver errors: confirm Gurobi is available and licensed; check `work/optimization/logs/gurobi/`.
- Pandapower convergence issues: validate the input net, check demand magnitudes, and inspect `src/gridexpand/powerflow/grid_topol.py` helpers.

---

## Licenses

- Project license: see `LICENSE`
- Third-party notices: see `THIRD_PARTY_LICENSES`
- urbs license: see `src/gridexpand/optimization/urbs/LICENSE`

