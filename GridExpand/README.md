# GridExpand

GridExpand simulates low-voltage (LV) distribution grids of a region before and after electrification (heat
pumps, EVs, rooftop PV, batteries) and estimates the grid reinforcement they need. It reads synthetic grids and
buildings from a [pylovo](https://github.com/tum-ens/pylovo) database (usually an InfDB) and optionally real DSO
grids for validation.

| step | package | does |
|---|---|---|
| 1 grid sampling | `gridexpand.sampling` | export pylovo grids to HDF5 (optional; the database pipelines read pylovo directly) |
| 2 demand allocation | `gridexpand.allocation` | hourly building/bus time series and asset plans, urbs inputs |
| 3 optimization | `gridexpand.optimization` | building model (urbs or PyPSA optimizer): dispatch (and sizing for the optimized case) |
| 4 power flow | `gridexpand.powerflow` | pandapower time series before and after electrification |
| 5 analysis | `gridexpand.analysis` | cable and transformer expansion costs, loaders, plots, notebooks |
| API | `gridexpand.api` | HTTP API for the GridPlanner UI: jobs, results, scenario files |

Everything is one Python package (`src/gridexpand/`) with one uv environment and one command, `gridexpand`.
Results go to the `surrogrid` schema of the same database and to HDF5 hand-off files below `work/`.

## Install

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed). From `GridExpand/`:

```bash
uv sync                          # environment in .venv, incl. the dev tools (pytest, ruff)
uv sync --extra notebooks        # + JupyterLab, seaborn, kaleido
uv sync --extra api          # + FastAPI/uvicorn for gridexpand api
cp .env.example .env             # database credentials (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
uv run gridexpand --help
```

Step 3 uses Gurobi by default (gurobipy is installed; a licence is needed for models above the size limit of the
bundled licence). `GUROBI_HOME` and `GRB_LICENSE_FILE` are taken from `.env` or the environment, otherwise
`/opt/gurobi1302/linux64` and `~/gurobi.lic` when they exist. Without a licence use
`GRIDEXPAND_SOLVER=appsi_highs` (HiGHS; results may differ, see [Step 3](docs/steps/3_urbs.md)).
Step 3 has two optimizers for the same building model: `urbs` (Pyomo, default) and `pypsa` (PyPSA/linopy, several
times faster, full-year inputs only); select one with `--optimizer`, `execution.optimizer` or
`GRIDEXPAND_OPTIMIZER` ([Step 3](docs/steps/3_urbs.md#pypsa-optimizer)).

### Large data assets

Not tracked; copy or symlink them into place (the `*_required.txt` files mark the locations):

| path (below `data/statistics/`) | needed by |
|---|---|
| `inhabited_buildings/elec_lps.h5` | Step 2 residential load profiles (all runs) |
| `general/mobility_profile_pool_old/mobility_demand_pool.csv`, `mobility_availability_pool.csv` | synthetic runs (legacy pool `emobpy_pool_v1`) |
| `general/mobility_profile_pool/` | paired runs (session pool with `mobility_pool_manifest.json`) |

The database must provide the `pylovo` schema (grids, buildings), `citydb` LoD2 roof surfaces for PV (without
them every building gets the PV fallback) and, for `infdb_ro_heat` scenarios, the INFDB `ro_heat` schema. Paired
validation needs confidential DSO data: SWF exports below `GRID_DATA_PATH` (`.env`) and, for aligned runs, the
alignment bundle and ÜZW delivery named in the run YAML.

## Database

- **Fresh database**: the `surrogrid` schema is created on first use (or explicitly with `uv run gridexpand db
  init-schema`).
- **Existing database** (created by older code or with pending migrations): pipeline runs stop with
  `SchemaMigrationRequired`. Back up, then `uv run gridexpand db migrate --plan` and `--apply`.
- **After a pylovo re-generation** (new grid ids): `uv run gridexpand db relink-pylovo --plan` and `--apply`
  (checks that every grid case still has the same buildings).
- Other maintenance: `gridexpand db compress` (TimescaleDB compression of raw tables), `gridexpand db
  delete-scenario <scenario_key>` (dry run unless `--execute`).

Details, tables and views: [docs/database.md](docs/database.md). Every `gridexpand` command parses its arguments
before it opens a database connection, so `--help` is safe; everything else writes to the database of `.env`.

## Quick start: one synthetic region

```bash
cp config/runs/00_run_scenario_template.yaml config/runs/my_region.yaml   # edit: run.id, run.scenario, ags, pylovo version, cases
uv run gridexpand grids --ags 9184137 --pylovo-version-id 1               # candidate grids and their numbers
uv run gridexpand config check config/runs/my_region.yaml                 # no database
uv run gridexpand run config/runs/my_region.yaml --dry-run                # plan and commands
uv run gridexpand run config/runs/my_region.yaml                          # Steps 2-4 + expansion per model case
uv run gridexpand status my_region                                        # work/runs/my_region/state.json
```

`gridexpand run` prepares one regional electrification assignment, runs one `gridexpand synthetic` batch per
execution group (`pre`, `heuristic-assets`, `post-hems-optimized`), materializes the expansion analyses and
refreshes the QGIS views; `--resume` continues an interrupted run. The YAML keys are in
[docs/configuration.md](docs/configuration.md), the model cases and scientific rules in
[docs/method.md](docs/method.md). `gridexpand synthetic --help` runs one batch directly (same flags).

## Quick start: paired and aligned validation

```bash
uv run gridexpand run config/runs/joint_2045_v1_smoke24.yaml --prepare-only   # build and check the paired datasets
uv run gridexpand run config/runs/joint_2045_v1_smoke24.yaml --skip-prepare   # real and synthetic runs + expansion
```

`pipeline: paired_validation` compares SWF grids, `pipeline: paired_aligned` SWF and ÜZW grids of a pylovo
alignment bundle with their synthetic counterparts. See [docs/paired_validation.md](docs/paired_validation.md).

## Quick start: API and Docker

```bash
uv sync --extra api
uv run gridexpand api                                                   # http://127.0.0.1:18766/docs
GRIDEXPAND_UID=$(id -u) GRIDEXPAND_GID=$(id -g) docker compose -f docker/compose.yaml up --build
```

The API starts synthetic pipeline jobs, serves expansion results and building assets, and edits copies of
scenario YAMLs: [docs/api.md](docs/api.md). The browser UI for pylovo and GridExpand lives in the GridPlanner
repository, which runs both images behind one proxy.

## Commands

| command | does |
|---|---|
| `gridexpand run <run.yaml>` | one run YAML: `synthetic`, `paired_validation` or `paired_aligned` (`run-aligned` is an older name) |
| `gridexpand status <run>` | state of a run directory (no database) |
| `gridexpand grids --ags <AGS> --pylovo-version-id <V>` | candidate grids of an AGS with the runner's numbering |
| `gridexpand config check <yaml...>` | validate run/scenario YAMLs, print hashes and keys (no database) |
| `gridexpand synthetic --ags ... --pylovo-version-id ... --scenario-config ... --run-dir ...` | Steps 2-4 (+ expansion) for the synthetic grids of one AGS and one model case |
| `gridexpand allocate <id> --scenario-config ...` | Step 2 for one grid ([docs](docs/steps/2_demand_allocation.md)) |
| `gridexpand optimize <id> --scenario-config ...` | Step 3 for one Step 2 file ([docs](docs/steps/3_urbs.md)) |
| `gridexpand powerflow <id>` | Step 4 for one scenario file ([docs](docs/steps/4_powerflow.md)) |
| `gridexpand expansion --run-name ... --stage ...` | Step 5 expansion materialization ([docs](docs/steps/5_postprocessing.md)) |
| `gridexpand db <init-schema\|migrate\|compress\|relink-pylovo\|delete-scenario>` | database maintenance |
| `gridexpand api` | HTTP API for the GridPlanner UI (`api` extra) |

`gridexpand <command> --help` lists the options. Module entry points without a command:
`python -m gridexpand.sampling.export_single_grid` (Step 1),
`gridexpand.allocation.electrification_preparation`, `gridexpand.allocation.generate_mobility_profile_pool`,
`gridexpand.paired.runner`, `gridexpand.analysis.expansion.aligned_expansion`,
`gridexpand.analysis.audits.topology_bottleneck` (all with `uv run python -m ... --help`).

## Layout

```text
GridExpand/
  pyproject.toml, uv.lock   one environment (Python 3.12)
  .env.example              database credentials template (copy to .env)
  src/gridexpand/           cli.py, paths.py (the only module that knows directories), common/, db/ (+ sql/),
                            sampling/, allocation/, optimization/, powerflow/, analysis/, scenario/, paired/, api/
  config/                   scenarios/ and runs/ YAMLs
  data/                     static inputs: sampling/, statistics/
  notebooks/                sampling/, analysis/, archive/ (historical)
  docs/                     documentation (index: docs/README.md)
  docker/                   Dockerfile and compose.yaml of the image (CLI + API)
  scripts/                  hpc/ Slurm templates, migrate_local_layout.py
  tests/                    unit tests, regression/ harness
  work/                     runtime artifacts (gitignored)
```

| `work/` directory | content |
|---|---|
| `sampling/results/` | Step 1 exports |
| `allocation/grids/`, `allocation/results/<scenario key>/` | Step 2 HDF5-mode inputs; Step 2 outputs |
| `allocation/outputs/scenario_calibration/` | paired datasets and `profile_libraries/` |
| `optimization/{input,result/<scenario key>,logs/<solver>}/` | Step 3 inputs, results, solver logs |
| `powerflow/{input,output}/` | Step 4 inputs and HDF5 outputs |
| `analysis/output/` | Step 5 plots and audit exports |
| `runs/<run id>/` | run directories of `gridexpand run` / `synthetic` (state, logs); `runs/slurm/` Slurm logs |
| `api/jobs/`, `api/terminal/` | job history of the API; run YAMLs prepared for terminal runs |

`GRIDEXPAND_WORK_DIR`, `GRIDEXPAND_DATA_DIR` and `GRIDEXPAND_ENV_FILE` relocate `work/`, `data/` and `.env`
(process environment only).

**Adopting this layout in an existing checkout** (files in the old step folders `1.grid_sampling/` ...
`5.postprocessing/`): merge the branch first (git removes the tracked files of the old folders), then run
`uv run python scripts/migrate_local_layout.py` as a dry run (planned moves, conflicts, uncovered files; it never
moves git-tracked files and never overwrites), then again with `--execute`, then delete the old per-step `.venv`
folders (and their `__pycache__` / `.ruff_cache`), which the script leaves in place.

HPC: `scripts/hpc/<allocation|optimization|powerflow>/run_cluster_serialstd.sh <id>` and
`scripts/hpc/start_batch_jobs.sh <step> <START> <END>` (templates; Steps 2 and 3 need `SCENARIO_CONFIG`).

## Testing

```bash
uv run pytest -q                   # unit tests; no database (tests point gridexpand at an unreachable one)
uv run ruff check src tests scripts   # lint with the project config
```

Opt-in database tests use a sandbox database only: `GRIDEXPAND_ANALYSIS_TEST_DATABASE=<sandbox db> uv run pytest
tests/analysis` (SQL parity), `GRIDEXPAND_API_TEST_DATABASE=<sandbox db> uv run pytest tests/api`.
`tests/regression/` runs the whole synthetic pipeline on a sandbox demo region and compares database snapshots
([tests/regression/README.md](tests/regression/README.md)); it drops and recreates its database.

## Documentation

| document | content |
|---|---|
| [docs/README.md](docs/README.md) | index |
| [docs/configuration.md](docs/configuration.md) | scenario and run YAML reference, grid selection, run directories, environment |
| [docs/method.md](docs/method.md) | model cases, sizing rules, heat, mobility, paired contract, acceptance checks |
| [docs/steps/](docs/steps/) | Steps 1-5: inputs, outputs, HDF keys, options, conventions |
| [docs/paired_validation.md](docs/paired_validation.md) | paired SWF and aligned SWF/ÜZW pipelines |
| [docs/database.md](docs/database.md) | `surrogrid` schema, migrations, maintenance |
| [docs/expansion_costs.md](docs/expansion_costs.md) | reinforcement rules and cost assumptions |
| [docs/api.md](docs/api.md) | HTTP API, its contract, container |
| [docs/research/](docs/research/) | historical run notes and audits |

## Licences

Project licence: [LICENSE](LICENSE). Vendored code (districtgenerator, emobpy, urbs) and data sources:
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
