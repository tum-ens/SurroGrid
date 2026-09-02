## Grid sampling / readout (Step 1)

This folder creates a **representative set of LV distribution grids** (pandapower networks) and enriches them with **location / regional metadata**, **building-to-bus mapping**, and **weather time series**. The output is stored as one HDF5 (`.h5`) file per sampled grid.

The workflow is notebook-driven and lives in [gridreadout/](gridreadout).

### When you can skip this step

This step typically requires **read access to a pylovo PostgreSQL database** that stores pandapower grids and associated building data.

If you already have your own grid(s) in the **same `.h5` format as the provided example** ([gridreadout/results/0_N2819500E4261500_86165_2_40.h5](gridreadout/results/0_N2819500E4261500_86165_2_40.h5)), you can skip Step 1 entirely and start with the next pipeline step that consumes these grid files.

(The naming format is arbitrary and a relic from the Germany wide analysis of the associated thesis - id_gridCoordinatesEPSG3035inGermany_zipCodeOfPylovoGridAssigned_kcidPylovo_bcidPylov.h5)

### Repository layout

- [gridreadout/](gridreadout): notebooks + Python helper modules
- [gridreadout/input_data/](gridreadout/input_data): census + shapefiles used for sampling and geo lookups
- [gridreadout/results/](gridreadout/results): example output `.h5` grid file(s)

## Setup

### 1) Create the environment

Use uv in this folder:

```bash
cd GridExpand/1.grid_sampling
uv sync
```

This uses [pyproject.toml](pyproject.toml) and creates `.venv` locally.

Legacy fallback: [environment.yml](environment.yml) is still available for conda-based setups.

### 2) (Optional) Configure pylovo DB access

DB access is configured via environment variables that are loaded in [gridreadout/config.py](gridreadout/config.py).

Create a `.env` file at the GridExpand repository root (`GridExpand/.env`) with:

```bash
DB_HOST=...
DB_PORT=5432
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
PYLOVO_VERSION_ID=1  # optional; leave empty to use the latest pylovo version
```

`config.py` loads `GridExpand/.env` explicitly, so all GridExpand steps use the same database and data-path configuration. `PYLOVO_VERSION_ID` is also read from this file and is used to pin all pylovo grid readout queries to one generated version when set.

The notebooks expect the pylovo DB schema to provide at least the tables queried in `src/db_read.py`:

- `pylovo.grid_result` (grid JSON + grid identifiers)
- `pylovo.transformer_positions` (transformer point geometry)
- `pylovo.buildings_result` (building attributes incl. `type`; no separate `res`/`oth` tables required)
- `pylovo.municipal_register` (regional stats)

If you do not have DB access, see **“Skip DB / use your own .h5 grids”** below.

## How to run (notebooks)

Run the notebooks from within `gridreadout/` so imports like `import src.db_read as dbrd` work as expected.

```bash
cd GridExpand/1.grid_sampling/gridreadout
uv run --project .. jupyter lab
```

### Notebook 1: Filter valid grids

Notebook: [gridreadout/1_filter_valid_grids.ipynb](gridreadout/1_filter_valid_grids.ipynb)

What it does:

- Reads candidate grids (PLZ/KCID/BCID + transformer location) from the pylovo DB.
- Reads census population grid from `input_data/Zensus2022_Bevoelkerungszahl_1km-Gitter.csv`.
- Filters out grids far away from populated census cells.
- Writes the remaining grid identifiers to `input_data/valid_grids` (HDF5 via `pandas.to_hdf`).

Output:

- `gridreadout/input_data/valid_grids` (HDF5 store with key `grids`)

### Notebook 2: Sample and export grids

Notebook: [gridreadout/2_sample_grids.ipynb](gridreadout/2_sample_grids.ipynb)

What it does:

- Loads `input_data/valid_grids` and the census grid.
- Samples a target number of census cells weighted by population.
- Maps sampled cells to pylovo grids (directly if a grid exists in the cell, otherwise via population density matching).
- For each selected grid:
	- Loads the pandapower net from the pylovo DB.
	- Loads building data and maps buildings to consumer buses.
	- Fetches weather time series (PVGIS TMY + Open-Meteo soil temperature) for the grid location.
	- Writes everything into a single `.h5` file in `gridreadout/results/`.

Output:

- One `.h5` file per sampled grid in `gridreadout/results/`.

### Optional CLI for single-grid pilots (e.g. PLZ 80803)

For cheap debugging runs, you can export exactly one grid without editing notebooks:

```bash
cd GridExpand/1.grid_sampling/gridreadout
uv run --project .. python export_single_grid.py --plz 80803 --list-candidates
uv run --project .. python export_single_grid.py --plz 80803 --candidate-index 0 --cell-id 0
```

Notes:

- `--list-candidates` shows available `(plz, kcid, bcid)` tuples from the new `pylovo` grid tables.
- Candidate grids are filtered by a minimum building threshold (`--min-buildings`, default `5`).
- You can pin an exact grid with `--kcid <...> --bcid <...>`.
- `--cell-id` controls the output filename prefix used by downstream step selection.
- Use `--skip-weather` if API calls are not possible; then run Step 2 with `weather_data_exists=False`.

## Output file format (`.h5`)

The sampling step writes a single HDF5 file per grid via `src/save_grid.py`.

At minimum, a compatible grid file must contain:

- A **pandapower network** serialized to JSON (created by `pandapower.to_json(net)` and loaded by `pandapower.from_json_string(...)`).
- **Location / region information** (latitude/longitude; optionally altitude and regional classification).

### Expected contents written by this step

This step writes the following entries (paths are HDF5 keys; Pandas objects are stored via `pandas.HDFStore`):

- `/raw_data/net` (HDF5 dataset): UTF-8 JSON string of the pandapower `net`.
- `/raw_data/consumers` (Pandas table): consumer bus mapping as returned by `src/grid_topol.get_consumers(net)`.
- `/raw_data/region` (Pandas table): one-row DataFrame with at least `lat`, `lon` (and typically `plz`, `regio7`, `altitude`, plus sampling metadata).
- `/raw_data/buildings` (Pandas table): exactly one physical-building row per `objectid`, with its shared consumer `bus`, source building classification, effective Residential/non-residential areas, component peaks, and mixed-use provenance.
- `/raw_data/building_components` (Pandas table): the required mixed-capable component manifest. It contains one Residential and/or Commercial/Public row per positive source component; an MV-direct non-residential row remains present with `included_in_lv=False`.
- `/raw_data/weather` (Pandas table): hourly TMY-like weather including `temp_air`, `relative_humidity`, `dew_point`, `soil_temp`, etc.

Filename convention used by the notebook:

- `N{y}E{x}_{plz}_{kcid}_{bcid}.h5` where `{x,y}` are derived from the sampled census cell centroid in EPSG:3035.

### Skip DB / use your own `.h5` grids

If you already have a compatible `.h5` file like the example in `gridreadout/results/`, you can:

- Skip running the notebooks in this folder.
- Place your `.h5` files where the downstream pipeline expects them.

To be compatible with downstream steps, ensure your `.h5` contains at least:

- A pandapower net JSON dataset (either at `/raw_data/net` as created here, or at the key expected by the downstream consumer).
- Location metadata (`lat`, `lon`) somewhere accessible (this repository stores it in `/raw_data/region`).
- `/raw_data/buildings` and `/raw_data/building_components` generated from the pinned mixed-capable PyLoVo schema.
- Furthermore, it is recommended to already assign weather data once in this step according to `weather.py` to prevent running out of weather API calls later in the pipeline

Pre-mixed PyLoVo schemas and old Step-1 HDF files are intentionally unsupported. Step 2 fails with a clear regeneration error when the component manifest is missing.

If a downstream step expects a different key (e.g. `grid_top/net`), adapt your file accordingly or add a lightweight conversion step.

## Code overview

Helper modules live in [gridreadout/src/](gridreadout/src):

- `db_read.py`: SQLAlchemy-based readout of grids, buildings, and metadata from the pylovo DB.
- `export_grid.py`: shared grid export flow used by Notebook 2 and the single-grid CLI.
- `save_grid.py`: creates `.h5` files and writes pandapower + pandas objects.
- `grid_topol.py`: small topology cleanup helpers (line lengths, duplicate loads, consumer buses).
- `weather.py`: fetch PVGIS TMY and Open-Meteo soil temperature; computes dew point.
- `powerflow.py`: utilities for running simple pandapower power flows (used for quick checks / experiments).