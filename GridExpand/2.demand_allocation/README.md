# Step 2: Demand allocation (GridExpand)

This step takes sampled low-voltage grids from **Step 1** (stored as `.h5`) and enriches them with **hourly time series** for:

- Weather (if not already present in the input)
- Rooftop PV supply (SupIm)
- Electricity demand
- Space-heating + domestic hot water demand
- EV charging (mobility demand) + EV availability

It then **writes URBS-ready input tables** into the same `.h5` file (stored in `gridalloc/results/`) under the HDF5 group `urbs_in/`.

---

## What this code does (high level)

The executable entrypoint is `gridalloc/main.py`.

1. **Select input grid file** from `gridalloc/data/grids/` by matching the `inputfile_id` (see “Input selection”).
1. **Load raw input tables** from the `.h5`: `raw_data/buildings`, `raw_data/region`, optionally `raw_data/weather`.
1. **Generate time series** in a strict order (dependencies matter): Weather → PV → Electricity → Heat → Mobility.
1. **Assemble URBS input sheets** (demand, supply, processes, commodities, storages, etc.).
1. **Copy the input `.h5` to `gridalloc/results/`** and append/overwrite tables (update `raw_data/buildings`/`raw_data/weather`, add `urbs_in/*`).

The `Grid` orchestration logic is implemented in `gridalloc/src/classes/grid.py`.

---

## Required inputs

### 1) Grid input file(s) (`.h5`)

Place input grid files in:

- `gridalloc/data/grids/*.h5`

Each file must contain at least these HDF5 keys (written by Step 1 in this project):

- `raw_data/buildings` (pandas table)
- `raw_data/region` (pandas table; typically a single-row DataFrame)

Recommended (and assumed by default in `main.py`):

- `raw_data/weather` (pandas table)

If `raw_data/weather` is missing, you must run with `weather_data_exists=False` (see “Weather data handling”), otherwise the run will fail when PV/heat/mobility try to access weather columns.

#### Expected columns (minimum)

The exact schema depends on the Step 1 generator, but Step 2 expects at least:

`raw_data/region` (single row):

- `lat`, `lon` (float): location used for weather/PV
- `altitude` (float): used for PV modeling
- `plz` (int/str): ZIP code used by the heat generator
- `regio7` (int): region class used for mobility statistics
- `bcid`, `kcid` (int): used to build deterministic per-vehicle seeds

`raw_data/buildings` (one row per building):

- `bus` (int): node/site id
- `use` (str): expects values like `Residential`, `Public`, `Commercial`
- `type` (str): building type code (e.g. `SFH`, `MFH`, `TH`, `AB` for residential; non-res types for GHD)
- `houses_per_building` (int): number of flats/households
- `occupants` (int/float): total occupants in the building (used to distribute to flats)
- `floors` (int): used to scale non-residential demands
- `area` (float): building footprint area (m²)
- `constructi` (str/int/NaN): construction year category for heat model (missing values are sampled)

### 2) Statistical input data

The code reads multiple statistics files from:

- `gridalloc/data/statistics/`

Examples (non-exhaustive):

- Household size distributions and electricity CDFs
- Non-residential (GHD) electricity/DHW per m² profiles
- Roof tilt distributions for PV
- EV specs and trip statistics for mobility

These are already included in the repository under `gridalloc/data/statistics/**`.

---

## Generated outputs

### 1) Result `.h5` file

For each run, the input file is **copied** to:

- `gridalloc/results/<same_filename_as_input>.h5`

and then augmented with additional tables.

### 2) HDF5 keys written by this step

This step writes (or overwrites) the following keys using `pandas.HDFStore`:

#### Raw data updates

- `raw_data/weather` (written even if it existed)
- `raw_data/buildings` (written with additional sampled fields)

#### URBS inputs (new group)

- `urbs_in/weather` (ambient time series with `Tamb`, `Irradiation`)
- `urbs_in/supim` (PV supply time series; multi-indexed columns `(bus, solar_tilt_azim)`)
- `urbs_in/demand` (electricity, heat, water heat, mobility demand)
- `urbs_in/eff_factor` (efficiency factors: heat pump COP and EV availability)
- `urbs_in/buy_sell_price` (import/feed-in prices)
- `urbs_in/process`
- `urbs_in/commodity`
- `urbs_in/process_commodity`
- `urbs_in/storage`

### 3) Logs (HPC)

The SLURM scripts write logs to:

- `gridalloc/logs/normal/<jobid>_output.log`
- `gridalloc/logs/errors/<jobid>_error.log`

---

## Expected HDF5 schema (input and output)

This step reads and writes pandas objects via `pandas.HDFStore`. Paths below are HDF5 keys.

### Input file schema (required)

#### `raw_data/region` (pandas DataFrame)

Expected to be a single-row table (the code reads it as a DataFrame and accesses columns like a series).

Required columns:

- `lat` (float): latitude of the transformer / representative grid location
- `lon` (float): longitude
- `altitude` (float): meters above sea level
- `plz` (int or str): ZIP code used by the heat generator
- `regio7` (int): region class used for EV ownership statistics

Strongly recommended (used for deterministic EV seeding; missing values can break mobility generation):

- `bcid` (int)
- `kcid` (int)

#### `raw_data/buildings` (pandas DataFrame)

One row per building. Minimum required columns:

- `bus` (int): unique node/site id for the building in the LV grid
- `use` (str): building use class; the code expects values like `Residential`, `Public`, `Commercial`
- `type` (str): building type code; residential types are typically `SFH`, `MFH`, `TH`, `AB`, non-res types match the GHD profile tables
- `houses_per_building` (int): number of flats / households in the building
- `occupants` (int/float): total occupants in the building (used to distribute household sizes)
- `area` (float): building footprint area in m²
- `floors` (int): number of floors (used to scale non-res GHD profiles)

Recommended:

- `constructi` (str/int/NaN): construction year class; if missing, this step samples missing values for non-res buildings

#### `raw_data/weather` (pandas DataFrame) (optional but recommended)

If present, the run can be executed fully offline (recommended on HPC). If missing, weather must be fetched from PVGIS/OpenMeteo.

Minimum columns used by this step:

- `time(inst)` (datetime-like): used as the PV model time index in `src/functions/solar.py`
- `ghi`, `dni`, `dhi` (float): irradiance components (W/m²)
- `temp_air` (float): ambient temperature (°C)
- `relative_humidity` (float): % (0–100)
- `pressure` (float): Pa

Additional columns required/used by other generators:

- `dew_point` (float): °C (mobility)
- `soil_temp` (float): °C (heat pump COP calculation; currently air-source COP uses `temp_air`, but soil temp is referenced)

Expected length:

- 8760 hourly rows (reference year handling is based on `config.REF_YEAR` and DST offsets are hard-coded for 2009)

### Output file schema (written by this step)

All outputs are written into a copy of the input file under `gridalloc/results/`.

#### Updated raw data tables

`raw_data/buildings` is overwritten with additional columns created during sampling, typically including:

- `roofs`: list of roof sections per building; each element is `(cap_kW, tilt_deg, azimuth_deg)`
- `occ_list`: list of household sizes per flat (residential)
- `demand_tot_list`: list of sampled annual electricity demands per flat (residential), or derived totals (non-res)
- `heating_type`: sampled as `radiator` or `floor` (used for COP)
- `cars_by_flat`, `n_cars_tot`, `car_dict`: mobility sampling results

`raw_data/weather` is written/overwritten with the final weather table used by the generators.

#### URBS inputs (`urbs_in/*`)

The following keys are written. Most time series are stored as DataFrames with 8760 rows and index name `t`.

- `urbs_in/weather`: ambient time series with MultiIndex columns `('ambient', 'Tamb')` and `('ambient', 'Irradiation')`
- `urbs_in/supim`: PV supply time series (SupIm), MultiIndex columns `(bus, solar_<tilt>_<azim>)`
- `urbs_in/demand`: concatenated demand time series
  - electricity: columns `(bus, 'electricity')`
  - heat: columns `(bus, 'space_heat')` and `(bus, 'water_heat')`
  - mobility: columns for each simulated EV, typically `(bus, 'mobility<id>')`
- `urbs_in/eff_factor`: efficiency factors
  - heat pump COP: columns `(bus, 'heatpump_air')`
  - EV availability: columns derived from emobpy, typically `(bus, 'charging_station<id>')` with values 0/1
- `urbs_in/buy_sell_price`: columns `electricity_import` and `electricity_feed_in`
- `urbs_in/process`: process table with URBS columns like `Site`, `Process`, `inst-cap`, `cap-up`, `inv-cost-fix`, `inv-cost`, `fix-cost`, `var-cost`, `wacc`, `depreciation`, `pf-min`
- `urbs_in/commodity`: commodity table with columns `Site`, `Commodity`, `Type`, `price`
- `urbs_in/process_commodity`: mapping table with columns `Process`, `Commodity`, `Direction`, `ratio`
- `urbs_in/storage`: storage table with columns `Site`, `Storage`, `Commodity`, capacities, efficiencies, and cost parameters

Notes on column naming:

- Bus-level tables use the building `bus` ids as the URBS `Site`.
- Mobility creates per-vehicle commodities/processes (`mobility<id>`, `charging_station<id>`, `mobility_storage<id>`). These ids are local to the grid/run.

---

## Source code tour (`gridalloc/src`)

This section documents the first-party code in `gridalloc/src`. Vendored third-party libraries under `gridalloc/src/external/` are intentionally not described here.

### `gridalloc/src/classes/`

- `grid.py`
  - Main orchestration class `Grid(settings)`.
  - Loads input HDF5 tables via `SaveFile`, runs generators in the required order, builds URBS tables (`create_*`), and writes everything back via `save_grid_data()`.
  - Implements daylight-saving alignment helpers (`_add_input_data_daylight_saving_shift`, `_add_output_data_daylight_saving_shift`) and a greedy workload partitioner (`partition_df_by_cpu`) for parallel heat generation.

- `save_grid.py`
  - Defines `SaveFile`, a small wrapper around pandas HDF5 I/O.
  - Reads required input tables (`get_input_data()`), copies the input file into `results/` (`copy_save_file()`), and appends tables with compression (`save_df()`).

- `resource_report.py`
  - Provides `resource_report(...)` / `ResourceReport` context manager.
  - Prints wall-clock time, CPU time and peak RSS (best-effort across platforms). Used for profiling pipeline sections.

### `gridalloc/src/functions/`

- `weather.py`
  - Fetches typical meteorological year weather data from PVGIS (SARAH3) and soil temperatures from OpenMeteo.
  - Adds helper `get_dew_point()` used by the mobility generator.

- `solar.py`
  - Samples roof sections (`roofs`) per building from tilt distributions and assigns flat vs. gabled roofs.
  - Uses `pvlib` to compute PV AC power time series per `(tilt, azimuth)` combination and builds the URBS `supim` (Supply-Import) table.
  - Creates URBS process/commodity mappings for PV.

- `electricity.py`
  - Samples household occupancy distribution per building and total annual electricity per household from CDFs.
  - Builds hourly electricity demand time series using residential load profiles (`elec_lps.h5`) and non-residential GHD profiles (per m²).
  - Creates URBS processes/commodities/storages for import/feed-in and electricity demand.

- `heat.py`
  - Uses the vendored districtgenerator interface (`Datahandler`) to generate hourly space-heating and DHW demand for residential buildings.
  - Adds non-residential DHW using GHD DHW profiles and generates heat pump COP time series (air-source, with radiator/floor sink temperature assumptions).
  - Creates URBS processes/commodities/storages for heat supply.

- `mobility.py`
  - Samples number of cars per household and assigns vehicle models + commuter/non-commuter schedules.
  - Runs EV simulation via emobpy (per-vehicle temporary DB dirs), resamples to hourly, reallocates non-home charging to home, and consolidates charging into the end of home-stretches.
  - Produces hourly EV charging demand (`mobility<id>`) and availability (`charging_station<id>`), and creates URBS process/commodity/storage definitions based on the simulated battery capacities.

---

## Folder and file structure

Top-level (this step):

- `gridalloc/main.py` – CLI entrypoint, selects input file and runs the pipeline
- `gridalloc/config.py` – all constants and file paths; instantiates `config = Config()`
- `gridalloc/run_cluster_serialstd.sh` – SLURM job script (single grid per job)
- `gridalloc/start_batch_jobs_serialstd.sh` – submits multiple SLURM jobs over an index range

Data:

- `gridalloc/data/grids/` – input grid `.h5` files
- `gridalloc/data/statistics/` – statistical datasets used for sampling and profiles
- `gridalloc/results/` – output `.h5` files (copy of inputs + URBS sheets)

Code:

- `gridalloc/src/classes/grid.py` – `Grid` class orchestrating the generation and URBS table creation
- `gridalloc/src/classes/save_grid.py` – HDF5 read/copy/write helpers
- `gridalloc/src/functions/` – domain generators:
  - `weather.py` (PVGIS/OpenMeteo fetching)
  - `solar.py` (pvlib PV modeling)
  - `electricity.py` (residential + GHD electrical load assignment)
  - `heat.py` (districtgenerator-based heat profiles + COP)
  - `mobility.py` (emobpy-based EV demand + availability)
- `gridalloc/src/external/` – vendored third-party code (districtgenerator, emobpy)

---

## How to run

All commands below assume you are in the `gridalloc/` directory.

### 1) Create the environment

On Linux/HPC, use:

```bash
conda env create -f environment_HPC.yml
conda activate grid_alloc
```

The `environment.yml` in this folder is a fully pinned export from a development machine and may include OS-specific artifacts; prefer `environment_HPC.yml` on clusters.

### 2) Run a single grid locally

```bash
python3 main.py <inputfile_id> --n_cpu 1
```

Example (if your grid file name starts with `0_`):

```bash
python3 main.py 0 --n_cpu 8
```

### 3) Run on HPC (SLURM)

Single job:

```bash
sbatch run_cluster_serialstd.sh <inputfile_id>
```

Submit a range (inclusive):

```bash
bash start_batch_jobs_serialstd.sh 0 24
```

---

## Input selection (`inputfile_id`)

`main.py` does **not** take a full path. It takes an `inputfile_id` and searches in `data/grids/`.

Matching rule:

- For each `*.h5`, take the substring before the **first underscore** (`_`).
- If it equals `str(inputfile_id)`, that file is selected.

So a call like `python3 main.py 0` matches files like:

- `0_N2819500E4261500_86165_2_40.h5`

Important:

- If multiple files share the same prefix, the code currently picks the first match.
- If no file matches, the run errors.

---

## Weather data handling (HPC vs local)

The pipeline supports two modes:

- **Weather already in the `.h5`** (`raw_data/weather`)
  - This is the default in `main.py` (`weather_data_exists=True`).
  - Recommended for HPC environments with restricted internet access.

- **Fetch weather from online APIs** (PVGIS + OpenMeteo)
  - Requires outbound internet.
  - Uses PVGIS SARAH3 TMY and OpenMeteo archive data.

Notes:

- The code assumes a fixed UTC+1 alignment (see `config.TIME_ZONE` and comments).
- Holidays/daylight-saving logic is hard-coded for the reference year `2009`.

---

## Details to keep in mind

- **Run directory matters**: paths like `data/grids` and `data/statistics` are relative; run from `gridalloc/`.
- **Time resolution**: all outputs are designed around **8760 hourly** time steps (1 year). The DST correction in `Grid` uses fixed indices for 2009.
- **Randomness / reproducibility**:
  - PV roof sections and several demand assignments use NumPy random sampling without a global seed.
  - EV simulation uses deterministic per-vehicle seeds derived from `(bcid, plz, kcid, bus, vehicle_id)` with retries that increment the seed if emobpy fails.
- **Parallel execution**:
  - `--n_cpu > 1` parallelizes heat generation (multiprocessing) and mobility simulation (process pool).
  - Mobility can be memory-heavy; keep `n_cpu` within available RAM.
- **Units**:
  - Electricity/heat/mobility demands are generated as per-timestep energies compatible with hourly URBS timesteps.
  - PV supply is derived from pvlib AC power in W and stored scaled by `1/1000` (kW). For hourly URBS timesteps this is typically interpreted as kWh per hour.

---

## Troubleshooting

- Missing weather columns (`temp_air`, `ghi`, `dni`, …): ensure `raw_data/weather` exists in the input `.h5`, or switch to API fetching.
- `IndexError` during input selection: no file matched your `inputfile_id`, or multiple matches exist and the selected one is unexpected.
- Mobility failures: emobpy may throw sporadic errors for certain sampled vehicles; the code retries up to 3 times with bumped seeds.

