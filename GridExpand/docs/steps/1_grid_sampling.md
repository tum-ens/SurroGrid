# Step 1: grid sampling and HDF5 export

Step 1 selects low-voltage grids from a pylovo database and exports each one as an HDF5 file with the
pandapower network, the physical buildings with their component manifest, region metadata and (optionally)
weather. It is only needed for the HDF5 hand-off: the database-backed pipelines (`gridexpand run`,
`gridexpand synthetic`, `gridexpand allocate --storage db`) read grids directly from pylovo and skip Step 1.

Code: `src/gridexpand/sampling/` (`db_read.py` readers, `export_grid.py` shared export, `save_grid.py` HDF5
writer, `grid_topol.py` topology clean-up, `export_single_grid.py` CLI). Notebooks:
`notebooks/sampling/`. Inputs: `data/sampling/` (census grid, VG250 municipalities, PLZ shapes, RegioStaR
reference files). Output: `work/sampling/results/`.

## Inputs

- A pylovo database (read-only; credentials from `GridExpand/.env`). Readers use `pylovo.grid_result`,
  `transformer_positions`, `buildings_result`, `municipal_register`, `pandapower_bus` and `pandapower_load`
  through `gridexpand.db`. `PYLOVO_VERSION_ID` in `.env` pins the version; without it the numerically latest
  version of a grid is used. Coordinates are transformed in SQL (`ST_Transform`) from the pylovo CRS
  (EPSG:25832) to WGS84.
- Internet access to PVGIS for weather (unless `--skip-weather`).

## Notebooks (regional sampling)

```bash
uv sync --extra notebooks
uv run jupyter lab notebooks/sampling
```

1. `1_filter_valid_grids.ipynb`: reads candidate grids (PLZ/KCID/BCID and transformer location), keeps those near
   populated cells of `data/sampling/Zensus2022_Bevoelkerungszahl_1km-Gitter.csv` and writes
   `data/sampling/valid_grids` (HDF5, key `grids`).
2. `2_sample_grids.ipynb`: samples census cells weighted by population, maps them to pylovo grids and exports each
   grid with `export_grid.export_pylovo_grid` as `N<y>E<x>_<plz>_<kcid>_<bcid>.h5` (cell centroid in EPSG:3035).

## Single-grid export

```bash
uv run python -m gridexpand.sampling.export_single_grid --plz 80803 --list-candidates
uv run python -m gridexpand.sampling.export_single_grid --plz 80803 --candidate-index 0 --cell-id 0
```

| option | meaning |
|---|---|
| `--plz` | postcode (required) |
| `--kcid`, `--bcid` | one exact grid (both together) |
| `--candidate-index` | 0-based candidate of the PLZ when `--kcid/--bcid` are not given (default 0) |
| `--min-buildings` | minimum buildings of a candidate (default 5) |
| `--cell-id` | file-name prefix (default `pilot<PLZ>`) |
| `--skip-weather` | no PVGIS download, no `raw_data/weather` |
| `--list-candidates` | print the `(plz, kcid, bcid)` candidates and exit |

The output is `work/sampling/results/<cell_id>_<plz>_<kcid>_<bcid>.h5` (an existing file is overwritten).

## Output (HDF5 keys)

| key | content |
|---|---|
| `raw_data/net` | pandapower network as JSON string (`pandapower.to_json`); zero line lengths set to 1e-6 km, one zeroed load row per demand bus |
| `raw_data/consumers` | `Consumer Nodebus <vertice_id>` buses of the network |
| `raw_data/region` | one row: `municipal_register` fields of the PLZ plus `lat`, `lon`, `kcid`, `bcid`, and `altitude` (from PVGIS) |
| `raw_data/buildings` | one row per physical building (`objectid` unique) with `bus`, `floor_area`, `floor_number`, `residential_floor_area`, `nonresidential_floor_area`, `nonresidential_use`, `households`, `occupants` (imputed where pylovo has households but no occupants, flag `occupants_imputed`), `building_use`, `building_type`, mixed-use fields and peak loads |
| `raw_data/building_components` | one Residential and/or Commercial/Public component per positive component; MV-direct non-residential components are kept with `included_in_lv = False` |
| `raw_data/weather` | hourly PVGIS SARAH3 TMY moved to the reference year 2009 in UTC+1 (e.g. `ghi`, `dni`, `dhi`, `temp_air`, `relative_humidity`, `pressure`, `time(UTC+1)`, `time(inst)`, plus `dew_point`; see `src/gridexpand/common/weather.py`) |

Step 2 in HDF5 mode reads these files from `work/allocation/grids/` and selects a file by the part of its name
before the first underscore (the notebook's `N<y>E<x>` or the single-grid `--cell-id`), see
[Step 2](2_demand_allocation.md#input-selection).

Limitations:

- The export writes no `raw_data/pv_roof_sections`, which the post (electrification) cases need in HDF5 mode; HDF5
  inputs therefore support the `pre` case only (open question: extend the export or retire the HDF5 mode).
- Step 2 in HDF5 mode never downloads weather; a file exported with `--skip-weather` cannot be used for Step 2
  unless `raw_data/weather` is added.
- Old one-use HDF files without `raw_data/building_components` are rejected by Step 2.

## Conventions

- Weather and all time series use fixed UTC+1 and the reference year 2009 ([method.md](../method.md#time-axis)).
- Nothing is written to the database; `data/sampling/valid_grids` is written by notebook 1 into the repository
  data folder.
