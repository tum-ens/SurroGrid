Goal:
- The final goal is to change file based storage in GridExpand from h5 files to the existing infdb postgresql database ecosystem where it makes sense and it is possible. 

Background:
- At the moment, for each ags-kcid-bcid combination that describes a grid from pylovo, we get a h5 file for multiple steps in the pipeline
- Statistical, more static data like in 2.demand_allocation/gridalloc/data/statistics should stay in the current format
- All grid related h5 files generate a lot of files e.g. in 2.demand_allocation/gridalloc/data/grids, 2.demand_allocation/gridalloc/results, 4.powerflow/Input, 4.powerflow/Output could contain duplicate data, also the generated grids are already present at the pylovo schema there (Due to our latest changes, the grids are actually fetched from this database). I think a much more efficient, object-relational approach is possible to make the pipeline of this repo more efficcient

Task:
- First analyze representative existing HDF5 files and current HDF5 read/write points to identify duplicated grid-related data and propose a target database structure. Before editing code, inspect the current HDF5 read/write points and propose 2-3 migration candidates.
For each candidate, explain:
- which files/tables are affected
- whether the data already exists in pylovo/infdb
- what new table/schema would be needed
- expected benefit
- risk

Wait for approval before implementation.

- Afer approval start testing that structure going through it step by step. Do not migrate all HDF5 files in one pass.
- Do not migrate static statistics files, weather cache files, notebooks, or external prepared artifacts such as elec_lps.h5 unless explicitly discussed first.
- Test your changes with ags 09278140 as a test. A successful first implementation should:
    - remove or bypass at least one duplicated grid-related HDF5 artifact
    - preserve the existing pipeline behavior for one example AGS
    - use the existing database connection from GridExpand/1.grid_sampling/.env
    - add the minimal database schema/tables needed under schema surrogrid and make this transparent (e.g. by writing an extra ddl file for the surrogrid schema creation as in /home/breveron/git/github/pylovo/src/pylovo/database/config_table_structure.py)
    - include a short migration note explaining what still remains file-based

Infos:
- I stored the path to the new database in GridExpand/1.grid_sampling/.env
- Feel free to add a new schema named surrogrid to that database
- to briefly understand the pylovo grid structure have a look into the other repos to generate grids have a look into ../pylovo
- to briefly understand the infdb database ecosystem have a look into the repo ../infdb/infdb-june
- Prefer normalized relational tables for grid metadata, buses, lines, transformers, buildings, consumers, and time-series references.
- Do not store opaque pickles or whole HDF5 blobs in the database unless explicitly justified.
- Use PostGIS geometry types where spatial data is present and timescaledb for timeseries data (this extension should be available in the target database)

