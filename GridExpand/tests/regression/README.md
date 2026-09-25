# End-to-end regression harness

Runs the ordinary synthetic pipeline (Steps 2-4 and the expansion materialization) on the four grids of an
OSM-derived demo region and snapshots every `surrogrid` table, so that refactorings can be proven to leave
results unchanged. The pipeline is deterministic (fixed `profile_seed`, TEASER heat seeded per building,
Gurobi deterministic concurrent LP), so two runs of the same code are `IDENTICAL`.

**Never point this at a production database**: the script drops and recreates the database named in `.env`
and refuses unless host, port and database-name prefix match the sandbox settings below.

## What it runs

- Region: AGS 9184137 / PLZ 85653 (Aying, © OpenStreetMap contributors, ODbL), pylovo version 1, 4 grids
  with 63-171 buildings each (`--min-buildings 60`).
- Scenario: `scenario_sandbox.yaml` (Schweinfurt 2045 assumptions, TEASER heat, see its header).
- Three model runs: `pre`, `post-hems-heuristic`, `post-hems-optimized`; timeframe
  `max_base_electricity_demand_week` (168 h); `--powerflow-output both`.
- Runtime on a 32-core workstation: about 12 min (`HARNESS_QUICK=1`: one grid, about 3 min).
- Not covered: paired/aligned pipelines (confidential DSO data), full-year runs, TSAM, INFDB heat, h5 mode.

## Sandbox database

A PostgreSQL container with PostGIS, pgRouting and TimescaleDB (`infdb/db` image) holding a template database
with the pylovo schema (setup plus generated versions 1-3 for PLZ 85653, built from OSM data with an
InfDB-shaped `basedata`/`opendata` layout) and `sandbox_citydb_roofs.sql` applied (synthetic LoD2 roofs; the
real InfDB provides `citydb`). Settings (environment variables, defaults in brackets): `HARNESS_CONTAINER`
[`pylovo-fable-sandbox`], `HARNESS_TEMPLATE_DB` [`surrogrid_fable_base`], `HARNESS_DB_HOST` [`127.0.0.1`],
`HARNESS_DB_PORT` [`55439`], `HARNESS_DB_PREFIX` [`sg_`], `HARNESS_DB_USER`/`HARNESS_DB_PASSWORD`
[`sandbox`]. The demo-region builder lives in the GridPlanner repository (`demo/`).

## Usage

```bash
cd GridExpand
cp .env.example .env.sandbox        # DB_HOST=127.0.0.1, DB_PORT=55439, DB_NAME=sg_<you>, sandbox user
GRIDEXPAND_ENV_FILE=$PWD/.env.sandbox tests/regression/run_harness.sh /tmp/run_a
GRIDEXPAND_ENV_FILE=$PWD/.env.sandbox tests/regression/run_harness.sh /tmp/run_b
uv run python tests/regression/compare.py /tmp/run_a/snapshot.pkl /tmp/run_b/snapshot.pkl   # IDENTICAL
```

`compare.py` aligns rows on natural keys (serial ids are replaced by grid/run names in `snapshot.py`) and
prints, per table and column, how many rows differ and by how much. JSON `assumptions` columns are compared per
key; absolute paths inside them are reduced to their basename (run directories differ between runs).
