# SurroGrid Database Schema

This document summarizes the PostgreSQL tables created by `GridExpand/surrogrid_schema.sql` for DB-backed GridExpand runs. The schema is named `surrogrid` and is intended to store grid identity, scenario identity, run metadata, selected demand allocation outputs, and power-flow outputs.

The reusable pregenerated mobility profile pool is not stored in the database. It remains CSV-backed under `GridExpand/2.demand_allocation/gridalloc/data/statistics/general/mobility_profile_pool/` or an equivalent external directory. The database stores only the selected, run-specific vehicle/profile allocations and resulting per-bus time series.

## Run Identity

```text
grid_case + scenario
        |
        v
pipeline_run
   |             |
   v             v
demand_allocation_run   powerflow_run
   |                     |
   v                     v
allocated_*             powerflow_*
```

### `surrogrid.grid_case`

One physical LV grid case. It links SurroGrid runs back to the source pylovo grid and building data.

Key columns:

- `grid_case_id`: surrogate primary key.
- `ags`, `plz`, `kcid`, `bcid`: municipality, postal code, and pylovo grid identifiers.
- `pylovo_grid_result_id`, `pylovo_version_id`: source pylovo grid version.
- `cell_id`: readable GridExpand grid label, for example `9278140-00`.

### `surrogrid.scenario`

One reusable scenario definition or assumption set.

Key columns:

- `scenario_id`: surrogate primary key.
- `scenario_key`: readable unique key, currently `baseline_static` by default.
- `scenario_label`, `description`: human-readable metadata.
- `assumptions`: JSONB scenario assumptions.

### `surrogrid.pipeline_run`

Shared parent for one grid and scenario pipeline execution. This table is the bridge that keeps Step 2 and Step 4 results coupled without merging their detailed metadata.

Key columns:

- `pipeline_run_id`: surrogate primary key.
- `grid_case_id`, `scenario_id`: parent grid and scenario.
- `run_name`: readable pipeline-level name, currently `baseline_static_pipeline` by default.

## Step 2 Demand Allocation Tables

### `surrogrid.demand_allocation_run`

Metadata for one Step 2 demand allocation run.

Key columns:

- `demand_allocation_run_id`: surrogate primary key.
- `pipeline_run_id`: shared parent pipeline run.
- `grid_case_id`, `scenario_id`: repeated for direct filtering and uniqueness.
- `run_name`: Step 2 run name, for example `baseline_static_all_pool_demand_allocation`.
- `bridge_filename`: temporary HDF5 bridge filename used while Step 3 still consumes HDF5.
- `profiles`: Step 2 profile scope, for example `status_quo`, `electricity_heat`, `electricity_mobility`, or `all`.
- `mobility_source`: `emobpy` or `pool`.

### `surrogrid.allocated_demand`

Per-bus hourly demand time series written by Step 2. This includes electricity, heat, mobility, and any other demand commodities present in `/urbs_in/demand`.

Key columns:

- `demand_allocation_run_id`: parent Step 2 run.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `bus`: pandapower bus index.
- `commodity`: demand label from the URBS input table.
- `value`: demand value.

### `surrogrid.allocated_eff_factor`

Per-bus hourly efficiency or availability factors written by Step 2 from `/urbs_in/eff_factor`.

Key columns:

- `demand_allocation_run_id`: parent Step 2 run.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `bus`: pandapower bus index.
- `component`: component label.
- `value`: efficiency or availability factor.

### `surrogrid.allocated_vehicle`

Vehicle metadata selected during Step 2 mobility allocation. This is the database record of which pregenerated pool profile, or generated emobpy vehicle, was assigned to each building bus.

Key columns:

- `demand_allocation_run_id`: parent Step 2 run.
- `bus`, `vehicle_id`: vehicle identity within the allocated grid.
- `model`: sampled EV model.
- `schedule`: sampled driver schedule, for example commuter or freetime.
- `seed`: vehicle sampling seed.
- `profile_id`: pregenerated mobility pool profile ID when `mobility_source = 'pool'`.
- `battery_cap_kwh`: vehicle battery capacity used by the allocation.

## Step 4 Power-Flow Tables

### `surrogrid.powerflow_run`

Metadata for one Step 4 power-flow run.

Key columns:

- `powerflow_run_id`: surrogate primary key.
- `pipeline_run_id`: shared parent pipeline run.
- `grid_case_id`, `scenario_id`: repeated for direct filtering and uniqueness.
- `run_name`: Step 4 run name, for example `baseline_static_pre_powerflow` or `baseline_static_full_powerflow`.
- `urbs_input_file`: temporary HDF5 bridge file read by Step 4.
- `pre_only`: true when the run skipped URBS post-expansion demand.

### `surrogrid.powerflow_demand`

Per-bus active and reactive demand reconstructed by Step 4 for power-flow simulation.

Key columns:

- `powerflow_run_id`: parent Step 4 run.
- `stage`: `pre` or `post`.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `bus`: pandapower bus index.
- `p_kw`, `q_kvar`: active and reactive demand.

### `surrogrid.powerflow_import`

External-grid import time series.

Key columns:

- `powerflow_run_id`: parent Step 4 run.
- `stage`: `pre` or `post`.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `p_mw`, `q_mvar`: active and reactive import.

### `surrogrid.powerflow_bus_voltage`

Per-bus voltage results.

Key columns:

- `powerflow_run_id`: parent Step 4 run.
- `stage`: `pre` or `post`.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `bus`: pandapower bus index.
- `vm_pu`: voltage magnitude in per unit.

### `surrogrid.powerflow_line_result`

Per-line loading and flow results.

Key columns:

- `powerflow_run_id`: parent Step 4 run.
- `stage`: `pre` or `post`.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `line`: pandapower line index.
- `p_from_mw`, `q_from_mvar`, `i_from_ka`: line flow and current from the from-bus side.

### `surrogrid.powerflow_reactive_component`

Reactive power contribution by component/source, written for the power-flow input reconstruction.

Key columns:

- `powerflow_run_id`: parent Step 4 run.
- `ts`, `t_index`: timestamp and zero-based timestep.
- `bus`: pandapower bus index.
- `component`, `source`: component labels.
- `q_kvar`: reactive power contribution.

## Views

### `surrogrid.grid_building_bus`

Convenience view joining `surrogrid.grid_case` to pylovo building, bus, and load metadata. Use it to map allocated demand and power-flow results back to building attributes and coordinates.

## Example Queries

List pipeline runs and their child run IDs:

```sql
SELECT
    pipe.pipeline_run_id,
    gc.cell_id,
    gc.plz,
    sc.scenario_key,
    pipe.run_name AS pipeline_run_name,
    dar.demand_allocation_run_id,
    dar.run_name AS demand_allocation_run_name,
    pr.powerflow_run_id,
    pr.run_name AS powerflow_run_name
FROM surrogrid.pipeline_run pipe
JOIN surrogrid.grid_case gc USING (grid_case_id)
JOIN surrogrid.scenario sc USING (scenario_id)
LEFT JOIN surrogrid.demand_allocation_run dar USING (pipeline_run_id)
LEFT JOIN surrogrid.powerflow_run pr USING (pipeline_run_id)
ORDER BY pipe.created_at DESC;
```

Inspect selected mobility profiles for a Step 2 run:

```sql
SELECT
    av.bus,
    av.vehicle_id,
    av.model,
    av.schedule,
    av.profile_id,
    av.battery_cap_kwh
FROM surrogrid.allocated_vehicle av
JOIN surrogrid.demand_allocation_run dar USING (demand_allocation_run_id)
WHERE dar.run_name = 'baseline_static_all_pool_demand_allocation'
ORDER BY av.bus, av.vehicle_id;
```

Inspect allocated mobility demand at one bus:

```sql
SELECT ad.ts, ad.bus, ad.commodity, ad.value
FROM surrogrid.allocated_demand ad
JOIN surrogrid.demand_allocation_run dar USING (demand_allocation_run_id)
WHERE dar.run_name = 'baseline_static_all_pool_demand_allocation'
  AND ad.bus = 1
  AND ad.commodity ILIKE '%mobility%'
ORDER BY ad.t_index;
```

Inspect maximum post-expansion line current for a power-flow run:

```sql
SELECT line, MAX(i_from_ka) AS max_i_from_ka
FROM surrogrid.powerflow_line_result plr
JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
WHERE pr.run_name = 'baseline_static_full_powerflow'
  AND plr.stage = 'post'
GROUP BY line
ORDER BY max_i_from_ka DESC;
```
