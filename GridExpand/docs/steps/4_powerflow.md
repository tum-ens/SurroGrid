# Step 4: time-series power flow

Step 4 reconstructs per-bus active and reactive demand before (`pre`) and after (`post`) electrification, solves a
pandapower power flow for every timestep and stores the raw time series and/or compact summary statistics.

Entry point: `gridexpand powerflow` = `src/gridexpand/powerflow/run_pwrflw.py` (synthetic pylovo grids).

| module | content |
|---|---|
| `demands.py` | `reconstruct_demands`: pre, flexible and INFLEX post demand with reactive power |
| `network.py` | load-bus normalization, relaxed limits, transformer handling, evaluation scopes (`full`, `backbone`) |
| `engine.py` | `run_timeseries` (chunked, parallel), raw tables, `summarize_powerflow_matrices` |
| `io.py` | `HDF_KEYS`, `ScenarioResultReader`, database and HDF5 sinks |
| `config.py` | power factors |
| `run_real_swf_powerflow.py`, `run_real_swf_scenario_powerflow.py` | real-grid (SWF, ÜZW with `--provider`) runners called by the paired runner |

## Inputs

`inputfile_id` is a path, an exact file name in `work/powerflow/input/` or the unique prefix before the first
underscore of one file there. With `--pre-only` it may be a Step 2 file; otherwise it is a Step 3 result. Keys read
(`io.HDF_KEYS`):

| key | used for |
|---|---|
| `urbs_in/demand` (or `urbs_out/reduced_data/demand` after TSAM) | pre demand (base electricity) |
| `urbs_out/MILP/tau_pro` | flexible post demand: `import − feed_in`, `heatpump_air`, `Rooftop*` PV |
| `urbs_out/MILP/cap_pro`, `urbs_in` or `urbs_out/reduced_data` `eff_factor`, `supim`, `process`, `storage`, `urbs_in/ev_sessions`, `ev_session_hours` | INFLEX post demand |
| `urbs_out/temporal_method`, `urbs_out/tsam/hoursPerPeriod` | temporal provenance and alignment (`--expect-temporal-method`) |
| `urbs_out/solver_audit` | Step 3 provenance copied into the run assumptions |
| `raw_data/allocation_plan` | paired inputs: projection of scenario units onto buses |
| `raw_data/net` | the network in HDF5 mode (without it, the pylovo grid is read from the database) |
| `metadata/timeframe` | scenario key, timeframe, model case (run assumptions) |

In DB mode the pandapower grid comes from pylovo, resolved from the file name or `--grid-case-id`.

## Command line

```bash
uv run gridexpand powerflow <inputfile_id> --storage db --outputs raw,summary --n_cpu 4
uv run gridexpand powerflow <inputfile_id> --storage db --pre-only --outputs summary
```

| option | default | meaning |
|---|---|---|
| `--storage` | `h5` | `h5`: raw tables into `work/powerflow/output/<file>`; `db`: runs in the `surrogrid` schema |
| `--outputs` | `raw` | comma list of `raw`, `summary`; both come from one power-flow pass; summaries need `--storage db` |
| `--summary-only` | | same as `--outputs summary` |
| `--run-name`, `--summary-run-name` | DB default | run name of the raw tables (or of the summary with `--summary-only`); separate summary run with `--outputs raw,summary` |
| `--pre-only` | off | only the pre stage from `urbs_in/demand` (no Step 3 result needed) |
| `--post-demand-mode` | `flexible` | `flexible` (optimized urbs import) or `inflex` (fixed heat, PV and EV charging; needs EV sessions, i.e. paired inputs) |
| `--inflex-ev-charger-kw` | none | INFLEX only: cross-check of the charger rating of every vehicle |
| `--expect-temporal-method` | none | reject a Step 3 result that does not record `full_year_no_tsam` / `shared_weather_tsam` |
| `--summary-grid-scope` | `full` | `full` or `backbone` (terminal service lines removed, terminal voltages mapped one bus upstream) |
| `--summary-nonconvergence` | `auto` | summary runs: `nan`/`auto` record failed timesteps and continue, `raise` aborts; raw runs always raise |
| `--grid-case-id`, `--pylovo-version-id` | | DB mode: explicit grid case; pylovo version (default `PYLOVO_VERSION_ID`) |
| `--hh-only`, `--hh-annual-demand-scale` | off, 1.0 | DB mode: household components only; scale of pre household demand (needs `--hh-only --pre-only`) |
| `--max-timesteps` | none | smoke-test limit |
| `--n_cpu` | 1 | time chunks solved in parallel processes |

## Outputs

| output | HDF5 key (`--storage h5`) | database table (`--storage db`) |
|---|---|---|
| per-bus demand | `pwrflw/input/demand_pre`, `pwrflw/input/demand_post` | `powerflow_demand` |
| external-grid import | `pwrflw/output/<stage>/demand_import` | `powerflow_import` |
| bus voltage | `pwrflw/output/<stage>/vm` | `powerflow_bus_voltage` |
| line flow and current | `pwrflw/output/<stage>/line_loads` | `powerflow_line_result` |
| reactive components (household, heat pump, PV) | `pwrflw/urbs_out/MILP/reactive` | `powerflow_reactive_component` |
| summary (database only) | | `powerflow_summary`, `powerflow_cable_summary`, `powerflow_bus_voltage_summary`, `powerflow_tail_value`, `powerflow_transformer_diagnostic` |
| installed building assets (database only, post cases) | (the input's `urbs_out/MILP/cap_*`) | `powerflow_asset` |

`<stage>` is `pre` or `post`. The HDF5 output is a copy of the input with these keys appended (an existing output
of the same name is replaced). In the database each run is one `powerflow_run` row with the run assumptions
(timeframe metadata, Step 4 settings, temporal method, Step 3 solver summary). Rows are written to a staging run
that replaces the previous run of the same name in one transaction only when the whole pass succeeded; a failed or
cancelled Step 4 keeps the previous results ([database.md](../database.md)). INFLEX runs with a stationary battery
also write a battery-state audit `work/powerflow/output/<file stem>[.<run name>].component_audit.h5` (key
`component_audit/inflex_battery_state`).

## Method

**Demand.** Pre: base electricity from `urbs_in/demand`, reactive power with the fixed power factor
`PF_ELC = 0.959`. Flexible post: net import `import − feed_in` per bus from `tau_pro`; heat-pump reactive power with
`PF_HP = 0.95`; PV compensates reactive power locally within `|Q| ≤ P_PV · tan(arccos(PF_PV_MIN))`, `PF_PV_MIN =
0.95` (a model assumption, not a voltage-dependent inverter control). INFLEX post: pre electricity plus fixed heat
(heat pump up to its fixed electric capacity times COP, residual on the auxiliary heater), fixed PV and EV charging
from the session table, and the fixed battery under causal self-consumption control. After TSAM the initialization
row is dropped (6 × 168 = 1,008 simulated hours); paired inputs are projected from scenario units onto buses.

**Network.** Static pylovo loads are replaced by one zeroed load row per demand bus (duplicate demand buses are
rejected); static generators, generators and storages are switched off; line, load and voltage limits are relaxed
(1000 kA, 1000 MW, 0–10 pu) and the rated cable currents are kept for the evaluation. Synthetic grids: the single
MV/LV transformer is replaced by a closed bus-bus switch and the external grid bus gets the LV nominal voltage;
transformer loading is evaluated from the external-grid import against the station rating (`sn_mva` × parallel units).

**Solve.** Synthetic grids use `pandapower.runpp(algorithm="bfsw")` per timestep (real grids `nr`, then
`iwamoto_nr`), `tolerance_mva = 1e-6`; timesteps are split into `--n_cpu` chunks solved in parallel processes, and
results do not depend on the chunking.

## Conventions

- Units: demand in kW/kvar, converted to MW/Mvar for pandapower.
- Signs (pandapower load convention): positive P consumes active power, positive Q absorbs inductive reactive power;
  PV compensation contributes negative Q to the net load.
- Time: `t_index` 0-based hours of the run's timeframe; database time stamps are `timeframe_start + t_index`.
- Non-convergence is never dropped silently: summary runs record the failed timesteps (`powerflow_tail_value`,
  metric `Power-flow convergence`), raw runs abort.

## HPC

```bash
sbatch scripts/hpc/powerflow/run_cluster_serialstd.sh <inputfile_id>
bash scripts/hpc/start_batch_jobs.sh powerflow 0 24
```
