# TASK: Optional Timeslice Mode For Large AGS Studies

You are working in the SurroGrid repository.

Before changing code, read:

- `AGENTS.md`
- `MEMORY.md`
- `ERRORS.md`
- `TASK.md`
- the relevant GridExpand Step 2, Step 3, Step 4, database, and AGS runner code

Do not implement immediately. First inspect the current code paths, then present 2-3 implementation approaches with pros and cons and wait for user approval.

## Goal

Design and implement an optional timeslice mode for GridExpand.

The purpose is to make large-region studies, especially Munich AGS `09162000`, computationally manageable by running selected critical one-week horizons instead of full-year simulations.

The default full-year pipeline must remain unchanged.

This mode is for targeted operational stress studies and screening, not for full annual scenario conclusions unless the model is later adapted and documented for that interpretation.

## Context From The Munich Full-Year Pilot

The Munich AGS contains many individual grid candidates. A candidate is one grid case identified by `AGS-index_PLZ_KCID_BCID.h5`; it is not a whole PLZ and not the full AGS.

The robust full-year runner proved stable for the first Munich grids, but runtime is too high for the full AGS:

- Candidate 0 full-year Step 3 required `--n_cpu 16` and `URBS_CLUSTER_CONCURRENCY=1`.
- with lower timesteps we can also increase the URBS_CLUSTER_CONCURRENCY substantially as RAM was the problem here
- Lower cluster counts created too-large Pyomo/Gurobi models and were killed by memory pressure.
- Step 4 DB writing needed chunking and validation to avoid partial post-result rows.
- Step 2 persistent allocated time-series storage is expensive, so `--timeseries-storage temp` is preferred for large batches.
- The full Munich run is technically possible but too slow in full-year mode.

Timeslicing should reduce:

- Step 3 model size
- Step 4 powerflow runtime
- DB/HDF5 time-series volume
- memory pressure during Pyomo/Gurobi and DB writes

It will not reduce the number of buildings/vehicles by itself, so Step 3 clustering and DB write chunking remain necessary.

## Relevant Files

Inspect at least:

- `GridExpand/2.demand_allocation/gridalloc/main.py`
- `GridExpand/2.demand_allocation/gridalloc/src/classes/grid.py`
- `GridExpand/2.demand_allocation/gridalloc/src/classes/save_grid.py`
- `GridExpand/2.demand_allocation/gridalloc/src/functions/electricity.py`
- `GridExpand/2.demand_allocation/gridalloc/src/functions/heat.py`
- `GridExpand/2.demand_allocation/gridalloc/src/functions/solar.py`
- `GridExpand/2.demand_allocation/gridalloc/src/functions/mobility.py`
- `GridExpand/2.demand_allocation/gridalloc/config.py`
- `GridExpand/3.urbs/run_urbs_cluster.py`
- `GridExpand/3.urbs/urbs/runfunctions.py`
- `GridExpand/4.powerflow/run_pwrflw.py`
- `GridExpand/4.powerflow/src/save_grid.py`
- `GridExpand/4.powerflow/src/demands.py`
- `GridExpand/database.py`
- `GridExpand/surrogrid_schema.sql`
- `GridExpand/ags_pipeline_runner.py`

## Supported Timeframe Modes

Keep `full_year` as default.

Start with a small, non-circular set of one-week modes:

- `full_year`
- `min_temperature_week`
- `max_solar_radiation_week`
- `max_base_electricity_demand_week`

Avoid initially:

- `max_mobility_demand_week`, because this is circular if mobility generation is what we want to avoid
- `max_net_load_week`, because net load depends on heat, PV, mobility, and optimization outputs
- any mode that requires full expensive demand generation before selecting the week

A useful next extension may be explicit manual slicing:

```bash
--timeframe-mode manual_week --timeframe-start 2009-01-12T00:00:00
```

Add manual mode only if the implementation remains simple.

## Critical Design Point

To reduce runtime, the selected week must be known before expensive generation steps whenever possible.

Expected flow for exogenous modes:

1. Read grid/building metadata.
2. Retrieve/read weather.
3. Select week from cheap exogenous data:
   - min temperature week from weather temperature
   - max solar radiation week from weather GHI
   - max base electricity demand week only if electricity generation is cheap enough
4. Slice weather and all generated profiles to the selected 168-hour horizon.
5. Run heat, solar, electricity, and mobility only on the selected horizon where feasible.
6. Save the resulting HDF5/DB outputs with explicit timeframe metadata.

## Daylight Saving Time

The current daylight-saving helpers assume a full 8760-hour year and fixed indices.

For timeslice mode:

- Do not blindly apply full-year DST insertion/deletion indices to a 168-hour slice.
- Prefer selecting after the profile time basis is defined and you know if DST applies
- Preserve correct timestamps for the selected week.
- If DST weeks are unsupported initially, reject them explicitly with a clear error.

## Mobility

Always use the `--mobility-source pool`.

Reason:

- The pregenerated pool already exists.
- It avoids full emobpy generation.
- It allows slicing selected vehicle profiles to the chosen week.

Do not support expensive full emobpy generation in timeslice mode.

Need to inspect:

- whether pool profiles are full-year indexed
- how availability and charging demand are sliced
- how initial battery state and final storage state are handled
- whether URBS storage initial/final assumptions are acceptable for a one-week operational stress run

If assumptions are simplified, document them.

## URBS Requirements

Step 3 must support 168-hour inputs explicitly.

Inspect and adjust:

- timestep derivation from HDF5 input length
- storage initialization
- storage final-state assumptions
- annualized cost/investment interpretation
- objective scaling
- result saving

Important:

If costs remain annualized or investment decisions are still enabled, document that one-week runs are operational/stress-screening outputs and not investment-valid annual optima.

## Step 4 Requirements

Step 4 must work for any HDF5 horizon length, especially 168 and 8760.

Adjust:

- demand reconstruction from `urbs_in/demand` and `urbs_out/MILP/tau_pro`
- timestamp generation in DB writers
- validation checks
- powerflow output row expectations

The Step 4 DB validator must expect:

- 8760 rows per year horizon
- 168 rows per one-week horizon

Do not hard-code `0..8759` in new validation logic without checking timeframe metadata or actual input length.

## DB/HDF5 Requirements

Create a new scenario so we can select the new run input/result data for the reduced timeframe. We also need to adjust the h5 filenames in step3 to distinguish between both runs also for the h5 files.

HDF5 and DB behavior must remain comparable.

Add explicit metadata for timeframe assumptions. Prefer metadata over encoding meaning only in filenames or run names.

At minimum store:

- `timeframe_mode`
- `horizon_hours`
- `timeframe_start`
- `timeframe_end`
- `source_year_or_reference_year`
- `methodological_note`
- whether the run is `full_year` or `timeslice`
- whether cost/investment interpretation is annual-valid or operational-only

Potential storage locations:

- HDF5 key such as `/metadata/timeframe`
- DB columns on pipeline/demand/powerflow runs, or a small scenario-assumption table

Do not reintroduce stale timeslice run names or ambiguous old metadata.

If adding schema columns, include migration logic in:

- `GridExpand/database.py`
- `GridExpand/surrogrid_schema.sql`

Explain why the metadata belongs in schema.

## AGS Runner Requirements

Update `GridExpand/ags_pipeline_runner.py` to accept and pass through timeslice settings.

Example:

```bash
uv run --project GridExpand/4.powerflow python GridExpand/ags_pipeline_runner.py \
  --repo-root /path/to/SurroGrid \
  --ags 09162000 \
  --timeframe-mode min_temperature_week \
  --step2-timeseries-storage temp \
  --workers 1
```

Runner status/logs should include:

- timeframe mode
- horizon hours
- selected week start/end
- failed grids
- per-stage runtime

Keep resume behavior compatible.

## Acceptance Criteria

1. `full_year` output is unchanged.
2. One timeslice mode runs Step 2, Step 3, and Step 4 for one small grid.
3. DB-backed and HDF5-backed behavior are comparable for the same grid.
4. Output time-series lengths are correct:
   - 8760 for `full_year`
   - 168 for one-week modes
5. Step 3 derives modeled timesteps from input length or explicitly supports 168.
6. Step 4 runs and writes complete pre/post DB results for 168-hour horizons.
7. The AGS runner can pass the timeframe mode through the full pipeline.
8. Metadata clearly states that one-week mode is a targeted operational/stress study, not a full annual result.
9. Munich candidate pilot works with one selected timeslice before any larger Munich subset is attempted.

## Suggested Validation

Use small known AGS first:

```bash
cd GridExpand/2.demand_allocation/gridalloc
uv run --project .. python main.py 09278140 \
  --storage db \
  --profiles all \
  --mobility-source pool \
  --timeseries-storage temp \
  --timeframe-mode min_temperature_week \
  --n_cpu 1
```

Then:

```bash
cp results/<generated>.h5 ../../3.urbs/Input/

cd ../../3.urbs
URBS_CLUSTER_CONCURRENCY=1 uv run python run_urbs_cluster.py <input_id> --n_cpu 4

cp result/<generated_urbs>.h5 ../4.powerflow/Input/

cd ../4.powerflow
uv run python run_pwrflw.py <generated_urbs>.h5 --storage db --n_cpu 1
```

Then validate:

- HDF5 time-series length is 168.
- DB powerflow rows exist for pre and post stages.
- Step 4 validation accepts 168-hour horizon.
- Metadata records selected week and method warning.

After that, test Munich AGS `09162000` with:

- candidate 0 only
- then a small subset, e.g. `--limit 5`
- only then consider a broader Munich timeslice batch
