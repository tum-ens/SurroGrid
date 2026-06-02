# Project Memory

## 2026-05-26 - Session start

- What was decided: Start by verifying the handoff in `plan.md` before changing pipeline code.
- Why: `plan.md` documents already implemented Step 1 and Step 2 work plus a known external-data blocker, and current workspace has user-owned/untracked changes.
- What was rejected and why: Skipping directly to code edits was rejected because `AGENTS.md` requires reading memory first and asking before non-trivial implementation when intent is unclear.


## 2026-05-26 - Step 2 blocker confirmed

- What was decided: Do not synthesize or silently replace `GridExpand/2.demand_allocation/gridalloc/data/statistics/inhabited_buildings/elec_lps.h5`.
- Why: Step 2 currently fails because this externally prepared residential load-profile HDF5 is missing, and the repo does not contain a clear generator or enough provenance to recreate the scientific artifact without assumptions.
- What was rejected and why: Creating a placeholder load-profile file was rejected because it would let the pipeline run with unvalidated demand data and could distort downstream URBS and power-flow results.


## 2026-05-27 - Step 2 mobility metadata fix

- What was decided: Extract the single region row inside `GridExpand/2.demand_allocation/gridalloc/src/functions/mobility.py::sample_statistics` before reading `regio7`, `plz`, `kcid`, and `bcid`.
- Why: Step 2 reached mobility after `elec_lps.h5` was added, but failed because `df_region["regio7"]` is a one-row Series and cannot be compared directly to the statistics table's `region` Series.
- What was rejected and why: Passing separate scalar metadata through the call stack was rejected for now because the one-row extraction is the smallest direct fix for the current bug.

## 2026-05-27 - Step 3 support timeframe mismatch found

- What was decided: Diagnose before editing Step 3 because the failure is a model/index consistency issue.
- Why: The HDF5 URBS input time series are indexed with support timeframe 2026, while `insert_scenario()` hardcodes global properties to support timeframe 2025. This makes model cost weighting look for `(2025, t)` keys that do not exist in the fallback `type_period` weights.
- What was rejected and why: Changing cost weighting directly was not chosen yet because the mismatch originates earlier in scenario support-timeframe construction.


## 2026-05-27 - Electricity-only status-quo powerflow path

- What was decided: Add `--profiles electricity` to Step 2 and `--pre-only` to Step 4.
- Why: Full Step 2 mobility allocation is too slow for interactive status-quo testing, while Step 4 can validate current grid behavior from `urbs_in/demand` without URBS post-expansion output.
- What was rejected and why: Running full mobility/URBS before testing powerflow was rejected because the user explicitly wanted electrical-profile status-quo testing first. A broader profile framework was deferred to keep this change minimal.

## 2026-05-27 - Step 4 legacy pandapower load schema

- What was decided: Normalize missing pandapower load columns in `GridExpand/4.powerflow/src/powerflow.py::prepare_grid`.
- Why: The saved grid lacks `const_z_percent` and related columns expected by the installed pandapower version, causing status-quo powerflow to fail before solving.
- What was rejected and why: Regenerating the topology solely to update pandapower schema was rejected because adding missing default columns is the smaller compatibility fix.


## 2026-05-29 - Step 3 Gurobi Python interface

- What was decided: Add `gurobipy` to the Step 3 uv environment after the `gurobi_cl`-only path did not return usable Pyomo solutions.
- Why: Pyomo 6.9 resolves `SolverFactory("gurobi")` to its supported Gurobi file/Python interface when `gurobipy` is installed, and license access succeeds with `/home/breveron/gurobi.lic`.
- What was rejected and why: A custom `gurobi_cl` adapter was rejected for now because it would require exporting LP/MPS files, invoking the CLI, parsing solution files, and integrating that back into Pyomo result handling.

## 2026-05-29 - Step 3 solve killed after model setup

- What was decided: Treat the current Step 3 blocker as runtime resource pressure, not missing Gurobi.
- Why: With `gurobipy` installed, model setup completes and the main-process diagnostic exits with code 137 during solve, which indicates the process was killed below Python, commonly by the OS due memory pressure.
- What was rejected and why: Continuing to debug as a Python exception was rejected because no Python traceback is produced when the worker is killed.


## 2026-05-29 - Step 4 plotting module approach

- What was decided: Add a minimal plotting CLI at `GridExpand/4.powerflow/plotting/powerflow_plotting.py` that reads saved Step-4 result tables and calls pandapower's built-in `pf_res_plotly` for heatmap visualization.
- Why: This directly satisfies the request to use prebuilt pandapower plotting on a map-like network view while keeping implementation simple and avoiding custom plotting logic.
- What was rejected and why: Building custom matplotlib/plotly traces from scratch was rejected because it increases maintenance and duplicates existing pandapower functionality.


## 2026-05-29 - Plotting package moved to repository root

- What was decided: Move the plotting package from `GridExpand/4.powerflow/plotting` to top-level `plotting/` and add `plotting/powerflow_plotting_test.ipynb` for direct interactive testing.
- Why: The user requested easier review and a single obvious location for plotting code plus a runnable notebook.
- What was rejected and why: Keeping plotting under step 4 was rejected because it made review/discovery less direct for this workflow.


## 2026-05-29 - Step 4 plotting location and display defaults

- What was decided: Move plotting back to `GridExpand/4.powerflow/plotting`, keep using the step-4 uv environment, force-display both voltage and line-loading colorbars, and hide household/load buses by default.
- Why: This keeps tooling aligned with the step where dependencies live, guarantees both requested legend bars are visible, and reduces visual clutter in dense LV household networks.
- What was rejected and why: Keeping plotting at repo root was rejected after workflow feedback; relying only on `create_line_trace(..., show_colorbar=True)` was rejected because the line-loading colorbar was not rendered in this pandapower version.


## 2026-05-29 - Step 4 plotting displays inline

- What was decided: Change `plot_powerflow_heatmap` to return a Plotly figure and call `fig.show()` by default instead of creating HTML files.
- Why: The plotting workflow is being used from Jupyter, where inline display is the intended review path.
- What was rejected and why: Keeping `output_html` plus an `IFrame` display in the notebook was rejected because it creates extra generated files and makes notebook viewing indirect.


## 2026-05-29 - Step 4 matching heatmap colorbars

- What was decided: Normalize the bus-voltage and line-loading Plotly colorbars to the same title side, vertical anchor, length, and thickness.
- Why: Jupyter display showed mismatched colorbar geometry, making the two heatmap legends look inconsistent.
- What was rejected and why: Leaving the fallback line-loading colorbar with Plotly defaults was rejected because its title rendered above the bar while the bus-voltage title rendered beside the bar.


## 2026-05-29 - Step 4 plot hover values restored

- What was decided: Use pandapower's `infofunc` hook to include voltage and line-loading values in Plotly hover labels, and shift the line-loading colorbar farther right.
- Why: The notebook plot should show both legend titles clearly and expose the plotted result values on hover.
- What was rejected and why: Manually editing generated Plotly trace text after creation was rejected because pandapower already provides `infofunc` for this purpose.


## 2026-05-29 - Step 4 timestep slider plotting

- What was decided: Add `ipywidgets` to the Step 4 uv environment and update the plotting notebook with an `IntSlider` that redraws `plot_powerflow_heatmap` for the selected timestep.
- Why: This is the simplest notebook-native way to inspect multiple power-flow timesteps without building a heavier Plotly animation frame system.
- What was rejected and why: Building Plotly animation frames first was rejected because the current network plot has many traces and may become heavy; a manual slider is easier to validate and debug.


## 2026-05-29 - Step 4 transformer import time series

- What was decided: Add a Plotly time-series plot in the Step 4 plotting notebook using `/pwrflw/output/{stage}/demand_import` as the transformer import proxy, with P import and apparent S import across all timesteps.
- Why: Step 4 removes the transformer during power-flow preparation and stores external-grid import instead; plotting this series helps identify critical timesteps before inspecting the network map.
- What was rejected and why: Plotting `res_trafo` directly was rejected because transformer result tables are not saved in the current HDF5 output and the transformer is replaced by a switch during simulation.

## 2026-06-01 - GridExpand DB-backed storage slice

- What was decided: Store AGS as an integer without the leading zero, keep `urbs_in/*` file-based for now, and add `--storage {h5,db}` flags so DB mode can bypass raw-grid HDF5 input and write Step 4 power-flow results to PostgreSQL instead of HDF5.
- Why: The first migration should remove duplicated grid-related artifacts while preserving the existing URBS HDF5 contract and keeping the original HDF5 workflow available.
- What was rejected and why: Migrating `urbs_in/*` in this slice was rejected because Step 3 still reads pandas HDFStore inputs and the task explicitly deferred that migration.



## 2026-06-01 - GridExpand scenario/run identity cleanup

- What was decided: Remove HDF5 filename identity from `surrogrid.grid_case`, keep a file name only as the temporary `urbs_input_file` bridge on `powerflow_run`, and add the static `baseline_static` scenario plus readable powerflow `run_name`.
- Why: Future DB-first workflows should identify grids by DB keys and scenario assumptions, not by synthetic HDF5 names. Step 4 still needs a bridge reference only because `urbs_in/*` remains file-based.
- What was rejected and why: Treating `powerflow_run_id` as a scenario-defining value was rejected because it is just a database surrogate key; reruns now overwrite rows for the same `(grid_case_id, scenario_id, run_name)`.


## 2026-06-01 - Scenario integer key and building-bus view

- What was decided: Make `surrogrid.scenario.scenario_id` an integer surrogate key, add unique `scenario_key` for readable labels such as `baseline_static`, add `timeframe_mode`, and create the `surrogrid.grid_building_bus` view.
- Why: Integer scenario IDs are better relational keys, while `scenario_key` and `run_name` keep queries interpretable. The view makes the implicit `powerflow_demand.bus` to pylovo building relation explicit.
- What was rejected and why: Encoding the scenario solely in `run_name` or `powerflow_run_id` was rejected because it makes joins and future scenario assumptions ambiguous. Reducing the simulated time horizon was not implemented yet because the selection method needs to be explicit before it changes results.


## 2026-06-02 - Operational one-week timeframe modes and DB/H5 smoke test

- What was decided: Keep only the requested timeframe modes and make them operational in Step 2 by slicing generated time-series outputs to either `full_year` or one 168-hour rolling week. Step 3 now derives timesteps from the HDF5 input horizon, and Step 4 records the selected timeframe in the scenario/run metadata.
- Why: The pipeline needs to run fast for targeted checks while preserving the DB-first scenario/run identity and keeping the HDF5 bridge contract for URBS.
- What was rejected and why: Treating timeframe modes as metadata only was rejected because it would not reduce runtime or exercise the changed pipeline. Migrating `urbs_in/*`/`urbs_out/*` to DB was still rejected for this slice because Step 3 remains HDF5-based.
- Verification: `max_electricity_demand_week` was smoke-tested for one DB-backed grid (`9278140-00_94342_1_-1`) and one H5 comparison grid (`9278192-00_94342_1_-1`) through Step 2, Step 3, and full Step 4 pre/post power flow.


## 2026-06-02 - Step 4 no-URBS flag naming

- What was decided: Rename the user-facing Step 4 status-quo flag from `--pre-only` to `--no-urbs`.
- Why: The flag is meant to say that Step 4 should skip URBS-derived post-expansion demand and only use `urbs_in/demand`, which is clearer than describing the saved stage as pre-only.
- What was rejected and why: Renaming the database `pre_only` column was rejected for this small cleanup because it already stores the pre-expansion stage flag and changing it would create unnecessary schema churn.


## 2026-06-02 - Reduced operational timeframe modes for mobility-aware slicing

- What was decided: Drop `max_mobility_demand_week` and `max_reverse_power_flow_week` from the supported timeframe modes for now.
- Why: `max_mobility_demand_week` is circular if the goal is to avoid generating full-year mobility first, and `max_reverse_power_flow_week` is not useful until PV/feed-in-heavy scenarios are implemented and tested. The remaining modes are better candidates for selecting a week before expensive mobility generation.
- What was rejected and why: Keeping every earlier mode as a CLI choice was rejected because it suggested runtime support for modes that cannot yet reduce mobility-generation time in a methodologically clear way.


## 2026-06-02 - Dropped max-net-load timeframe mode

- What was decided: Drop `max_net_load_week` from the supported timeframe modes for now.
- Why: If net load includes mobility, it is circular for the goal of selecting a week before expensive mobility generation. If it excludes mobility, it needs a clearer methodology before being exposed as a runnable scenario mode.
- What was rejected and why: Keeping `max_net_load_week` as a partial/proxy mode was rejected because it could be misinterpreted as full net load including mobility.


## 2026-06-02 - Timeslice modes reverted

- What was decided: Remove the operational timeslice concept from GridExpand after deciding that a methodology-consistent pregenerated emobpy mobility profile pool is the better scaling path.
- Why: Timeslice modes speed up targeted tests but change the research question from full-year mobility behavior to critical-week behavior. The thesis explicitly identifies pregenerated mobility profiles as the desirable future scaling approach when designed with the right metadata and assignment logic.
- What was rejected and why: Keeping timeslice CLI modes, Step 2 rolling-week slicing, Step 3 variable-horizon support, and Step 4 timeframe run metadata was rejected because these paths would distract from the pregenerated-profile implementation and leave misleading workflow options.


## 2026-06-02 - Electricity-only path simplified to no-URBS contract

- What was decided: Keep `--profiles electricity` as a slim Step 2 mode that writes only sampled building metadata and `urbs_in/demand`, and require Step 4 `--no-urbs` for status-quo power-flow checks.
- Why: Electricity-only runs are not meant to feed URBS optimization, so fabricating empty URBS input tables and adding empty HP/PV robustness in `demands.py` made the codebase harder to understand without supporting the intended workflow.
- What was rejected and why: Keeping `Grid.create_electricity_only_urbs()` and special post-URBS empty-component handling was rejected because full `--profiles all` should produce the normal URBS inputs, while electricity-only bypasses URBS entirely.

## 2026-06-02 - Step 1 shared single-grid export helper

- What was decided: Extract the duplicated pylovo grid export flow into `GridExpand/1.grid_sampling/gridreadout/src/export_grid.py` and have both `export_single_grid.py` and Notebook 2 call it.
- Why: The pilot CLI should not duplicate the original notebook pipeline logic for topology cleanup, building readout, weather enrichment, and HDF5 writing.
- What was rejected and why: Keeping the full export sequence inside `export_single_grid.py` was rejected because it creates two places to maintain the same Step 1 output contract. Replacing the notebook workflow with a larger CLI was rejected for now because the smallest cleanup is a shared helper.
