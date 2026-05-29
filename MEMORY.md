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
