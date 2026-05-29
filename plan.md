# SurroGrid Pipeline Handoff Plan (Agent Notes)

## 1. Scope and Intent
This document summarizes the current technical state of the GridExpand pipeline work, with emphasis on:
- Step 1 (grid sampling/readout) DB/schema migration and pilot export flow
- Step 2/3/4 runtime checks performed after Step 1 updates
- Verified blockers and concrete continuation steps for the next agent

Date of handoff: 2026-05-21

---

## 2. User Constraints and Final Direction
The user explicitly requested the following direction and simplifications:
- Use uv-based environments and commands.
- Remove legacy fallback logic (no old public-schema fallback paths).
- Use pylovo schema only.
- Replace old filler-cluster filtering logic with a minimum building threshold.
- Candidate selection should be based on available transformer positions and practical filtering.
- Building classification should not depend on `res`/`oth` tables anymore.
- Building use/type should come from `buildings_result.type`:
  - Residential: `AB`, `MFH`, `TH`, `SFH`
  - Other/non-res: public and commercial

---

## 3. What Was Implemented

### 3.1 Step 1 DB readout and candidate logic
Main file: `GridExpand/1.grid_sampling/gridreadout/src/db_read.py`

Implemented behavior:
- Pylovo-only queries.
- Candidate retrieval via `pylovo.grid_result` + `pylovo.transformer_positions` + `pylovo.buildings_result`.
- Candidate filter by `min_buildings` (default 5).
- Query binding fix for `:min_buildings` via SQLAlchemy text + params.
- `read_buildings` migrated to `buildings_result`-only (no `res`/`oth` dependency).
- Explicit type/use mapping from `buildings_result.type`:
  - `AB|MFH|TH|SFH` -> `type` normalized uppercase + `use=Residential`
  - strings containing `public` -> `type=public`, `use=Public`
  - strings containing `commercial` -> `type=commercial`, `use=Commercial`
  - fallback -> `type=commercial`, `use=Commercial`

### 3.2 Step 1 single-grid export CLI
Main file: `GridExpand/1.grid_sampling/gridreadout/export_single_grid.py` (currently untracked)

Implemented behavior:
- Pylovo-only selection path.
- `--min-buildings` argument (default 5).
- Clear output separating global pool count vs PLZ-scoped candidates.
- `--skip-weather` option for cheap tests.

### 3.3 Step 1 notebook and docs alignment
- Notebook updated at one point to call threshold method directly (current notebook state may have since changed by user).
- `GridExpand/1.grid_sampling/README.md` updated to reflect:
  - pylovo-only tables
  - no `res`/`oth` requirement
  - candidate threshold behavior
  - uv command examples

### 3.4 Step 2 fix discovered during integration test
Main file: `GridExpand/2.demand_allocation/gridalloc/src/classes/grid.py` (currently modified)

Bug fixed:
- `Grid.__init__` was storing `region/lat/lon/altitude/plz` as pandas Series (from one-row DataFrame), which broke downstream pvlib operations.
- Fixed by selecting row 0 scalar values (`region_row = self.df_region.iloc[0]`) and converting to proper scalar types.

This fix is required for Step 2 to proceed past solar generation with current inputs.

---

## 4. Runtime Validation Results

### 4.1 Step 1 validations
Validated commands (uv):
- Candidate listing worked and showed PLZ-local counts correctly.
- Export worked for PLZ 80803.

Important verified artifact:
- `GridExpand/1.grid_sampling/gridreadout/results/900_80803_2_-1.h5`
- Verified HDF keys in this file:
  - `/raw_data/buildings`
  - `/raw_data/consumers`
  - `/raw_data/region`
  - `/raw_data/weather`

### 4.2 Step 2 integration status
Input prepared:
- Copied Step 1 output into `GridExpand/2.demand_allocation/gridalloc/data/grids/`

Result:
- After scalar fix in `grid.py`, Step 2 progressed further.
- Current blocker is **missing external data artifact**:
  - `gridalloc/data/statistics/inhabited_buildings/elec_lps.h5`
- Error thrown in:
  - `GridExpand/2.demand_allocation/gridalloc/src/functions/electricity.py`
  - while loading keys `df_normalized_scaled` and `df_sums` from `ELEC_LPS_PATH`.

### 4.3 Step 3 smoke run status
Run attempted on sample input (`id=0`).

Result:
- Fails during model construction with `KeyError: (2025, 1)` in cost constraint path.
- Trace points to:
  - `GridExpand/3.urbs/urbs/model.py` around `def_costs_rule`
  - missing key in `m.typeperiod['weight_typeperiod'][(m.stf_list[0], tm)]`

### 4.4 Step 4 smoke run status
Initial run failed due no matching input file in `4.powerflow/Input`.
After copying sample input into `4.powerflow/Input`, run progressed and then failed with:
- `KeyError: 'No object named urbs_out/MILP/tau_pro in the file'`

Interpretation:
- Expected until Step 3 successfully writes URBS outputs.

---

## 5. Provenance Note: `elec_lps.h5`
Evidence in repo indicates `elec_lps.h5` is an externally generated/prepared artifact:
- Referenced in config:
  - `GridExpand/2.demand_allocation/gridalloc/config.py` (`ELEC_LPS_PATH`)
- Explicitly ignored in git:
  - `GridExpand/2.demand_allocation/.gitignore`
- Required by Step 2 electricity generator:
  - `GridExpand/2.demand_allocation/gridalloc/src/functions/electricity.py`

Given prior project discussions, this likely comes from the previously analyzed GridStatistics/GridStatisticsPrep workflow, but this repo itself does not provide a generator script for it.

---

## 6. Current Git Workspace State (at handoff)
`git status --short` showed:
- `M GridExpand/1.grid_sampling/gridreadout/1_filter_valid_grids.ipynb`
- `M GridExpand/1.grid_sampling/gridreadout/input_data/valid_grids`
- `M GridExpand/2.demand_allocation/gridalloc/src/classes/grid.py`
- `?? GridExpand/1.grid_sampling/gridreadout/export_single_grid.py`
- `?? thesis.pdf`

Important: do not revert user-owned changes unless asked.

---

## 7. Immediate Next Actions for Future Agent

### Priority A: unblock Step 2
1. Obtain/provide `GridExpand/2.demand_allocation/gridalloc/data/statistics/inhabited_buildings/elec_lps.h5`.
2. Confirm HDF keys exist:
   - `df_normalized_scaled`
   - `df_sums`
3. Re-run Step 2 for test prefix `900`:
   - `cd GridExpand/2.demand_allocation/gridalloc`
   - `uv run --project .. python main.py 900 --n_cpu 1`

### Priority B: continue full pipeline check
4. Copy Step 2 result file to Step 3 input and run:
   - `cd GridExpand/3.urbs`
   - `uv run --project . python run_urbs_cluster.py 900 --n_cpu 1`
5. If Step 3 still fails with typeperiod key error, debug `typeperiod` index consistency vs time indices in `urbs_in` sheets.
6. After Step 3 success, copy result file to Step 4 input and run:
   - `cd GridExpand/4.powerflow`
   - `uv run --project . python run_pwrflw.py 900 --n_cpu 1`

---

## 8. Known Non-Blocking Warnings
- Pandas FutureWarning in Step 1 `db_read.py` around `fillna` downcasting.
- Step 2 warning about `int(Series)` no longer appears after scalar fix.

---

## 9. Kernel/Notebook Environment Note
A prior notebook import error (`ModuleNotFoundError: dotenv`) was traced to selecting the wrong `.venv` kernel, not a missing dependency in Step 1 `pyproject.toml`.
User reported notebook is fixed and runs now.

---

## 10. Quick Command Snippets (Reference)

Step 1 export with weather:
```bash
cd GridExpand/1.grid_sampling/gridreadout
uv run --project .. python export_single_grid.py --plz 80803 --candidate-index 0 --cell-id 900
```

Step 2 run:
```bash
cd GridExpand/2.demand_allocation/gridalloc
uv run --project .. python main.py 900 --n_cpu 1
```

Step 3 run:
```bash
cd GridExpand/3.urbs
uv run --project . python run_urbs_cluster.py 900 --n_cpu 1
```

Step 4 run:
```bash
cd GridExpand/4.powerflow
uv run --project . python run_pwrflw.py 900 --n_cpu 1
```

---

## 11. Bottom Line
- Step 1 migration goals are implemented and validated.
- Step 2 handoff logic is fixed at code level (`grid.py` scalar extraction).
- End-to-end progression is currently blocked by missing external statistics artifact `elec_lps.h5`.
- After that artifact is restored, continue with Step 3 typeperiod-key debugging, then Step 4.
