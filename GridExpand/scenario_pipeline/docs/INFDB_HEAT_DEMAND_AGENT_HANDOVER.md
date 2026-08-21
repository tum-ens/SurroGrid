# Agent Handover: INFDB `ro_heat` Space-Heat Pipeline

> Status: the implementation stages in this original plan have been completed
> or superseded. The approved 8,736-to-8,760 final-day duplication,
> progressively broadened similar-building fallback, and temporary OpenDHW
> nondeterminism replace the stricter failure/reproducibility gates below.
> Use `GridExpand/paired_validation/INFDB_FULL_RUN_READINESS_AGENT_HANDOVER.md`
> for the remaining pre-run work.

## Objective and priority

This is the prioritized heat-demand workflow. Make the general scenario pipeline
use building-level space-heat time series from the INFDB `ro_heat` schema while
retaining OpenDHW for domestic hot water (DHW). Rebuild the paired Forchheim
dataset against the fresh database before running real/synthetic comparisons.

Do not regenerate TEASER profiles in this workflow. The deferred TEASER work is
specified in `TEASER_HEAT_PROFILE_AGENT_HANDOVER.md`.

## Established facts and constraints

- PostgreSQL/PostGIS connection settings are in `GridExpand/.env`.
- The fresh database currently uses `PYLOVO_VERSION_ID=1`.
- The prepared directory
  `swf_2045_paired_v5_91301_station_hybrid_v2` belongs to a different pylovo
  topology. Its grid filenames and buses do not identify the same buildings in
  the fresh database and must not be reused.
- On 2026-08-20, `citydb.feature` and `citydb.property` existed but both had zero
  rows. The Forchheim LoD2 data must be imported before paired allocation or an
  ordinary LoD2-PV scenario can be prepared.
- Space heat may come from `ro_heat`; DHW remains OpenDHW.
- Heat assets remain one central system per physical building.
- Source-specific libraries must have explicit names. Use a name such as
  `forchheim_2045_infdb_ro_heat_v1.h5`, not a generic `physical_heat_v2.h5`.
- Preserve `forchheim_2045_physical_heat_v1.h5` until replacement artifacts
  have passed all audits.
- Do not use `--resume` with the failed status file in the obsolete V5 paired
  directory.
- Keep general source import and scenario logic outside `paired_validation`.
  Paired validation may consume prepared common profiles but must not implement
  the INFDB adapter.
- Do not add persistent unit-test modules. Use temporary smoke checks and store
  audit results with the existing run logs.

## Agent workflow and gates

Execute stages in order. Parallel work is allowed only where explicitly noted.
An agent must stop at a failed gate rather than introducing a fallback that
mixes heat methods or stale topologies.

### Stage 0 - Coordinator and repository state

Owner: coordinator agent.

1. Read `AGENTS.md` and inspect `git status` and the relevant diffs.
2. Preserve all existing user changes.
3. Confirm that these recent changes are present and compile:
   - pylovo footprint is multiplied by `floor_number` before TEASER;
   - `paired_heat_profile_regeneration` supports `--force-all`;
   - physical-library construction supports `--source-mode exact`;
   - Step 2 supports the dedicated `heat_library` profile mode.
4. Record the database host, database name, and selected pylovo version without
   printing credentials.
5. Reserve a new paired dataset ID, for example:

   ```text
   swf_2045_paired_pylovo_v1_91301_station_hybrid_v2
   ```

Gate 0:

- The working tree is understood and unrelated changes are untouched.
- No planned command references the obsolete V5 paired directory.
- `PYLOVO_VERSION_ID=1` is intentional.

### Stage 1 - Fresh-database data contract

Owner: database/data agent. This stage is read-only except for the externally
authorized LoD2 import.

1. Verify the selected pylovo version and enumerate the Forchheim grid results.
2. Verify non-empty `citydb.feature` and `citydb.property` tables after the LoD2
   import.
3. Measure joins between pylovo building object IDs and CityDB roof features.
4. Verify availability of `Dachneigung`, `Dachorientierung`, and `Flaeche`.
5. Inspect the recreated `ro_heat` schema and identify:
   - building key;
   - profile/run/version key;
   - timestamp and timezone;
   - interval length and calendar;
   - value unit and whether it is power or interval energy;
   - whether values represent space heat only;
   - duplicate and missing-value behavior.
6. Compare a sample of `ro_heat` building IDs with current pylovo building IDs.
7. Write a concise data-contract audit under the normal run-log/audit area.

Gate 1:

- CityDB is populated and the configured 0% LoD2 fallback requirement can be
  met for the retained paired buildings.
- Each required `ro_heat` building has one unambiguous complete time series.
- Units and timestamp semantics are proven from schema/data, not guessed.
- Stop and request direction if multiple `ro_heat` model runs exist without a
  defensible selector.

### Stage 2 - Rebuild the paired allocation

Owner: paired-allocation agent. Start only after Gate 1.

From `GridExpand/2.demand_allocation/gridalloc`, run:

```bash
uv run --project .. python \
  -m src.scenario_calibration.allocation.paired_allocation \
  --plz 91301 \
  --final-year 2045 \
  --min-buildings 5 \
  --pv-location-mode swf \
  --grid-data-path /home/breveron/data/swf_split_station_hybrid_v2 \
  --output-dir outputs/scenario_calibration/swf_2045_paired_pylovo_v1_91301_station_hybrid_v2
```

Then rebuild or validate the paired PV profile library from the new LoD2 roof
catalog and current weather input. Do not copy the old PV library solely to
satisfy a readiness check.

Gate 2:

- Metadata records pylovo version `1`.
- Real and synthetic scenario-unit populations and base-demand totals match.
- Every retained building maps to the current synthetic grid and bus.
- The LoD2 fallback share is exactly zero.
- The PV library covers every angle bin referenced by the new roof catalog.

### Stage 3 - Implement the general `ro_heat` adapter

Owner: general scenario-pipeline agent.

Before editing, inspect the Stage-1 contract and present any unavoidable design
choice to the user. Implement the smallest adapter that satisfies that contract.

1. Add the scientific source selection to the scenario YAML because two space-
   heat methods are supported:

   ```yaml
   heat:
     space_heat_source: infdb_ro_heat
   ```

2. Keep fixed behavior out of YAML:
   - DHW remains OpenDHW;
   - the building-central heat system remains fixed;
   - do not add storage-representation or heuristic-method selectors with only
     one implementation.
3. Add the adapter under the general demand/heat pipeline, not under paired
   validation.
4. Normalize database values to the existing hourly URBS demand convention.
5. Fail clearly on missing buildings, duplicate timestamps, incomplete years,
   negative demand, unsupported resolution, or ambiguous units.
6. Retain common downstream components for OpenDHW, heat-pump COP, heat-asset
   sizing, and URBS materialization.
7. Ensure selecting `infdb_ro_heat` does not import or execute TEASER.

Gate 3:

- An ordinary non-paired scenario can select `infdb_ro_heat`.
- Its space-heat values reproduce the source database annual energy within
  numerical tolerance.
- OpenDHW remains a separate hourly DHW series.
- TEASER is not invoked.

### Stage 4 - Build the source-specific physical library

Owner: profile-library agent. This may run in parallel with documentation after
Gate 3, but not with changes to the same profile modules.

Build `forchheim_2045_infdb_ro_heat_v1.h5`, keyed by stable
`building_objectid`, for the exact paired heat-building population. Store:

- `space_heat` from `ro_heat`;
- `water_heat` from deterministic building-level OpenDHW generation;
- `heatpump_air_cop` from the common weather/COP method.

Record metadata for source schema/run/version, units, resolution, weather,
OpenDHW seed convention, scenario year, hours, building count, creation time,
and relevant scenario/code hashes.

Gate 4:

- Exact coverage of the new paired heat-building set.
- Expected hourly length for every series.
- No NaN, infinite, or negative heat values.
- COP is finite and positive.
- Annual stored space heat equals the selected `ro_heat` source.
- Rebuilding with the same inputs reproduces DHW exactly.

### Stage 5 - Paired readiness and automatic resolution

Owner: integration agent.

Run paired readiness using the new paired directory and INFDB library. Confirm
that it rewrites the catalog with:

```text
profile_source_kind = physical_heat_library
profile_set_id = forchheim_2045_infdb_ro_heat_v1
```

Verify that `paired_validation.datasets.resolve_paired_dataset()` resolves the
library automatically from the dataset ID. Do not add a library path to the run
YAML if the catalog can provide it.

Gate 5:

- All heat profiles are publication-ready exact physical-building profiles.
- No diagnostic or area-scaled fallback is present.
- Dataset resolution returns the intended INFDB library.

### Stage 6 - Smoke runs and conservation audits

Owner: verification agent. Run sequentially so failures remain attributable.

1. One real-grid `post-inflex-heuristic` smoke run.
2. The matching synthetic `post-inflex-heuristic` smoke run.
3. One real-grid `post-hems-optimized` smoke run.
4. The matching synthetic `post-hems-optimized` smoke run.
5. If required by the comparison, exercise `post-hems-heuristic` as well.

Audit:

- identical scenario-unit thermal energy on real and synthetic sides;
- `space_heat + water_heat` equals total thermal demand;
- heat-pump plus auxiliary output satisfies demand in every timestep;
- no demand clipping or unexplained energy creation;
- identical heuristic-capacity contracts across paired networks;
- PV and battery results are not changed by profile-source plumbing alone;
- no TEASER execution appears in logs.

Gate 6:

- All smoke runs and capacity/energy-conservation audits pass.
- Only after this gate may the complete paired Forchheim run be launched.

### Stage 7 - Documentation and handoff

Owner: documentation agent.

Update `SCENARIO_METHOD.md` and the configuration reference with:

- the INFDB 1R1C space-heat source and exact run/version;
- identifier join, unit conversion, calendar, and failure policy;
- OpenDHW remaining the DHW source;
- common COP and heat-sizing behavior;
- scenario-YAML versus run-YAML responsibilities;
- source-specific library name and provenance;
- verification results and known limitations.

Do not claim that 1R1C is more or less accurate than TEASER until the deferred
comparison has been completed.

## Completion definition

This workflow is complete only when a new-user database can prepare the current
paired dataset, build the INFDB heat library, pass readiness, and complete the
real/synthetic smoke and conservation gates without relying on old SurroGrid
tables, old pylovo topology artifacts, or TEASER profiles.
