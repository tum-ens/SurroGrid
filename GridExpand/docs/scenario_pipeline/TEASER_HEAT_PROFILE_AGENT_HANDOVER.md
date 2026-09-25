# Agent Handover: Deferred TEASER Heat-Profile Regeneration and Comparison

## Objective and priority

This workflow is intentionally deferred. Execute it only after the general
scenario pipeline using `infdb_ro_heat` passes its complete smoke and
conservation gates. Regenerate corrected TEASER/DistrictGenerator profiles for
the same current paired building population and then compare space-heat methods.

The correction under evaluation is that pylovo `floor_area` is the LoD2
footprint and must be multiplied by `floor_number` before it is passed to
TEASER's total `net_leased_area` input.

## Established facts and constraints

- The fresh database currently uses `PYLOVO_VERSION_ID=1`.
- The old `swf_2045_paired_v5_91301_station_hybrid_v2` heat catalog is invalid
  for this database. For example, it expected heat buildings on buses 29, 34,
  37, and 39 where the fresh database resolved one commercial building on bus
  3.
- Never run regeneration from the old V5 catalog and never resume its failed
  status file.
- Reuse the new paired dataset prepared by the INFDB workflow, expected to have
  a name such as
  `swf_2045_paired_pylovo_v1_91301_station_hybrid_v2`.
- Step 2 now has a dedicated `heat_library` profile mode. It retains weather and
  base electricity needed by heat generation but skips PV, battery, and
  mobility generation.
- The regeneration helper supports `--force-all`; the library builder supports
  `--source-mode exact`.
- OpenDHW remains the DHW source in both INFDB and TEASER libraries.
- Keep both source-specific libraries; do not overwrite the INFDB library.
- Do not add persistent unit-test modules. Use temporary checks and existing
  run-log/audit locations.

## Preconditions inherited from the prioritized workflow

Before assigning implementation agents, the coordinator must prove:

1. CityDB LoD2 data has been imported and audited.
2. The paired allocation has been rebuilt for pylovo version `1`.
3. The new paired PV library and allocation gates pass.
4. `forchheim_2045_infdb_ro_heat_v1.h5` is complete and publication-ready.
5. The INFDB real/synthetic heuristic and optimized smoke runs pass.
6. Capacity and energy-conservation audits pass.

Stop if any precondition is false. TEASER work must not delay stabilization of
the prioritized INFDB pipeline.

## Agent workflow and gates

### Stage T0 - Coordinator and isolated artifacts

Owner: coordinator agent.

1. Read `AGENTS.md`, inspect current diffs, and preserve user changes.
2. Confirm the new paired dataset ID and current pylovo version.
3. Reserve the explicit output name:

   ```text
   forchheim_2045_teaser_5r1c_v2.h5
   ```

4. Confirm that the existing V1 library remains untouched.
5. Archive evidence from the successful INFDB workflow for later comparison.

Gate T0:

- Inputs are current and source-specific output names cannot collide.
- No command references the stale V5 paired directory.

### Stage T1 - Fix the fresh-database regeneration bootstrap

Owner: profile-regeneration agent.

The current regeneration helper assumes an existing heat catalog. That is not a
safe fresh-database bootstrap because readiness may have no exact profiles from
which to construct one. Implement the simplest explicit refresh path:

1. Derive required heat buildings from the new paired real allocation using the
   combined residential and GHD heat-pump row evidence already used by
   readiness.
2. Read each building's current `synthetic_bridge_filename` and `synthetic_bus`.
3. Produce regeneration requirements containing at least:
   - `building_objectid`;
   - `exact_source_hdf`;
   - `exact_source_bus`;
   - readiness state.
4. Do not derive mappings from an old HDF library or old topology.
5. Ensure successful exact-source files can be used to build a library before
   final readiness rewrites the catalog against that library.
6. Keep this preparation in the common profile tooling. Do not put it in the
   paired runner.

Before editing, inspect the new paired CSV columns and show any unavoidable
choice to the user. Do not invent column names or fallback rules.

Gate T1:

- Every required building has exactly one current source HDF and bus.
- Source filenames resolve to the same current buildings in pylovo version `1`.
- No diagnostic area-based substitution is present.
- The bootstrap works when no old physical heat library exists.

### Stage T2 - One-grid TEASER validation

Owner: heat-profile agent.

Select a current grid containing both single-storey and multi-storey residential
buildings. Run only the dedicated heat-library path, initially with one worker
and one CPU.

Verify from inputs and output:

- pylovo footprint and `floor_number` are positive;
- TEASER receives `floor_area * floor_number`;
- a 100 m2 footprint with three floors would be passed as 300 m2;
- DistrictGenerator/TEASER produces `space_heat`;
- OpenDHW produces hourly `water_heat`;
- COP output contains `heatpump_air` for every heat building;
- output building IDs and buses match the refreshed regeneration catalog;
- no PV, battery, mobility, or CityDB query occurs in `heat_library` mode.

Perform preliminary plausibility checks by building type, construction year,
floor count, annual kWh, kWh/m2 of total assumed heated area, and peak kW.

Gate T2:

- All required demand/COP columns exist for the one-grid sample.
- The floor-area correction is demonstrated from actual inputs.
- No unrelated profile stage executes.
- Stop if annual or peak results show an unexplained order-of-magnitude error.

### Stage T3 - Full regional TEASER regeneration

Owner: batch-execution agent. Start only after Gate T2.

From `GridExpand/2.demand_allocation/gridalloc`, run against the new paired
directory:

```bash
uv run --project .. python \
  -m src.scenario_calibration.profiles.paired_heat_profile_regeneration \
  --paired-dir outputs/scenario_calibration/swf_2045_paired_pylovo_v1_91301_station_hybrid_v2 \
  --force-all \
  --workers 4 \
  --n-cpu 1
```

Monitor the status CSV and per-source logs. If interrupted, repeat the same
command with `--resume` as well as `--force-all`. Do not use a status file from
another paired dataset.

Gate T3:

- Every current exact source reports `done`.
- Every required bus has `space_heat`, `water_heat`, and `heatpump_air` COP.
- Failures are explained and corrected; none are hidden by reduced coverage.
- Record total runtime, worker settings, and resource observations.

### Stage T4 - Build and register the TEASER library

Owner: library agent.

Build the source-specific exact library:

```bash
uv run --project .. python \
  -m src.scenario_calibration.profiles.physical_heat_profile_library \
  --source-catalog outputs/scenario_calibration/swf_2045_paired_pylovo_v1_91301_station_hybrid_v2/paired_heat_profile_catalog.csv \
  --source-hdf-dir ../../3.urbs/Input \
  --source-mode exact \
  --output outputs/scenario_calibration/profile_libraries/forchheim_2045_teaser_5r1c_v2.h5 \
  --profile-set-id forchheim_2045_teaser_5r1c_v2
```

Then run paired readiness using this library. Confirm that the catalog records
the TEASER profile-set ID and that paired dataset resolution selects it when the
TEASER dataset variant is requested. Do not silently replace the INFDB catalog;
if both sources must remain runnable, use separate paired dataset manifests or
another explicit existing selection mechanism approved by the user.

Gate T4:

- Exact building coverage and expected hourly length.
- No NaN, infinite, negative heat values, or non-positive COP.
- Metadata identifies TEASER/DistrictGenerator, 5R1C, corrected total floor
  area, weather, scenario year, and OpenDHW seed convention.
- INFDB and old V1 libraries remain intact.

### Stage T5 - Controlled INFDB versus TEASER comparison

Owner: comparison/audit agent.

Use the intersection of identical building IDs and identical timestamps. Match
weather and deterministic OpenDHW seeds. Compare space heat first, separately
from DHW, because DHW is intended to be identical.

Per building and grouped by building type, construction year, and floor count,
calculate:

- annual space-heat demand;
- kWh/m2 using total assumed heated area;
- peak thermal power;
- full-load hours;
- hourly and daily correlation;
- duration-curve differences;
- peak timing and peak coincidence;
- regional aggregate annual energy and peak.

Explicitly test whether deviations correlate with `floor_number`. Separate the
effects of:

- the former footprint-only TEASER regression;
- 1R1C versus TEASER/DistrictGenerator 5R1C behavior;
- envelope and refurbishment assumptions;
- occupancy and internal gains;
- weather and temporal processing.

Do not attribute a difference to model order without evidence.

Gate T5:

- DHW equality is confirmed or every difference is explained.
- Space-heat comparison tables contain identical populations and units.
- Multi-storey deviations are quantified rather than inferred from examples.

### Stage T6 - Downstream scenario-impact comparison

Owner: scenario verification agent.

Using otherwise identical scenario settings, compare the effects of the two
space-heat sources on:

- heuristic heat-pump thermal and electrical capacity;
- auxiliary-heater capacity and annual output;
- buffer-storage capacity;
- heat-pump electricity consumption;
- building and regional simultaneous peaks;
- HEMS dispatch and available flexibility.

Start with matched real/synthetic smoke targets. A full paired rerun is justified
only if the source comparison and conservation checks pass.

Gate T6:

- Capacity changes are traceable to heat-demand differences.
- Heat balance holds in every modeled timestep.
- Network comparisons do not mix INFDB capacities with TEASER profiles or vice
  versa.

### Stage T7 - Decision record and documentation

Owner: documentation agent.

Update `SCENARIO_METHOD.md` with:

- corrected TEASER total-floor-area interpretation;
- source-specific profile-set names and provenance;
- comparison population, metrics, and results;
- limitations of both heat models;
- the selected default source and rationale;
- whether the alternative remains supported.

Do not remove either reproducible library until the scientific choice has been
reviewed. If one method is retired later, remove its code and artifacts only in
a separately authorized cleanup.

## Completion definition

This deferred workflow is complete only when corrected TEASER profiles are
regenerated from the current paired allocation, pass exact-coverage and energy
audits, and are compared to the validated INFDB profiles on identical buildings
and timestamps. The final result must distinguish data/model effects from the
fixed multi-storey-area regression and document the selected default method.
