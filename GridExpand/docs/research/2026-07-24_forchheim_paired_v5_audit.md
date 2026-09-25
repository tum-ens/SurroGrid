> **Historical record** (not maintained). Run: paired SWF allocation `swf_2045_paired_v5_91301_station_hybrid_v2` and the diagnostic runs named below. pylovo version: 5. Written: 2026-07-17 to 2026-07-24 (later edits until September 2026). Split from `docs/PAIRED_SCENARIO.md` on 2026-09-25: the contract (scenario unit, shared and network-specific layers, evidence rules, heat-profile library, publication criteria) is maintained in [method.md](../method.md#paired-validation-contract); this note keeps the audit numbers, the LV113 diagnostic and the commands of that time. See [the documentation index](../README.md) for the current code.

# Paired SWF scenario: pylovo v5 audit (Forchheim, PLZ 91301)

## Scope

For PLZ 91301 and pylovo version 5, the revised station-hybrid-v2 scope contains 7,994 scenario units covering 7,643 physical buildings. The same units are projected to 88 real grids and 91 synthetic grids.

## Current Regional Audit

The station-hybrid-v2 allocation generated:

| Quantity | Real SWF | Synthetic |
|---|---:|---:|
| Target grids | 88 | 91 |
| Physical buildings | 7,643 | 7,643 |
| Scenario units | 7,994 | 7,994 |
| HH rows | 14,398 | 14,398 |
| HH annual demand | 40.847 GWh | 40.847 GWh |
| Calibrated GHD annual demand | 9.147 GWh | 9.147 GWh |

Pylovo version 5 covers every physical building retained by the paired scope; the unmapped-building audit is empty.

### Electrification Penetration and Capacity

For external communication, the 2045 electrification assumptions are summarized at physical-building level:

| Electrification assumption | 2045 scenario |
|---|---|
| Heat pump | 47% of buildings; residential locations follow the SWF inventory, while capacities follow the selected shared scenario sizing mode |
| PV and stationary battery | SWF rows retain predefined locations; PV potential comes from LoD2 roofs; battery capacity follows the shared HTW rule |
| EV charging | 76% of buildings; 11 kW per charging point |

The SWF inventory provides heat-pump locations but no positive heat-pump capacities. It is therefore location evidence only. Both heuristic cases use the shared building-level full-load-hour HP/auxiliary/buffer plan; `post-hems-optimized` instead uses its calculated monovalent values as finite optimization bounds. SWF PV capacity is not used as an installation limit. The SWF pandapower file is cumulative: existing and future PV rows with different `Baujahr` values coexist in the final network and are `in_service`. In `swf` location mode, any PV row with `Baujahr <= 2045` establishes building/location eligibility; legacy rows without a usable year are retained as existing assets. Repeated rows at the same connection collapse to one eligibility record. In `all_buildings` mode, every retained physical building is eligible. Charging-station capacities are fixed; a building can carry more than one 11 kW charging point.

## GHD and Mixed-Use Calibration

Pylovo's open building layer contains many more Commercial/Public polygons than SWF contains GHD customer rows. These quantities are not directly comparable: an ALKIS building polygon is not necessarily an active independent electricity customer, while one electrical connection can represent a mixed-use building. The paired scenario therefore applies the following evidence rules:

1. SWF GHD demand is retained only where the GHD row can be matched to a physical building.
2. SWF HH rows that match a Commercial/Public polygon are retained as mixed-use household proxies.
3. Unsupported pylovo per-square-metre GHD defaults are not added to either target network.
4. An unmatched SWF GHD row is excluded and audited at row level; it does not reject the otherwise valid LV grid.

The full audit contains 2,991 generic Commercial structures without direct SWF load evidence. Most are unaddressed and small, and all use the broad ALKIS `31001_2000` category. Conversely, 719 Commercial/Public buildings already receive 1,647 SWF HH rows and are represented as mixed-use proxies. A blanket conversion of the remaining generic structures to households would duplicate demand or displace demand from explicitly residential buildings. See [2026-07-21_forchheim_ghd_calibration_v5.md](2026-07-21_forchheim_ghd_calibration_v5.md) for the complete evidence table and interpretation.

## Heat-Profile Readiness

SWF contains 10,851 heat-pump rows but only 3,617 `(lv_id, bus, name, Baujahr)` installation identities. Every installation occurs as three component records: the heat-pump/COP reference (`load_type=hp`), space-heating demand (`load_type=heat`), and domestic-hot-water demand (`load_type=dhw`). These records share the same adoption year and must be combined as one physical heat-pump system before a complete heat profile is assigned; they are not three scenario-year installations.

The revised pylovo-version-5 paired scope contains 3,580 physical heat-profile buildings. The additional scope initially exposed one Public building without an exact source profile (`DEBY_LOD2_3229113`, source `9474126-45_91301_8_1.h5`, bus 248).

The validated physical heat and COP time series are generated once and stored in a network-independent regional HDF5 library keyed by `building_objectid`. A library represents one explicit physical-profile assumption set, so topology changes can reuse it while refurbishment or weather changes require a new profile-set identifier. The current `forchheim_2045_physical_heat_v1` library contains 3,582 buildings and 8,760 hours in approximately 196 MB. It is a slight superset of the 3,580 profiles required by the version-5 paired scope.

Paired readiness checks building coverage against this library, and URBS input generation reads the same building profile before projecting it to the current real or synthetic target bus. Changing the pylovo grid version therefore requires a new paired allocation but no repeated heat-profile generation when the physical profile assumptions are unchanged. The legacy per-grid HDF workflow remains available only for constructing a new library or explicitly diagnostic fallbacks.

The paired materializer then applies the same residential heat-asset method documented in [method.md](../method.md#residential-heat-assets): the original hourly OpenDHW demand and its matching COP are retained, one central system is assigned per physical building, and the resulting fixed capacities or optimization bounds are projected through scenario-unit sites. The real and synthetic targets consume the same physical profiles and sizing assumptions.

## Diagnostic Pilot: LV113

The previous full-local run triplicated heat demand through duplicate SWF WP markers. Correcting the allocation changed LV113 as follows:

| Run | Transformer max | Minimum voltage | Cable max | Failed timesteps |
|---|---:|---:|---:|---:|
| Old post-flex | 274.1% | 0.455 p.u. | 292.1% | 136 |
| Corrected diagnostic post-flex | 153.6% | 0.658 p.u. | 187.4% | 0 |
| Old post-inflex | 283.9% | 0.482 p.u. | 308.8% | 132 |
| Corrected diagnostic post-inflex | 239.2% | 0.469 p.u. | 302.7% | 0 |

The remaining inflex peak is not an initialization artifact. At the critical reduced timestep, LV113 carries approximately 319 kW base electricity, 378 kW electric heat, and 375 kW EV charging.

Topology also matters. Restoring all 15 lines removed by radialization improves the inflex minimum voltage from 0.469 to 0.738 p.u. and cable loading from 303% to 189%, while transformer loading remains about 203%. The transformer is the original SWF 630 kVA unit; it was not lost during splitting. LV113 is therefore a mixed case:

- the old post-flex extreme was substantially inflated by a demand-allocation defect;
- radialization exaggerates cable and voltage stress;
- the original single-transformer supply area is still genuinely overloaded under the paired inflex scenario.

## Commands

Build and audit the paired allocation. Creating the physical heat-profile library is a one-time regional preparation step; skip that command when the named profile set already exists:

The allocation command requires the AGS and PyLoVo version explicitly, registers every eligible regional grid, and records both in paired-scenario metadata. Prefer the staged `gridexpand run --prepare-only` command for normal operation; the commands below remain useful for component diagnostics. The shown V4 catalog is needed only for the one-time migration of already validated physical profiles into the reusable library; later topology versions use the library directly.

```bash
cd GridExpand
uv run python -m gridexpand.allocation.scenario_calibration.allocation.paired_allocation \
  --ags 9474126 \
  --plz 91301 \
  --pylovo-version-id 1 \
  --final-year 2045 \
  --min-buildings 5 \
  --grid-data-path /home/breveron/data/swf_split_station_hybrid_v2 \
  --output-dir work/allocation/outputs/scenario_calibration/swf_2045_paired_v5_91301_station_hybrid_v2

uv run python -m gridexpand.allocation.scenario_calibration.profiles.physical_heat_profile_library \
  --source-catalog work/allocation/outputs/scenario_calibration/swf_2045_paired_v4_91301_station_hybrid_v2/paired_heat_profile_catalog.csv \
  --source-hdf-dir work/optimization/input \
  --output work/allocation/outputs/scenario_calibration/profile_libraries/forchheim_2045_physical_heat_v1.h5 \
  --profile-set-id forchheim_2045_physical_heat_v1

uv run python -m gridexpand.allocation.scenario_calibration.profiles.paired_profile_readiness \
  --paired-dir work/allocation/outputs/scenario_calibration/swf_2045_paired_v5_91301_station_hybrid_v2 \
  --heat-profile-library work/allocation/outputs/scenario_calibration/profile_libraries/forchheim_2045_physical_heat_v1.h5
```


### Rebuild physical heat profiles after a method change

A change to TEASER inputs or another physical heat-profile assumption requires
regenerating every exact source grid, even when the existing catalog marks its
profiles ready. Keep the previous library until the replacement passes the
readiness audit, and use a new profile-set version so results remain traceable.
From `GridExpand` run:

```bash
uv run python -m gridexpand.allocation.scenario_calibration.profiles.paired_heat_profile_regeneration \
  --paired-dir work/allocation/outputs/scenario_calibration/swf_2045_paired_v5_91301_station_hybrid_v2 \
  --force-all \
  --workers 4 \
  --n-cpu 1

uv run python -m gridexpand.allocation.scenario_calibration.profiles.physical_heat_profile_library \
  --source-catalog work/allocation/outputs/scenario_calibration/swf_2045_paired_v5_91301_station_hybrid_v2/paired_heat_profile_catalog.csv \
  --source-hdf-dir work/optimization/input \
  --source-mode exact \
  --output work/allocation/outputs/scenario_calibration/profile_libraries/forchheim_2045_physical_heat_v2.h5 \
  --profile-set-id forchheim_2045_physical_heat_v2

uv run python -m gridexpand.allocation.scenario_calibration.profiles.paired_profile_readiness \
  --paired-dir work/allocation/outputs/scenario_calibration/swf_2045_paired_v5_91301_station_hybrid_v2 \
  --heat-profile-library work/allocation/outputs/scenario_calibration/profile_libraries/forchheim_2045_physical_heat_v2.h5
```

The regeneration helper uses Step 2's dedicated `heat_library` profile mode:
weather and base electricity are retained as heat-model inputs, while unrelated
PV and battery generation is skipped. `--workers` controls concurrently
regenerated source grids. `--n-cpu` controls heat-generation processes inside
each grid job; avoid multiplying both values without checking available RAM.
If a regional batch is interrupted,
repeat the first command with `--resume` in addition to `--force-all`.

The paired runner produces pre electricity-only, post-flex, and post-inflex power-flow summaries for every selected target. The publication run uses six representative weeks selected only from ambient temperature and irradiation. One canonical mapping is stored in the run directory and every real and synthetic optimization result must reproduce it before its power flows are accepted.

```bash
cd GridExpand
uv run gridexpand run \
  --run-config config/runs/forchheim_2045_paired_full_year.yaml
```

The paired run YAML is the authoritative dataset, model-case, and
concurrency definition. Its model-case list materializes the shared heuristic
asset plan once for post-hems-heuristic and post-inflex-heuristic, then runs the
separate endogenous post-hems-optimized solve. Output directories are derived
from the run ID. A single-case CLI override remains available for diagnostics,
but it is not required for the configured complete comparison.

Omit `--allow-diagnostic-heat-fallback` for the publication run. The strict runner blocks before computation if any paired heat profile is not publication-ready. Use `--resume` only with the same run directory and unchanged TSAM settings; the runner validates those settings against the saved reference.

## Publication Gate

Do not use the existing `real_swf_2045_full_local_sector_flex` or `real_swf_2045_full_local_sector_inflex` regional runs for the final comparison. They predate exact WP deduplication and contain 11 grids with failed power-flow timesteps.

A publication run requires:

1. equal paired-plan buildings and demand totals;
2. no diagnostic heat-profile fallbacks;
3. identical scenario-unit inputs and optimization settings for both target networks;
4. identical temporal horizon and representative periods for both targets;
5. zero silently skipped power-flow timesteps; non-convergence is reported as a result, not removed from the comparison;
6. a separate meshed-versus-radial sensitivity for critical real grids where radialization materially changes stress.
