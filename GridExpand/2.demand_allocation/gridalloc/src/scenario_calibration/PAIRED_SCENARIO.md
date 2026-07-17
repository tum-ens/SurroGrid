# Paired SWF Scenario Contract

## Purpose

The paired scenario is the authoritative path for comparing real SWF and synthetic pylovo grids under post-electrification demand. It holds the physical demand and technology realization constant and changes only the electrical network and the mapping of scenario units to network buses.

## Contract

| Layer | Shared between real and synthetic | Network-specific |
|---|---|---|
| Scope | Physical buildings retained by both network models and the same minimum-building rule | Grid partition |
| Base demand | HH rows, measured annual HH energy, calibrated GHD energy, and sampled profile realization | Allocation bus |
| Sector assets | Deduplicated SWF 2045 PV, EV, and heat-pump inventory | Allocation bus |
| Time series | Profile seed, weather, mobility pool, heat demand, COP, and temporal horizon | None |
| Optimization | Identical scenario-unit inputs, model formulation, technology assumptions, solver settings, and TSAM mapping | Target-grid batch partition |
| Power flow | Active-power time series and metric definitions | Pandapower topology, impedance, equipment capacity, and bus mapping |

The comparison must fail before optimization when the paired plans contain different physical buildings or different HH/GHD totals.

## Scenario Unit

A physical building can be associated with several real SWF connection buses. Therefore, a physical building alone is not always a sufficiently precise demand-allocation unit. The stable scenario-unit identity is:

```text
(source_lv_id, source_allocation_bus, building_objectid)
```

For PLZ 91301 and pylovo version 3, the station-hybrid-v2 scope contains 7,493 scenario units covering 7,156 physical buildings. The same units are projected to 84 real grids and 81 synthetic grids.

Optimization remains at scenario-unit resolution. Aggregating units to real or synthetic buses before URBS can change flexibility and therefore violates the paired comparison contract. Bus aggregation belongs at the power-flow boundary. Each target-grid batch recreates the same deterministic scenario-unit inputs and uses the same optimization and TSAM settings; only the target-grid grouping and final bus projection differ.

## Current Regional Audit

The station-hybrid-v2 allocation generated:

| Quantity | Real SWF | Synthetic |
|---|---:|---:|
| Target grids | 84 | 81 |
| Physical buildings | 7,156 | 7,156 |
| Scenario units | 7,493 | 7,493 |
| HH rows | 13,730 | 13,730 |
| HH annual demand | 38.718 GWh | 38.718 GWh |
| Calibrated GHD annual demand | 8.931 GWh | 8.931 GWh |

One additional physical building from the calibrated source scope is absent from pylovo version 3. It represents one HH row and about 5.2 MWh/a and is reported separately rather than silently assigned.

## Heat-Profile Readiness

SWF contains 10,851 heat-pump rows but only 3,617 `(lv_id, bus, name, Baujahr)` installation identities. Every installation occurs as three component records: the heat-pump/COP reference (`load_type=hp`), space-heating demand (`load_type=heat`), and domestic-hot-water demand (`load_type=dhw`). These records share the same adoption year and must be combined as one physical heat-pump system before a complete heat profile is assigned; they are not three scenario-year installations.

The hybrid-v2 paired scope contains 3,359 physical heat-pump buildings. Before repair, 244 Public or Commercial buildings used area-scaled diagnostic substitutes because the residential-only Step 2 sources omitted their `space_heat`, `water_heat`, and heat-pump COP columns.

The targeted `paired_heat_profile_regeneration` command regenerated the 61 affected source HDFs with all building uses and validated all 244 missing building buses. The pilot also exposed a temporal-index bug: residential profiles used a `DatetimeIndex`, while GHD profiles used a `RangeIndex`; pandas consequently produced 17,520 rows when both sectors were combined. Step 2 now normalizes both sources to one positional 8,760-hour index before bus aggregation.

The final readiness catalog contains 3,359 `exact_physical_building` rows, all with `publication_ready=true`, and no diagnostic substitutions. Reproduce or resume the repair with:

```bash
uv run --project .. python -m src.scenario_calibration.profiles.paired_heat_profile_regeneration --paired-dir outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2 --pylovo-version-id 3 --workers 4 --n-cpu 1 --resume
```

## Diagnostic Pilot: LV113

The previous full-local run triplicated heat demand through duplicate SWF WP markers. Correcting the allocation changed LV113 as follows:

| Run | Transformer max | Minimum voltage | Cable max | Failed timesteps |
|---|---:|---:|---:|---:|
| Old post-flex | 274.1% | 0.455 p.u. | 292.1% | 136 |
| Corrected diagnostic post-flex | 153.6% | 0.658 p.u. | 187.4% | 0 |
| Old post-no-flex | 283.9% | 0.482 p.u. | 308.8% | 132 |
| Corrected diagnostic post-no-flex | 239.2% | 0.469 p.u. | 302.7% | 0 |

The remaining no-flex peak is not an initialization artifact. At the critical reduced timestep, LV113 carries approximately 319 kW base electricity, 378 kW electric heat, and 375 kW EV charging.

Topology also matters. Restoring all 15 lines removed by radialization improves the no-flex minimum voltage from 0.469 to 0.738 p.u. and cable loading from 303% to 189%, while transformer loading remains about 203%. The transformer is the original SWF 630 kVA unit; it was not lost during splitting. LV113 is therefore a mixed case:

- the old post-flex extreme was substantially inflated by a demand-allocation defect;
- radialization exaggerates cable and voltage stress;
- the original single-transformer supply area is still genuinely overloaded under the paired no-flex scenario.

## Commands

Build and audit the paired allocation:

```bash
cd GridExpand/2.demand_allocation/gridalloc
uv run --project .. python -m src.scenario_calibration.allocation.paired_allocation \
  --plz 91301 \
  --final-year 2045 \
  --pylovo-version-id 3 \
  --min-buildings 5 \
  --grid-data-path /home/breveron/data/swf_split_station_hybrid_v2 \
  --output-dir outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2

uv run --project .. python -m src.scenario_calibration.profiles.paired_profile_readiness \
  --paired-dir outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2
```

The paired runner produces pre electricity-only, post-flex, and post-no-flex power-flow summaries for every selected target. The publication run uses six representative weeks selected only from ambient temperature and irradiation. One canonical mapping is stored in the run directory and every real and synthetic optimization result must reproduce it before its power flows are accepted.

```bash
cd /home/breveron/git/github/SurroGrid
uv run --project GridExpand/2.demand_allocation \
  python GridExpand/runme/paired_swf_pipeline_runner.py \
  --repo-root /home/breveron/git/github/SurroGrid \
  --plz 91301 \
  --paired-dir GridExpand/2.demand_allocation/gridalloc/outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2 \
  --grid-data-path /home/breveron/data/swf_split_station_hybrid_v2 \
  --weather-source-hdf GridExpand/2.demand_allocation/gridalloc/results/9474126-00_91301_1_2.h5 \
  --target both \
  --tsam \
  --tsam-periods 6 \
  --tsam-hours-per-period 168 \
  --tsam-extreme-method replace_cluster_center \
  --workers 4 \
  --step3-cpus 4 \
  --step3-cluster-concurrency 1 \
  --step4-cpus 2 \
  --cleanup-intermediates \
  --scenario-label forchheim_paired_tsam \
  --run-name-prefix forchheim_paired_tsam \
  --run-dir GridExpand/run_logs/forchheim_paired_tsam_$(date -u +%Y%m%dT%H%M%SZ)
```

Omit `--allow-diagnostic-heat-fallback` for the publication run. The strict runner blocks before computation if any paired heat profile is not publication-ready. Use `--resume` only with the same run directory and unchanged TSAM settings; the runner validates those settings against the saved reference.

## Publication Gate

Do not use the existing `real_swf_2045_full_local_sector_flex` or `real_swf_2045_full_local_sector_no_flex` regional runs for the final comparison. They predate exact WP deduplication and contain 11 grids with failed power-flow timesteps.

A publication run requires:

1. equal paired-plan buildings and demand totals;
2. no diagnostic heat-profile fallbacks;
3. identical scenario-unit inputs and optimization settings for both target networks;
4. identical temporal horizon and representative periods for both targets;
5. zero silently skipped power-flow timesteps; non-convergence is reported as a result, not removed from the comparison;
6. a separate meshed-versus-radial sensitivity for critical real grids where radialization materially changes stress.
