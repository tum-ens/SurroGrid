# Scenario Run Summary

This document is the concise source of truth for GridExpand scenario runs. It records which runs are authoritative, the assumptions shared between scenarios, database names used by postprocessing, known limitations, and the commands needed to reproduce or resume the analysis.

Detailed methodology remains in:

- [Paired SWF scenario contract](../../2.demand_allocation/gridalloc/src/scenario_calibration/PAIRED_SCENARIO.md)
- [Pipeline runner documentation](../../runme/README.md)
- [Power-flow methodology](../../4.powerflow/README.md)
- [Postprocessing and expansion analysis](../README.md)
- [Expansion cost assumptions](assumptions_costs.md)

## Current Source of Truth

The current comparison run is the paired Forchheim battery-aware TSAM run completed on 20 July 2026.

| Item | Value |
|---|---|
| Region | Forchheim, PLZ 91301 |
| Scenario year | 2045 |
| Synthetic grid version | pylovo version 3 |
| Real-grid preparation | `swf_split_station_hybrid_v2` |
| Scenario scope | Paired full local demand: residential plus calibrated GHD |
| Physical buildings | 7,647 on each network side |
| Scenario units | 8,001 on each network side |
| Household rows | 14,408 on each network side |
| Household annual demand | 40.888 GWh on each network side |
| Calibrated GHD annual demand | 9.275 GWh on each network side |
| Target grids | 88 real SWF and 83 synthetic |
| Temporal reduction | Six representative weeks, 168 hours each |
| Power-flow horizon | 1,008 representative hours |
| TSAM variables | Ambient temperature and irradiation only |
| Power-flow storage | Compact database summaries |
| Final runner state | All 171 target jobs complete with `status=done` and `message=ok` |

Two additional calibrated buildings are absent from pylovo version 3. Together they contain two household rows, 8.610 MWh/a household demand, and 23.755 MWh/a GHD demand. They are reported separately and are not silently assigned to either paired network.

### Run Artifacts

| Artifact | Location or identifier |
|---|---|
| Paired allocation | `2.demand_allocation/gridalloc/outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2` |
| Real grid files | `/home/breveron/data/swf_split_station_hybrid_v2/station_radialized` |
| Runner directory | `run_logs/forchheim_paired_v2_tsam_20260720T105015Z` |
| Shared TSAM mapping | `run_logs/forchheim_paired_v2_tsam_20260720T105015Z/shared_tsam_reference.json` |
| Analysis prefix | `forchheim_paired_battery_tsam` |

The first execution attempts contained materialization and power-flow failures. The run was resumed in the same directory, and its final `status.tsv` contains 171 jobs with `status=done` and `message=ok`. Historical entries remain in `failed_grids.jsonl` for auditability but do not describe the final run state.

The paired-battery run contains the synthetic static-load scope correction and rule-based stationary-battery dispatch. It supersedes the earlier `forchheim_paired_tsam` and `forchheim_paired_v2_tsam` power-flow summaries.

## Compared Scenarios

All three cases use the same paired physical buildings, annual base demand, sector assets, profile realizations, weather, mobility pool, TSAM mapping, and power-flow metric definitions. Only the network model and allocation bus differ between the real and synthetic sides.

| Display name | Meaning | Power-flow stage |
|---|---|---|
| Status quo | Base electricity demand before sector-coupling electrification | `pre` |
| SWF 2045 electrification with HEMS | Post-electrification demand after building-level URBS optimization | `post` |
| SWF 2045 electrification without HEMS | Fixed post-electrification demand reconstructed without flexible dispatch | `post` |

### Status Quo

- Uses paired household and calibrated GHD electricity profiles.
- Does not add post-electrification heat, EV charging, or PV dispatch to the power-flow demand.
- Provides the common electrical-demand baseline for both grid models.

### Post-Flex / HEMS

- Represents the 2045 SWF sector-asset inventory after deduplication and projection to the paired physical buildings.
- URBS optimizes every scenario unit with the same formulation and technology assumptions on both network sides.
- Heat pumps, auxiliary electric heating, EV charging, PV, and storage dispatch follow the optimized result.
- Auxiliary heating is an economic capacity and dispatch decision for peak heat coverage rather than a blanket heat-pump-only assumption.

### Post-No-Flex

- Depends on the post-flex optimization for installed heat-pump and auxiliary-heater capacities.
- Reconstructs fixed heat demand using the optimized heat-pump/auxiliary capacity split and COP time series.
- Reuses Step 2 mobility profiles; emobpy is not rerun.
- Redistributes EV energy within cyclic home-availability windows and applies the configured home-charger power cap.
- Uses exogenous rooftop PV availability and installed capacity without optimized flexible dispatch.
- Applies rule-based stationary-battery dispatch without forecasts: local PV surplus charges the battery and later local demand discharges it within the configured power, energy, efficiency, and state-of-charge limits.
- Acts as the counterfactual stress case for estimating the grid benefit of HEMS flexibility.

## Database and Analysis Names

| Scenario | Synthetic power-flow run | Real power-flow run | Synthetic expansion key | Real expansion key |
|---|---|---|---|---|
| Status quo | `forchheim_paired_battery_tsam_synthetic_pre` | `forchheim_paired_battery_tsam_real_swf_pre` | `forchheim_paired_battery_tsam_pre` | `forchheim_paired_battery_tsam_real_pre` |
| Post-flex | `forchheim_paired_battery_tsam_synthetic_flex` | `forchheim_paired_battery_tsam_real_swf_flex` | `forchheim_paired_battery_tsam_post` | `forchheim_paired_battery_tsam_real_post` |
| Post-no-flex | `forchheim_paired_battery_tsam_synthetic_no_flex` | `forchheim_paired_battery_tsam_real_swf_no_flex` | `forchheim_paired_battery_tsam_post_no_flex` | `forchheim_paired_battery_tsam_real_post_no_flex` |

These are the identifiers used by `5.postprocessing/notebooks/analysis_expansion.ipynb`.

## Expansion Cost Comparison

The same `de_lv_heuristic_2026` thermal-reinforcement heuristic is materialized independently for the synthetic and real grids. Existing cables are retained, while added capacity is selected from the common `NAYY_4_150`, `NAYY_4_185`, and `NAYY_4_240` catalogue. Parallel SWF line rows with identical endpoints and route lengths within 5% are costed as one physical corridor; their installed capacities and currents are aggregated, and civil works are charged once. The real totals include only complete grid-stage simulations: 87 real grids are complete in the status quo, while 85 are complete in each post scenario. LV 113 is explicitly excluded in all stages; LV 38 and LV 47 are additionally incomplete in both post scenarios because some timesteps did not converge. Synthetic coverage is 83 complete grids in every stage.

| Source | Scenario | Cable cost | Transformer cost | Total cost |
|---|---|---:|---:|---:|
| Synthetic | Status quo | EUR 0 | EUR 0 | EUR 0 |
| Synthetic | Post-no-flex | EUR 843,270 | EUR 830,000 | EUR 1,673,270 |
| Synthetic | Post-flex / HEMS | EUR 532,413 | EUR 544,000 | EUR 1,076,413 |
| Real SWF | Status quo | EUR 27,674 | EUR 33,000 | EUR 60,674 |
| Real SWF | Post-no-flex | EUR 937,028 | EUR 856,000 | EUR 1,793,028 |
| Real SWF | Post-flex / HEMS | EUR 705,572 | EUR 492,000 | EUR 1,197,572 |

HEMS reduces the modeled total thermal-reinforcement cost by 35.7% on the synthetic networks and 33.2% on the complete real-SWF networks. These are heuristic, loading-driven screening costs; voltage mitigation remains outside the cost total. Because source coverage differs, publication tables must show complete, incomplete, and excluded grid counts alongside totals and should include a per-complete-grid comparison.

## Important Methodological Constraints

1. Scenario units remain separate through demand creation and optimization. They are aggregated only when projected onto pandapower buses.
2. Real and synthetic runs must use identical scenario-unit inputs, random/profile realizations, optimization assumptions, and temporal aggregation.
3. Heat-pump component rows are deduplicated by physical installation. The `hp`, `heat`, and `dhw` records are components of one installation, not three independent heat pumps.
4. All 3,359 heat-pump buildings in the paired scope have exact physical-building heat profiles. Diagnostic area-scaled substitutions are not allowed in publication runs.
5. TSAM periods are selected from weather only and stored once. Every real and synthetic URBS result must reproduce the shared mapping.
6. Non-converged power-flow timesteps are reported as failed and represented by missing summary observations. They must not be silently removed or interpreted as valid low-stress states.
7. Direct service connections are excluded from the backbone cable and voltage comparison. The comparison focuses on demand-carrying feeder backbones.

## Known Open Issue: LV 113 Partition

The completed paired run is computationally complete, but LV 113 is not yet publication-ready from a topology perspective.

- The current station split assigns 513 household rows to the 630 kVA LV 113 transformer and only 69 rows to the neighboring 400 kVA LV 163 transformer.
- About 47% of LV 113 household demand and 52% of its calibrated GHD demand lie behind a 0.156 kA `NYY-J 4x35` corridor.
- The original SWF component also contains two closed feeders from the LV 163 transformer, but a corrected connectivity audit shows that they do not supply the territory behind line 23963. Moving that territory to LV 163 would disconnect it and overload the smaller neighboring transformer.
- The critical branch has no parallel cable. Its alternative physical connections to LV 59, LV 136, and LV 174 are all explicitly source-open.
- Radialization and station partitioning do not create this particular bottleneck: with source switch states respected, line 23963 is already the only active power-cable supply to the branch.
- The remaining ambiguity is therefore in the source model itself: the cable type or topology may be incomplete, the recorded switch state may not represent the relevant operating condition, or an additional supply asset may be absent.

Until the LV 113 source topology has been clarified or an explicit sensitivity has been defined, critical-tail real-grid voltage and cable results must be presented as provisional. LV 113 should not be silently repaired by reassignment. It is currently excluded explicitly from the real-SWF power-flow distribution and voltage analyses in `analysis_expansion.ipynb`; synthetic grids are unaffected. The technical findings and DSO clarification questions are documented in [`failed_grids_analyisis.md`](../audits/failed_grids_analyisis.md). The paired demand totals and optimization comparison remain valid.

## Reproduction Command

Run from the repository root after the paired allocation and heat-profile readiness checks have passed:

```bash
uv run --project GridExpand/2.demand_allocation \
  python GridExpand/runme/paired_swf_pipeline_runner.py \
  --repo-root /home/breveron/git/github/SurroGrid \
  --plz 91301 \
  --paired-dir GridExpand/2.demand_allocation/gridalloc/outputs/scenario_calibration/swf_2045_paired_v3_91301_station_hybrid_v2 \
  --grid-data-path /home/breveron/data/swf_split_station_hybrid_v2 \
  --weather-source-hdf GridExpand/2.demand_allocation/gridalloc/results/9474126-00_91301_1_2.h5 \
  --target both \
  --scenario-config GridExpand/scenario_pipeline/config/scenarios/forchheim_2045.yaml \
  --model-case post-hems-heuristic \
  --workers 4 \
  --step3-cpus 4 \
  --step3-cluster-concurrency 1 \
  --step4-cpus 2 \
  --cleanup-intermediates \
  --scenario-label forchheim_paired_battery_tsam \
  --run-name-prefix forchheim_paired_battery_tsam \
  --run-dir GridExpand/run_logs/forchheim_paired_battery_tsam_$(date -u +%Y%m%dT%H%M%SZ)
```

Do not add `--allow-diagnostic-heat-fallback` to a publication run.

To resume an interrupted run, use the same run directory and unchanged scenario and TSAM settings, then add:

```bash
--resume --rerun-failed
```

These flags are for resuming an existing run, not for a clean first execution.

## Superseded or Diagnostic Runs

| Run family | Status | Use |
|---|---|---|
| Earlier synthetic-only Forchheim HH runs | Superseded by the paired full-local-demand run | Method development and historical sensitivity only |
| `real_swf_2045_full_local_sector_flex` | Superseded | Do not use for publication; predates exact heat-pump deduplication |
| `real_swf_2045_full_local_sector_no_flex` | Superseded | Do not use for publication; predates exact heat-pump deduplication |
| Earlier LV 113 diagnostic runs | Diagnostic | Useful for tracing demand duplication and radialization effects, not regional comparison |
| Munich pre/post-flex/post-no-flex pilot | Deleted and non-authoritative | Exposed disk-volume and initial no-flex EV reconstruction problems; must be regenerated with the current pipeline before use |

## Publication Checklist

Before using a run in the paper, verify:

- paired real and synthetic building identities and annual HH/GHD totals are equal;
- all heat profiles are marked `publication_ready=true`;
- the real and synthetic runs use the same TSAM signature;
- every target job is complete;
- failed power-flow timesteps are reported and investigated;
- expansion analyses reference the intended run names above;
- known topology anomalies such as LV 113 are corrected or transparently excluded with a documented rule;
- intermediate HDF files are cleaned only after database summaries and required audit artifacts have been verified.

## Maintenance

Update this file whenever a scenario becomes the new source of truth. Record the run directory, power-flow run names, analysis keys, final target counts, temporal settings, demand scope, unresolved anomalies, and whether earlier results were retained or deleted.
