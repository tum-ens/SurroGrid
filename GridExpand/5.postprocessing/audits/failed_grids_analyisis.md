# Failed Real-SWF Grid Analysis

## Scope

This document consolidates the investigation of real-SWF grids `LV_038`, `LV_047`, and `LV_113` for the paired Forchheim power-flow scenarios. The analyzed runs use the station-hybrid-v2 split, path-preserving radialization, identical paired demands for the synthetic and real targets, and 1,008 TSAM timesteps. A failed timestep means that the power flow did not reach a valid operating point; it is not interpreted as a zero-load or zero-cost result.

## Summary

| Grid | Original transformer | Status quo | HEMS | No-flex | Main finding | Current treatment |
|---|---|---:|---:|---:|---|---|
| `LV_038` | `MSNS_TrSt_0062`, 400 kVA | 0 failed | 5 failed | 14 failed | Electrification exposes a long, weak single-feeder path; no evidence of lost loads or a harmful radialization | Retain as a stressed grid; expansion must restore feasibility |
| `LV_047` | `MSNS_TrSt_0101`, 630 kVA | 0 failed | 33 failed | 44 failed | Electrification overloads an early 185 A corridor; the recorded open point is electrically consequential and the station boundary remains uncertain | Retain as incomplete pending reinforcement and DSO clarification |
| `LV_113` | `MSNS_TrSt_0150`, 630 kVA | 0 failed | 218 failed | 289 failed | Severe bottleneck already exists in the status quo behind one 156 A cable, suggesting missing or ambiguous source topology/equipment data | Explicitly exclude pending DSO clarification |

The failed post-electrification states in `LV_038` and `LV_047` are not random solver failures. Alternative algorithms also fail, while proportional demand reduction restores convergence. The failures therefore represent voltage-collapse or overload states in the modeled networks. They should normally trigger reinforcement rather than exclusion.

## Comparative Power-Flow Evidence

| Grid | Scenario | Failed timesteps | Maximum transformer loading | Maximum cable loading | Minimum converged voltage |
|---|---|---:|---:|---:|---:|
| `LV_038` | Status quo | 0 / 1,008 | 43.7% | 107.4% | 0.871 p.u. |
| `LV_038` | HEMS | 5 / 1,008 | 145.9% | 358.4% | 0.557 p.u. |
| `LV_038` | No-flex | 14 / 1,008 | 157.5% | 387.0% | 0.501 p.u. |
| `LV_047` | Status quo | 0 / 1,008 | 36.0% | 111.0% | 0.864 p.u. |
| `LV_047` | HEMS | 33 / 1,008 | 93.6% | 357.2% | 0.507 p.u. |
| `LV_047` | No-flex | 44 / 1,008 | 122.0% | 358.6% | 0.471 p.u. |
| `LV_113` | Status quo | 0 / 1,008 | 85.9% | 299.5% | 0.600 p.u. |
| `LV_113` | HEMS | 218 / 1,008 | 99.8%* | 348.4%* | 0.502 p.u.* |
| `LV_113` | No-flex | 289 / 1,008 | 150.0%* | 357.4%* | 0.519 p.u.* |

`*` Extrema are conditional on converged timesteps and do not describe the failed operating points.

For `LV_038` and `LV_047`, nearly all failures occur in the cold first TSAM period, rather than at an initialization timestep. The failed no-flex hours combine high heat-pump and EV demand. Mean demand during these failed hours is approximately 400 kW for `LV_038` and 456 kW for `LV_047`. Calibrated commercial/public demand is zero in `LV_038` and only about 37 MWh/a in `LV_047`, so GHD allocation does not explain these failures.

## LV 038

### Topology and bottleneck

- Original transformer: `MSNS_TrSt_0062`, `chr_name` `6038038_000000_000000_000000_05001`, 400 kVA.
- Transformer LV bus: original bus 23212, `NS_Kn(n)_TrSt_000062`.
- All 131 scenario load buses are supplied through one feeder.
- The transformer-adjacent physical corridor starts with original line 29030, `NS_Kb_001673_001`, type `NKBA 4x70sm 0.6/1kV`, rated 235 A. Several following sections have the same rating.
- A 235 A three-phase LV cable carries about 163 kVA at 400 V, substantially less than the 400 kVA transformer rating. This is a plausible feeder bottleneck even before the transformer is fully utilized.
- The critical path is about 1.07 km long and contains many overloaded sections under electrification.

### Data-quality checks

- No scenario load buses are unsupplied or dropped.
- No active cable has missing or infinite capacity.
- The station split retained the complete name-implied territory; no fallback reassignment was applied.
- Radialization opened only line 29400, `NS_Kb_001799_001`, and preserved every original minimum-impedance transformer-to-load path (`1.00x` maximum path stretch).
- Restoring the logical ring improves convergence only partially and does not remove the severe voltage and cable-loading problem.
- Closing the alternative source-switch line 27290, `NS_Kb_001179_001`, and opening line 33061 recovers most failed snapshots, but leaves minimum voltages down to about 0.47 p.u. and cable loading above 300%.

### Assessment

The evidence does not identify a splitter, load-allocation, radialization, or solver defect. `LV_038` is best treated as a genuinely weak represented feeder under the electrification scenario, unless the DSO confirms a missing parallel cable or an additional station feeder. Its failed hours should be included in an iterative reinforcement calculation rather than discarded.

## LV 047

### Topology and bottleneck

- Original transformer: `MSNS_TrSt_0101`, `chr_name` `6047047_000000_000000_000000_05001`, 630 kVA.
- Transformer LV bus: original bus 23135, `NS_Kn(n)_TrSt_000101`.
- Four feeder groups leave the transformer. The dominant feeder starts with line 26286, `NS_Kb_000833_001`, type `NYY-J 3x185SM`, rated 483 A, and supplies approximately 74% of the grid peak.
- Shortly downstream, the path crosses three 185 A cables: lines 30095, 30094, and 30093 (`NS_Kb_002041_003`, `_002`, and `_001`). These sections reach approximately 357-359% loading.
- Source-open line 26518, `NS_Kb_000907_001`, is a 483 A `NYY-J 3x185SM` tie between `NS_KVS_419_(S)` and `NS_KVS_420_(S)`. Its recorded switch state routes supply through the much smaller 50 mm2 corridor.

### Data-quality checks

- No scenario load buses are unsupplied or dropped, and active cable ratings are finite.
- Radialization opened only line 28979, `NS_Kb_001660_004`, while preserving minimum-impedance supply paths (`1.00x` maximum path stretch).
- Reusing the logical ring does not restore convergence for the failed snapshots.
- A diagnostic open-point swap, closing line 26518 and opening line 30094, restores only 20 of 33 HEMS failures and 20 of 44 no-flex failures. Remaining states are still severely stressed.
- The station split is more uncertain than for `LV_038`: the original electrically coupled component contains `LV_047`, `LV_062`, and `LV_173`, and the station assignment retained 457 buses compared with 1,091 buses implied by the logical name. Explicit switch states nevertheless provide evidence for the current supply territory, so stress alone is not a sound basis for moving loads to neighboring transformers.

### Assessment

`LV_047` contains a credible physical bottleneck under the exported switch state, but the intended operational open point and transformer-area boundary require DSO confirmation. It should not be silently reassigned or excluded. Until reinforcement can restore all failed timesteps, its expansion cost is incomplete and must be reported as such.

## LV 113

### Topology and bottleneck

- Original transformer: table index 157, `MSNS_TrSt_0150`, `chr_name` `6113113_000000_000000_000000_05001`, 630 kVA.
- Critical first cable: original line 23963, `NS_Kb_000121_001`, `chr_name` `7113113_001001_002002_485488_06001`, one `NYY-J 4x35` cable rated 156 A.
- Removing line 23963 separates a territory with 308 buses, 240 household rows, approximately 0.723 GWh/a household demand, and 0.123 GWh/a calibrated commercial/public demand.
- Under the recorded source-switch states, line 23963 is the only active physical power-cable connection to this territory. Six subsequent `NS_Kb_000500_*` sections continue the same small-cable chain.
- Several physical ties to neighboring named grids exist but are recorded open. Closed lines from `LV_163` in the wider component do not supply the territory behind line 23963.

### Assessment

Unlike `LV_038` and `LV_047`, `LV_113` is already severely implausible in the status quo: cable loading reaches about 300% and voltage falls to 0.600 p.u. This persists after the revised station split and path-preserving radialization. It is therefore excluded from comparative distributions until the DSO clarifies whether a feeder, parallel cable, transformer, normal switch state, or station assignment is missing or incorrect. No source record is modified automatically.

## DSO Clarification Questions

### LV 038

1. Is `MSNS_TrSt_0062` normally connected to its complete LV territory through only the 235 A corridor beginning with `NS_Kb_001673_001`?
2. Is a parallel cable, additional outgoing feeder, or operational tie missing from the export?

### LV 047

1. Is `NS_Kb_000907_001` between `NS_KVS_419_(S)` and `NS_KVS_420_(S)` intentionally open in the relevant normal operating state?
2. Is the territory behind lines `NS_Kb_002041_003`, `_002`, and `_001` normally supplied through these 185 A cables?
3. Is the exported station boundary between `LV_047`, `LV_062`, and `LV_173` correct, and may the open point change during high load?

### LV 113

1. Is `NS_Kb_000121_001` correctly represented as one 156 A `NYY-J 4x35` cable?
2. Does it normally supply the full downstream territory, or is another feeder, parallel cable, transformer, or closed tie missing?
3. Which neighboring tie is the intended operational open point, and do the exported switch states represent normal operation?
4. Is the downstream territory correctly assigned to `MSNS_TrSt_0150`?
5. Are all `NS_StLt` / `NS-Leitungstyp_fiktiv` records control connections rather than load-carrying LV paths?

## Methodological Consequences

1. Keep `LV_113` explicitly excluded until its status-quo inconsistency is clarified.
2. Keep `LV_038` and `LV_047` visible as incomplete stressed grids; do not classify their failed electrification hours as data errors solely because they do not converge.
3. Extend the expansion workflow with feasibility restoration: use the last converged or continuation state to identify cable and transformer bottlenecks, add capacity, rerun failed snapshots, and iterate until convergence.
4. Do not repair failed grids by reducing demand, changing station assignments based only on loading, reclosing recorded source-open switches without evidence, or interpreting missing cost rows as zero cost.
5. Report aggregate real-grid expansion costs as incomplete while any included grid-stage still lacks a feasible reinforced solution. Omitting failed grids would bias the estimated real-grid cost downward.

This document consolidates the detailed `LV_113` evidence and the clarification questions for communication with the DSO.
