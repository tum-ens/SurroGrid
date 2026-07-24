# Synthetic and Real Feeder-Structure Comparison

This audit explains the remaining synthetic/real cable-loading difference for the paired `forchheim_paired_battery_tsam` scenario. It covers 83 synthetic grids and 87 real SWF grids after excluding `LV 113`. Annual demand is the paired household plus calibrated GHD demand. Terminal service connections are excluded consistently.

## Normalization

- Parallel physical line rows between the same buses are treated as one electrical edge.
- Consecutive physical edges through degree-two buses without allocated demand form one feeder section.
- `NS_StLt` rows with `NS-Leitungstyp_fiktiv` are classified as control-line connectors rather than physical LV cable capacity.
- Downstream demand is accumulated along the minimum-impedance root-to-load path.
- Section loading is the maximum capacity-weighted cable-row peak among the physical edges in that section. Row maxima need not be perfectly simultaneous, so this remains a screening metric.

Generated tables and figures are written to `output/audits/feeder_structure/<AGS>/<scenario>/`.

## Results

| Median grid metric | Synthetic | Real SWF |
|---|---:|---:|
| Outgoing physical feeders | 4 | 5 |
| Physical feeder sections | 53 | 81 |
| Backbone demand attachment buses | 53 | 70 |
| Downstream demand per installed ampere | 166.5 kWh/a/A | 86.2 kWh/a/A |
| Outgoing-feeder demand per installed ampere | 346.5 kWh/a/A | 231.8 kWh/a/A |
| Downstream-demand-weighted cable capacity | 306.5 A | 325.7 A |
| Demand-weighted root-to-load depth | 8.75 sections | 6.18 sections |
| Demand-weighted root-to-load length | 0.210 km | 0.220 km |

The analyzed regional annual demand is 50.16 GWh/a for the synthetic grids and 48.41 GWh/a for the retained real grids. The 3.6% difference results mainly from excluding real `LV 113` without removing its scenario units from the synthetic network and is too small to explain the cable-loading gap.

The normalized physical topology contains 8,460 synthetic and 8,946 real cable rows. These become 8,460 and 8,892 parallel-normalized edges, respectively. Parallel-row representation is therefore negligible: only 54 additional physical rows are consolidated on the real side. Mean rows per feeder section are also similar, with median grid values of 1.30 synthetic and 1.19 real.

| Median physical-section maximum loading | Synthetic | Real SWF | Synthetic / real |
|---|---:|---:|---:|
| Status quo | 9.04% | 5.28% | 1.71 |
| No-flex | 27.67% | 17.06% | 1.62 |
| HEMS | 25.35% | 15.92% | 1.59 |

Consolidating parallel rows and degree-two cable chains therefore does not remove the loading difference.

## Interpretation

The evidence does not support the hypothesis that pylovo simply creates too few outgoing feeders: the median is four synthetic versus five real. Nor is the result mainly caused by different numbers of pandapower cable rows. Instead, three related structural differences remain:

1. Real demand is connected at more backbone attachment points and distributed over more feeder sections.
2. A typical synthetic feeder section carries about twice as much annual downstream demand per installed ampere.
3. Synthetic root-to-load paths contain about 42% more feeder sections despite having a similar physical length, so demand remains aggregated across more serial cable sections.

The synthetic cables are not generally much smaller when capacity is weighted by downstream demand: the median grid values differ by only about 6%. The important difference is where branching, load attachment, and cable capacity occur. The pylovo networks use fewer physical feeder and attachment sections, and their typical sections consequently remain responsible for more downstream demand.

## SWF Control Lines

The retained real grids contain 718 `NS_StLt` rows used as connector-only links and eight rows parallel to physical cable edges. None of these rows is present in the compact cable-summary scope used by the current loading plot, so they do not directly explain its distributional difference. The eight parallel cases affect only two grids and should still be removed from the electrical power-cable model in a future rerun because zero-impedance control lines must not carry power in parallel with physical cables.

## Conclusion

The remaining cable-loading difference is primarily a feeder-structure and demand-distribution result, not a raw equipment-catalogue or line-row-count artifact. The current evidence indicates that pylovo produces fewer branch and load-attachment sections and concentrates downstream demand over longer serial paths. The next model-level review should therefore focus on connection-point aggregation, feeder branching, and the feeder-splitting criterion rather than globally increasing cable capacity.

## Version 4 Branching Calibration

Version 4 uses a 15 m connection-point aggregation radius, at most four aggregated buildings, a 0.850 kA feeder-split threshold, and a 100 m minimum shared prefix. The corrected metric export contains grid-result IDs 2950--3056; the earlier export with IDs 2843--2949 represented the accidental 850 kA configuration and must not be used.

Across the six comparison metrics, mean normalized Wasserstein distance increased from 0.206 for the version 3 large-cable generation used by the previous paired run to 0.227 for version 4 (the older default export scores 0.197). Average transformer distance moved close to the real mean (0.234 km synthetic versus 0.232 km real), but mean feeder count overshot the reference (7.10 versus 5.58) and graph length remained high (1.40 km versus 1.01 km). Thus the global topology score did not improve.

The paired feeder audit nevertheless shows targeted structural progress:

| Median paired-grid metric | Previous synthetic | Version 4 | Real SWF |
|---|---:|---:|---:|
| Outgoing physical feeders | 4 | 5 | 5 |
| Physical feeder sections | 53 | 60 | 81 |
| Backbone demand attachment buses | 53 | 60 | 70 |
| Outgoing-feeder demand per installed ampere | 346.5 kWh/a/A | 276.7 kWh/a/A | 231.8 kWh/a/A |
| Downstream-demand-weighted cable capacity | 306.5 A | 315.8 A | 325.7 A |
| Demand-weighted root-to-load depth | 8.75 sections | 7.63 sections | 6.18 sections |

Stored design-point power flows across all 83 paired synthetic grids also move in the expected direction: median grid-maximum backbone loading falls from 82.4% to 78.1%, while the median of grid-level cable medians falls from 23.8% to 21.3%. Three TSAM pre-power-flow checks on territories with identical physical-building membership show one strong P95 cable-loading improvement (25.8% to 16.8%) and two nearly unchanged critical tails (70.9% to 71.1% and 39.8% to 37.0%). Transformer loading and voltage remain effectively unchanged, as expected under identical territory and demand.

Version 4 therefore improves feeder-level demand sharing, but not enough to remove the loading discrepancy, and the 100 m shared-prefix threshold creates too many feeders in the unpaired regional metric population. Further calibration should retain the reduced connection-point aggregation while lowering the shared-prefix threshold before changing cable capacities.
