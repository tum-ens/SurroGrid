# Synthetic and Real SWF Equipment Capacity Comparison

This audit compares the equipment represented in the current paired power-flow runs. Cable statistics cover the demand-carrying backbone, exclude final service connections, and omit real LV 113 consistently with the analysis notebook.

| Quantity | Synthetic | Real SWF |
|---|---:|---:|
| Transformer grids | 83 | 87 |
| Total transformer capacity | 47.000 MVA | 47.305 MVA |
| Mean transformer capacity | 0.566 MVA | 0.544 MVA |
| Median transformer capacity | 0.630 MVA | 0.630 MVA |
| Backbone cable rows | 8,460 | 8,918 |
| Mean installed cable capacity | 291 A | 290 A |
| Median installed cable capacity | 242 A | 276 A |
| P90 installed cable capacity | 425 A | 426 A |
| P95 installed cable capacity | 626 A | 426 A |

## Interpretation

- Transformer capacity is closely aligned: both distributions have a median of 630 kVA and their retained regional totals differ by less than 1%.
- Mean installed backbone-cable capacity is also effectively equal. Real cable rows have higher ratings through P90, while pylovo has the higher P95 because its upper tail uses parallel feeder circuits.
- These unweighted equipment distributions do not describe where capacity occurs relative to downstream demand. The graph-normalized feeder audit finds median downstream-demand-weighted capacities of 307 A synthetic and 326 A real.
- More importantly, the median grid-level downstream demand per installed ampere is 166.5 kWh/a/A synthetic versus 86.2 kWh/a/A real. The loading difference is therefore tied to feeder branching, load attachment, and capacity placement rather than the equipment catalogue alone.

## Cable Representation Note

The graph-normalized audit consolidates parallel physical rows between the same buses and degree-two cable chains. It finds 8,460 synthetic and 8,946 real physical backbone line rows, which become 8,460 and 8,892 parallel-normalized edges. Thus, separate real parallel rows account for only 54 additional rows and do not explain the loading difference.

SWF NS_StLt rows using NS-Leitungstyp_fiktiv are classified as control connectors, not physical cable capacity. Their role and the complete feeder-structure findings are documented in feeder_structure_comparison.md.
