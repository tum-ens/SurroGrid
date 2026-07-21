# Synthetic and Real SWF Equipment Capacity Comparison

This audit compares the equipment represented in the paired Forchheim power-flow runs. Cable statistics cover the demand-carrying backbone and exclude final service connections. Real grid `LV 113` is excluded where noted because it is temporarily excluded from the power-flow comparison.

| Quantity | Synthetic | Real SWF |
|---|---:|---:|
| Transformer grids | 81 | 84 |
| Total transformer capacity | 45.970 MVA | 45.645 MVA |
| Total transformer capacity excluding real `LV 113` | 45.970 MVA | 45.015 MVA |
| Mean transformer capacity | 0.568 MVA | 0.543 MVA |
| Median transformer capacity | 0.630 MVA | 0.630 MVA |
| Backbone cable rows excluding real `LV 113` | 9,077 | 8,321 |
| Mean installed cable capacity | 289 A | 291 A |
| Median installed cable capacity | 242 A | 276 A |
| P90 installed cable capacity | 357 A | 426 A |
| P95 installed cable capacity | 626 A | 426 A |

## Interpretation

- Transformer capacity is closely aligned: both distributions have a median of 630 kVA and their regional totals differ by less than 1% before excluding `LV 113`.
- Mean installed backbone-cable capacity is effectively equal. Real single cables tend to have higher ratings through P90, whereas the synthetic P95 is higher because pylovo uses parallel feeder cables in its upper capacity tail.
- Equipment capacity does not explain the substantially lower loading observed in the real SWF results.
- The loading comparison was found to be contaminated by original static loads retained on unselected synthetic buses. Across the 81 synthetic grids, 13.407 MW of static load remained active outside the paired allocation scope. The synthetic power-flow preparation must therefore be corrected and rerun before interpreting loading differences as structural grid differences.

## Cable Representation Note

Synthetic installed capacity is `max_i_ka * parallel`. Of 9,077 synthetic backbone cable rows, 682 use `parallel > 1`. Almost all real cable rows use `parallel = 1`; physical parallel cables may instead appear as separate line rows in the SWF model.
