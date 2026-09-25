# Forchheim 2045 Paired v11 Forensic Audit

## Scope

This report audits the results consumed by
notebooks/analysis/analysis_expansion.ipynb for the paired
Forchheim run forchheim_2045_paired_v11. It addresses:

1. whether terminal building-connection lines are included in power-flow and
   expansion analysis;
2. why heuristic HEMS produces worse critical loading and expansion cost than
   INFLEX despite improving typical-hour loading;
3. why synthetic feeder sections have higher downstream demand although both
   networks receive the same paired buildings; and
4. why the former Real SWF expansion-envelope row was constant across scenarios.

The audit used the committed run configuration, paired allocation artifacts,
materialized database summaries, feeder-audit exports, power-flow and URBS
implementation, and the plotting implementation. No full pipeline stage was
rerun.

## Run identity and readiness

The notebook selects AGS 9474126, scenario prefix
forchheim_2045_paired_v11, 45 synthetic grids and 49 real SWF grids. The run
configuration fixes PyLoVo version 11, paired dataset
swf_2045_paired_v11_91301_swf_clear, and power-flow scope full.

The synthetic power-flow runs are complete. The real SWF compact summaries
contain failed timesteps:

| Scenario | Real grids with failed timesteps | Failed timesteps |
|---|---:|---:|
| Status quo | 1 | 1 |
| INFLEX | 2 | 306 |
| Heuristic HEMS | 4 | 320 |
| Optimized HEMS | 1 | 297 |

Consequently, the notebook publication gate correctly labels the current
outputs diagnostic. This affects the number of real grids for which expansion
cost can be materialized, but it does not explain the synthetic HEMS reversal.

## Terminal connection-line scope

### What the v11 power flow includes

The run configuration sets powerflow_grid_scope to full. Database assumptions
confirm that every selected synthetic and real v11 compact summary was executed
with summary_grid_scope equal to full.

The full-scope implementation selects every active pandapower line and observes
voltage directly at terminal load buses. Therefore:

- transformer loading is calculated from the full network solution;
- cable loading is stored for every active cable in that solution; and
- minimum-voltage and voltage-distribution metrics use terminal load-bus
  voltages.

The full compact cable inventory contains 8,329 rows for the 45 synthetic grids
and 9,185 rows for the 48 real grids retained after excluding LV 113.

### What the feeder-structure audit excludes

The feeder-structure audit has a different purpose. For each terminal demand
bus, it maps the demand one bus upstream and removes the final physical edge
from the retained path. It also excludes SWF NS_StLt and
NS-Leitungstyp_fiktiv control placeholders from physical-cable metrics.

The resulting backbone inventory contains 4,108 synthetic and 4,625 real line
rows. The large difference from the full cable inventory is therefore expected
and is dominated by terminal connections.

### What can be switched in the notebook

There are two materially different operations:

1. Filtering already-computed cable rows is a notebook/postprocessing
   operation. A line-role classification can label service, backbone and
   control rows and allow cable-loading or cost distributions to include or
   exclude service lines without rerunning power flow.
2. Recomputing transformer loading or terminal voltage after electrically
   removing service lines is not a plotting filter. Those quantities come from
   a solved network state. A backbone-only counterfactual requires a separate
   Step 4 power-flow run and expansion rematerialization. Demand allocation and
   URBS dispatch do not need to be rerun.

At present the feeder helper hard-codes service-line removal and has no notebook
switch. Adding an explicit include_service_connections or
exclude_terminal_service_connections option would allow the structural audit
to be rerun from the notebook without rerunning the pipeline.

The expansion materialization consumes the full compact cable summaries.
Synthetic materialization maps 8,284 of 8,329 cable rows; the 45 omitted rows
are one non-visible, non-overloaded connector-like row per grid. For
cost-complete real grids, every source cable-summary row is represented in the
materialization, with parallel physical rows possibly grouped into corridors.

## INFLEX versus heuristic HEMS

### The compared asset inventory

INFLEX and heuristic HEMS share heuristic asset sizing. The model-case
definition differs only in dispatch:

- INFLEX: heuristic assets with rule-based dispatch;
- heuristic HEMS: heuristic assets with optimized dispatch.

The comparison is therefore a dispatch comparison, not a PV, battery or heat
asset-capacity comparison.

### The optimization objective is not grid-aware

The building models minimize operating and investment cost independently. The
scenario uses a constant electricity import price of 0.398 EUR/kWh and zero PV
feed-in remuneration. The grid-connection process has a fixed 2,000 kW
capacity with zero investment, fixed and variable cost. The URBS runner
forcibly disables variable tariffs and the power-price term.

There is no feeder topology, transformer loading, coincident regional peak or
network expansion term in the HEMS objective. HEMS is therefore allowed to
reduce individual purchase cost while increasing simultaneous power across
buildings. Grid-friendly behavior was expected conceptually but is not
specified mathematically.

### Observed distributional reversal

The complete 45-grid synthetic comparison is:

| Metric | INFLEX | Heuristic HEMS |
|---|---:|---:|
| Median temporal transformer P50 | 18.199% | 14.891% |
| Median temporal transformer P90 | 36.089% | 32.851% |
| Median temporal transformer P95 | 41.073% | 35.800% |
| Median temporal transformer P99 | 47.558% | 42.103% |
| Median annual transformer maximum | 56.550% | 69.014% |
| P90 annual transformer maximum | 97.278% | 148.029% |
| Transformers above 100% | 4 | 15 |
| Cables above 100% | 23 | 187 |
| Expansion cost | 191,135.58 EUR | 1,054,750.76 EUR |

HEMS improves typical hours but worsens the extreme tail. Annual transformer
maximum increases in 38 of 45 synthetic grids, with a median increase of
10.773 percentage points. Of the 187 overloaded HEMS cable rows, 164 cross
from at or below 100% under INFLEX to above 100% under HEMS; no overloaded
INFLEX cable returns below 100%.

Expansion decisions use annual P100 loading and discrete cable/transformer
increments. A small number of synchronized hours can therefore dominate
capital cost even when P50 through P99 loading improves.

### Representative-week boundary synchronization

The strongest root-cause indicator is the critical-hour distribution:

- 31 of 45 synthetic HEMS transformer maxima occur at an exact 168-hour
  representative-week boundary;
- 21 occur at t=840, the first hour of the sixth representative week;
- nine occur at t=0 and one at t=168;
- no synthetic INFLEX transformer maximum occurs on a 168-hour boundary;
- 32 of 49 real HEMS maxima occur on such a boundary, compared with zero for
  real INFLEX.

The TSAM implementation constructs representative periods using weather
features only, currently ambient temperature and irradiation. Demand and
mobility profiles are not part of clustering despite the broader function
docstring. Storage content is constrained to return to the common initial
content at the end of every representative period.

With a flat price, many building-level dispatch schedules are economically
equivalent. Identical representative periods and cyclic boundary conditions
give the optimizer common hours in which to place flexible charging or heating
actions. Aggregating thousands of independently optimized buildings then
creates artificial coincidence.

This evidence establishes a grid-unaware objective and a representative-period
boundary artifact. It does not identify the exact component share of the peak.
The run enabled cleanup_intermediates, and the component-level v11 URBS HDF
results no longer exist. A targeted rerun retaining outputs for synthetic grid
3172 and real LV 073 would be sufficient to decompose t=0, t=168 and t=840
into EV, stationary battery, heat pump and thermal-storage flows.

### Real-grid expansion-cost check

Raw real totals use different complete-grid coverage:

| Scenario | Complete | Incomplete | Excluded | Total cost |
|---|---:|---:|---:|---:|
| Status quo | 48 | 0 | 1 | 43,625 EUR |
| INFLEX | 47 | 1 | 1 | 526,465 EUR |
| Heuristic HEMS | 45 | 3 | 1 | 815,424 EUR |
| Optimized HEMS | 48 | 0 | 1 | 524,373 EUR |

The raw INFLEX and heuristic HEMS totals must not be treated as equal-coverage
estimates. Restricting both to their 45 jointly complete grids still gives
heuristic HEMS 496,194 EUR more expansion cost: 12 grids cost more, one costs
less and 32 remain unchanged. The direction is therefore genuine even though
the unrestricted real totals are not publication-ready.

## Paired demand and downstream-demand discrepancy

### Building-level parity

The paired scope contains exactly 4,193 physical buildings/scenario units in
each target network. Row-level comparison by scenario_unit_id gives zero
maximum difference for household rows and annual demand, calibrated GHD
demand, EV charger capacity, PV capacity and battery capacity.

Both target networks therefore receive exactly:

- 7,392 residential-equivalent household rows;
- 21.747302 GWh/a residential demand; and
- 5.132155 GWh/a calibrated GHD demand.

The regional total is 26.879457 GWh/a in each network before exclusions.

### Effect of restoring LV 113

LV 113 contains 319 paired scenario units and 1.757231 GWh/a. Excluding it only
from the real audit creates the previously observed regional totals:

- synthetic: 26.879457 GWh/a;
- retained real: 25.122226 GWh/a.

Restoring LV 113 makes the regional annual totals exactly equal. It does not
make grid-level medians equal:

| Metric | Real with LV 113 | Synthetic |
|---|---:|---:|
| Number of grids | 49 | 45 |
| Regional annual demand | 26.879457 GWh/a | 26.879457 GWh/a |
| Mean annual demand/grid | 548.560 MWh/a | 597.321 MWh/a |
| Median annual demand/grid | 464.391 MWh/a | 557.426 MWh/a |
| Median section downstream demand | 24.382 MWh/a | 37.618 MWh/a |
| Median downstream demand/capacity | 93.504 kWh/a/A | 152.628 kWh/a/A |

Equal regional demand divided over 45 rather than 49 territories necessarily
raises synthetic mean demand per grid by 8.9%. The larger 20.0% median
difference also reflects different territory boundaries and the shape of the
grid-demand distribution.

LV 113 itself has median section downstream demand of 25.268 MWh/a, close to
the retained real median. Adding it consequently changes the real downstream
median only from 24.352 to 24.382 MWh/a. LV 113 solves the regional-total
asymmetry but essentially none of the section-level discrepancy.

### Why section downstream demand remains different

Downstream demand is derived after buildings have been assigned to buses and
routed over each network graph. For every normalized feeder section, the audit
takes the maximum accumulated edge demand in that section. It then calculates
a median across sections inside each grid and finally a median across grids.
This median-of-medians is neither additive nor invariant to repartitioning.

With LV 113 included, the topology medians remain:

| Metric | Real SWF | Synthetic |
|---|---:|---:|
| Backbone demand buses | 63 | 84 |
| Feeder sections | 78 | 84 |
| Demand-weighted section depth | 6.243 | 10.546 |
| Demand-weighted cable capacity | 333.634 A | 286.979 A |

The same buildings are therefore attached and branched differently. Synthetic
demand remains aggregated along more serial sections and those sections have
lower demand-weighted capacity. This, rather than mismatched building demand,
is the principal reason for the 37.618 versus 24.382 MWh/a downstream median.

A publication-quality comparison should report:

1. regional demand parity before and after exclusions;
2. mean and median demand per grid, acknowledging 45 versus 49 territories;
3. a global section-weighted distribution in addition to the current
   median-of-grid-medians; and
4. separate backbone and terminal-service metrics, because including thousands
   of one-building service edges changes the statistical population.

## Expansion-envelope plotting defect and correction

### Defect

The former notebook call passed the synthetic-only available_analysis_keys
mapping. The plotting function loaded synthetic expansion values for each
scenario, but the real row loaded only load-bus geometry. It then redrew the
same real envelopes with a fixed orange face color in every scenario column.
The returned real table contained only grid id, point count and polygon status,
not expansion cost.

The apparent constant Real SWF cost was therefore produced by construction and
did not reflect the real materialized data.

### Implemented correction

The plot now accepts real_analysis_keys in addition to the synthetic keys. It:

- loads source-specific Real SWF grid-cost summaries;
- retains cost-complete grids and joins them to real LV geometry by normalized
  LV id;
- creates real scenario envelopes with the actual selected cost metric;
- computes one shared color normalization from both synthetic and real values;
- labels each real panel with its number of cost-complete grids; and
- returns a scenario-specific real grid_metrics table for notebook inspection.

The notebook now passes ANALYSIS_KEYS_BY_SOURCE for both Synthetic and Real SWF.
If the plotting function is called without real analysis keys, its backward
compatible geometry-only row is explicitly titled Real SWF geometry reference
so it cannot be mistaken for a cost map.

No power-flow or expansion rerun is needed to correct the cost data. The
existing real materializations already contain varying costs. Direct
validation of the corrected loader produced 48, 47, 45 and 48 cost-complete
real grids and respectively 4, 12, 16 and 12 distinct per-grid cost values for
status quo, INFLEX, heuristic HEMS and optimized HEMS.

A separate reproducibility issue prevents the complete figure from being
rendered again from the database in its current state: the
surrogrid.grid_building_bus view is absent, and the underlying historical v11
PyLoVo building, bus and line geometry rows for grid-result ids 3396 and above
have also been cleaned. The expansion and power-flow summaries remain present,
but they do not contain building coordinates from which the synthetic convex
hulls can be reconstructed. Restoring a database snapshot or regenerating the
PyLoVo v11 spatial grids is therefore required to redraw the synthetic
envelopes. This is a geometry-retention issue, not a need to rerun demand,
URBS, power flow or expansion calculations. The previously embedded notebook
figure remains the old, defective rendering until the plot cell is rerun with
restored geometry.

LV 113 will remain absent from cost-colored panels until it has a complete
power-flow summary and the corresponding expansion analysis is rematerialized
without an explicit exclusion.

## Recommended follow-up

1. Add a feeder-audit option that returns and labels backbone, terminal-service
   and control lines instead of discarding terminal edges.
2. If an electrical backbone counterfactual is required, run Step 4 with scope
   backbone under new run names and rematerialize expansion.
3. Add a grid-aware peak term or transformer/import constraint to HEMS, then
   reassess representative-period storage boundary handling.
4. Retain component outputs for a small set of high-delta grids before another
   full run.
5. Keep real expansion comparisons coverage-matched until all failed SWF
   timesteps are resolved.
6. Persist or export synthetic envelope geometry with each run so historical
   plots do not depend on mutable PyLoVo working tables.
