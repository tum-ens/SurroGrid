# Street-Path Junction Refinement: Open Methodological Issue

## Status

Implementation is currently **on hold**. The structural difference documented here is real, but the available evidence does not show that adding more internal splitting joints would reduce the higher cable loading observed in the synthetic grids. This note records the problem, the previously considered topology changes, their limitations, and the analyses required before the generator is changed.

This decision note complements the measured results in `feeder_structure_comparison.md` and records the relevant KVS comparison findings directly below. It supersedes the earlier interpretation that a global union of all shortest street paths should directly be introduced as the next generator change.

## Observed Structural Difference

The paired synthetic and real SWF networks have broadly comparable transformer capacities, demand-weighted cable capacities, and outgoing feeder counts. Parallel synthetic cables are represented correctly in pandapower, and the finalized synthetic feeder sections are sized for their calculated downstream simultaneous current.

Nevertheless, the retained SWF grids contain substantially more internal electrical splitting nodes:

| Median metric | Synthetic | Real SWF |
|---|---:|---:|
| Electrical splitting nodes per grid | 3.5 | 11.0 |
| Load-weighted splitting depth | 1.56 | 2.41 |
| Distance from preceding split | 99 m | 55 m |
| Downstream demand per split | 127 MWh/a | 63 MWh/a |
| Downstream demand per incoming ampere | 360 kWh/a/A | 181 kWh/a/A |

The explicit SWF KVS layer does not explain most of this difference. Synthetic splitting nodes already resemble the explicitly labelled SWF KVS nodes reasonably well. The larger discrepancy is caused by 818 unlabelled SWF splitting nodes, which appear to represent ordinary cable joints or trunk-and-tap branch points. These nodes are more frequent, usually have two downstream branches, and commonly retain one dominant trunk continuation with a smaller lateral branch.

Pylovo currently represents many street vertices only as cable geometry. An intermediate street vertex without a building connection is normally not created as an electrical bus. If routed cable paths diverge at such a vertex, that physical-looking divergence is therefore not necessarily represented as an electrical splitting node.

## Current Pylovo Method

The current generator:

1. Selects the furthest remaining connection-point path.
2. Groups connection points into a logical branch until `FEEDER_SPLIT_MAX_CURRENT_KA` is reached.
3. Allows a later branch to reuse an already installed connection point when it is farther from the transformer than `MIN_SHARED_PREFIX_LENGTH_M`.
4. Finalizes the resulting radial branch topology.
5. Divides the topology into sections between transformer, splitting, and terminal nodes.
6. Sizes every section for its calculated downstream simultaneous demand, including parallel cables where required.

This means that branch reuse is currently possible only at connection points already represented in the electrical model. A street-path divergence located between those connection points remains geometry rather than an explicit cable-joint node.

## Previously Proposed Street-Path Refinement

The proposed refinement was intended to make the electrical feeder topology consistent with the street routes that pylovo already calculates:

1. Preserve the existing logical feeder groups and independent transformer feeders.
2. Expand every planned backbone connection into its complete routed street path.
3. Overlay paths belonging to the same logical feeder.
4. Identify their exact common prefixes and divergence vertices.
5. Promote meaningful divergence vertices to electrical planning nodes.
6. Keep ordinary degree-two street vertices as geometry only.
7. Build cable sections between the retained planning nodes.
8. Recalculate downstream simultaneous current and size all sections after the refined topology is complete.

For example, two branches in the same logical feeder may use:

```text
Transformer -> A -> B -> C1
Transformer -> A -> B -> C2
```

The refined electrical representation would be:

```text
                         -> C1
Transformer -> A -> B --
                         -> C2
```

The shared upstream section would carry the combined downstream demand, while the two downstream sections would carry their respective branch demands.

## Essential Constraint: Independent Feeders Must Remain Independent

Geometric overlap does not prove that cables are electrically shared. Two independently planned transformer feeders may follow the same road while remaining separate physical cables:

```text
Transformer -----------> C1
Transformer -----------> C2
```

A refinement must therefore never merge paths solely because their street geometries overlap. Every branch attached directly to the transformer would retain its logical feeder identity. Only paths already assigned to the same logical feeder could share a promoted junction node.

Without this constraint, a global shortest-path union could reduce the number of independent feeder roots, aggregate demand onto common trunks, and substantially change the current pylovo methodology.

## Why More Junctions Do Not Necessarily Reduce Cable Loading

The structural audit originally associated fewer synthetic splitting nodes with higher downstream demand per installed ampere. This is a correlation, not proof that the missing joints cause the loading difference.

Consider two independent branches, each carrying 100 A and dimensioned with a 142 A cable:

```text
Independent loading = 100 A / 142 A = 70% per cable
```

If they were incorrectly combined into one shared 242 A upstream cable:

```text
Shared loading = 200 A / 242 A = 83%
```

The shared section is more heavily loaded. Discrete cable sizes and parallel cables can occasionally produce the opposite result, but there is no general reason for additional sharing to lower utilization.

A correct street-path refinement could:

- increase loading on a shared upstream section;
- reduce demand on downstream branch-specific sections;
- change the distribution of cable rows and section lengths;
- leave critical loading almost unchanged;
- occasionally reduce utilization when the aggregated demand selects a larger cable class or parallel capacity.

Its defensible purpose is therefore **topological consistency**, not automatic loading reduction.

## More Direct Approach to Excessive Cable Loading

If the objective is to prevent unrealistic demand concentration, the more direct control is a post-topology section check:

1. Finalize the candidate radial feeder topology.
2. Calculate downstream simultaneous current for every section.
3. Identify highly utilized shared prefixes.
4. Determine whether the affected branch should remain shared or become an independent transformer feeder.
5. Use an independent feeder only where a separate electrical route is methodologically justified.
6. Otherwise retain the shared topology and dimension it with an adequate cable or parallel capacity.
7. Verify the final topology with the actual power-flow demand profiles.

The generator already performs most of the section-current calculation and capacity selection. The unresolved methodological question is when a concentrated shared prefix should trigger another independent feeder rather than only a larger or parallel cable.

## Audit Required Before Reconsidering Implementation

Before implementing the street-path refinement, current routed paths should be classified into:

1. **Independent feeder overlap:** separate transformer feeders whose geometries happen to follow the same street; these must remain electrically separate.
2. **Hidden divergence within one logical feeder:** paths that are already intended to share a feeder but diverge at a street vertex not represented as an electrical node.
3. **Explicit shared topology:** branches whose common prefix and split are already represented correctly.

For each category, the audit should quantify:

- number of affected grids and feeders;
- shared route length;
- downstream simultaneous current;
- installed and required capacity;
- expected change in section count and splitting depth;
- counterfactual loading before and after junction refinement;
- effect on total cable-route and cable-material length.

Only the second category is a clear candidate for street-path junction refinement.

## Possible Future Implementation

If the audit demonstrates a material number of hidden same-feeder divergences, the smallest suitable implementation would:

- retain `_plan_backbone_branches()` as the logical feeder planner;
- assign a stable feeder identity to every planned branch;
- add a local refinement pass after branch planning;
- retrieve full street paths only for planned backbone connections;
- promote only same-feeder divergence vertices;
- create electrical buses only for original connection points and promoted junctions;
- pass finalized directed feeder edges to the existing section-grouping and cable-sizing logic;
- leave transformer placement, demand allocation, connection-point aggregation, service connections, cable selection, parallel handling, and pandapower export otherwise unchanged.

No new tuning parameter should be introduced initially. The refinement should preserve every genuine same-feeder divergence and simplify only degree-two vertices without demand or equipment.

## Decision

The street-path junction refinement is not implemented at this stage. The current evidence establishes a representational difference between synthetic and SWF feeder structures, but it does not establish that introducing more shared cable sections would resolve the synthetic cable-loading deviation.

The next justified step, if this topic is resumed, is the route-overlap classification and counterfactual loading audit described above. Generator changes should follow only if that audit identifies a meaningful population of hidden divergence points within existing logical feeders.
