# Expansion Cost Assumptions

This note documents the cost assumptions used by `expansion/grid_expansion.py` and `expansion/schema.sql`.

Default assumption key: `de_lv_heuristic_2026`

These values are transparent screening assumptions for spatial postprocessing. They are not construction offers, DSO work-order costs, or a substitute for site-specific grid planning. The capacity heuristic remains deliberately simple: reinforce only where the simulated peak exceeds nominal existing capacity.

Important topology note: cable capacity must be derived from the raw electrical pandapower/pylovo line components, not from `pylovo.lines_result_view`. The `lines_result_view` object is a QGIS-friendly display layer. It can contain artificial helper geometries, offset geometries, merged feeder chains, and visual lines that share geometry without being electrically parallel. It is suitable for displaying and joining final results in QGIS, but it must not be used as the source of installed electrical capacity.

## Current Defaults Used in the Calculation

| Parameter | Value used | Unit | Use in this model |
| --- | ---: | --- | --- |
| Expansion threshold | 100 | % nominal loading | No target loading margin. Reinforcement is triggered only when simulated peak loading exceeds the existing rating. |
| Parallel LV cable <=150 mm2 in existing route/duct | 25,000 | EUR/km | Applied per overloaded raw electrical cable component whose `std_type` indicates <=150 mm2, including small service cable classes. |
| Parallel LV cable 185 mm2 in existing route/duct | 45,000 | EUR/km | Applied per overloaded raw electrical cable component when `std_type` indicates 185 mm2. |
| Parallel LV cable 240 mm2 in existing route/duct | 70,000 | EUR/km | Applied per overloaded raw electrical cable component when `std_type` indicates 240 mm2. Also used as the default for unknown cable size. |
| Reopened-route rural reference | 90,000 | EUR/km | Stored in the assumption table as context/sensitivity, not used by the default cost formula. |
| Reopened-route suburban reference | 95,000 | EUR/km | Stored in the assumption table as context/sensitivity, not used by the default cost formula. |
| Reopened-route urban reference | 165,000 | EUR/km | Stored in the assumption table as context/sensitivity, not used by the default cost formula. |
| All-in transformer replacement to 400 kVA | 33,000 | EUR/unit | Used when required transformer capacity is > existing rating and <=400 kVA. |
| All-in transformer replacement to 630 kVA | 38,000 | EUR/unit | Used when required transformer capacity is >400 and <=630 kVA. |
| All-in transformer replacement to 800 kVA | 42,000 | EUR/unit | Used when required transformer capacity is >630 and <=800 kVA. |
| All-in transformer replacement to 1,000 kVA | 48,000 | EUR/unit | Used when required transformer capacity is >800 and <=1,000 kVA. |
| Full station rebuild boundary case | 100,000 | EUR/station | Used when required transformer capacity exceeds 1,000 kVA. |
| Transformer capacity step | 50 | kVA | Required transformer capacity is rounded up to this increment. |

## Why These Defaults Fit This Heuristic

The model does not replace existing cables with a newly routed underground line. It estimates the capacity/cost impact of adding enough parallel cable capacity on already represented LV cable components. Therefore the default cable cost should not be the broad dense-urban reopened-trench value. The direct brownfield parallel-cable values are the best match for this specific calculation.

The reopened-route values are still important, especially for dense urban streets, but they are a sensitivity/boundary case. If the result is later interpreted as a full trench reopening or route rebuild, use the reopened-route references instead of the direct-parallel defaults.

For transformers, the previous `additional_kVA * EUR/kVA + fixed handling` rule was too abstract and too low for brownfield replacement. The model now uses all-in replacement bins, which better match the planning question: if a local transformer is overloaded, what replacement size is implied and what is the approximate all-in replacement cost?

## Capacity Heuristic

Cable capacity is evaluated on raw electrical line components, ideally `pylovo.pandapower_line` joined to `surrogrid.powerflow_line_result`. The line current from the power-flow result is compared with the installed capacity of the same electrical component.

```text
installed_capacity_ka = max_i_ka * component_parallel
loading_percent = peak_current_ka / installed_capacity_ka * 100
required_parallel = ceil(peak_current_ka / max_i_ka)
additional_parallel = max(required_parallel - component_parallel, 0)
requires_expansion = additional_parallel > 0
```

Cable cost:

```text
component_cost_eur = additional_parallel * component_length_km * selected_parallel_cable_cost_eur_per_km
estimated_cost_eur = sum(component_cost_eur) for the displayed QGIS feature
```

The selected line cost for the most critical mapped raw component is stored per row in:

- `critical_component_cost_eur_per_km`
- `critical_component_cost_basis`

If several raw electrical components are mapped to one visible QGIS segment, `estimated_cost_eur` and `additional_parallel` are aggregated over all mapped components. The `critical_component_*` fields describe the component with the highest loading, and `component_cost_basis_count` / `component_std_type_count` indicate whether the visible segment combines heterogeneous component assumptions.

`pylovo.lines_result_view` should only be used after the raw electrical calculation, to attach final component or aggregated component results to QGIS-friendly geometries. In particular, do not derive `existing_parallel`, `required_parallel`, or installed capacity from a merged helper row in `lines_result_view`. A visual helper can combine multiple feeder pieces using display-oriented rules such as `max(parallel)` and `sum(length_km)`, which can overstate or misassign electrical capacity when independent lines share a lane or when predefined pandapower parallel cables are not visualized as separate geometries.

Transformer capacity:

```text
peak_s_kva = sqrt(P_mW^2 + Q_mvar^2) * 1000
loading_percent = peak_s_kva / rated_kva * 100
required_transformer_kva = ceil(peak_s_kva / transformer_capacity_step_kva) * transformer_capacity_step_kva
additional_transformer_kva = max(required_transformer_kva - rated_kva, 0)
requires_expansion = additional_transformer_kva > 0
```

Transformer cost:

```text
if required_transformer_kva <= existing_rated_kva: 0 EUR
elif required_transformer_kva <= 400: 33,000 EUR
elif required_transformer_kva <= 630: 38,000 EUR
elif required_transformer_kva <= 800: 42,000 EUR
elif required_transformer_kva <= 1000: 48,000 EUR
else: 100,000 EUR boundary case
```

The selected transformer basis is stored per row in `transformer_cost_basis`.

## Literature Synthesis Behind the Defaults

| Cost item | Literature values from synthesis | Value selected here | Reason |
| --- | ---: | ---: | --- |
| Parallel LV cable 150 mm2, existing route/duct | Wintzek: +20k EUR/km; 2025 qualitative uplift 20k-35k EUR/km | 25k EUR/km | Direct match to this model's added-parallel-cable interpretation. |
| Parallel LV cable 185 mm2, existing route/duct | Wintzek: +40k EUR/km; 2025 qualitative range 40k-55k EUR/km | 45k EUR/km | Direct source plus modest current-cost uplift. |
| Parallel LV cable 240 mm2, existing route/duct | Wintzek: +60k EUR/km; Agora/FfE and WEI/GridSim material benchmarks about 25k-28k EUR/km | 70k EUR/km | Direct brownfield value with 2025 uplift and material-cost consistency check. |
| Reopened rural brownfield cable route | Agora/FfE decomposition and WEI/GridSim installation anchor, roughly 67k-120k EUR/km | 90k EUR/km | Stored as sensitivity/context. |
| Reopened suburban brownfield cable route | Agora/FfE rural/suburban laying plus WEI/GridSim German-average installation, roughly 67k-130k EUR/km | 95k EUR/km | Stored as sensitivity/context. |
| Reopened dense urban brownfield cable route | Agora/FfE about 115k EUR/km, Wintzek urban all-in 240 mm2 about 200k EUR/km, dena 2025 Musterhausen NS-line benchmark 182k EUR/km | 165k EUR/km | Stored as sensitivity/context for paved-street reopening. |
| DSO-survey difficult NS line boundary | dena 2025: 80k / 182k / 380k EUR/km low / Musterhausen / high | 182k EUR/km reference only | Useful boundary check, not default because scope is broader than added parallel cable. |
| Cable material only, NAYY-J 4x240 | Agora/FfE 28k EUR/km; WEI/GridSim 25k EUR/km | 28k EUR/km reference only | Confirms material is not the dominant driver. |
| Transformer equipment only 630-1000 kVA | Wintzek 10k/12.5k/15k EUR; Agora/FfE 15k EUR; dena 2012 10k EUR | not used directly | Equipment-only lower bound, insufficient for brownfield replacement. |
| All-in transformer replacement to 400 kVA | WEI/GridSim 30k EUR plus planning uplift | 33k EUR | Direct all-in replacement bin. |
| All-in transformer replacement to 630 kVA | WEI/GridSim 35k EUR, checked against equipment lower bounds | 38k EUR | Direct all-in replacement bin with uplift. |
| All-in transformer replacement to 800 kVA | Interpolation between 630 kVA and 1 MVA; Wintzek 800 kVA equipment 12.5k EUR | 42k EUR | Interpolated all-in replacement bin. |
| All-in transformer replacement to 1,000 kVA | WEI/GridSim 45k EUR, checked against Wintzek equipment 15k EUR | 48k EUR | Direct all-in replacement bin with uplift. |
| rONT equipment / incremental cost | Wintzek 21.2k-27.5k EUR equipment, Agora/FfE 28k EUR, WEI/GridSim +4k EUR all-in premium | not used by default | rONT can be a voltage/flexibility measure, but this capacity-only overload heuristic does not decide voltage-control measures. |
| Full ONS renewal / rebuild | dena 2025 80k/101k/140k EUR station incl. transformer; WEI/GridSim rebuild 100k EUR | 100k EUR | Boundary case when required transformer size exceeds the simple <=1 MVA replacement bins. |

## Source Notes

The source synthesis behind these defaults emphasizes that civil works dominate LV cable cost. Cable material-only values around 25k-28k EUR/km are much lower than full underground-cable costs. Therefore the decisive modeling choice is whether an added parallel cable can use an existing route/duct or whether a paved route must be reopened.

The default calculation assumes the former because the model output is an additional-parallel capacity estimate on existing pylovo electrical cable components. For urban construction budgeting, the `line_reopen_urban_eur_per_km = 165000` value should be used as a sensitivity.

Primary/context sources and how they are used:

- Bundesnetzagentur, "Zustand und Ausbau der Strom-Verteilernetze": context for why distribution-grid expansion is driven by renewable generation, electromobility, and heat-sector electrification. The page also states that the 2024 distribution-grid expansion plans use regional scenarios and include the legal 2045 climate-neutrality targets. https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/VerteilerNetz/start.html
- Deutsche Energie-Agentur, dena-Verteilnetzstudie II Gutachten, 2025: recent DSO-informed boundary values for broad NS-line and station-cost assumptions. Table 12 reports low / Musterhausen / high cost assumptions of 80k / 182k / 380k EUR per km for NS lines and 80k / 101k / 140k EUR per network station including transformer. These values are used as conservative boundary checks, not as the default direct-parallel-cable cost. https://www.dena.de/fileadmin/dena/Publikationen/PDFs/2025/Gutachten_VNSII.pdf
- Consentec, Fraunhofer ISI, Fraunhofer IEG, "Planung von Verteilnetzen der Zukunft", 2025: context for future planning practice and flexibility. The study frames flexibility as a lever that can reduce dimensioning-relevant peaks and long-term grid expansion quantities, rather than as a direct reduction in civil-work unit costs. https://consentec.de/app/uploads/2025/08/Consentec_ISI_IEG_BMWK_VN-Zukunft_AbschlussBer_20250627-1.pdf
- Statistisches Bundesamt, construction/civil-engineering price indices: context for treating older cable and civil-work values cautiously and applying a qualitative 2025 planning uplift where the direct source is older. https://www.destatis.de/DE/Themen/Wirtschaft/Konjunkturindikatoren/Preise/bpr210.html

Additional source names used in the synthesis and retained for traceability:

- Wintzek 2021: direct brownfield parallel-cable values and transformer equipment anchors. Used as the strongest match for the default existing-route parallel-cable cost tiers.
- Agora/FfE 2023: cable material and laying decomposition, transformer and rONT equipment anchors, and rONT/flexibility context. Used for consistency checks and reopened-route sensitivity values.
- WEI/GridSim 2025: recent modelling assumptions for cable material, installation, transformer replacements, and station rebuilds. Used for all-in transformer replacement bins and reopened-route sensitivity checks.
- dena 2012: older DSO-validated lower-bound plausibility check. Not used as the primary default because the cost base is historical.
- FfE MONA and related FfE distribution-grid modelling: retained for methodological traceability around LV-grid simulations and technology options.

## Voltage Consideration

The current line expansion estimate is a thermal-capacity heuristic: it adds parallel cable capacity only when the simulated line current exceeds the installed nominal capacity. Voltage violations are intentionally not converted directly into segment-level cable costs yet, because the cheapest technically plausible remedy may be a different transformer tap, transformer relocation, feeder reconfiguration, local cable reinforcement, or a new feeder route. Treating every low-voltage case as an additional parallel cable on the most loaded segment would therefore be too confident.

The recommended next layer is to keep the thermal result as the base cost and add a separate voltage-stress classification per grid or supply envelope. For grids with voltage minima below the accepted threshold but no thermal overload, the first robust estimate should flag the affected supply area and report voltage-mitigation need as a sensitivity class. A later segment-level voltage heuristic should only be added once the downstream path to the critical bus is reconstructed reliably, so candidate reinforcement sections can be ranked by path impedance contribution rather than by loading alone.

## Interpretation Guidance

Use the default result as an order-of-magnitude spatial screening layer:

- Good for mapping which cable routes and transformer positions become critical.
- Good for comparing regional cost pressure across scenarios.
- Not suitable for final construction budgeting without checking route feasibility, trench reopening, switchgear, protection, voltage constraints, station constraints, and DSO-specific planning rules.
- Not suitable for inferring electrical parallel cables from QGIS helper geometries. Raw electrical line components must remain the source for installed capacity and additional cable counts.

The output deliberately keeps `critical_component_cost_basis`, `critical_component_cost_eur_per_km`, and `transformer_cost_basis` in the result tables so QGIS users can see why a feature received its cost.
