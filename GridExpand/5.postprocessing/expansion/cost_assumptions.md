# Expansion Cost Assumptions

This note documents the LV expansion cost assumptions used by `expansion/grid_expansion.py` and `expansion/schema.sql`.

Default assumption key: `de_lv_heuristic_2026`

The result is a transparent screening estimate for spatial postprocessing. It is not a construction offer, a DSO work-order cost, or a substitute for site-specific grid planning. The current heuristic prices overload-driven cable and transformer reinforcement in existing settlement structures. Flexibility costs are intentionally excluded because flexibility is represented by the separate `Post-flex` and `Post-no-flex` power-flow scenarios.

Important topology note: cable capacity must be derived from raw electrical pandapower/pylovo line components, not from `pylovo.lines_result_view`. The `lines_result_view` object is a QGIS-friendly display layer. It can contain artificial helper geometries, offset geometries, merged feeder chains, and visual lines that share geometry without being electrically parallel. It is suitable for displaying and joining final results in QGIS, but it must not be used as the source of installed electrical capacity.

## Benchmark, Defaults, and Sources

This table is the single source of truth for the numerical assumptions used by the current LV expansion heuristic. Rows marked as "used directly" are read by `schema.sql` / `grid_expansion.py`; rows marked as "not included" are retained only for interpretation or possible later sensitivity layers.

| Model component | Parameter / category | Default used | Unit | Literature benchmark or status | Source / rationale | Used directly in code? |
| --- | --- | ---: | --- | --- | --- | --- |
| Loading trigger | Expansion threshold | 100 | % nominal loading | Technical screening criterion, not a unit-cost literature value | Reinforcement is triggered only when simulated peak loading exceeds the installed nominal rating. This keeps the heuristic tied to pandapower/pylovo asset ratings and avoids adding an implicit planning margin. | yes |
| Duct availability | Existing duct share | 0.20 | share of reinforced LV routes | No robust published national share found | WEI/GridSim and Wintzek/PuBStadt recommend laying empty ducts during first reinforcement, but the review found no national statistic for existing spare ducts. Therefore this is an explicit scenario parameter, not an empirical statistic. | yes |
| Duct availability | Trenching share | 0.80 | share of reinforced LV routes | Complement of existing duct share | Derived as `1 - existing_duct_share`. With the default assumption, 80% of reinforced routes require trenching / street reopening. | yes, derived |
| LV cable reinforcement with trenching | Rural settlement (`settlement_type = 1`) | 90,000 | EUR/km | about 80k-120k EUR/km | Verteilnetzstudie Baden-Wuerttemberg reports rural NS cable costs around 80k EUR/km; Agora/FfE and WEI/GridSim support a broader 67k-120k EUR/km corridor. | yes |
| LV cable reinforcement with trenching | Semi-urban settlement (`settlement_type = 2`) | 100,000 | EUR/km | about 100k EUR/km | Verteilnetzstudie Baden-Wuerttemberg semi-urban NS cable benchmark. Also used as fallback if settlement type is missing. | yes |
| LV cable reinforcement with trenching | Urban settlement (`settlement_type = 3`) | 165,000 | EUR/km | about 125k-200k EUR/km; difficult cases up to about 380k EUR/km | Verteilnetzstudie Baden-Wuerttemberg, Wintzek/PuBStadt, and dena Verteilnetzstudie II. The 380k EUR/km value is treated as difficult-case boundary, not the default. | yes |
| LV parallel cable in existing duct / empty pipe | <=150 mm2 equivalent | 25,000 | EUR/km | about 20k EUR/km | Wintzek/PuBStadt first-vs-parallel cable distinction; value includes a modest current-cost uplift and is also used for smaller LV service/backbone classes. | yes |
| LV parallel cable in existing duct / empty pipe | 185 mm2 equivalent | 45,000 | EUR/km | about 40k-55k EUR/km | Wintzek/PuBStadt plus current-cost uplift. | yes |
| LV parallel cable in existing duct / empty pipe | 240 mm2 equivalent | 70,000 | EUR/km | about 60k EUR/km; material checks around 25k-28k EUR/km | Wintzek/PuBStadt, Agora/FfE, and WEI/GridSim. Also used as default for unknown larger LV cable type. | yes |
| Transformer capacity upgrade in existing station context | Replacement / upgrade to 400 kVA | 33,000 | EUR/unit | all-in replacement/model bins about 30k EUR; equipment-only lower bounds about 10k-15k EUR | WEI/GridSim all-in transformer replacement bins, checked against Wintzek and BW-study equipment lower bounds. | yes |
| Transformer capacity upgrade in existing station context | Replacement / upgrade to 630 kVA | 38,000 | EUR/unit | all-in replacement/model bins about 35k EUR; equipment-only lower bounds about 10k-15k EUR | WEI/GridSim all-in transformer replacement bins, checked against Wintzek and BW-study equipment lower bounds. | yes |
| Transformer capacity upgrade in existing station context | Replacement / upgrade to 800 kVA | 42,000 | EUR/unit | interpolated between 630 kVA and 1,000 kVA all-in bins | Interpolation between WEI/GridSim 630 kVA and 1 MVA bins, checked against Wintzek 800 kVA equipment anchor. | yes |
| Transformer capacity upgrade in existing station context | Replacement / upgrade to 1,000 kVA | 48,000 | EUR/unit | all-in replacement/model bins about 45k EUR; equipment-only lower bounds about 10k-15k EUR | WEI/GridSim all-in transformer replacement bin with uplift, checked against Wintzek equipment lower bounds. | yes |
| Transformer capacity rounding | Capacity step | 50 | kVA | Screening discretization, not a direct cost benchmark | Required transformer capacity is rounded up in 50 kVA increments before mapping to the 400/630/800/1,000 kVA cost bins. This avoids over-interpreting continuous simulated peak values while preserving the standard-size bin logic. | yes |
| Full ONS / station rebuild boundary | Station-level boundary case above 1 MVA | 100,000 | EUR/station | dena low / central / high values around 80k / 101k / 140k EUR; other LV station references about 45k-60k EUR | dena Verteilnetzstudie II and WEI/GridSim station rebuild checks. Used only when required transformer capacity exceeds the <=1 MVA replacement bins. | yes |
| rONT for remaining voltage issues | Voltage-control option | not included | EUR/unit | about +30k-45k EUR premium or about 45k EUR/unit depending on source boundary | BW-study, Wintzek/PuBStadt, and Agora/FfE. Kept as a possible later voltage-mitigation layer, not part of the current loading-cost estimate. | no |
| LV cable Opex | Annual operating cost | not included | % CapEx/a | about 2.5% of CapEx per year | Wintzek/PuBStadt. Excluded because the current output is CapEx screening only. | no |
| MS cable / MS station reinforcement | Medium-voltage level | not included | EUR/km or EUR/station | MS cables about 130k-160k EUR/km; MS/LV station upgrades can reach 1.3M-3.2M EUR | Verteilnetzstudie Baden-Wuerttemberg. Excluded because this analysis is LV-focused. | no |

## Cable Reinforcement Heuristic

Cable capacity is evaluated on raw electrical line components, ideally `pylovo.pandapower_line` joined to `surrogrid.powerflow_cable_summary`. The line current from the power-flow result is compared with the installed capacity of the same electrical component.

```text
installed_capacity_ka = max_i_ka * component_parallel
loading_percent = peak_current_ka / installed_capacity_ka * 100
required_parallel = ceil(peak_current_ka / max_i_ka)
additional_parallel = max(required_parallel - component_parallel, 0)
requires_expansion = additional_parallel > 0
```

The route-cost component is now explicitly split between routes with existing ducts and routes requiring trenching. The settlement class comes from `pylovo.postcode_result.settlement_type`:

```text
settlement_type = 1 -> rural reopened-route cost
settlement_type = 2 -> semi-urban reopened-route cost
settlement_type = 3 -> urban reopened-route cost
missing settlement_type -> semi-urban reopened-route cost
```

For a reinforced component, the one-time expansion cost is estimated as:

```text
if additional_parallel == 0:
    component_cost_eur = 0
else:
    duct_cost = selected cable-in-duct cost by cable std_type
    reopen_cost = selected reopened-route cost by settlement_type
    trenching_share = 1 - existing_duct_share

    component_cost_eur = component_length_km * (
        additional_parallel * duct_cost
        + trenching_share * (reopen_cost - duct_cost)
    )
```

This formula reflects the current modelling choice: all required parallel capacity is added in one expansion step. Therefore the trenching premium is charged once per reinforced route, while the cable/material component scales with the number of additional parallel equivalents. If a route already has ducts, only the cable-in-duct component remains.

For reporting, the result table stores the effective cost basis for the critical mapped component:

- `settlement_type`
- `line_existing_duct_share`
- `line_trenching_share`
- `critical_component_cost_eur_per_km`
- `critical_component_cost_basis`
- `critical_component_duct_cost_eur_per_km`
- `critical_component_reopen_cost_eur_per_km`

If several raw electrical components are mapped to one visible QGIS segment, `estimated_cost_eur` and `additional_parallel` are aggregated over all mapped components. The `critical_component_*` fields describe the component with the highest loading, and `component_cost_basis_count` / `component_std_type_count` indicate whether the visible segment combines heterogeneous assumptions.

## Transformer Heuristic

Transformer capacity is evaluated on the annual peak apparent power imported through the transformer position.

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

The transformer bins are interpreted as all-in screening values for capacity replacement or upgrade in an existing station context. The equipment-only literature values are lower, but they do not include the planning and brownfield integration scope represented here. The full station rebuild value is a boundary case, not a detailed station design.

## Voltage and rONT Consideration

The current expansion estimate is still primarily a thermal-capacity heuristic. It adds cable capacity when simulated current exceeds nominal installed capacity and replaces transformers when apparent-power loading exceeds the rated transformer capacity.

Voltage violations are intentionally not converted into cable costs by default. The cheapest technically plausible voltage remedy may be transformer tap adjustment, rONT installation, feeder reconfiguration, local cable reinforcement, transformer relocation, or a new feeder route. Treating every low-voltage case as an additional parallel cable on the most loaded segment would be too confident.

A useful next layer would be separate from the thermal result:

```text
thermal_cost_eur = cable_loading_cost_eur + transformer_loading_cost_eur
optional_voltage_mitigation_cost_eur = rONT_or_voltage_measure_proxy for grids with remaining voltage violations
```

For the paper comparison, this keeps the current result interpretable as loading-driven reinforcement cost. Remaining voltage issues can be reported as a separate voltage-mitigation need or sensitivity class. rONT costs are therefore noted in the table above, but not included in the default expansion-cost materialization.

## Source Notes

The cost review behind the current defaults emphasizes that civil works dominate LV cable costs. Cable material-only values around 25k-28k EUR/km are much lower than full underground-cable costs. The decisive modelling choice is therefore whether an added parallel cable can use an existing duct/empty pipe or whether a paved route must be reopened.

No robust official or academic percentage was found for how many existing German LV routes already have spare ducts. The `existing_duct_share = 0.20` default is therefore a transparent scenario assumption, not an observed national statistic. It should be varied in sensitivity runs.

Primary/context sources and how they are used:

- WEI202 / GridSim / Candas-related LV expansion modelling: supports recursive line reinforcement, transformer replacement steps, and the distinction between first reinforcement with trenching and later parallel cables in empty pipes.
- Wintzek 2021 / PuBStadt: provides urban LV planning guidance, first-line vs parallel-line cable cost benchmarks, empty-pipe strategy, rONT context, and Opex assumptions.
- Verteilnetzstudie Baden-Wuerttemberg 2017: provides rural, semi-urban, and urban NS cable benchmarks, transformer/station/rONT references, and MS-level boundary values.
- Bundesnetzagentur, Zustand und Ausbau der Verteilernetze 2022: macro-level context for distribution-grid expansion needs and the use of aggregated lower-voltage planning estimates. https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/VerteilerNetz/start.html
- Deutsche Energie-Agentur, dena-Verteilnetzstudie II Gutachten, 2025: recent DSO-informed boundary values for NS-line and station-cost assumptions. https://www.dena.de/fileadmin/dena/Publikationen/PDFs/2025/Gutachten_VNSII.pdf
- Consentec, Fraunhofer ISI, Fraunhofer IEG, Planung von Verteilnetzen der Zukunft, 2025: context for future planning practice and flexibility as a lever to reduce dimensioning-relevant peaks. https://consentec.de/app/uploads/2025/08/Consentec_ISI_IEG_BMWK_VN-Zukunft_AbschlussBer_20250627-1.pdf
- Agora/FfE and related FfE distribution-grid modelling: cable material and laying decomposition, transformer and rONT equipment anchors, and method context.
- WEI/GridSim 2025 modelling assumptions: cable material, installation, transformer replacement, and station rebuild consistency checks.
- dena 2012: historical DSO-validated lower-bound plausibility check; not used as the primary default because the cost base is old.

## Interpretation Guidance

Use the default result as an order-of-magnitude spatial screening layer:

- Good for mapping which cable routes and transformer positions become thermally critical.
- Good for comparing regional cost pressure across scenarios.
- Good for sensitivity analysis around the existing-duct share, because this is the least observable but most cost-relevant cable parameter.
- Not suitable for final construction budgeting without checking route feasibility, trench reopening constraints, switchgear, protection, voltage constraints, station constraints, and DSO-specific planning rules.
- Not suitable for inferring electrical parallel cables from QGIS helper geometries. Raw electrical line components must remain the source for installed capacity and additional cable counts.

The output deliberately keeps `critical_component_cost_basis`, `critical_component_cost_eur_per_km`, `critical_component_duct_cost_eur_per_km`, `critical_component_reopen_cost_eur_per_km`, and `transformer_cost_basis` in the result tables so QGIS users can see why a feature received its cost.
