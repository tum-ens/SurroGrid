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
| Standard LV reinforcement cable | NAYY_4_150, 270 A | 25,000 | EUR/km | about 20k EUR/km | Wintzek/PuBStadt first-vs-parallel cable distinction; value includes a modest current-cost uplift and is also used for smaller LV service/backbone classes. | yes |
| Standard LV reinforcement cable | NAYY_4_185, 313 A | 45,000 | EUR/km | about 40k-55k EUR/km | Wintzek/PuBStadt plus current-cost uplift. | yes |
| Standard LV reinforcement cable | NAYY_4_240, 357 A | 70,000 | EUR/km | about 60k EUR/km; material checks around 25k-28k EUR/km | Wintzek/PuBStadt, Agora/FfE, and WEI/GridSim. Also used as default for unknown larger LV cable type. | yes |
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

Cable capacity is evaluated on P100 loading: the maximum current represented by the complete power-flow horizon. No grid or asset percentile cutoff is applied to the expansion decision.

```text
installed_capacity_ka = existing_max_i_ka * existing_parallel
required_added_capacity_ka = max(peak_current_ka - installed_capacity_ka, 0)
```

The existing cable is retained regardless of its SWF or pylovo type. Every added circuit must be selected from the same catalogue on both network sources:

| Added cable | Ampacity | Cable-in-duct cost |
| --- | ---: | ---: |
| `NAYY_4_150` | 0.270 kA | 25,000 EUR/km |
| `NAYY_4_185` | 0.313 kA | 45,000 EUR/km |
| `NAYY_4_240` | 0.357 kA | 70,000 EUR/km |

All nonnegative integer combinations are considered. The selected combination is the least-cost option whose added nominal ampacity covers `required_added_capacity_ka`. Ties are resolved by fewer circuits, then lower excess capacity. This is a thermal screening approximation: actual current sharing between unlike parallel cables is not recalculated.

For real SWF grids, parallel physical line rows are normalized to construction corridors before this selection is applied. Rows are grouped only when they connect the same unordered bus pair and their recorded lengths differ by no more than 5%. The exported SWF station files do not contain route geometries, so agreement in endpoints and length is the conservative available proxy for a common physical route; same-bus rows with materially different lengths remain separate. Installed ampacity and row-level P100 currents are summed, the longest member length represents the corridor, and trenching is charged once. The result retains the member cable IDs and grouping method for auditability. Synthetic parallel circuits already use pandapower's `parallel` attribute and therefore enter the same calculation as one electrical corridor.

The settlement class controls the reopened-route cost:

```text
settlement_type = 1 -> rural
settlement_type = 2 -> semi-urban
settlement_type = 3 -> urban
missing settlement_type -> semi-urban
```

For a selected combination:

```text
duct_total = sum(selected cable-in-duct costs)
primary_duct_cost = highest cable-in-duct cost among selected cables
trenching_share = 1 - existing_duct_share

route_cost_eur_per_km =
    duct_total
    + trenching_share * (reopened_route_cost - primary_duct_cost)

component_cost_eur = component_length_km * route_cost_eur_per_km
```

The reopened-route cost represents the first cable plus civil works. Further selected circuits add their cable-in-duct cost, so trenching is charged once per reinforced route.

The result tables store:

- `reinforcement_150_count`
- `reinforcement_185_count`
- `reinforcement_240_count`
- `reinforcement_added_capacity_ka`
- `reinforcement_catalog`
- route cost basis, duct/trenching shares, and estimated cost

For synthetic display geometries that combine several raw electrical components, cable counts, added capacity, and cost are summed over those components. Critical-component fields still identify the electrically most loaded mapped component.

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

## Application to Synthetic and Real Networks

The same `de_lv_heuristic_2026` assumption row and the same asset-level formulas are applied to both network sources. The implementation differs only where source-specific identifiers and geometry must be read:

- Synthetic assets are joined through pylovo grid, line, and transformer identifiers.
- Real SWF assets are joined through `real_grid_case_id`, pandapower line indices, transformer ratings, and geometries from the exported station grid.
- Cable reinforcement uses the P100 current, existing installed capacity, common three-cable reinforcement catalogue, line length, settlement class, and duct/trenching blend on both sides.
- Transformer reinforcement uses the P100 apparent-power loading, installed rating, 50 kVA rounding step, and the same replacement/rebuild cost bins on both sides.

A real grid-stage is priced only when every simulated timestep converged. A grid with failed timesteps is stored as `incomplete`, receives no cable or transformer cost rows, and is excluded from aggregate costs rather than being interpreted as a zero-cost grid. Methodological exclusions are stored separately as `excluded`. Cost comparisons must therefore report both total cost and coverage; per-complete-grid values are useful when source coverage differs.

The current real-SWF implementation deliberately follows the same thermal-only boundary as the synthetic calculation. It does not add a separate cost for voltage violations, meshing, switchgear changes, or rONT installation.

## Interpretation Guidance

Use the default result as an order-of-magnitude spatial screening layer:

- Good for mapping which cable routes and transformer positions become thermally critical.
- Good for comparing regional cost pressure across scenarios.
- Good for sensitivity analysis around the existing-duct share, because this is the least observable but most cost-relevant cable parameter.
- Not suitable for final construction budgeting without checking route feasibility, trench reopening constraints, switchgear, protection, voltage constraints, station constraints, and DSO-specific planning rules.
- Not suitable for inferring electrical parallel cables from QGIS helper geometries. Raw electrical line components must remain the source for installed capacity and additional cable counts.

The output deliberately keeps `critical_component_cost_basis`, `critical_component_cost_eur_per_km`, `critical_component_duct_cost_eur_per_km`, `critical_component_reopen_cost_eur_per_km`, and `transformer_cost_basis` in the result tables so QGIS users can see why a feature received its cost.
