# Expansion Cost Assumptions

This note documents the rules and cost assumptions of `gridexpand expansion` ([Step 5](steps/5_postprocessing.md)).

**Rules.** `gridexpand expansion` applies the staged rules of `src/gridexpand/analysis/expansion/staged.py`
(rule set `staged_2026`) with the assumption row `de_lv_staged_2026` (default) or one of its sensitivity rows. The
former overload heuristic was retired on 2026-09-29 (see [History](#history)).

**One implementation for both sources.** The rules run in Python for synthetic and real grids alike:
- `synthetic_materialization.py`: the SQL keeps only the scope and the mapping to the visible pylovo lines.
- `real_materialization.py`: the real SWF and ÜZW grids.

**Where the numbers come from.** The rows are seeded by migration `0007_staged_expansion.sql` (`ON CONFLICT DO
NOTHING`, so manual edits survive). Every row documents each parameter in the JSON column `parameter_provenance`:
value, unit, method and sources. The tables below are the readable form of that
column, and a test (`tests/analysis/test_staged_assumption.py`) keeps both in step.

**Methods used for the values:**
- **Central value:** the median of the listed sources, rounded.
- **Upper value (conservative):** the upper end of recent sources, used where prices have risen since the sources'
  price base or where a low value would understate costs.
- **Norm value:** a cable rating of DIN VDE 0276-603, read from the named secondary source because the norms are
  paywalled.
- **Decision:** a choice of the project owner (decisions D1–D10 of 2026-09-29), with the literature that supports
  it.
- **Assumption:** no source found; the derivation is stated and a sensitivity row covers it.

All prices keep the price base of their sources (2007–2025); nothing is inflation-adjusted.

**What the result is.** A transparent screening estimate for spatial postprocessing, not a construction offer or a
DSO work order. Flexibility costs are excluded: flexibility is represented by the model cases (`post-hems-*`,
`post-inflex-heuristic`).

**Topology note.** Cable capacity must come from the raw electrical pandapower/pylovo line components, not from
`pylovo.lines_result_view`. That view is a QGIS display layer with helper and merged geometries.

## Rules (`staged_2026`)

The rules follow the order of measures of Niederle et al. (EnInnov 2026, TUM/SWM) and German planning practice:
dena 2012, Verteilnetzstudie BW 2017, FfE/Agora 2023, Kerber 2011, PuBStadt 2021. They keep the load-based core:
the P100 values of the Step 4 time series.

### Stages

The inputs per grid are:
- the P100 current and installed capacity of each route;
- the P100 station import and the station rating;
- the minimum voltage of each evaluated bus (Step 4 summaries);
- the grid topology (`topology.py`).

A route is a synthetic line component, or a real cable corridor (parallel rows between the same buses, lengths
within 5 %).

**1. Station.**
- S_P100 ≤ τ·S_r: no measure.
- S_P100 ≤ τ·S_max: exchange for the smallest standard size (100, 160, 250, 400, 630, 800, 1000 kVA) that carries
  the load. S_max is 1000 kVA in every settlement type (the columns per settlement type allow other limits in
  sensitivity rows).
- Otherwise the grid is **over the limit** by S_P100 − τ·S_max:
  - its transformer is exchanged up to S_max;
  - the rest goes to stage 4.
- A station whose unit is already above S_max keeps it as its ceiling.

**2. Routes.** Each overloaded route gets the least-cost combination of parallel NAYY 4×150/185/240 cables, with at
most `line_max_added_cables`.
- **Ratings.** All cables of a route, existing and added, count at their nominal ratings (no grouping derating,
  see below).
- **Route cost.** On a trenched route the most expensive added cable pays the settlement's trench cost
  `line_reopen_*` (trench incl. that cable); further cables pay their increment `line_parallel_*`. The share
  `line_existing_duct_share` of a route lies in existing ducts and needs no trench.
- **Escalation.** A route that the cap cannot resolve escalates the grid. Its excess is √3·0.4 kV times the
  current that is still missing with the maximum number of cables.
- **Service lines.** A service line is the edge into a load bus with at most one line neighbour, the terminal
  service connection of `comparison_backbone_scope`. It is costed as `service_cost_eur` and stays out of the total
  (decision D9).

**3. Panels.**
- Panels = existing station outlets (cables) + cables added on outlet routes, one NH way each.
- Reported per grid. An escalation only if `panel_trigger` (default off, decision D5).

**4. Grids over the limit or escalated** (per analysis, all grids of the run and region together).
- **(a) Load transfer.**
  - *Neighbours:* grids whose buses come within `load_transfer_adjacency_m`. Bus coordinates are in EPSG:25832;
    pylovo's WGS84 coordinates are projected.
  - *Spare capacity* of a neighbour: τ·rating after its own stage 1, minus its P100.
  - *Order:* grids by descending excess; the largest spare is used first, and each kVA only once.
  - *If the spare covers the excess:* `load_transfer_eur` plus the own exchange to S_max.
- **(b) Whole new substations.**
  - The remaining neighbouring grids form a cluster that shares ⌈Σ excess / (τ·S_new)⌉ new substations.
  - Each substation costs `new_station_eur` + `new_station_mv_loop_in_km`·`mv_cable_eur_per_km` +
    `new_station_lv_connection_km`·trench cost of the settlement.
  - The cluster cost is allocated to the grids by excess.
- **In both cases:**
  - routes that need two or more added cables, and unresolved routes, count as relieved (measure
    `relieved_by_transfer` / `relieved_by_new_station`, no cable cost);
  - routes with one added cable keep their cost.

**5. Residual voltage.** Step 4 already sets the busbar to 0.96 pu and uses up to two off-load tap steps
(`powerflow/station_voltage.py`, the convention of pylovo). A bus still below 0.90 pu:
- counts as resolved if a measure touches a route on its path to the station, or if the grid gets a new
  substation;
- otherwise gets an rONT, if the rONT headroom 0.96·(1 + range) − busbar voltage lifts it to 0.90 pu. The rONT
  costs `ront_premium_eur` where stage 1 buys a new transformer anyway. Where it replaces the kept transformer, it
  costs a full unit: the conventional price of the station's size plus the premium (PuBStadt, BW 2017 and dena 2012
  price rONTs as full units). The 100, 160 and 250 kVA units share one price, so a small station pays for the
  smallest rONT of the product data (160 kVA);
- otherwise gets a feeder split: a new NAYY 4×240 route from the station to 2/3 of the distance to the farthest
  critical bus, priced like a one-cable reinforcement.

Runs solved before the station voltage existed (no `lv_busbar_vm_pu`) get `not_assessed`.

### Parameters of `de_lv_staged_2026`

PuBStadt 2021 and dena 2012 are cited by printed page (their PDF pages are 21 and 22 higher). Prices keep the price
base of their source; nothing is inflation-adjusted.

| Parameter (column) | Value | Method | Sources |
|---|---:|---|---|
| `transformer_planning_limit` | 1.0 × Sr | decision D4, DSO practice (all studies agree) | Niederle et al. 2026, p. 7–8; PuBStadt 2021 §7.5.2, p. 50 (100 % of Sr); dena 2012, p. 89–90; BW 2017, p. 37; FfE/Agora 2023, p. 53, 111 (reinforced even after short overloads); Nobis 2016 (no (n-1) in LV). IEC 60076-7 allows up to 1.5 p.u. normal cyclic loading, so 1.1/1.2 are sensitivities |
| `station_max_kva_rural` / `_suburban` / `_urban` | 1000 kVA | decision D2, revised 2026-09-29: as FfE/Agora 2023 | FfE/Agora 2023, p. 85, 111 (1000 kVA in every settlement type, after PuBStadt and DSO feedback; above that an additional station); PuBStadt 2021, p. 157 (1000 kVA or 2 × 630 kVA in urban grids); dena 2012, BW 2017, Agora 2019 / NRW 2021 (a second or more parallel units); Schneider Planungskompendium (MV fuse-switch protection up to about 1000 kVA, VDE 0670-402); Schneider EIG (urban substations with one or two 1000 kVA units). Stricter: PuBLIK 2016 / Harnisch 2019 (rural: new station from 800 kVA), Kerber 2011 (630 kVA), Munich (630/800 kVA). The exchange cost has no larger station housing, which small rural stations may need |
| `transformer_replace_100_eur` / `_160_eur` | 8,000 € | conservative: no 100 kVA source; the 250 kVA value, above Kerber's 160 kVA value | Kerber 2011, p. 142: 160 kVA 6.5 k€ (2007) |
| `transformer_replace_250_eur` | 8,000 € | upper value of the size-specific sources | Kerber 2011: 8.0 k€ (2007); FfE MONA 2030: 7.0 k€ (2015); Verteilnetzstudie Hessen 2018: 7 k€ (2015). FfE/Agora 2023 charges 15 k€ per exchange whatever the size (upper bound) |
| `transformer_replace_400_eur` | 10,000 € | upper range, rounded | Kerber 2011: 10.5 k€; MONA: 8.5 k€; Hessen 2018: 9 k€; BMWi 2014: 8 (6–10) k€; FfE/Agora 2023: 15 k€ per exchange (upper bound) |
| `transformer_replace_630_eur` | 12,000 € | central value (median of ten sources) | BMWi 2014, MONA, Hessen 2018: 12 k€; PuBStadt 2021, p. 213: 10 k€; dena 2012, p. 147: 10 k€ (2011); Kerber 2011: 14 k€; Schlömer 2017: 11.25 k€; Arnold 2019: 8.6 k€; FfE/Agora 2023 (any size) and BW 2017: 15 k€ |
| `transformer_replace_800_eur` | 12,500 € | central value (median of three sources) | PuBStadt 2021, p. 213: 12.5 k€; Arnold 2019: 9.8 k€ (2014); FfE/Agora 2023: 15 k€ (any size) |
| `transformer_replace_1000_eur` | 15,000 € | upper value of four sources | PuBStadt 2021, p. 213 and FfE/Agora 2023: 15 k€; Schlömer 2017: 13.25 k€; Arnold 2019: 10.9 k€ |
| `line_parallel_150/185/240_eur_per_km` | 25/45/70 k€/km | PuBStadt value rounded up by 5/5/10 k€/km (kept from `heuristic_2026`) | PuBStadt 2021, p. 212: +20/40/60 k€/km for a parallel cable in the same trench, cable and installation (2021). Lower bound: FfE/Agora 2023, p. 112–113: 28 k€/km material for NAYY 4×240. Upper bound: dena 2012 and BW 2017 price every added cable at the full trench rate |
| `line_reinforcement_150/185/240_max_i_ka` | 0.270/0.313/0.357 kA | kept for consistency with the grid data | SimBench/pandapower (270/357 A); DIN VDE 0276-603 via Siemens TIP 12 (2023): 275/313/364 A at m = 0.7 |
| `line_reopen_rural_eur_per_km` | 80 k€/km | BW 2017, the only source with rural, semi-urban and urban values (80/100/120) | BW 2017, p. 91: 80; dena 2012, p. 147: 60 (≤ 500 inhabitants/km², 2011); FfE/Agora 2023, p. 113: 67 (39 laying + 28 material). Median of the three rural sources: 67 |
| `line_reopen_suburban_eur_per_km` | 100 k€/km | central value: the two sources with a semi-urban class | BW 2017, p. 91: 100; Gutachten NRW 2021, Tab. 7-3: 100; dena 2012: 60/100 (≤/> 500 inhabitants/km², 2011); FfE/Agora 2023: 67 (rural and suburban) |
| `line_reopen_urban_eur_per_km` | 165 k€/km | central value: median of the six sources with 2021–2025 prices (115, 139, 150, 182, 250, 312) = 166, rounded | FfE/Agora 2023: 115; ef.Ruhr/EWI 2024: 139; PuBStadt 2021, p. 212: 150/175/200 (NAYY 150/185/240); dena VNS II 2025, p. 276: 182 (80–380); NAP 2024 (derived): SWM about 250, enercity about 312. Older: BW 2017: 120; dena 2012: 100 |
| `line_existing_duct_share` | 0.20 | assumption (no source) | PuBStadt recommends empty ducts but gives no share; `--line-existing-duct-share` for 0 / 0.5 |
| `line_max_added_cables` | 3 | decision D6 (revised 2026-09-29: nominal ratings, no derating) | Kerber 2011 (at most 3 parallel cables per street side); Agora 2019 / NRW 2021 (at most 4 parallel LV cables). No other German study caps the number: PuBStadt usually needs one more cable; BW 2017 and FfE/Agora 2023 allow any number (FfE: 1.0–1.3 cables per reinforced trench-km on average, derived) |
| `panel_max`, `panel_trigger` | 12, off | catalogue practice; decision D5 | Niederle 2026, p. 17 (8 or 12 panels); Jean Müller 2012 (8/10/12 NH2 ways); Kenter (12 NH2); no standard limits panels |
| `new_station_eur` | 85,000 € | central value: median of the eight German sources with 2020–2025 prices (51, 57.5, 83.5, 84, 86.5, 101, 150.5, 225 k€) | dena VNS II 2025, Gutachten p. 276: 101 k€ (80–140); ef.Ruhr/EWI 2024: 86.5 k€; NAP 2024 (derived): Netze BW 77–91, e-netz Südhessen 74–93, enercity about 150 k€, Munich 200–250 k€ per site; PuBStadt 2021, p. 213: 45 k€ building + 10–15 k€ transformer; FfE/Agora 2023: 51 k€ (probably incl. connection cables); BW 2017: 60 k€ incl. transformer; Langfristszenarien 3: 50 k€ (2018 prices); dena 2012: 30/40 k€ (2011). Sources with 2006–2018 prices give 18–60 k€ |
| `new_station_mv_loop_in_km` | 0.2 km | assumption (no source) | Two MV cables of 100 m to the nearest ring cable. No study gives a length; FfE counts connection cables in its station price, dena 2012 names their cost only qualitatively |
| `mv_cable_eur_per_km` | 250,000 €/km | central value: median of the three study values with 2021–2025 prices | ef.Ruhr/EWI 2024: 230 k€/km; PuBStadt 2021, p. 214: 250 k€/km (NA2XS2Y 3×1×240, incl. civil works); dena VNS II 2025: 278 k€/km (161–520); NAP 2024 (derived): 137–440 k€/km. Older: BW 2017: 130/145/160; dena 2012: 80/140 |
| `new_station_lv_connection_km` | 0.1 km at the settlement's trench cost | assumption | Two LV links of 50 m (dena 2012, p. 93–94: critical feeders are cut at half length and connected to the new station) |
| `new_station_kva` | 630 kVA | standard size of a new compact station | dena 2012, p. 97; BW 2017, p. 44; eDisGo `config_grid_expansion_default.cfg`. PuBStadt 2021 recommends 800 kVA as the new urban standard |
| `load_transfer_eur` | 20,000 € | assumption from unit costs, rounded up | LV link of about 50 m at 165 k€/km = 8.3 k€, plus a cable distribution cabinet of 5 k€ (PuBStadt 2021, p. 212) = 13.3 k€, rounded up for planning and switching. No study prices a load transfer; dena 2012 treats switching changes as operating reserve |
| `load_transfer_adjacency_m` | 50 m | assumption | Niederle 2026, p. 17: feeders or feeder sections move to less loaded stations. 0 disables the transfer |
| `ront_premium_eur` | 12,000 € | central value (median of twelve derived premiums, 5.5–30 k€) | MR/BUW 2023: 5.5; IEE/BWP 2022: 8; PuBLIK 2016: 8.5; Arnold 2019: 10–11; MONA: 10–11.5; PuBStadt 2021, p. 213: 11–12.5; Schlömer 2017: 12; FfE/Agora 2023: 13; BMWi 2014: 15; Hessen 2018: 16–17; dena 2012, p. 169: about 20 (30 k€ conversion incl. measurement, derived); BW 2017: 30 k€. Where the rONT replaces a kept transformer, stage 5 adds the conventional price of its size |
| `ront_control_range_percent` | ±10 % | product value | Schneider Minera SGrid (2015): 5 positions ±5 % or 9 positions ±10 % |
| `service_lines_in_total` | no | decision D9 | NAV § 9: the connectee usually reimburses changes of the house connection |

The baseline columns `transformer_station_rebuild_boundary_eur` and `transformer_capacity_step_kva` belong to the
retired rule set; the staged rows keep their defaults and do not use them.

**No grouping derating.** DIN VDE 0276-1000 reduces the rating of cables that lie side by side in one trench
(multicore cables 7 cm apart at load factor 0.7: two cables 0.85, three 0.75, four 0.70 of their rating; lower at
higher load factors). The German planning studies do not apply it to LV cables: PuBStadt 2021 (p. 87) argues
sufficient spacing and peaks that are not sustained; BW 2017, FfE/Agora 2023 and dena 2012 do not derate either.
pylovo's cable sizing and the Step 4 overload check use the nominal ratings as well, so derating in Step 5 alone
would count existing double cables as overloaded from 85 % loading and would weigh most on the real rural grids,
which have the most parallel cables.

### Sensitivity rows

| Assumption key | Change against `de_lv_staged_2026` |
|---|---|
| `de_lv_staged_2026_low` | new substation 60 k€ (BW 2017 incl. transformer; the upper end of the sources with 2006–2018 prices), MV cable 200 k€/km, load transfer 10 k€ |
| `de_lv_staged_2026_high` | new substation 150 k€ (NAP 2024 up to 151 k€; dena VNS II high 140 k€), MV cable 400 k€/km (NAP enercity/SWM), load transfer 30 k€ |
| `de_lv_staged_2026_trafo_allin` | transformer exchange at the all-in bins of `heuristic_2026` (28–48 k€; source not verifiable) |
| `de_lv_staged_2026_station_800` | station limit 800 kVA in every settlement type (PuBLIK/Harnisch rural 800 kVA, Kerber 630 kVA, Munich 630/800 kVA); also covers small rural stations that need a new housing for 1000 kVA |
| `de_lv_staged_2026_no_transfer` | no load transfer (`load_transfer_adjacency_m` = 0): whole new substations only |

Further sensitivities: `--line-existing-duct-share`, or new rows with another planning limit, station limit or
cable cap.

### What the result tables store

- **Line rows** (`expansion_line_result`, `expansion_real_line_result`):
  - `measure`: `none`, `local`, `outlet`, `service`, `relieved_by_transfer`, `relieved_by_new_station`,
    `unresolved_service`;
  - `is_station_outlet`, `is_service_line`, `route_cable_count`;
  - `service_cost_eur`: a service line's cost, which is not in `estimated_cost_eur`.
- **Transformer rows** (`expansion_transformer_result`, `expansion_real_transformer_result`):
  - `station_measure`: `none`, `exchange`, `over_limit`, `transfer`, `new_station`;
  - `station_limit_kva`, `excess_kva`;
  - the cost breakdown `transformer_exchange_cost_eur`, `load_transfer_cost_eur`, `new_station_cost_eur`,
    `voltage_cost_eur`;
  - `voltage_measure`.

  `estimated_cost_eur` is the sum of the breakdown, so line plus transformer rows still add up to the grid total.
- **`expansion_grid_result`:** one row per grid and analysis with the decision trail. It holds the
  escalation reason, transfer partners, new-station cluster and share of stations, panels, route counts, voltage
  measure and every cost component.
  - A grid without a transformer rating (one ÜZW area) has no transformer row; its station-level costs are only in
    this table.

## History

Until 2026-09-29 the default was `de_lv_heuristic_2026` (rule set `heuristic_2026`, seeded by migration 0001): the
least-cost parallel NAYY cables on every overloaded route, without a cap or derating; the P100 transformer load
rounded up to 50 kVA and priced with all-in bins of 28–48 k€ whose source could not be verified; a flat 100 k€
station rebuild above 1000 kVA. `staged_2026` replaced it. The row stays in the database for the analyses written
with it; `gridexpand expansion` refuses it. The code is in the git history (SurroGrid `dev` 8005175).

## Application to synthetic and real grids

The same assumption row and the same rules apply to both network sources; only the reading of identifiers and
geometry differs:

- **Synthetic grids.**
  - The components are joined through pylovo grid, line and transformer identifiers.
  - The staged rules read the pylovo network (`pylovo.grid_result.grid`) for the topology and coordinates.
  - Unmapped root connectors of at most 5 m count as station busbar.
- **Real grids.**
  - The rows are joined through `real_grid_case_id`, pandapower line indices, transformer ratings and the grid file
    of the power-flow run (SWF: Excel workbook, ÜZW: pandapower JSON).
  - The grid file also gives the topology and the coordinates.

**Coverage.**
- A real grid-stage is priced only when every simulated timestep converged. A grid with failed timesteps is
  `incomplete`, gets no cost rows and takes no part in the staged stage 4 (it offers no spare capacity).
- Methodological exclusions are `excluded`.
- Cost comparisons must report both total cost and coverage.

**Structural differences between the sources.** Real stations have a median of 6–7 outlets, synthetic a1 stations
3. The panel count is therefore reported, not a trigger. Real grids have more parallel cables than synthetic ones
(route length with two or more cables: real SWF 1.9 %, real ÜZW 7.5 %, synthetic a1 0.7 %); they count at their
nominal capacity.

## Interpretation guidance

Use the result as an order-of-magnitude screening layer:

- **Good for:**
  - mapping which routes and stations become critical;
  - comparing scenarios and flexibility cases;
  - sensitivity analysis. The most uncertain inputs are the new-substation cost, the load transfer (cost and
    neighbour distance) and the existing-duct share.
- **Not suitable for:**
  - budgeting without route, site and protection checks. New substations are not sited, and MV feasibility (ring
    capacity, n-1) is not checked;
  - inferring electrical parallel cables from QGIS helper geometries.
- **Lumpy measures.** Whole substations make the cost a step function of the peak load. Report measure counts
  (transfers, new substations, `expansion_grid_result`) next to the euros. The load transfer softens the steps
  where neighbours have spare capacity.

Cost-basis labels such as `catalog_rural_duct20_trench80_150x1_185x0_240x0` round the shares to whole percent. The
result tables keep the cost basis, per-km costs, measures and breakdown columns, so QGIS users can see why a feature
received its cost.

## Sources

- Niederle, S. et al. (2026): Ermittlung des Netzausbaubedarfes im urbanen Verteilnetz durch Sektorenkopplung
  mittels vereinfachter Netzberechnung. 19. Symposium Energieinnovation, Graz.
  https://www.tugraz.at/fileadmin/user_upload/Events/Eninnov/EnInnov2026/files/lf/262_LF_Niederle.pdf
- Wintzek, P. et al. (2021): Planungs- und Betriebsgrundsätze für städtische Verteilnetze (PuBStadt). Neue Energie
  aus Wuppertal 35. https://d-nb.info/1252809050/34
- Harnisch, S. et al. (2016): Planungs- und Betriebsgrundsätze für ländliche Verteilungsnetze (PuBLIK). Neue Energie
  aus Wuppertal 8; Harnisch, S. (2019): dissertation, BU Wuppertal.
- Kerber, G. (2011): Aufnahmefähigkeit von Niederspannungsverteilnetzen für die Einspeisung aus
  Photovoltaikkleinanlagen. Dissertation, TU München. https://mediatum.ub.tum.de/doc/998003/998003.pdf
- dena (2012): dena-Verteilnetzstudie. Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland bis 2030
  (TU Dortmund/ef.Ruhr, Brunekreeft). dena (2025): dena-Verteilnetzstudie II, Gutachten (BET, BMU Energy Consulting,
  BU Wuppertal); the cost values are in the Gutachten, not in the project report.
  https://www.dena.de/fileadmin/dena/Publikationen/PDFs/2025/Gutachten_VNSII.pdf
- ef.Ruhr et al. (2017): Verteilnetzstudie Baden-Württemberg; E-Bridge/IAEW/OFFIS (2014): Moderne Verteilernetze
  für Deutschland (BMWi).
- FfE for Agora Energiewende (2023): Haushaltsnahe Flexibilitäten nutzen; FfE (2017): MONA 2030.
- ef.Ruhr/EWI (2024): Verteilnetzausbau BW/DE 2045 (slides); Consentec (2024): Langfristszenarien 3, Stromnetze.
- Consentec/Fraunhofer ISI/IEG (2025): Planung von Verteilnetzen der Zukunft (planning practice and simultaneity,
  no unit costs); Bundesnetzagentur (2023): Bericht zum Zustand und Ausbau der Verteilernetze 2022 (DSO survey:
  planning simultaneity, expected thermal and voltage problems, rONT use, LV budgets).
- Verteilnetzstudie Hessen (2018); Gutachten Verteilnetze NRW (2021); Agora Verkehrswende/Energiewende (2019):
  Verteilnetzausbau für die Energiewende – Elektromobilität im Fokus.
- Arnold (2019), Schlömer (2017), Samweber (2018): dissertations with LV unit costs; MR/BUW (2023): rONT
  meta-study.
- DSO network expansion plans under § 14d EnWG (2024): enercity, SWM, Netze BW, e-netz Südhessen (derived unit
  costs).
- DIN VDE 0276-603 / -1000 (via Siemens TIP Technische Schriftenreihe 12, 2023; Nexans Starkstromkabel 2012);
  IEC 60076-7 (via VDE ETG 2024); DIN EN 50160.
- Schneider Electric: Electrical Installation Guide (wiki), Planungskompendium Energieverteilung, Medium Voltage
  Technical Guide AMTED300014EN (2022), Minera SGrid brochure (2015).
- eDisGo (open_eGo), `config_grid_expansion_default.cfg`.
- NAV § 9 (Niederspannungsanschlussverordnung).

The research notes with pages, price bases and the full URL list are kept outside the repository
(`AI/SurroGrid/GridExpand/expansion-heuristic-refinement-2026-09-28/`).
