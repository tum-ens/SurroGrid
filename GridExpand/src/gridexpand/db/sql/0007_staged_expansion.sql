-- 0007 staged LV expansion rules (analysis/expansion/staged.py, docs/expansion_costs.md).
--
-- expansion_cost_assumption gets a rule_set: 'staged_2026' for the rows of this
-- migration (station limit, capped cables, load transfer to neighbouring stations,
-- new substations, residual voltage), 'heuristic_2026' for the row of 0001. That
-- rule set was retired on 2026-09-29: its row stays for the analyses written with
-- it, its code is in the git history (SurroGrid dev 8005175).
-- Every staged row documents each parameter in parameter_provenance: value, unit,
-- the method that produced it (central value, conservative choice, decision,
-- assumption) and the sources. The default row de_lv_staged_2026 is
-- followed by five sensitivity rows that each change one group of parameters.
--
-- The line and transformer results get the measure of each row and a cost
-- breakdown; station-level measures (exchange, load transfer, new substations,
-- voltage) are carried by the transformer row, so estimated_cost_eur of the line
-- and transformer rows still adds up to the grid total. expansion_grid_result
-- holds one row per grid with the full decision trail.

ALTER TABLE surrogrid.expansion_cost_assumption
    ADD COLUMN IF NOT EXISTS rule_set text NOT NULL DEFAULT 'heuristic_2026'
        CHECK (rule_set IN ('heuristic_2026', 'staged_2026')),
    ADD COLUMN IF NOT EXISTS transformer_planning_limit double precision,
    ADD COLUMN IF NOT EXISTS station_max_kva_rural double precision,
    ADD COLUMN IF NOT EXISTS station_max_kva_suburban double precision,
    ADD COLUMN IF NOT EXISTS station_max_kva_urban double precision,
    ADD COLUMN IF NOT EXISTS line_max_added_cables integer,
    ADD COLUMN IF NOT EXISTS panel_max integer,
    ADD COLUMN IF NOT EXISTS panel_trigger boolean,
    ADD COLUMN IF NOT EXISTS new_station_eur double precision,
    ADD COLUMN IF NOT EXISTS new_station_mv_loop_in_km double precision,
    ADD COLUMN IF NOT EXISTS mv_cable_eur_per_km double precision,
    ADD COLUMN IF NOT EXISTS new_station_lv_connection_km double precision,
    ADD COLUMN IF NOT EXISTS new_station_kva double precision,
    ADD COLUMN IF NOT EXISTS load_transfer_eur double precision,
    ADD COLUMN IF NOT EXISTS load_transfer_adjacency_m double precision,
    ADD COLUMN IF NOT EXISTS ront_premium_eur double precision,
    ADD COLUMN IF NOT EXISTS ront_control_range_percent double precision,
    ADD COLUMN IF NOT EXISTS service_lines_in_total boolean,
    ADD COLUMN IF NOT EXISTS parameter_provenance jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE surrogrid.expansion_cost_assumption
SET parameter_provenance = '{"rule_set": {"value": "heuristic_2026", "method": "retired 2026-09-29", "sources": ["docs/expansion_costs.md, section History"], "note": "The row of 0001 (former overload heuristic), kept for the analyses written with it. Its rules are in the git history (SurroGrid dev 8005175)."}}'::jsonb
WHERE assumption_key = 'de_lv_heuristic_2026'
  AND parameter_provenance = '{}'::jsonb;

ALTER TABLE surrogrid.expansion_cost_assumption ALTER COLUMN rule_set SET DEFAULT 'staged_2026';

INSERT INTO surrogrid.expansion_cost_assumption (
    assumption_key, description, rule_set,
    line_parallel_150_eur_per_km, line_parallel_185_eur_per_km, line_parallel_240_eur_per_km,
    line_reinforcement_150_max_i_ka, line_reinforcement_185_max_i_ka, line_reinforcement_240_max_i_ka,
    line_existing_duct_share,
    line_reopen_rural_eur_per_km, line_reopen_suburban_eur_per_km, line_reopen_urban_eur_per_km,
    transformer_replace_100_eur, transformer_replace_160_eur, transformer_replace_250_eur,
    transformer_replace_400_eur, transformer_replace_630_eur, transformer_replace_800_eur,
    transformer_replace_1000_eur,
    transformer_planning_limit,
    station_max_kva_rural, station_max_kva_suburban, station_max_kva_urban,
    line_max_added_cables,
    panel_max, panel_trigger,
    new_station_eur, new_station_mv_loop_in_km, mv_cable_eur_per_km,
    new_station_lv_connection_km, new_station_kva,
    load_transfer_eur, load_transfer_adjacency_m,
    ront_premium_eur, ront_control_range_percent,
    service_lines_in_total,
    source_note, parameter_provenance
)
VALUES (
    'de_lv_staged_2026',
    'Staged German LV expansion screening after Niederle et al. (EnInnov 2026): station limit, capped cables, load transfer to neighbouring stations, new substations, residual voltage.',
    'staged_2026',
    25000.0, 45000.0, 70000.0,
    0.270, 0.313, 0.357,
    0.20,
    80000.0, 100000.0, 165000.0,
    8000.0, 8000.0, 8000.0,
    10000.0, 12000.0, 12500.0,
    15000.0,
    1.0,
    1000.0, 1000.0, 1000.0,
    3,
    12, FALSE,
    85000.0, 0.2, 250000.0,
    0.1, 630.0,
    20000.0, 50.0,
    12000.0, 10.0,
    FALSE,
    'Staged rules (docs/expansion_costs.md). Each parameter carries its value, unit, method and sources in parameter_provenance. Prices keep the price base of their sources (2007-2025); nothing is inflation-adjusted.',
    $prov$
{
  "rule_set": {"value": "staged_2026", "method": "decision", "sources": ["User decisions D1-D10 of 2026-09-29 (AI/SurroGrid/GridExpand/expansion-heuristic-refinement-2026-09-28/PLAN.md)"]},
  "line_parallel_150_eur_per_km": {"value": 25000, "unit": "EUR/km", "method": "PuBStadt value (cable and installation in the same trench) rounded up by 5 kEUR/km in de_lv_heuristic_2026", "sources": ["PuBStadt 2021, p. 212 (PDF 233): +20 kEUR/km for a parallel NAYY 150 in the same trench, 'Kabel und Montage' (2021)", "dena-Verteilnetzstudie 2012, derived from Tables 6.2/6.5, p. 150/157 (PDF 172/179): parallel LV cables priced at the full rate incl. earthworks (upper bound)"], "note": "The first cable of a trenched route pays line_reopen_* (trench incl. that cable); only further cables pay this increment, as in PuBStadt and FfE."},
  "line_parallel_185_eur_per_km": {"value": 45000, "unit": "EUR/km", "method": "PuBStadt value rounded up by 5 kEUR/km in de_lv_heuristic_2026", "sources": ["PuBStadt 2021, p. 212 (PDF 233): +40 kEUR/km for a parallel NAYY 185 (2021)"]},
  "line_parallel_240_eur_per_km": {"value": 70000, "unit": "EUR/km", "method": "PuBStadt value rounded up by 10 kEUR/km in de_lv_heuristic_2026", "sources": ["PuBStadt 2021, p. 212 (PDF 233): +60 kEUR/km for a parallel NAYY 240 (2021)", "FfE for Agora 2023, p. 112-113: NAYY-J 4x240 material only 28 kEUR/km, and only in a trench opened by the same reinforcement (lower bound)", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 51: no cost reduction for several cables in one route (stated for HV)"], "note": "PuBStadt prices mixed sections as the larger cable at its single price plus the smaller at its parallel price."},
  "line_reinforcement_150_max_i_ka": {"value": 0.27, "unit": "kA", "method": "kept for consistency with the grid data", "sources": ["SimBench LineType / pandapower std_types: NAYY 4x150 SE 270 A", "DIN VDE 0276-603 via Siemens TIP 12 (2023), Tab. A.7: 275 A at load factor 0.7, 20 degC, 1.0 K m/W"]},
  "line_reinforcement_185_max_i_ka": {"value": 0.313, "unit": "kA", "method": "norm value", "sources": ["DIN VDE 0276-603 via Siemens TIP 12 (2023), Tab. A.7: 313 A"]},
  "line_reinforcement_240_max_i_ka": {"value": 0.357, "unit": "kA", "method": "kept for consistency with the grid data", "sources": ["SimBench LineType: NAYY 4x240 SE 357 A", "DIN VDE 0276-603 via Siemens TIP 12 (2023): 364 A"]},
  "line_existing_duct_share": {"value": 0.2, "unit": "share of reinforced routes", "method": "assumption, no source found", "sources": [], "note": "Sensitivity 0 and 0.5 via --line-existing-duct-share. PuBStadt recommends laying empty ducts during civil works but gives no share."},
  "line_reopen_rural_eur_per_km": {"value": 80000, "unit": "EUR/km (trench incl. first cable)", "method": "Verteilnetzstudie BW 2017, the only source with rural, semi-urban and urban values (80/100/120 kEUR/km)", "sources": ["Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 80 kEUR/km, cable and earthworks, NAYY 4x150", "dena-Verteilnetzstudie 2012, p. 147 (PDF 169): 60 kEUR/km for municipalities up to 500 inhabitants/km2 (2011, incl. overheads)", "FfE for Agora 2023, p. 113: 39 kEUR/km laying (rural and suburban) + 28 kEUR/km NAYY-J 4x240 material = 67 kEUR/km"], "note": "The three rural sources give 60, 67 and 80 kEUR/km (median 67; price bases 2011-2023). The semi-urban value 100 kEUR/km is BW's as well."},
  "line_reopen_suburban_eur_per_km": {"value": 100000, "unit": "EUR/km (trench incl. first cable)", "method": "central value: the two sources with a semi-urban class (100 kEUR/km each)", "sources": ["Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 100 kEUR/km", "Gutachten Verteilnetze NRW 2021, Tab. 7-3: 100 kEUR/km", "dena-Verteilnetzstudie 2012, p. 147 (PDF 169): 60 kEUR/km up to 500 inhabitants/km2, which includes semi-urban municipalities; 100 kEUR/km above (2011)", "FfE for Agora 2023, p. 113: 67 kEUR/km (rural and suburban)"]},
  "line_reopen_urban_eur_per_km": {"value": 165000, "unit": "EUR/km (trench incl. first cable)", "method": "central value: median of the six sources with 2021-2025 prices (115, 139, 150, 182, 250, 312 kEUR/km) = 166, rounded", "sources": ["FfE for Agora 2023, p. 113: 87 kEUR/km laying + 28 kEUR/km material = 115 kEUR/km", "ef.Ruhr/EWI 2024, slide 38: 139 kEUR/km (no settlement split)", "PuBStadt 2021, p. 212 (PDF 233): 150/175/200 kEUR/km for a single NAYY 150/185/240 incl. civil works", "dena-Verteilnetzstudie II 2025, Gutachten p. 276: 182 kEUR/km (spread of partner-DSO values 80-380)", "NAP 2024 (derived): SWM about 250, enercity about 309-316 kEUR/km", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 120 kEUR/km; dena-Verteilnetzstudie 2012, p. 147 (PDF 169): 100 kEUR/km above 500 inhabitants/km2 (older price bases)"], "note": "NAP-implied urban values of 2024 reach 250-316 kEUR/km (high sensitivity)."},
  "transformer_replace_100_eur": {"value": 8000, "unit": "EUR per unit incl. installation", "method": "conservative choice: no source for 100 kVA, set to the 160/250 kVA value", "sources": ["Kerber 2011, p. 142: 160 kVA 6.5 kEUR (2007)"]},
  "transformer_replace_160_eur": {"value": 8000, "unit": "EUR per unit incl. installation", "method": "conservative choice: the 250 kVA value, above Kerber's 160 kVA value", "sources": ["Kerber 2011, p. 142: 160 kVA 6.5 kEUR (2007)"]},
  "transformer_replace_250_eur": {"value": 8000, "unit": "EUR per unit incl. installation", "method": "upper value of the recent sources (7-8 kEUR), conservative because prices rose since", "sources": ["Kerber 2011, p. 142: 8.0 kEUR (2007)", "FfE MONA 2030 (2017), p. 143: 7.0 kEUR (2015)", "Verteilnetzstudie Hessen 2018, Tab. 20: 7 kEUR (2015)"], "note": "FfE for Agora 2023, p. 112-113 charges 15 kEUR per exchange whatever the size (installation-dominated), an upper bound."},
  "transformer_replace_400_eur": {"value": 10000, "unit": "EUR per unit incl. installation", "method": "upper range of the recent sources (8-10.5 kEUR), rounded, conservative", "sources": ["Kerber 2011: 10.5 kEUR (2007)", "FfE MONA 2030: 8.5 kEUR (2015)", "Verteilnetzstudie Hessen 2018: 9 kEUR (2015)", "BMWi Moderne Verteilernetze 2014, p. 52: 8 (6-10) kEUR"], "note": "FfE for Agora 2023, p. 112-113 charges 15 kEUR per exchange whatever the size, an upper bound."},
  "transformer_replace_630_eur": {"value": 12000, "unit": "EUR per unit incl. installation", "method": "central value: median of ten sources", "sources": ["BMWi 2014: 12 (10-15) kEUR", "FfE MONA 2030: 12 kEUR", "Verteilnetzstudie Hessen 2018: 12 kEUR", "PuBStadt 2021, p. 213 (PDF 234): 10 kEUR", "Kerber 2011: 14 kEUR", "Schloemer 2017: 11.25 kEUR", "Arnold 2019: 8.6 kEUR", "FfE for Agora 2023, p. 112-113: 15 kEUR per exchange, any size", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 15 kEUR (exchange to the 630 kVA standard, incl. secondary equipment)", "dena-Verteilnetzstudie 2012, p. 147 (PDF 169): 10 kEUR (2011, exchange to the 630 kVA standard)"]},
  "transformer_replace_800_eur": {"value": 12500, "unit": "EUR per unit incl. installation", "method": "central value: median of three sources", "sources": ["PuBStadt 2021, p. 213 (PDF 234): 12.5 kEUR", "Arnold 2019: 9.8 kEUR (2014)", "FfE for Agora 2023, p. 112-113: 15 kEUR per exchange, any size"]},
  "transformer_replace_1000_eur": {"value": 15000, "unit": "EUR per unit incl. installation", "method": "upper value of four sources (PuBStadt 2021, FfE 2023)", "sources": ["PuBStadt 2021, p. 213 (PDF 234): 15 kEUR", "FfE for Agora 2023, p. 112-113: 15 kEUR per exchange, any size", "Schloemer 2017: 13.25 kEUR", "Arnold 2019: 10.9 kEUR (2014)"], "note": "Only reachable in urban grids (station_max_kva_urban)."},
  "transformer_planning_limit": {"value": 1.0, "unit": "share of the rated power", "method": "decision D4, DSO planning practice (all studies agree)", "sources": ["Niederle et al. 2026, p. 7-8: overloaded above Sr", "PuBStadt 2021, section 7.5.2, p. 50 (PDF 71): MV/LV transformers at 100 % of Sr", "dena-Verteilnetzstudie 2012, p. 89-90 (PDF 111-112): no (n-1) in LV, transformers and cables up to 100 %", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 37: all LV equipment up to 100 % of its rating", "FfE for Agora 2023, p. 53 and 111: overloaded above 100 %, reinforced even after short overloads (p. 72)", "Nobis 2016 (TUM): no (n-1) in LV, 100 % loading"], "note": "IEC 60076-7 allows up to 1.5 p.u. normal cyclic loading; 110/120 % are sensitivities."},
  "station_max_kva_rural": {"value": 1000, "unit": "kVA", "method": "decision D2, revised 2026-09-29: 1000 kVA in every settlement type, as FfE/Agora 2023", "sources": ["FfE for Agora 2023, p. 85 and 111: at most 1000 kVA in every settlement type (after PuBStadt and DSO feedback); above that an additional station splits the grid", "dena-Verteilnetzstudie 2012, p. 94 (PDF 116): a second standard unit in parallel (2 x 630 kVA)", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 44: one identical parallel unit, then any number of standard units", "Agora 2019 / Gutachten NRW 2021: at most two 630 kVA transformers", "Schneider Electric Planungskompendium: MV fuse-switch combinations up to about 1000 kVA (VDE 0670-402)"], "note": "Stricter: PuBLIK 2016 / Harnisch 2019 (new rural station from 800 kVA), Kerber 2011 (new station above 630 kVA). The exchange cost has no larger station housing, which small rural stations may need (PuBStadt 2021, p. 86)."},
  "station_max_kva_suburban": {"value": 1000, "unit": "kVA", "method": "decision D2, revised 2026-09-29: 1000 kVA in every settlement type, as FfE/Agora 2023", "sources": ["FfE for Agora 2023, p. 85 and 111: at most 1000 kVA in every settlement type (after PuBStadt and DSO feedback); above that an additional station splits the grid", "dena-Verteilnetzstudie 2012, p. 94 (PDF 116): a second standard unit in parallel (2 x 630 kVA)", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 44: one identical parallel unit, then any number of standard units", "Agora 2019 / Gutachten NRW 2021: at most two 630 kVA transformers", "Schneider Electric Planungskompendium: MV fuse-switch combinations up to about 1000 kVA (VDE 0670-402)"], "note": "Stricter: Kerber 2011 (400/630 kVA in suburban grids, 800 kVA and more usually as parallel units)."},
  "station_max_kva_urban": {"value": 1000, "unit": "kVA", "method": "decision D2, revised 2026-09-29: 1000 kVA in every settlement type, as FfE/Agora 2023", "sources": ["PuBStadt 2021, p. 157 (PDF 178): 1000 kVA or 2 x 630 kVA case by case; 5 % of the upgrades need more than 1000 kVA", "FfE for Agora 2023, p. 85 and 111: at most 1000 kVA in every settlement type (after PuBStadt and DSO feedback); above that an additional station splits the grid", "Schneider Electric, Electrical Installation Guide (wiki, Low-voltage distribution networks): urban substations with one or two 1000 kVA transformers", "Schneider Electric Planungskompendium: MV fuse-switch combinations up to about 1000 kVA (VDE 0670-402)"], "note": "Munich uses 630/800 kVA only (Niederle et al. 2026)."},
  "line_max_added_cables": {"value": 3, "unit": "cables per route", "method": "decision D6 (revised 2026-09-29: nominal ratings, no grouping derating)", "sources": ["Kerber 2011: at most 3 parallel cables per street side", "Agora 2019 / Gutachten NRW 2021: at most 4 parallel LV cables"], "note": "A route that needs more escalates to stage 4 (load transfer, new substation). Cables are sized at their nominal ratings, as in all German planning studies (PuBStadt 2021, p. 87: no reduction factors; BW 2017, FfE 2023, dena 2012: none), in pylovo's cable sizing and in the Step 4 overload check. No German study caps the number: PuBStadt 2021, p. 156 (PDF 177): usually one more NAYY 150 suffices; Verteilnetzstudie Baden-Wuerttemberg 2017, p. 44: one identical cable, then any number of standard cables; FfE for Agora 2023, p. 111: no limit (1.0-1.3 cables per reinforced trench-km on average, derived from Abb. 27)."},
  "panel_max": {"value": 12, "unit": "LV feeder panels", "method": "catalogue practice", "sources": ["Niederle et al. 2026, p. 17: 8 or 12 panels depending on the station", "Jean Mueller NH distribution catalogue 2012: 8/10/12 NH2 ways", "Kenter compact station: up to 12 NH2 feeders"], "note": "Reported only (panel_trigger false, decision D5); no standard limits the number of panels."},
  "panel_trigger": {"value": false, "unit": "", "method": "decision D5", "sources": [], "note": "Real stations already have a median of 6-7 outlets, synthetic ones 3; a trigger would treat the sources differently."},
  "new_station_eur": {"value": 85000, "unit": "EUR per compact substation incl. transformer", "method": "central value: median of the eight German sources with 2020-2025 prices (51, 57.5, 83.5, 84, 86.5, 101, 150.5, 225 kEUR) = 85.25, rounded", "sources": ["dena-Verteilnetzstudie II 2025, Gutachten p. 276 (Tab. 12): 101 kEUR incl. transformer, spread of partner-DSO values 80-140", "ef.Ruhr/EWI 2024, slide 38: 86.5 kEUR (without land)", "NAP 2024 (derived): Netze BW 77-91, e-netz Suedhessen 74-93, enercity about 150 kEUR per measure; SWM 200-250 kEUR per site", "PuBStadt 2021, p. 213 (PDF 234): 45 kEUR station building + 10-15 kEUR transformer (2021)", "FfE for Agora 2023, p. 112-113: 51 kEUR incl. transformer, housing, secondary equipment and usually the connection cables", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 60 kEUR incl. transformer, MV switchgear, LV distribution and building", "Langfristszenarien 3 (Consentec 2024), Abb. 12: 50 kEUR (2018 prices)", "dena-Verteilnetzstudie 2012, p. 147 (PDF 169): 30/40 kEUR (2011), without the MV cable"], "note": "Sources with 2006-2018 prices give 18-60 kEUR (low sensitivity 60 kEUR); dena VNS II's 101 kEUR and the urban DSO plans are in the high sensitivity (150 kEUR). The MV connection is priced separately (new_station_mv_loop_in_km); only FfE may include it."},
  "new_station_mv_loop_in_km": {"value": 0.2, "unit": "km of MV cable", "method": "assumption, no source found", "sources": [], "note": "Two MV cables of 100 m to the nearest MV ring cable (loop-in). No study gives a length: FfE for Agora 2023, p. 112 names connection cables as a cost reason of its station; dena-Verteilnetzstudie 2012, p. 94 (PDF 116) mentions the cost of the cable connection qualitatively."},
  "mv_cable_eur_per_km": {"value": 250000, "unit": "EUR/km incl. civil works", "method": "central value: median of the three study values with 2021-2025 prices (230, 250, 278 kEUR/km)", "sources": ["ef.Ruhr/EWI 2024, slide 38: 230 kEUR/km", "PuBStadt 2021, p. 214 (PDF 235): NA2XS2Y 3x1x240 250 kEUR/km incl. civil works (150/185/300 mm2: 225/237.5/275)", "dena-Verteilnetzstudie II 2025, Gutachten p. 276: 278 kEUR/km (spread 161-520)", "NAP 2024 (derived): 137-440 kEUR/km", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 130/145/160 kEUR/km; dena-Verteilnetzstudie 2012, p. 146 (PDF 168): 80/140 kEUR/km (older price bases)"]},
  "new_station_lv_connection_km": {"value": 0.1, "unit": "km of LV route at the settlement's trench cost", "method": "assumption", "sources": ["dena-Verteilnetzstudie 2012, p. 93-94 (PDF 115-116): critical feeders are cut at half length and connected to the new station"], "note": "Two LV links of 50 m from the new station into the existing feeders."},
  "new_station_kva": {"value": 630, "unit": "kVA", "method": "standard size of a new compact station", "sources": ["dena-Verteilnetzstudie 2012, p. 97 (PDF 119): standard 630 kVA", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 44: 630 kVA standard unit", "PuBStadt 2021, p. 157 (PDF 178): 630 kVA smallest size, 800 kVA the new urban standard", "eDisGo config_grid_expansion_default.cfg: mv_lv_transformer 630 kVA"]},
  "load_transfer_eur": {"value": 20000, "unit": "EUR per grid that hands load to neighbours", "method": "assumption derived from unit costs, rounded up (conservative)", "sources": ["LV link of about 50 m at the urban trench cost 165 kEUR/km = 8.3 kEUR", "cable distribution cabinet 5 kEUR (PuBStadt 2021, p. 212, PDF 233)"], "note": "13.3 kEUR rounded up to 20 kEUR for planning and switching. No study prices a load transfer: dena-Verteilnetzstudie 2012 treats switching changes as operating reserve (p. 94 and 133, PDF 116 and 155); the closest PuBStadt measure is a feeder reinforced up to the next cable cabinet (p. 87, PDF 108)."},
  "load_transfer_adjacency_m": {"value": 50, "unit": "m between the nearest buses of two grids", "method": "assumption", "sources": ["Niederle et al. 2026, p. 17: moving feeders or feeder sections to less loaded stations"], "note": "Grids this close can be linked with one short LV cable; 0 disables the transfer."},
  "ront_premium_eur": {"value": 12000, "unit": "EUR over a conventional transformer", "method": "central value: median of twelve derived premiums (5.5-30 kEUR)", "sources": ["MR/BUW rONT meta-study 2023: 5.5 kEUR", "IEE/BWP 2022: 8 kEUR", "PuBLIK 2016: 8.5 kEUR", "Arnold 2019: 10-11 kEUR", "FfE MONA 2030: 10-11.5 kEUR", "PuBStadt 2021, p. 213 (PDF 234): rONT 21.2/24.2/27.5 kEUR against 10/12.5/15 kEUR for 630/800/1000 kVA, premium 11-12.5 kEUR", "Schloemer 2017: 12 kEUR", "FfE for Agora 2023, p. 87 and 113: rONT 28 kEUR instead of a 15 kEUR exchange, premium 13 kEUR", "BMWi 2014: 15 kEUR", "Verteilnetzstudie Hessen 2018: 16-17 kEUR", "dena-Verteilnetzstudie 2012, p. 169 (PDF 191): conversion to a regulated station 30 kEUR incl. measurement and communication, about 20 kEUR over a 10 kEUR exchange (derived)", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: rONT 45 kEUR against a 15 kEUR exchange, premium 30 kEUR"], "note": "Stage 5 charges the premium where stage 1 buys a new transformer anyway, and the conventional price of the station's size plus the premium (a full rONT, as priced by PuBStadt, BW and dena 2012) where the rONT replaces the existing transformer."},
  "ront_control_range_percent": {"value": 10, "unit": "% of the LV reference voltage", "method": "product value, the wider of two variants", "sources": ["Schneider Electric Minera SGrid brochure (2015): on-load LV tappings 5 positions +/-5 % or 9 positions +/-10 %"]},
  "service_lines_in_total": {"value": false, "unit": "", "method": "decision D9", "sources": ["NAV section 9: the connectee usually reimburses changes of the house connection"], "note": "Service-line reinforcements are costed and reported separately (service_cost_eur)."}
}
$prov$::jsonb
)
ON CONFLICT (assumption_key) DO NOTHING;

-- Sensitivity rows: the default row with one group of parameters changed.
INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_low',
        'description', 'Sensitivity of de_lv_staged_2026: low new-substation and load-transfer costs.',
        'new_station_eur', 60000.0,
        'mv_cable_eur_per_km', 200000.0,
        'load_transfer_eur', 10000.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "new_station_eur": {"value": 60000, "unit": "EUR", "method": "low sensitivity: the upper end of the sources with 2006-2018 prices (18-60 kEUR)", "sources": ["Verteilnetzstudie Baden-Wuerttemberg 2017, p. 91: 60 kEUR incl. transformer", "PuBStadt 2021, p. 213 (PDF 234): 45 kEUR building + 10-15 kEUR transformer (2021)"]},
          "mv_cable_eur_per_km": {"value": 200000, "unit": "EUR/km", "method": "low sensitivity", "sources": ["NAP 2024 (derived): Netze BW 234-236, e-netz Suedhessen 137-153 kEUR/km"]},
          "load_transfer_eur": {"value": 10000, "unit": "EUR", "method": "low sensitivity (assumption)", "sources": []}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_high',
        'description', 'Sensitivity of de_lv_staged_2026: high (urban DSO plan) new-substation and load-transfer costs.',
        'new_station_eur', 150000.0,
        'mv_cable_eur_per_km', 400000.0,
        'load_transfer_eur', 30000.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "new_station_eur": {"value": 150000, "unit": "EUR", "method": "high sensitivity", "sources": ["NAP 2024 (derived): up to 151 kEUR per new station (enercity), Munich 200-250 kEUR per site", "dena-Verteilnetzstudie II 2025: high value 140 kEUR"]},
          "mv_cable_eur_per_km": {"value": 400000, "unit": "EUR/km", "method": "high sensitivity", "sources": ["NAP 2024 (derived): enercity 439-440, SWM about 300 kEUR/km"]},
          "load_transfer_eur": {"value": 30000, "unit": "EUR", "method": "high sensitivity (assumption)", "sources": []}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_trafo_allin',
        'description', 'Sensitivity of de_lv_staged_2026: transformer exchange at the all-in bins of de_lv_heuristic_2026.',
        'transformer_replace_100_eur', 28000.0, 'transformer_replace_160_eur', 28800.0,
        'transformer_replace_250_eur', 30000.0, 'transformer_replace_400_eur', 33000.0,
        'transformer_replace_630_eur', 38000.0, 'transformer_replace_800_eur', 42000.0,
        'transformer_replace_1000_eur', 48000.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || jsonb_build_object(
          'transformer_replace_100_eur', '{"value": 28000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (source not verifiable)", "sources": []}'::jsonb,
          'transformer_replace_160_eur', '{"value": 28800, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (source not verifiable)", "sources": []}'::jsonb,
          'transformer_replace_250_eur', '{"value": 30000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (source not verifiable)", "sources": []}'::jsonb,
          'transformer_replace_400_eur', '{"value": 33000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (WEI/GridSim, not verifiable)", "sources": []}'::jsonb,
          'transformer_replace_630_eur', '{"value": 38000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (WEI/GridSim, not verifiable)", "sources": []}'::jsonb,
          'transformer_replace_800_eur', '{"value": 42000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (interpolated there)", "sources": []}'::jsonb,
          'transformer_replace_1000_eur', '{"value": 48000, "method": "upper sensitivity: bin of de_lv_heuristic_2026 (WEI/GridSim, not verifiable)", "sources": []}'::jsonb)))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_station_800',
        'description', 'Sensitivity of de_lv_staged_2026: station limit 800 kVA in every settlement type (stricter planning practice).',
        'station_max_kva_rural', 800.0, 'station_max_kva_suburban', 800.0, 'station_max_kva_urban', 800.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "station_max_kva_rural": {"value": 800, "unit": "kVA", "method": "stricter sensitivity", "sources": ["Harnisch 2019 / PuBLIK 2016: new rural station from 800 kVA", "Kerber 2011: new station above 630 kVA"], "note": "Also covers small rural stations that cannot take a 1000 kVA unit without a new housing."},
          "station_max_kva_suburban": {"value": 800, "unit": "kVA", "method": "stricter sensitivity", "sources": ["PuBStadt 2021, p. 157 (PDF 178): 800 kVA as the new standard size", "Kerber 2011, p. 40-41: 800 kVA and more usually as parallel units"]},
          "station_max_kva_urban": {"value": 800, "unit": "kVA", "method": "stricter sensitivity", "sources": ["Niederle et al. 2026: Munich uses 630/800 kVA only"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_no_transfer',
        'description', 'Sensitivity of de_lv_staged_2026: no load transfer to neighbouring stations (whole new substations only).',
        'load_transfer_adjacency_m', 0.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "load_transfer_adjacency_m": {"value": 0, "unit": "m", "method": "sensitivity: load transfer disabled", "sources": []}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

-- Measures and cost breakdown of the result rows (NULL/0 for the analyses written before).
ALTER TABLE surrogrid.expansion_line_result
    ADD COLUMN IF NOT EXISTS measure text,
    ADD COLUMN IF NOT EXISTS is_station_outlet boolean,
    ADD COLUMN IF NOT EXISTS is_service_line boolean,
    ADD COLUMN IF NOT EXISTS route_cable_count integer,
    ADD COLUMN IF NOT EXISTS service_cost_eur double precision NOT NULL DEFAULT 0.0;
ALTER TABLE surrogrid.expansion_real_line_result
    ADD COLUMN IF NOT EXISTS measure text,
    ADD COLUMN IF NOT EXISTS is_station_outlet boolean,
    ADD COLUMN IF NOT EXISTS is_service_line boolean,
    ADD COLUMN IF NOT EXISTS route_cable_count integer,
    ADD COLUMN IF NOT EXISTS service_cost_eur double precision NOT NULL DEFAULT 0.0;
ALTER TABLE surrogrid.expansion_transformer_result
    ADD COLUMN IF NOT EXISTS station_measure text,
    ADD COLUMN IF NOT EXISTS station_limit_kva double precision,
    ADD COLUMN IF NOT EXISTS excess_kva double precision,
    ADD COLUMN IF NOT EXISTS transformer_exchange_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS load_transfer_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS new_station_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS voltage_measure text,
    ADD COLUMN IF NOT EXISTS voltage_cost_eur double precision NOT NULL DEFAULT 0.0;
ALTER TABLE surrogrid.expansion_real_transformer_result
    ADD COLUMN IF NOT EXISTS station_measure text,
    ADD COLUMN IF NOT EXISTS station_limit_kva double precision,
    ADD COLUMN IF NOT EXISTS excess_kva double precision,
    ADD COLUMN IF NOT EXISTS transformer_exchange_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS load_transfer_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS new_station_cost_eur double precision NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS voltage_measure text,
    ADD COLUMN IF NOT EXISTS voltage_cost_eur double precision NOT NULL DEFAULT 0.0;

-- One row per analysis and grid: the decision trail of the staged rules.
CREATE TABLE surrogrid.expansion_grid_result (
    expansion_analysis_run_id bigint NOT NULL REFERENCES surrogrid.expansion_analysis_run (expansion_analysis_run_id) ON DELETE CASCADE,
    grid_key text NOT NULL,
    scenario_id bigint NOT NULL,
    powerflow_run_id bigint REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    grid_case_id bigint,
    real_powerflow_run_id bigint REFERENCES surrogrid.real_powerflow_run (real_powerflow_run_id) ON DELETE CASCADE,
    real_grid_case_id bigint,
    plz integer,
    grid_label text,
    settlement_type integer,
    rule_set text NOT NULL,
    transformer_rated_power_kva double precision,
    peak_kva double precision,
    station_limit_kva double precision,
    excess_kva double precision,
    station_measure text,
    escalation_reason text,
    transfer_kva double precision,
    transfer_partners text,
    new_station_cluster text,
    new_stations double precision,
    existing_outlet_cables integer,
    added_outlet_cables integer,
    panels_total integer,
    routes_reinforced integer,
    cables_added integer,
    heavy_routes integer,
    relieved_routes integer,
    unresolved_routes integer,
    service_routes_reinforced integer,
    voltage_residual_buses integer,
    voltage_measure text,
    cable_cost_eur double precision NOT NULL DEFAULT 0.0,
    service_cost_eur double precision NOT NULL DEFAULT 0.0,
    transformer_exchange_cost_eur double precision NOT NULL DEFAULT 0.0,
    load_transfer_cost_eur double precision NOT NULL DEFAULT 0.0,
    new_station_cost_eur double precision NOT NULL DEFAULT 0.0,
    voltage_cost_eur double precision NOT NULL DEFAULT 0.0,
    total_cost_eur double precision NOT NULL DEFAULT 0.0,
    PRIMARY KEY (expansion_analysis_run_id, grid_key),
    CONSTRAINT ck_expansion_grid_result_source CHECK ((powerflow_run_id IS NULL) <> (real_powerflow_run_id IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_expansion_grid_result_powerflow_run ON surrogrid.expansion_grid_result (powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_grid_result_real_powerflow_run ON surrogrid.expansion_grid_result (real_powerflow_run_id);
CREATE INDEX IF NOT EXISTS idx_expansion_grid_result_scenario ON surrogrid.expansion_grid_result (scenario_id);
