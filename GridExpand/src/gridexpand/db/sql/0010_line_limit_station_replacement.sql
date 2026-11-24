-- 0010 cable planning limit and station replacement of the staged rules (Lee 2026).
--
-- line_planning_limit: a route is overloaded when its P100 current exceeds this
-- share of its capacity, and the added cables must bring it back below it
-- (1.0 = nominal ratings).
-- station_replacement_from_kva / station_replacement_eur: an exchange to a unit of
-- this size or larger also replaces the station (housing and switchgear for the
-- larger unit). NULL / 0 = no replacement.
-- Every staged_2026 row keeps the nominal cable ratings and replaces the station
-- from 1000 kVA for 70 kEUR. Three sensitivity rows: no station replacement, and
-- cable limits of 60 % and 80 %.

ALTER TABLE surrogrid.expansion_cost_assumption
    ADD COLUMN IF NOT EXISTS line_planning_limit double precision NOT NULL DEFAULT 1.0
        CHECK (line_planning_limit > 0 AND line_planning_limit <= 1),
    ADD COLUMN IF NOT EXISTS station_replacement_from_kva double precision
        CHECK (station_replacement_from_kva > 0),
    ADD COLUMN IF NOT EXISTS station_replacement_eur double precision NOT NULL DEFAULT 0.0
        CHECK (station_replacement_eur >= 0);

UPDATE surrogrid.expansion_cost_assumption
SET station_replacement_from_kva = 1000.0,
    station_replacement_eur = 70000.0,
    parameter_provenance = parameter_provenance || '{
  "line_planning_limit": {"value": 1.0, "unit": "share of the route capacity", "method": "nominal ratings, as transformer_planning_limit (decision D4)", "sources": ["PuBStadt 2021, section 7.5.2, p. 50 (PDF 71): LV cables up to 100 % of their rating", "dena-Verteilnetzstudie 2012, p. 89-90: transformers and cables up to 100 %", "Verteilnetzstudie Baden-Wuerttemberg 2017, p. 37: all LV equipment up to 100 % of its rating"]},
  "station_replacement_from_kva": {"value": 1000, "unit": "kVA", "method": "DSO experience", "sources": ["Candas 2024, section 3 and Table 3.8: a transformer replaced by one of 1000 kVA or more needs a new transformer station (rONT 800 kVA 40,315 EUR, 1000 kVA 191,447 EUR incl. the station)"], "note": "Covers the larger housing and switchgear that the exchange cost leaves out (station_max_kva_* note)."},
  "station_replacement_eur": {"value": 70000, "unit": "EUR per station", "method": "derived: new_station_eur (85 kEUR, station incl. transformer) minus transformer_replace_1000_eur (15 kEUR), which the exchange already pays", "sources": ["see new_station_eur", "PuBStadt 2021 (Wintzek et al.): station replacement 55 kEUR, new station 45 kEUR (2021 prices; via Lee 2026, Table 3.17)", "Candas 2024, Table 3.8: about 151 kEUR between the 800 and 1000 kVA rONT (incl. the station)"]}}'::jsonb
WHERE rule_set = 'staged_2026'
  AND NOT parameter_provenance ? 'line_planning_limit';

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_no_station_replacement',
        'description', 'Sensitivity of de_lv_staged_2026: no station replacement (the exchange to 1000 kVA pays the transformer only).',
        'station_replacement_from_kva', NULL,
        'station_replacement_eur', 0.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "station_replacement_from_kva": {"value": null, "unit": "kVA", "method": "sensitivity: no station replacement (the rules before 0010)", "sources": []},
          "station_replacement_eur": {"value": 0, "unit": "EUR per station", "method": "sensitivity: no station replacement (the rules before 0010)", "sources": []}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_line60',
        'description', 'Sensitivity of de_lv_staged_2026: cables reinforced above 60 % of their rating (planning premise).',
        'line_planning_limit', 0.6,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "line_planning_limit": {"value": 0.6, "unit": "share of the route capacity", "method": "planning premise, sensitivity", "sources": ["Spalthoff et al. 2022: planning premises with LV cables loaded up to 60 % of their rating", "Probst et al. 2024: cables are reinforced at 50-60 % in practice (via Lee 2026, section 4.2.5)"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_line80',
        'description', 'Sensitivity of de_lv_staged_2026: cables reinforced above 80 % of their rating (the feeder planning utilization of pylovo).',
        'line_planning_limit', 0.8,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "line_planning_limit": {"value": 0.8, "unit": "share of the route capacity", "method": "sensitivity: the planning utilization pylovo sizes its feeders with", "sources": ["Arnold 2019, Table 4, p. 45: maximum line utilization 80 % of the thermal limit", "pylovo CABLE_PLANNING_UTILIZATION 0.8, like its TRANSFORMER_PLANNING_UTILIZATION"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;
