-- 0010 cable planning limit and station replacement of the staged rules.
--
-- line_planning_limit: a route is overloaded when its P100 current exceeds this
-- share of its capacity, and the added cables must bring it back below it
-- (1.0 = nominal ratings, the behaviour of 0007).
-- station_replacement_from_kva / station_replacement_eur: an exchange to a unit of
-- this size or larger also replaces the station (housing and switchgear for the
-- larger unit). NULL / 0 = no replacement, the behaviour of 0007.
-- The defaults keep every existing row unchanged. Three sensitivity rows apply the
-- cable limit of 60 % and the station replacement from 1000 kVA.

ALTER TABLE surrogrid.expansion_cost_assumption
    ADD COLUMN IF NOT EXISTS line_planning_limit double precision NOT NULL DEFAULT 1.0
        CHECK (line_planning_limit > 0 AND line_planning_limit <= 1),
    ADD COLUMN IF NOT EXISTS station_replacement_from_kva double precision
        CHECK (station_replacement_from_kva > 0),
    ADD COLUMN IF NOT EXISTS station_replacement_eur double precision NOT NULL DEFAULT 0.0
        CHECK (station_replacement_eur >= 0);

UPDATE surrogrid.expansion_cost_assumption
SET parameter_provenance = parameter_provenance || '{
  "line_planning_limit": {"value": 1.0, "unit": "share of the route capacity", "method": "nominal ratings (behaviour of 0007)", "sources": ["PuBStadt 2021, section 7.5.2, p. 50 (PDF 71): LV cables up to 100 % of their rating"]},
  "station_replacement_from_kva": {"value": null, "unit": "kVA", "method": "no station replacement (behaviour of 0007)", "sources": [], "note": "See the station_max_kva_* note: the exchange cost has no larger station housing."},
  "station_replacement_eur": {"value": 0, "unit": "EUR", "method": "no station replacement (behaviour of 0007)", "sources": []}}'::jsonb
WHERE rule_set = 'staged_2026'
  AND NOT parameter_provenance ? 'line_planning_limit';

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_line60',
        'description', 'Sensitivity of de_lv_staged_2026: cables reinforced above 60 % of their rating (planning premise).',
        'line_planning_limit', 0.6,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "line_planning_limit": {"value": 0.6, "unit": "share of the route capacity", "method": "planning premise, sensitivity", "sources": ["Spalthoff et al. 2022: planning premises with LV cables loaded up to 60 % of their rating"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_station_replacement',
        'description', 'Sensitivity of de_lv_staged_2026: an exchange to 1000 kVA also replaces the station.',
        'station_replacement_from_kva', 1000.0,
        'station_replacement_eur', 55000.0,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "station_replacement_from_kva": {"value": 1000, "unit": "kVA", "method": "threshold of Candas 2024", "sources": ["Candas 2024: station replacement for units of 1000 kVA and more"], "note": "Covers the larger housing and switchgear that the exchange cost of 0007 leaves out (station_max_kva_* note)."},
          "station_replacement_eur": {"value": 55000, "unit": "EUR per station (2021 prices)", "method": "source value", "sources": ["Wintzek et al. 2021 (PuBStadt): station replacement 55 kEUR (2021)"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026'
ON CONFLICT (assumption_key) DO NOTHING;

INSERT INTO surrogrid.expansion_cost_assumption
SELECT (jsonb_populate_record(a, jsonb_build_object(
        'assumption_key', 'de_lv_staged_2026_station_replacement_line60',
        'description', 'Sensitivity of de_lv_staged_2026: station replacement from 1000 kVA and cables reinforced above 60 % of their rating.',
        'line_planning_limit', 0.6,
        'created_at', now(), 'updated_at', now(),
        'parameter_provenance', a.parameter_provenance || '{
          "line_planning_limit": {"value": 0.6, "unit": "share of the route capacity", "method": "planning premise, sensitivity", "sources": ["Spalthoff et al. 2022: planning premises with LV cables loaded up to 60 % of their rating"]}}'::jsonb))).*
FROM surrogrid.expansion_cost_assumption a
WHERE a.assumption_key = 'de_lv_staged_2026_station_replacement'
ON CONFLICT (assumption_key) DO NOTHING;
