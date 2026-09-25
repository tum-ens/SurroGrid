-- Synthetic power-flow runs of one expansion analysis (temp table of the materialization transaction).
CREATE TEMP TABLE expansion_selected_run ON COMMIT DROP AS
SELECT
    pr.powerflow_run_id,
    pr.grid_case_id,
    pr.scenario_id,
    gc.ags,
    gc.plz,
    gc.kcid,
    gc.bcid,
    gc.pylovo_grid_result_id,
    gc.pylovo_version_id,
    pcr.settlement_type,
    COALESCE(
        NULLIF(
            CASE WHEN pr.assumptions <> '{}'::JSONB THEN pr.assumptions ELSE sc.assumptions END
                ->> 'timeframe_start',
            ''
        ),
        '2009-01-01 00:00:00+00:00'
    )::TIMESTAMPTZ AS timeframe_start
FROM surrogrid.powerflow_run pr
JOIN surrogrid.grid_case gc USING (grid_case_id)
JOIN surrogrid.scenario sc ON sc.scenario_id = pr.scenario_id
LEFT JOIN pylovo.postcode_result pcr
  ON pcr.version_id = gc.pylovo_version_id
 AND pcr.postcode_result_plz = gc.plz
WHERE pr.run_name = :run_name
  AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
  AND (CAST(:ags AS BIGINT[]) IS NULL OR gc.ags = ANY(CAST(:ags AS BIGINT[])))
  AND (CAST(:plz AS INTEGER[]) IS NULL OR gc.plz = ANY(CAST(:plz AS INTEGER[])))
  AND (
      CAST(:pylovo_version_id AS TEXT) IS NULL
      OR gc.pylovo_version_id = CAST(:pylovo_version_id AS TEXT)
  )
