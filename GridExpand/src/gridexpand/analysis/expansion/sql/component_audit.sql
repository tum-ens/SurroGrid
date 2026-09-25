-- Active line components without a visible geometry. Root-connector-like components (no
-- pylovo line, at most 5 m) may be ignored; any other overloaded one is a hard failure.
WITH active_components AS (
    SELECT
        ecl.*,
        CASE
            WHEN ecl.max_i_ka IS NULL OR ecl.max_i_ka = 0.0 THEN NULL
            ELSE ecl.max_i_from_ka
                / (ecl.max_i_ka * COALESCE(ecl.component_parallel, 1))
                * 100.0
        END AS loading_percent
    FROM expansion_component_loading ecl
),
unmapped AS (
    SELECT *
    FROM active_components
    WHERE visible_line_id IS NULL
)
SELECT
    (SELECT COUNT(*) FROM expansion_selected_run) AS selected_runs,
    (SELECT COUNT(*) FROM active_components) AS active_components,
    (SELECT COUNT(*) FROM unmapped) AS unmapped_components,
    COUNT(*) FILTER (
        WHERE COALESCE(loading_percent, 0.0) > 100.0
          AND NOT (
              source_line_name IS NULL
              AND source_geom_missing
              AND component_length_km <= 0.005
          )
    ) AS overloaded_unmapped_components,
    COUNT(*) FILTER (
        WHERE COALESCE(loading_percent, 0.0) > 100.0
          AND source_line_name IS NULL
          AND source_geom_missing
          AND component_length_km <= 0.005
    ) AS overloaded_root_connector_like_components,
    COALESCE(MAX(loading_percent), 0.0) AS max_unmapped_loading_percent,
    COUNT(*) FILTER (
        WHERE source_line_name IS NULL
          AND source_geom_missing
          AND component_length_km <= 0.005
    ) AS root_connector_like_components
FROM unmapped
