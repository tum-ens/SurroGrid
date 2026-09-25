-- Least-cost cable reinforcement per component (SQL twin of heuristics.select_cable_reinforcement).
-- Reads the relations `component_loading` (required_added_capacity_ka, settlement_type,
-- component_length_km, component_parallel) and `assumption` (one expansion_cost_assumption row).
SELECT
    cl.*,
    cl.component_parallel + selection.additional_parallel AS required_parallel,
    selection.additional_parallel,
    selection.reinforcement_150_count,
    selection.reinforcement_185_count,
    selection.reinforcement_240_count,
    selection.reinforcement_added_capacity_ka,
    'NAYY_4_150|NAYY_4_185|NAYY_4_240'::TEXT AS reinforcement_catalog,
    selection.line_cost_eur_per_km,
    selection.line_cost_basis,
    selection.duct_cost_eur_per_km,
    selection.reopen_cost_eur_per_km,
    selection.existing_duct_share,
    selection.trenching_share,
    COALESCE(cl.component_length_km, 0.0)
        * selection.line_cost_eur_per_km AS estimated_component_cost_eur
FROM component_loading cl
CROSS JOIN assumption
CROSS JOIN LATERAL (
    SELECT LEAST(
        GREATEST(
            COALESCE(:line_existing_duct_share, assumption.line_existing_duct_share),
            0.0
        ),
        1.0
    ) AS existing_duct_share
) share
CROSS JOIN LATERAL (
    SELECT
        CASE
            WHEN cl.settlement_type = 1 THEN assumption.line_reopen_rural_eur_per_km
            WHEN cl.settlement_type = 3 THEN assumption.line_reopen_urban_eur_per_km
            ELSE assumption.line_reopen_suburban_eur_per_km
        END AS reopen_cost_eur_per_km,
        CASE
            WHEN cl.settlement_type = 1 THEN 'rural'
            WHEN cl.settlement_type = 3 THEN 'urban'
            ELSE 'semiurban'
        END AS settlement_label
) route
CROSS JOIN LATERAL (
    SELECT
        candidate.n150 AS reinforcement_150_count,
        candidate.n185 AS reinforcement_185_count,
        candidate.n240 AS reinforcement_240_count,
        candidate.n150 + candidate.n185 + candidate.n240 AS additional_parallel,
        candidate.added_capacity_ka AS reinforcement_added_capacity_ka,
        candidate.total_duct_cost_eur_per_km AS duct_cost_eur_per_km,
        route.reopen_cost_eur_per_km,
        share.existing_duct_share,
        1.0 - share.existing_duct_share AS trenching_share,
        candidate.total_cost_eur_per_km AS line_cost_eur_per_km,
        CASE
            WHEN candidate.n150 + candidate.n185 + candidate.n240 = 0
                THEN 'none_existing_capacity_sufficient'
            ELSE CONCAT(
                'catalog_', route.settlement_label,
                '_duct', ROUND((share.existing_duct_share * 100.0)::NUMERIC)::TEXT,
                '_trench', ROUND(((1.0 - share.existing_duct_share) * 100.0)::NUMERIC)::TEXT,
                '_150x', candidate.n150,
                '_185x', candidate.n185,
                '_240x', candidate.n240
            )
        END AS line_cost_basis
    FROM (
        SELECT
            n150,
            n185,
            n240,
            n150 * assumption.line_reinforcement_150_max_i_ka
                + n185 * assumption.line_reinforcement_185_max_i_ka
                + n240 * assumption.line_reinforcement_240_max_i_ka
                AS added_capacity_ka,
            n150 * assumption.line_parallel_150_eur_per_km
                + n185 * assumption.line_parallel_185_eur_per_km
                + n240 * assumption.line_parallel_240_eur_per_km
                AS total_duct_cost_eur_per_km,
            CASE
                WHEN n150 + n185 + n240 = 0 THEN 0.0
                ELSE
                    n150 * assumption.line_parallel_150_eur_per_km
                    + n185 * assumption.line_parallel_185_eur_per_km
                    + n240 * assumption.line_parallel_240_eur_per_km
                    + (1.0 - share.existing_duct_share) * (
                        route.reopen_cost_eur_per_km
                        - GREATEST(
                            CASE WHEN n150 > 0 THEN assumption.line_parallel_150_eur_per_km ELSE 0.0 END,
                            CASE WHEN n185 > 0 THEN assumption.line_parallel_185_eur_per_km ELSE 0.0 END,
                            CASE WHEN n240 > 0 THEN assumption.line_parallel_240_eur_per_km ELSE 0.0 END
                        )
                    )
            END AS total_cost_eur_per_km
        FROM generate_series(
            0,
            CEIL(
                cl.required_added_capacity_ka
                / assumption.line_reinforcement_150_max_i_ka
            )::INTEGER
        ) n150
        CROSS JOIN generate_series(
            0,
            CEIL(
                cl.required_added_capacity_ka
                / assumption.line_reinforcement_150_max_i_ka
            )::INTEGER
        ) n185
        CROSS JOIN generate_series(
            0,
            CEIL(
                cl.required_added_capacity_ka
                / assumption.line_reinforcement_150_max_i_ka
            )::INTEGER
        ) n240
        WHERE (
            cl.required_added_capacity_ka <= 1e-12
            AND n150 + n185 + n240 = 0
        ) OR (
            n150 + n185 + n240 > 0
            AND n150 * assumption.line_reinforcement_150_max_i_ka
                + n185 * assumption.line_reinforcement_185_max_i_ka
                + n240 * assumption.line_reinforcement_240_max_i_ka
                >= cl.required_added_capacity_ka - 1e-12
        )
    ) candidate
    ORDER BY
        candidate.total_cost_eur_per_km,
        candidate.n150 + candidate.n185 + candidate.n240,
        candidate.added_capacity_ka - cl.required_added_capacity_ka,
        candidate.n240 DESC,
        candidate.n185 DESC
    LIMIT 1
) selection
