-- Transformer size and cost (SQL twin of heuristics.required_transformer_kva and
-- transformer_upgrade_cost). Reads the relations `transformer_peak` (s_mva, rated_kva) and
-- `assumption`; required_kva is the P100 apparent power rounded up to the capacity step.
SELECT
    sized.*,
    CASE
        WHEN sized.required_kva <= sized.rated_kva THEN 0.0
        WHEN sized.required_kva <= 100.0 THEN assumption.transformer_replace_100_eur
        WHEN sized.required_kva <= 160.0 THEN assumption.transformer_replace_160_eur
        WHEN sized.required_kva <= 250.0 THEN assumption.transformer_replace_250_eur
        WHEN sized.required_kva <= 400.0 THEN assumption.transformer_replace_400_eur
        WHEN sized.required_kva <= 630.0 THEN assumption.transformer_replace_630_eur
        WHEN sized.required_kva <= 800.0 THEN assumption.transformer_replace_800_eur
        WHEN sized.required_kva <= 1000.0 THEN assumption.transformer_replace_1000_eur
        ELSE assumption.transformer_station_rebuild_boundary_eur
    END AS estimated_cost_eur,
    CASE
        WHEN sized.required_kva <= sized.rated_kva THEN 'none_existing_capacity_sufficient'
        WHEN sized.required_kva <= 100.0 THEN 'all_in_replacement_to_100kva'
        WHEN sized.required_kva <= 160.0 THEN 'all_in_replacement_to_160kva'
        WHEN sized.required_kva <= 250.0 THEN 'all_in_replacement_to_250kva'
        WHEN sized.required_kva <= 400.0 THEN 'all_in_replacement_to_400kva'
        WHEN sized.required_kva <= 630.0 THEN 'all_in_replacement_to_630kva'
        WHEN sized.required_kva <= 800.0 THEN 'all_in_replacement_to_800kva'
        WHEN sized.required_kva <= 1000.0 THEN 'all_in_replacement_to_1000kva'
        ELSE 'station_rebuild_boundary_case_gt_1000kva'
    END AS transformer_cost_basis
FROM (
    SELECT
        tp.*,
        -- 1e-12 tolerates float noise just above a step (as the cable search does).
        CEIL(
            (tp.s_mva * 1000.0)
            / assumption.transformer_capacity_step_kva
            - 1e-12
        ) * assumption.transformer_capacity_step_kva AS required_kva
    FROM transformer_peak tp
    CROSS JOIN assumption
) sized
CROSS JOIN assumption
