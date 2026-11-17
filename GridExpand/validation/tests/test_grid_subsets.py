"""Grid subsets of aligned runs (validation.grid_subsets)."""

from __future__ import annotations

import pandas as pd

from validation.grid_subsets import real_grid_number, select_grid_subset


def test_grid_subset(tmp_path):
    population = tmp_path / "population.json"
    population.write_text(__import__("json").dumps({"metric_cohort": {"dso_b": {"successful_components": [
        {"real": ["dso_b:area-1"], "synthetic": [101]},
        {"real": ["dso_b:area-2", "dso_b:area-3"], "synthetic": [102]},
    ]}}}))
    pd.DataFrame({"grid_result_id": [101, 102], "grid_case_id": [11, 12]}).to_csv(
        tmp_path / "paired_registered_synthetic_grids.csv", index=False)
    subset = select_grid_subset(population=population, provider="dso_b", paired_dir=tmp_path,
                                grid_subset={"method": "islands", "seed": 1, "real_grids_per_provider": None})
    assert subset["real_dso_b"] == [1] and subset["synthetic"] == [11]
    assert real_grid_number("dso_a", "LV_080__x") == 80
