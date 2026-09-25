"""Grid subsets of aligned (SWF + ÜZW) paired runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def real_grid_number(provider: str, name: str) -> int:
    """Real grid id of a pylovo cohort name (ÜZW ``…:area-12``, SWF ``LV_12__…``)."""
    number = name.split(":")[-1] if provider == "uzw" else name.split("__")[0]
    return int(number.removeprefix("area-").removeprefix("LV_"))


def select_grid_subset(
    *,
    population: Path,
    provider: str,
    paired_dir: Path,
    grid_subset: dict[str, Any],
) -> dict[str, Any]:
    """Pick whole overlap components so real and synthetic keep the same buildings.

    pylovo's metric cohort groups real and synthetic grids into components that
    share buildings. ``components`` takes them in a seeded random order while the
    provider's real-grid count stays at or below the requested number;
    ``islands`` keeps only one-real-one-synthetic components (identical building
    sets, directly comparable grid by grid), all of them unless a number is set.
    Needs the prepared dataset (``paired_registered_synthetic_grids.csv``).

    Returns:
        ``{seed, method, components, real_<provider>: [...], synthetic: [...]}``;
        the grid lists are the paired runner's ``--job-subset``.
    """
    document = json.loads(Path(population).read_text(encoding="utf-8"))
    components = list(document["metric_cohort"][provider]["successful_components"])
    if grid_subset["method"] == "islands":
        components = [c for c in components if len(c["real"]) == 1 and len(c["synthetic"]) == 1]
    random.Random(grid_subset["seed"]).shuffle(components)
    target = grid_subset["real_grids_per_provider"] or sum(len(c["real"]) for c in components)
    registered = pd.read_csv(Path(paired_dir) / "paired_registered_synthetic_grids.csv")
    case_by_result = dict(zip(registered["grid_result_id"].astype(str), registered["grid_case_id"].astype(int)))
    chosen, real, synthetic = [], [], []
    for component in components:
        if len(real) + len(component["real"]) > target:
            continue
        chosen.append(component)
        real.extend(real_grid_number(provider, name) for name in component["real"])
        synthetic.extend(case_by_result[str(value)] for value in component["synthetic"])
        if len(real) == target:
            break
    return {
        "seed": grid_subset["seed"],
        "method": f"{grid_subset['method']}: whole pylovo overlap components, seeded random order",
        "components": chosen,
        f"real_{provider}": sorted(real),
        "synthetic": sorted(synthetic),
    }
