"""Stratified sample of postal code areas in Bavaria (Lee, 2026, Section 3.9.1).

Strata are the seven RegioStaR 7 types. Within each type the postal code areas are sorted by population and split
into eight groups with the same number of areas (numpy.array_split). One area per group is sampled, close to the
group median, so the sample has 7 x 8 = 56 areas. Where the area closest to the median could not be used, another
area of the same group was taken. The sample used in the thesis is data/extrapolation/bavaria_sample.csv.

This script rebuilds the quantile groups from data/extrapolation/bavaria_postal_code_areas.csv, checks that every
sampled area lies in its group and writes one row per group with its population range and the sampled area.

Usage:
    uv run python scripts/bavaria_extrapolation/sampling.py [--out sample_check.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gridexpand import paths

DATA_DIR = paths.DATA_DIR / "extrapolation"
N_GROUPS = 8


def load_areas() -> pd.DataFrame:
    """All Bavarian postal code areas with population and RegioStaR type, one row per area.

    The file has one row per postal code area and municipality. The population of the postal code area
    (pop_postcode) is used where available, otherwise pop. An area that spans municipalities of different types
    takes the lowest type number (the most urban type).
    """
    a = pd.read_csv(DATA_DIR / "bavaria_postal_code_areas.csv", dtype={"plz": str})
    a["plz"] = a.plz.str.zfill(5)
    b = a.groupby("plz").agg(pop=("pop", "first"), pop_postcode=("pop_postcode", "first"), regio7=("regio7", "min"))
    b["pop"] = b.pop_postcode.fillna(b["pop"])
    return b.loc[b["pop"] > 0, ["pop", "regio7"]]


def load_sample() -> pd.DataFrame:
    """The 56 sampled areas: plz (index), regio7, bin (quantile group 1-8), population of the synthetic grids."""
    s = pd.read_csv(DATA_DIR / "bavaria_sample.csv", dtype={"plz": str})
    s["plz"] = s.plz.str.zfill(5)
    return s.set_index("plz")


def quantile_groups(areas: pd.DataFrame, n_groups: int = N_GROUPS) -> pd.DataFrame:
    """Quantile group of every postal code area within its RegioStaR type."""
    rows = []
    for t, g in areas.groupby("regio7"):
        g = g.sort_values("pop")
        for k, idx in enumerate(np.array_split(np.arange(len(g)), n_groups), start=1):
            part = g.iloc[idx]
            rows += [(plz, t, k, part["pop"].min(), part["pop"].median(), part["pop"].max()) for plz in part.index]
    return pd.DataFrame(rows, columns=["plz", "regio7", "group", "pop_min", "pop_median", "pop_max"]).set_index("plz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="sample_check.csv", help="output CSV (default: %(default)s)")
    args = ap.parse_args()

    areas = load_areas()
    sample = load_sample()
    groups = quantile_groups(areas)
    print(f"{len(areas)} postal code areas, {int(areas['pop'].sum()):,} inhabitants")
    print("areas per RegioStaR type:", areas.groupby("regio7").size().to_dict())

    check = sample[["regio7", "bin"]].join(groups[["group", "pop_min", "pop_median", "pop_max"]])
    check["pop"] = areas["pop"].reindex(check.index)
    check["in_group"] = check["bin"] == check["group"]
    check.sort_values(["regio7", "bin"]).to_csv(args.out)
    print(f"sampled areas: {len(check)}, outside their quantile group: {int((~check.in_group).sum())}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
