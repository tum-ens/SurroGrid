"""Leave-one-out validation of the Bavaria extrapolation (Lee, 2026, Section 4.2.3, Table 4.22).

The model of extrapolate.py is refitted 56 times, each time without one sampled postal code area. For each run:
  - the cost of the left-out area is predicted and compared with its calculated cost;
  - the Bavaria total is recomputed and compared with the total of the full-sample fit.
The centring value (mean log population of the full sample) is kept in every run.

Reported per scenario and level, as the range over the four cost cases: change of the Bavaria total, median
absolute error per inhabitant, median absolute percentage error, and the error of the sum over the left-out areas.

Usage:
    uv run python scripts/bavaria_extrapolation/validate.py --costs-dir <folder> [--out-dir <folder>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from extrapolate import CASES, CHECK_LEVELS, SCENARIOS, design, fit, load_costs, predict
from sampling import load_areas, load_sample


def leave_one_out(costs: pd.DataFrame, pop: pd.Series, mu: float, type_of: pd.Series, types: list[int],
                  areas: pd.DataFrame) -> pd.DataFrame:
    """One row per cost case, level and left-out area."""
    rows = []
    for case in CASES:
        d = costs[costs.case == case].reset_index(drop=True)
        r, _, smear = fit(d, pop, mu, type_of, types)
        for L in CHECK_LEVELS:
            full = predict(r.params, smear, areas, L, mu, types).sum()
            for plz in sorted(d.plz.unique()):
                r2, _, s2 = fit(d[d.plz != plz], pop, mu, type_of, types)
                total = predict(r2.params, s2, areas, L, mu, types).sum()
                x = design(pd.Series([plz]), pd.Series([L]), pd.Series([np.log(pop[plz]) - mu]), type_of, types)
                pred = float(np.exp(x[r2.params.index].to_numpy() @ r2.params.to_numpy())[0] * s2[L])
                calc = float(d[(d.plz == plz) & (d.level == L)].cost.iloc[0])
                rows.append((case, L, plz, pop[plz], calc, pred, 100 * (total / full - 1)))
    r = pd.DataFrame(rows, columns=["case", "level", "plz", "population", "cost_calculated", "cost_predicted",
                                    "total_change_pct"])
    r["abs_error_per_inhabitant"] = (r.cost_predicted - r.cost_calculated).abs() / r.population
    r["abs_pct_error"] = 100 * (r.cost_predicted - r.cost_calculated).abs() / r.cost_calculated
    return r


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Range over the cost cases of each statistic, one column per level."""
    g = runs.groupby(["level", "case"])
    c = pd.DataFrame({"chg_min": g.total_change_pct.min(), "chg_max": g.total_change_pct.max(),
                      "mae": g.abs_error_per_inhabitant.median(), "mape": g.abs_pct_error.median(),
                      "sum": 100 * (g.cost_predicted.sum() / g.cost_calculated.sum() - 1)}).groupby(level="level")

    def rng(lo, hi, fmt):
        return lo.map(fmt.format) + " to " + hi.map(fmt.format)

    return pd.DataFrame({
        "Change in the Bavaria total (%)": rng(c.chg_min.min(), c.chg_max.max(), "{:.1f}"),
        "Median absolute error (EUR per inhabitant)": rng(c.mae.min(), c.mae.max(), "{:.0f}"),
        "Median absolute percentage error (%)": rng(c.mape.min(), c.mape.max(), "{:.0f}"),
        "Error of the sum over the left-out areas (%)": rng(c["sum"].min(), c["sum"].max(), "{:.1f}"),
    }).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--costs-dir", required=True, type=Path, help="folder with the per-grid cost files")
    ap.add_argument("--out-dir", default="extrapolation_output", type=Path, help="output folder")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample, areas = load_sample(), load_areas()
    pop, type_of = sample.population, sample.regio7
    types = sorted(type_of.unique().tolist())
    mu = float(np.log(pop).mean())
    for sc in SCENARIOS:
        costs = load_costs(args.costs_dir, sc)
        runs = leave_one_out(costs[costs.plz.isin(pop.index)], pop, mu, type_of, types, areas)
        runs.to_csv(args.out_dir / f"leave_one_out_runs_{sc}.csv", index=False)
        s = summary(runs)
        s.to_csv(args.out_dir / f"leave_one_out_summary_{sc}.csv")
        print(f"\n{sc}: leave-one-out, range over the cost cases")
        print(s.to_string())
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
