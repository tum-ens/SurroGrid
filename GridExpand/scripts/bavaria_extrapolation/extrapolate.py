"""Extrapolation of the LV reinforcement cost of the sampled postal code areas to Bavaria (Lee, 2026, 3.9.2, 4.2).

Input: the reinforcement cost of every synthetic grid of the 56 sampled areas, one CSV per scenario (inflex, flex),
electrification level (25, 50, 100 %) and cable thermal loading limit (100, 60 %), named
bavaria_{scenario}_exp_{level}_TH{limit}.csv, with the columns file (<grid>_<plz>_...), total_cost_A, total_cost_B,
line_cost_A, line_cost_B and trafo_cost_AB (EUR), written by reinforcement_cost.py. These files are not in git.

Cost cases: A = parallel cable laying, B = separate laying; 1 = 100 %, 2 = 60 % cable limit.

Model, one fit per scenario and cost case:
    log(cost) = b * (log(pop) - mean log(pop)) + sum_t c_t * Type_t + e25 * E25 + e50 * E50
    - cost: sum of the reinforcement cost of all grids of the postal code area;
    - population centred on the sample mean; one dummy per RegioStaR type, no constant;
    - 100 % electrification is the reference level;
    - OLS with standard errors clustered by postal code area, as each area appears at every level;
    - F tests of population x level and type x level interactions check that the effects are the same at all levels.

Prediction: the model is applied to all 2,064 postal code areas. exp(prediction) estimates the median, so it is
multiplied by Duan's smearing factor (mean of exp(residual)), estimated per level.

Outputs in --out-dir: regression tables (Tables 4.16-4.18), Bavaria total and cost per inhabitant (Table 4.20), cost
by RegioStaR type (4.21), areas outside the sampled population range (4.23), +/-20 % unit-cost sensitivity (4.24).

Usage:
    uv run python scripts/bavaria_extrapolation/extrapolate.py --costs-dir <folder> [--out-dir <folder>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sampling import load_areas, load_sample

LEVELS = [25, 50, 100]                      # the last level is the reference
CASES = {"A1": ("A", 100), "B1": ("B", 100), "A2": ("A", 60), "B2": ("B", 60)}
SCENARIOS = ["inflex", "flex"]
CHECK_LEVELS = [50, 100]                    # levels of the range check and the sensitivity
UNIT_COST_VARIATION = 0.20


# ---------------------------------------------------------------- data
def load_costs(costs_dir: Path, scenario: str, scale_cable: float = 1.0, scale_trafo: float = 1.0) -> pd.DataFrame:
    """Cost per postal code area, case and level (columns case, level, plz, cost).

    Without scaling the totals of the grid files are summed; with scaling the cable and the transformer and
    substation cost are scaled separately (unit-cost sensitivity).
    """
    rows = []
    for level in LEVELS:
        for limit in (100, 60):
            d = pd.read_csv(Path(costs_dir) / f"bavaria_{scenario}_exp_{level}_TH{limit}.csv")
            d["plz"] = d["file"].str.split("_").str[1]
            for m in ("A", "B"):
                if scale_cable == 1.0 and scale_trafo == 1.0:
                    d[m] = d[f"total_cost_{m}"]
                else:
                    d[m] = d[f"line_cost_{m}"] * scale_cable + d["trafo_cost_AB"] * scale_trafo
            g = d.groupby("plz")[["A", "B"]].sum()
            for case, (m, lim) in CASES.items():
                if lim == limit:
                    rows += [(case, level, plz, v) for plz, v in g[m].items()]
    out = pd.DataFrame(rows, columns=["case", "level", "plz", "cost"])
    if (out.cost <= 0).any():
        raise ValueError(f"non-positive cost, log undefined:\n{out[out.cost <= 0]}")
    return out


# ---------------------------------------------------------------- model
def design(plz: pd.Series, level: pd.Series, lpop_c: pd.Series, type_of: pd.Series, types: list[int],
           interaction: str | None = None) -> pd.DataFrame:
    """Regressors: centred log population, type dummies, level dummies, optional interactions ('pop' or 'type')."""
    X = pd.DataFrame(index=plz.index)
    X["lpop"] = lpop_c
    t = plz.map(type_of)
    for ty in types:
        X[f"T{ty}"] = (t == ty).astype(float)
    for L in LEVELS[:-1]:
        X[f"E{L}"] = (level == L).astype(float)
    if interaction == "pop":
        for L in LEVELS[:-1]:
            X[f"lpop_E{L}"] = X.lpop * X[f"E{L}"]
    elif interaction == "type":
        for ty in types[1:]:
            for L in LEVELS[:-1]:
                X[f"T{ty}_E{L}"] = X[f"T{ty}"] * X[f"E{L}"]
    return X


def fit(d: pd.DataFrame, pop: pd.Series, mu: float, type_of: pd.Series, types: list[int],
        interaction: str | None = None, clustered: bool = False):
    """OLS fit of one cost case; returns the statsmodels result, the design and the smearing factor per level."""
    d = d.reset_index(drop=True)
    X = design(d.plz, d.level, np.log(d.plz.map(pop)) - mu, type_of, types, interaction)
    y = np.log(d.cost)
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": d.plz.astype("category").cat.codes}} if clustered else {}
    r = sm.OLS(y, X).fit(**kw)
    smear = {L: float(np.mean(np.exp(r.resid[(d.level == L).to_numpy()]))) for L in LEVELS}
    return r, X, smear


def predict(params: pd.Series, smear: dict, areas: pd.DataFrame, level: int, mu: float, types: list[int],
            interaction: str | None = None) -> pd.Series:
    """Predicted cost (EUR) of every postal code area at one level, with smearing."""
    plz = pd.Series(areas.index, index=areas.index)
    X = design(plz, pd.Series(level, index=areas.index), np.log(areas["pop"]) - mu, areas["regio7"], types,
               interaction)
    return np.exp(X[params.index] @ params) * smear[level]


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of each regressor (no-constant design, uncentred R2)."""
    A = X.to_numpy()
    out = {}
    for j, c in enumerate(X.columns):
        y, Z = A[:, j], np.delete(A, j, axis=1)
        r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        out[c] = 1.0 / (r @ r / (y @ y))
    v = pd.Series(out)
    return pd.DataFrame({"VIF": v, "Tolerance": 1 / v})


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def regression_column(d: pd.DataFrame, pop: pd.Series, mu: float, type_of: pd.Series, types: list[int]) -> dict:
    """Coefficients (clustered SE), R2 and the F and t tests of one cost case."""
    r, X, _ = fit(d, pop, mu, type_of, types, clustered=True)
    y = np.log(d.reset_index(drop=True).cost)
    n, k = X.shape
    r2 = 1 - ((y - r.fittedvalues) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def wald(res, R):
        w = res.wald_test(np.atleast_2d(R), use_f=True, scalar=True)
        return float(w.statistic), float(w.pvalue)

    E, ix = np.eye(k), list(X.columns).index
    contrasts = [E[ix(f"T{t}")] - E[ix(f"T{types[0]}")] for t in types[1:]]
    f_model = wald(r, [E[ix("lpop")], *contrasts, *[E[ix(f"E{L}")] for L in LEVELS[:-1]]])
    f_type = wald(r, contrasts)
    col = {c: f"{r.params[c]:.3f} ({r.bse[c]:.3f})" for c in X.columns}
    col.update({"R2": f"{r2:.3f}", "Adjusted R2": f"{1 - (1 - r2) * (n - 1) / (n - k):.3f}",
                "F model": f"{f_model[0]:.2f}{stars(f_model[1])}",
                "t log(population)": f"{r.tvalues['lpop']:.2f}{stars(r.pvalues['lpop'])}",
                "F RegioStaR type": f"{f_type[0]:.2f}{stars(f_type[1])}"})
    for inter in ("pop", "type"):
        ri, Xi, _ = fit(d, pop, mu, type_of, types, inter, clustered=True)
        added = [j for j, c in enumerate(Xi.columns) if c not in X.columns]
        f = wald(ri, np.eye(Xi.shape[1])[added])
        col[f"F {inter} x electrification level (p)"] = f"{f[0]:.2f}{stars(f[1])} ({f[1]:.3f})"
    return col


# ---------------------------------------------------------------- main
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
    inh = areas["pop"].sum()
    print(f"sample: {len(sample)} areas, centring value {np.exp(mu):.0f} inhabitants; "
          f"Bavaria: {len(areas)} areas, {inh:,.0f} inhabitants")

    for sc in SCENARIOS:
        costs = load_costs(args.costs_dir, sc)
        costs = costs[costs.plz.isin(pop.index)]

        # Tables 4.16-4.18
        table = {c: regression_column(costs[costs.case == c], pop, mu, type_of, types) for c in CASES}
        pd.DataFrame(table).to_csv(args.out_dir / f"regression_table_{sc}.csv")
        _, X, _ = fit(costs[costs.case == "A1"], pop, mu, type_of, types)
        vif_table(X).to_csv(args.out_dir / "vif.csv")

        total, by_type, outside, sens = [], [], [], []
        lo, hi = pop.min(), pop.max()
        t_lo, t_hi = pop.groupby(type_of).min(), pop.groupby(type_of).max()
        below_t, above_t = areas["pop"] < areas.regio7.map(t_lo), areas["pop"] > areas.regio7.map(t_hi)
        ranges = {"outside the sample range": (areas["pop"] < lo) | (areas["pop"] > hi),
                  "below the range of the own type": below_t, "above the range of the own type": above_t,
                  "outside the range of the own type": below_t | above_t}

        for case in CASES:
            d = costs[costs.case == case]
            r, _, smear = fit(d, pop, mu, type_of, types)
            ri, _, smear_i = fit(d, pop, mu, type_of, types, "type")
            for L in LEVELS:
                p = predict(r.params, smear, areas, L, mu, types)
                p_i = predict(ri.params, smear_i, areas, L, mu, types, "type").sum()
                total.append((case, L, smear[L], p.sum() / 1e9, p.sum() / inh, 100 * (p_i / p.sum() - 1)))
                if L == LEVELS[-1]:                                            # Table 4.21
                    per_type = p.groupby(areas.regio7).sum() / areas.groupby("regio7")["pop"].sum()
                    by_type += [(case, t, v) for t, v in per_type.items()]
                if L in CHECK_LEVELS:                                          # Table 4.23
                    outside += [(case, L, k, int(m.sum()), 100 * areas["pop"][m].sum() / inh,
                                 100 * p[m].sum() / p.sum()) for k, m in ranges.items()]
            v = UNIT_COST_VARIATION                                            # Table 4.24
            for label, s_cab, s_tr in [(f"cable +{v:.0%}", 1 + v, 1.0), (f"cable -{v:.0%}", 1 - v, 1.0),
                                       (f"transformer +{v:.0%}", 1.0, 1 + v), (f"transformer -{v:.0%}", 1.0, 1 - v)]:
                c2 = load_costs(args.costs_dir, sc, s_cab, s_tr)
                r2, _, smear2 = fit(c2[(c2.case == case) & c2.plz.isin(pop.index)], pop, mu, type_of, types)
                for L in CHECK_LEVELS:
                    base = predict(r.params, smear, areas, L, mu, types).sum()
                    sens.append((case, L, label, 100 * (predict(r2.params, smear2, areas, L, mu, types).sum()
                                                        / base - 1)))

        T = pd.DataFrame(total, columns=["case", "level", "smearing", "total_bn_eur", "eur_per_inhabitant",
                                         "type_x_level_model_diff_pct"])
        T.to_csv(args.out_dir / f"total_{sc}.csv", index=False)
        pd.DataFrame(by_type, columns=["case", "regio7", "eur_per_inhabitant"]).to_csv(
            args.out_dir / f"by_type_{sc}.csv", index=False)
        pd.DataFrame(outside, columns=["case", "level", "group", "n_areas", "population_share_pct",
                                       "cost_share_pct"]).to_csv(args.out_dir / f"outside_sample_range_{sc}.csv",
                                                                 index=False)
        pd.DataFrame(sens, columns=["case", "level", "variation", "change_pct"]).to_csv(
            args.out_dir / f"unit_cost_sensitivity_{sc}.csv", index=False)
        print(f"\n{sc}: EUR per inhabitant")
        print(T.pivot(index="level", columns="case", values="eur_per_inhabitant")[list(CASES)].round(2).to_string())
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
