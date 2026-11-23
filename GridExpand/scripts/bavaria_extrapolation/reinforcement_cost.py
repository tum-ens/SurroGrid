"""Reinforcement cost of each LV grid from the power-flow results (Lee, 2026, Section 3.7).

Reads the power-flow HDF5 files of Step 4 (``--storage h5``: ``raw_data/net`` and
``pwrflw/output/post/{line_loads, vm, demand_import}``) and writes one row per grid with the cable and transformer
reinforcement and its cost. These files are the input of extrapolate.py and validate.py.

Reinforcement criteria
    - Cables: the annual maximum current of each line is compared with the permissible current times the limit,
      100 % (Wintzek et al. 2021) or 60 % (planning premises of Spalthoff et al. 2022), set with --limit.
      Additional parallel cables = ceil(peak current / (permissible current x limit)) - existing cables.
    - Transformer: the annual maximum apparent power at the grid connection is compared with 100 % of the rated
      power (Wintzek et al. 2021, DIN EN 60076-1).
    - Voltage (0.9 to 1.1 pu) is flagged, not priced.

Cable cost (Wintzek et al. 2021, EUR per km, 2021 prices; cables and civil works x1.384 to 2026, Destatis
construction price index, local sewers)
    cross-section   parallel laying   separate laying
    150 mm2              20,000           150,000
    185 mm2              40,000           175,000
    240 mm2              60,000           200,000
    Cables below 150 mm2 take the 150 mm2 cost, cables above 240 mm2 the 240 mm2 cost.
    - Method A (parallel laying, a spare conduit is available): n x parallel cost x length.
    - Method B (separate laying): (separate cost + (n - 1) x parallel cost) x length.

Transformer and substation cost (Wintzek et al. 2021, EUR, 2021 prices; x1.521 to 2026, Destatis producer price
index GP19-27114, electrical transformers)
    rONT units: 400 kVA 18,054 (linear fit to the Wintzek prices of 630-1,600 kVA), 630 kVA 21,200,
    800 kVA 24,200, 1,000 kVA 27,500, 1,200 kVA 30,000, 1,600 kVA 36,000.
    - Up to 1,600 kVA: the smallest single unit that covers the required capacity.
    - Above 1,600 kVA: as many 1,600 kVA units as fit plus the smallest unit for the remainder, in parallel at the
      existing supply point (no grid split).
    - Substation replacement (55,000) if the first unit is 1,000 kVA or above (threshold following Candas 2024).
    - New substation (45,000) for each additional parallel unit.
    The same transformer cost applies to methods A and B.

Cost cases of the thesis: A1 = total_cost_A at 100 %, B1 = total_cost_B at 100 %, A2 = total_cost_A at 60 %,
B2 = total_cost_B at 60 %.

Usage (name the output as extrapolate.py expects, e.g. bavaria_inflex_exp_100_TH60.csv):
    uv run python scripts/bavaria_extrapolation/reinforcement_cost.py <folder with power-flow h5> \
        --limit 0.6 --out bavaria_inflex_exp_100_TH60.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
import pandapower as pp
import pandas as pd

# price indices 2021 -> 2026
CABLE_INDEX = 1.384
TRANSFORMER_INDEX = 1.521

# cable cost in EUR per km, 2021 prices
SEPARATE = {150: 150000.0, 185: 175000.0, 240: 200000.0}
PARALLEL = {150: 20000.0, 185: 40000.0, 240: 60000.0}

# rONT units (kVA) and cost in EUR, 2021 prices
UNIT_KVA = [400, 630, 800, 1000, 1200, 1600]
UNIT_COST = [18054.0, 21200.0, 24200.0, 27500.0, 30000.0, 36000.0]

# substations, EUR, 2021 prices
SUBSTATION_THRESHOLD_KVA = 1000.0
SUBSTATION_REPLACEMENT = 55000.0
NEW_SUBSTATION = 45000.0

V_MIN, V_MAX = 0.9, 1.1


def cable_class(std_type: str) -> int | None:
    """Cost class of a line (150, 185 or 240 mm2); None for MV cables."""
    st = str(std_type)
    if "kV" in st or "N2XS" in st or "NA2XS" in st:
        return None
    if "240" in st:
        return 240
    if "185" in st:
        return 185
    if any(s in st for s in ("150", "120", "95", "70", "50", "35", "25", "16")):
        return 150
    return 240


def cable_cost_per_km(size: int | None, n_add: int) -> tuple[float, float]:
    """Cost per km of n_add additional cables, method A (parallel) and B (separate laying)."""
    if size is None or n_add <= 0:
        return 0.0, 0.0
    a = CABLE_INDEX * n_add * PARALLEL[size]
    b = CABLE_INDEX * (SEPARATE[size] + (n_add - 1) * PARALLEL[size])
    return a, b


def transformer_units(required_kva: float) -> list[int]:
    """Units installed for a required capacity: single units up to 1,600 kVA, parallel units above."""
    top = UNIT_KVA[-1]
    if required_kva > top:
        n_full = int(required_kva // top)
        rest = required_kva - n_full * top
        units = [top] * n_full
        if rest > 0:
            units.append(next(k for k in UNIT_KVA if k >= rest))
        return units
    return [next((k for k in UNIT_KVA if k >= required_kva), top)]


def transformer_cost(required_kva: float, rated_kva: float) -> tuple[float, float]:
    """Transformer and substation cost (EUR, 2026) and installed capacity (kVA); zero if the rating suffices."""
    if np.isnan(required_kva):
        return 0.0, np.nan
    if rated_kva and required_kva <= rated_kva:
        return 0.0, rated_kva
    units = transformer_units(required_kva)
    cost = dict(zip(UNIT_KVA, UNIT_COST))
    equipment = sum(cost[k] for k in units)
    substation = (SUBSTATION_REPLACEMENT if units[0] >= SUBSTATION_THRESHOLD_KVA else 0.0) \
        + NEW_SUBSTATION * (len(units) - 1)
    return TRANSFORMER_INDEX * equipment + TRANSFORMER_INDEX * substation, float(sum(units))


def analyze_grid(h5path: str, limit: float) -> dict:
    """Cable and transformer reinforcement and cost of one power-flow HDF5 file."""
    with h5py.File(h5path, "r") as hf:
        raw = hf["raw_data/net"][()]
        if isinstance(raw, np.ndarray):
            raw = raw[0]
        if isinstance(raw, bytes):
            raw = raw.decode()
    net = pp.from_json_string(raw)
    with pd.HDFStore(h5path, "r") as s:
        line_loads = s["/pwrflw/output/post/line_loads"]
        vm = s["/pwrflw/output/post/vm"]
        imp = s["/pwrflw/output/post/demand_import"]

    # cables
    lines = net.line[["std_type", "length_km", "max_i_ka", "parallel"]].copy()
    lines["peak_i_ka"] = line_loads.xs("i_from_ka", axis=1, level=1).max().values
    lines["loading_pct"] = lines.peak_i_ka / (lines.max_i_ka * lines.parallel) * 100
    lines["n_add"] = (np.ceil(lines.peak_i_ka / (lines.max_i_ka * limit)).astype(int) - lines.parallel).clip(lower=0)
    cost_a = cost_b = 0.0
    for r in lines[lines.n_add > 0].itertuples():
        a, b = cable_cost_per_km(cable_class(r.std_type), int(r.n_add))
        cost_a += a * float(r.length_km)
        cost_b += b * float(r.length_km)

    # transformer: peak apparent power at the grid connection
    rated_kva = float(net.trafo.sn_mva.iloc[0] * 1000) if len(net.trafo) else np.nan
    p, q = imp[imp.columns[0]], imp[imp.columns[1]]
    peak_kva = float(np.sqrt(p ** 2 + q ** 2).max() * 1000)
    trafo, installed_kva = transformer_cost(peak_kva, rated_kva)

    vmin, vmax = float(vm.min().min()), float(vm.max().max())
    n_overloaded = int((lines.n_add > 0).sum())
    return {
        "grid": os.path.basename(h5path).split("_")[0],
        "file": os.path.basename(h5path),
        "n_lines": len(lines),
        "n_overloaded_lines": n_overloaded,
        "max_line_loading_pct": round(float(lines.loading_pct.max()), 1),
        "line_cost_A": round(cost_a),
        "line_cost_B": round(cost_b),
        "trafo_rated_kva": rated_kva,
        "trafo_peak_kva": round(peak_kva, 1),
        "trafo_loading_pct": round(peak_kva / rated_kva * 100, 1) if rated_kva else np.nan,
        "trafo_installed_kva": installed_kva,
        "trafo_cost_AB": round(trafo),
        "total_cost_A": round(cost_a + trafo),
        "total_cost_B": round(cost_b + trafo),
        "requires_expansion": bool(n_overloaded > 0 or trafo > 0),
        "vmin_pu": round(vmin, 4),
        "vmax_pu": round(vmax, 4),
        "voltage_violation": bool(vmin < V_MIN or vmax > V_MAX),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="folder with the power-flow HDF5 files")
    ap.add_argument("--limit", type=float, default=1.0, help="cable thermal loading limit, 1.0 or 0.6")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.h5")))
    if not files:
        raise SystemExit(f"no h5 files in {args.folder}")
    rows, failed = [], []
    for i, f in enumerate(files, 1):
        try:
            rows.append(analyze_grid(f, args.limit))
        except Exception as e:  # noqa: BLE001 (report the file and continue with the other grids)
            failed.append((os.path.basename(f), str(e)))
            print(f"[{i}/{len(files)}] {os.path.basename(f)} failed: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} grids, {int(df.requires_expansion.sum())} need reinforcement; "
          f"total A {df.total_cost_A.sum() / 1e6:.2f} M EUR, B {df.total_cost_B.sum() / 1e6:.2f} M EUR")
    if failed:
        print(f"{len(failed)} files failed")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
