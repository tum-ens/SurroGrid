"""Compare two surrogrid snapshots row by row on natural keys.

Usage: python compare.py <a.pkl> <b.pkl> [--rtol 1e-9] [--atol 1e-12] [--ignore t1,t2] [--verbose]

Rows are aligned on the natural key of each table (serial ids are already
replaced by natural keys in the snapshot). Prints per table: keys only in a/b,
and per value column the number of differing rows and the largest absolute and
relative difference. JSON 'assumptions' columns are compared per key.
Final line: IDENTICAL, EQUAL_WITHIN_TOLERANCE or DIFFERENT.
"""

from __future__ import annotations

import argparse
import ast
import pickle

import numpy as np
import pandas as pd

KEYS = {
    "allocated_demand": ["demand_allocation_run_id", "t_index", "bus", "commodity"],
    "allocated_eff_factor": ["demand_allocation_run_id", "t_index", "bus", "component"],
    "allocated_vehicle": ["demand_allocation_run_id", "bus", "vehicle_id"],
    "demand_allocation_run": ["demand_allocation_run_id"],
    "demand_component_audit": ["demand_allocation_run_id", "component_id", "commodity"],
    "electrification_assignment": ["demand_allocation_run_id", "building_objectid", "technology"],
    "expansion_analysis_run": ["analysis_key"],
    "expansion_cost_assumption": ["assumption_key"],
    "expansion_grid_result": ["expansion_analysis_run_id", "powerflow_run_id", "real_powerflow_run_id"],
    "expansion_line_qgis_mv": ["analysis_key", "powerflow_run_id", "visible_line_id"],
    "expansion_line_result": ["expansion_analysis_run_id", "powerflow_run_id", "visible_line_id"],
    "expansion_transformer_qgis_mv": ["analysis_key", "powerflow_run_id"],
    "expansion_transformer_result": ["expansion_analysis_run_id", "powerflow_run_id"],
    "grid_building_bus": ["grid_case_id", "objectid"],
    "grid_building_component": ["grid_case_id", "component_id"],
    "grid_case": ["grid_case_id"],
    "pipeline_run": ["pipeline_run_id"],
    "powerflow_bus_voltage": ["powerflow_run_id", "stage", "t_index", "bus"],
    "powerflow_bus_voltage_summary": ["powerflow_run_id", "stage", "bus"],
    "powerflow_cable_summary": ["powerflow_run_id", "stage", "cable"],
    "powerflow_demand": ["powerflow_run_id", "stage", "t_index", "bus"],
    "powerflow_import": ["powerflow_run_id", "stage", "t_index"],
    "powerflow_line_result": ["powerflow_run_id", "stage", "t_index", "line"],
    "powerflow_reactive_component": ["powerflow_run_id", "t_index", "bus", "component", "source"],
    "powerflow_run": ["powerflow_run_id"],
    "powerflow_summary": ["powerflow_run_id", "stage"],
    "powerflow_tail_value": ["powerflow_run_id", "stage", "metric", "asset_type", "asset_id", "tail", "t_index"],
    "powerflow_transformer_diagnostic": ["powerflow_run_id", "stage", "diagnostic", "point_index"],
    "scenario": ["scenario_key"],
}

parser = argparse.ArgumentParser()
parser.add_argument("a")
parser.add_argument("b")
parser.add_argument("--rtol", type=float, default=1e-9)
parser.add_argument("--atol", type=float, default=1e-12)
parser.add_argument("--ignore", default="")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
A = pickle.load(open(args.a, "rb"))
B = pickle.load(open(args.b, "rb"))
ignore = {name for name in args.ignore.split(",") if name}


def _as_dict(value):
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, list) and all(isinstance(i, tuple) and len(i) == 2 for i in parsed):
        # Absolute paths name the run directory of one harness run; compare basenames.
        return {k: (v.rsplit('/', 1)[-1] if isinstance(v, str) and v.startswith('/') else v) for k, v in parsed}
    return value


def _norm(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    return frame


exact = True
close = True
for name in sorted(set(A) | set(B)):
    if name in ignore:
        continue
    a, b = A.get(name), B.get(name)
    if a is None or b is None:
        only = "a" if b is None else "b"
        frame = a if b is None else b
        if len(frame):
            print(f"{name}: table only in {only} ({len(frame)} rows)")
            exact = close = False
        continue
    if len(a) == 0 and len(b) == 0:
        continue
    missing = sorted(set(a.columns) ^ set(b.columns))
    if missing:
        print(f"{name}: columns only on one side: {missing}")
        exact = close = False
    columns = [c for c in a.columns if c in b.columns]
    keys = [k for k in KEYS.get(name, []) if k in columns]
    if not keys:
        keys = [c for c in columns if a[c].dtype.kind not in "f"]
    a, b = _norm(a[columns]), _norm(b[columns])
    for frame in (a, b):
        frame["_dup"] = frame.groupby(keys, dropna=False).cumcount()
    keys = keys + ["_dup"]
    merged = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    only_a = int((merged["_merge"] == "left_only").sum())
    only_b = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    problems = []
    if only_a or only_b:
        problems.append(f"keys only in a: {only_a}, only in b: {only_b}")
        exact = close = False
    table_exact = not (only_a or only_b)
    for column in columns:
        if column in keys:
            continue
        x, y = both[f"{column}_a"], both[f"{column}_b"]
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            nan_x, nan_y = np.isnan(xv), np.isnan(yv)
            equal = (xv == yv) | (nan_x & nan_y)
            if equal.all():
                continue
            table_exact = False
            ok = equal | (~nan_x & ~nan_y & np.isclose(xv, yv, rtol=args.rtol, atol=args.atol))
            diff = np.abs(xv - yv)
            diff[nan_x | nan_y] = np.inf
            rel = diff / np.maximum(np.maximum(np.abs(xv), np.abs(yv)), 1e-300)
            text = f"{column}: {int((~equal).sum())}/{len(xv)} differ, max abs {np.nanmax(diff[~equal]):.3g}, max rel {np.nanmax(rel[~equal]):.3g}"
            if not ok.all():
                close = False
                problems.append(text)
            else:
                problems.append(text + " (within tol)")
        else:
            xs = x.map(_as_dict)
            ys = y.map(_as_dict)
            def _missing(v):
                return v is None or (not isinstance(v, (dict, list, str)) and bool(pd.isna(v)))
            neq = [i for i, (p, q) in enumerate(zip(xs, ys)) if not (_missing(p) and _missing(q)) and p != q]
            if not neq:
                continue
            table_exact = False
            close = False
            example = ""
            p, q = xs.iloc[neq[0]], ys.iloc[neq[0]]
            if isinstance(p, dict) and isinstance(q, dict):
                changed = sorted(k for k in set(p) | set(q) if p.get(k) != q.get(k))
                example = f"keys differ: {changed[:12]}"
            else:
                example = f"e.g. {str(p)[:80]!r} vs {str(q)[:80]!r}"
            problems.append(f"{column}: {len(neq)}/{len(both)} differ, {example}")
    if not table_exact:
        exact = False
    if problems:
        print(f"{name} ({len(a)} rows):")
        for line in problems:
            print(f"    {line}")
    elif args.verbose:
        print(f"{name} ({len(a)} rows): identical")

print("IDENTICAL" if exact else "EQUAL_WITHIN_TOLERANCE" if close else "DIFFERENT")
