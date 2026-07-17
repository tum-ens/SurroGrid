"""Audit readiness of real SWF sector-coupling assets for URBS handover."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pandapower as pp
from dotenv import load_dotenv

from ..paths import ENV_PATH, GRIDALLOC_DIR

DEFAULT_ALLOCATION_PLAN = (
    GRIDALLOC_DIR
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_building_match_91301"
    / "swf_2045_full_local_demand_bus_allocation_plan.csv"
)
DEFAULT_OUTPUT_DIR = (
    GRIDALLOC_DIR / "outputs" / "scenario_calibration" / "swf_2045_building_match_91301"
)
PROFILE_REFERENCE_COLUMNS = ["file", "file_add", "Char_P", "chr_name", "description"]


def _load_manifest(grid_data_path: Path, plz: int) -> pd.DataFrame:
    manifest_path = grid_data_path / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if "plz" in manifest.columns:
        manifest = manifest[
            pd.to_numeric(manifest["plz"], errors="coerce").eq(int(plz))
        ].copy()
    if "variant" in manifest.columns:
        manifest = manifest[manifest["variant"].astype(str).eq("radialized")].copy()
    elif "output_kind" in manifest.columns:
        manifest = manifest[manifest["output_kind"].astype(str).eq("radialized")].copy()
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"].astype(str).eq("exported")].copy()
    if "source_file" not in manifest.columns:
        path_column = next(
            (col for col in ["output_file", "path", "file"] if col in manifest.columns),
            None,
        )
        if path_column is None:
            raise ValueError(
                "split_manifest.csv has no source_file/output_file/path/file column."
            )
        manifest["source_file"] = manifest[path_column]
    manifest["source_file"] = manifest["source_file"].map(
        lambda value: _resolve_grid_file(grid_data_path, value)
    )
    return manifest


def _resolve_grid_file(grid_data_path: Path, value: Any) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    candidates = [grid_data_path / path, grid_data_path / "radialized" / path.name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _lv_id_from_row(row: pd.Series) -> int | None:
    for column in ["lv_id", "grid_id", "name"]:
        if column in row and not pd.isna(row[column]):
            text = str(row[column])
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits)
    source = str(row.get("source_file", ""))
    stem = Path(source).stem
    if stem.startswith("LV_"):
        digits = stem.split("__", 1)[0].replace("LV_", "")
        if digits.isdigit():
            return int(digits)
    return None


def _existing_profile_files(
    profile_roots: list[Path], values: pd.Series
) -> tuple[int, int, list[str]]:
    refs = sorted(
        {
            str(value)
            for value in values.dropna().unique()
            if str(value) not in {"", "None", "nan"}
        }
    )
    if not refs:
        return 0, 0, []
    missing = []
    existing = 0
    for ref in refs:
        if any((root / ref).exists() for root in profile_roots):
            existing += 1
        else:
            missing.append(ref)
    return len(refs), existing, missing[:10]


def _summarize_grid(
    source_file: Path, lv_id: int, plan: pd.DataFrame, profile_roots: list[Path]
) -> dict[str, Any]:
    net = pp.from_excel(source_file)
    load = net.load.copy() if hasattr(net, "load") else pd.DataFrame()
    sgen = net.sgen.copy() if hasattr(net, "sgen") else pd.DataFrame()
    plan_lv = plan[plan["lv_id"].astype(int).eq(int(lv_id))].copy()

    row: dict[str, Any] = {
        "lv_id": int(lv_id),
        "source_file": str(source_file),
        "plan_rows": int(len(plan_lv)),
        "plan_buses": int(plan_lv["allocation_bus"].nunique())
        if not plan_lv.empty
        else 0,
        "plan_hh_rows": int(
            pd.to_numeric(
                plan_lv.get("residential_equivalent_hh_rows"), errors="coerce"
            )
            .fillna(0)
            .sum()
        )
        if not plan_lv.empty
        else 0,
        "plan_ghd_kwh": float(
            pd.to_numeric(plan_lv.get("calibrated_annual_ghd_kwh"), errors="coerce")
            .fillna(0)
            .sum()
        )
        if not plan_lv.empty
        else 0.0,
    }

    for prefix in ["residential", "ghd", "unsupported_nonres"]:
        for metric in ["ev_charger_kw", "pv_kw", "battery_kwh"]:
            column = f"{prefix}_{metric}"
            row[f"plan_{column}"] = (
                float(
                    pd.to_numeric(plan_lv.get(column), errors="coerce").fillna(0).sum()
                )
                if column in plan_lv
                else 0.0
            )
        for metric in [
            "wp_rows",
            "ev_charger_rows",
            "pv_rows",
            "battery_rows",
            "heat_storage_rows",
        ]:
            column = f"{prefix}_{metric}"
            row[f"plan_{column}"] = (
                int(pd.to_numeric(plan_lv.get(column), errors="coerce").fillna(0).sum())
                if column in plan_lv
                else 0
            )

    if not load.empty and "type" in load.columns:
        load_type = load["type"].astype(str)
        row["swf_load_hh_rows"] = int(load_type.eq("HH").sum())
        row["swf_load_ghd_rows"] = int(load_type.eq("GHD").sum())
        row["swf_load_ev_rows"] = int(load_type.eq("Ladestation").sum())
        row["swf_load_wp_rows"] = int(load_type.eq("WP").sum())
        if "load_type" in load.columns:
            for load_type_name in ["ev", "hp", "heat", "dhw"]:
                row[f"swf_load_type_{load_type_name}_rows"] = int(
                    load["load_type"].astype(str).eq(load_type_name).sum()
                )
    else:
        row.update(
            {
                "swf_load_hh_rows": 0,
                "swf_load_ghd_rows": 0,
                "swf_load_ev_rows": 0,
                "swf_load_wp_rows": 0,
            }
        )

    if not sgen.empty and "type" in sgen.columns:
        sgen_type = sgen["type"].astype(str)
        row["swf_sgen_pv_rows"] = int(sgen_type.eq("Photovoltaik").sum())
        row["swf_sgen_battery_rows"] = int(sgen_type.eq("Batterie").sum())
        row["swf_sgen_heat_storage_rows"] = int(sgen_type.eq("Wärmespeicher").sum())
    else:
        row.update(
            {
                "swf_sgen_pv_rows": 0,
                "swf_sgen_battery_rows": 0,
                "swf_sgen_heat_storage_rows": 0,
            }
        )

    profile_values = []
    for table in [load, sgen]:
        for column in PROFILE_REFERENCE_COLUMNS:
            if column in table.columns:
                profile_values.append(table[column])
    if profile_values:
        refs, existing, missing_examples = _existing_profile_files(
            profile_roots, pd.concat(profile_values, ignore_index=True)
        )
    else:
        refs, existing, missing_examples = 0, 0, []
    row["profile_references"] = refs
    row["profile_references_existing"] = existing
    row["profile_references_missing"] = refs - existing
    row["profile_reference_missing_examples"] = json.dumps(
        missing_examples, ensure_ascii=False
    )

    row["can_use_swf_profile_files"] = refs > 0 and refs == existing
    row["can_materialize_base_electricity"] = bool(
        row["plan_hh_rows"] > 0 or row["plan_ghd_kwh"] > 0
    )
    row["can_materialize_sector_assets_without_extra_assumptions"] = bool(
        row["can_use_swf_profile_files"]
    )
    if not row["can_materialize_sector_assets_without_extra_assumptions"]:
        row["sector_blocker"] = (
            "SWF sector rows reference profile CSV files that are not available under the configured profile roots."
        )
    else:
        row["sector_blocker"] = ""
    return row


def build_sector_readiness_audit(
    *,
    grid_data_path: Path,
    allocation_plan: Path,
    plz: int,
    profile_roots: list[Path],
    lv_id: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    manifest = _load_manifest(grid_data_path, plz)
    plan = pd.read_csv(allocation_plan)
    if lv_id is not None:
        manifest = manifest[
            manifest.apply(lambda row: _lv_id_from_row(row) == int(lv_id), axis=1)
        ].copy()
    rows = []
    for _, manifest_row in manifest.iterrows():
        current_lv_id = _lv_id_from_row(manifest_row)
        if current_lv_id is None:
            continue
        source_file = Path(manifest_row["source_file"])
        if not source_file.exists():
            continue
        rows.append(_summarize_grid(source_file, current_lv_id, plan, profile_roots))
        if limit is not None and len(rows) >= int(limit):
            break
    return pd.DataFrame(rows).sort_values("lv_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real SWF sector asset readiness for URBS handover."
    )
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--grid-data-path", type=Path, default=None)
    parser.add_argument("--allocation-plan", type=Path, default=DEFAULT_ALLOCATION_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lv-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile-root", action="append", type=Path, default=[])
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=True)
    grid_data_path = args.grid_data_path or Path(os.environ["GRID_DATA_PATH"])
    profile_roots = args.profile_root or [
        grid_data_path,
        grid_data_path / "profiles",
        grid_data_path / "profile",
        grid_data_path.parent,
    ]
    audit = build_sector_readiness_audit(
        grid_data_path=grid_data_path,
        allocation_plan=args.allocation_plan,
        plz=args.plz,
        profile_roots=profile_roots,
        lv_id=args.lv_id,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_lv_{int(args.lv_id):03d}" if args.lv_id is not None else ""
    output_path = args.output_dir / f"swf_2045_sector_readiness_audit{suffix}.csv"
    audit.to_csv(output_path, index=False)
    print(output_path)
    if not audit.empty:
        summary = {
            "lv_grids": int(len(audit)),
            "base_ready": int(audit["can_materialize_base_electricity"].sum()),
            "sector_ready_without_extra_assumptions": int(
                audit["can_materialize_sector_assets_without_extra_assumptions"].sum()
            ),
            "profile_references": int(audit["profile_references"].sum()),
            "profile_references_missing": int(
                audit["profile_references_missing"].sum()
            ),
            "swf_load_ev_rows": int(
                audit.get("swf_load_ev_rows", pd.Series(dtype=int)).sum()
            ),
            "swf_load_wp_rows": int(
                audit.get("swf_load_wp_rows", pd.Series(dtype=int)).sum()
            ),
            "swf_sgen_pv_rows": int(
                audit.get("swf_sgen_pv_rows", pd.Series(dtype=int)).sum()
            ),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
