"""Delete DB-backed SurroGrid data and Step 3 files for one scenario key."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[2]
GRIDEXPAND_DIR = REPO_ROOT / "GridExpand"
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase

STEP3_DIR = REPO_ROOT / "GridExpand" / "3.urbs"
STEP3_ARTIFACT_DIRS = (
    STEP3_DIR / "Input",
    STEP3_DIR / "result",
    STEP3_DIR / "logs",
    STEP3_DIR / "logs" / "gurobi",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete SurroGrid data for one scenario_key. Defaults to dry-run."
    )
    parser.add_argument("scenario_key", help="Readable surrogrid.scenario.scenario_key to clean up.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows and Step 3 files. Without this flag, only counts are printed.",
    )
    parser.add_argument(
        "--keep-demands",
        action="store_true",
        help="Keep the scenario, pipeline, and Step 2 demand-allocation rows; delete downstream results only.",
    )
    parser.add_argument(
        "--no-refresh-expansion-views",
        action="store_true",
        help="Skip refreshing Step 5 QGIS materialized views after deletion.",
    )
    return parser


def _step3_artifact_names(db: SurroGridDatabase, scenario_key: str) -> tuple[set[str], set[str]]:
    query = text(
        """
        WITH selected_scenario AS (
            SELECT scenario_id
            FROM surrogrid.scenario
            WHERE scenario_key = :scenario_key
        ), artifact_names AS (
            SELECT bridge_filename AS filename
            FROM surrogrid.demand_allocation_run
            WHERE scenario_id = (SELECT scenario_id FROM selected_scenario)
            UNION
            SELECT urbs_input_file AS filename
            FROM surrogrid.powerflow_run
            WHERE scenario_id = (SELECT scenario_id FROM selected_scenario)
        )
        SELECT filename
        FROM artifact_names
        WHERE filename IS NOT NULL AND filename <> ''
        """
    )
    db.ensure_schema()
    with db.engine.connect() as conn:
        filenames = {Path(str(row.filename)).name for row in conn.execute(query, {"scenario_key": scenario_key})}

    log_prefixes = {f"{Path(filename).stem}_PV" for filename in filenames}
    log_prefixes.update(f"{Path(filename).stem}_" for filename in filenames if "_PV" in Path(filename).stem)
    return filenames, log_prefixes


def _step3_artifacts_for_scenario(db: SurroGridDatabase, scenario_key: str) -> list[Path]:
    exact_filenames, log_prefixes = _step3_artifact_names(db, scenario_key)
    matches: list[Path] = []
    for directory in STEP3_ARTIFACT_DIRS:
        if not directory.exists():
            continue
        is_log_dir = directory.name == "logs" or directory.parent.name == "logs"
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.name in exact_filenames:
                matches.append(path)
            elif is_log_dir and any(path.name.startswith(prefix) for prefix in log_prefixes):
                matches.append(path)
    return sorted(set(matches))


def _delete_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink()


def _kept_tables(keep_demands: bool) -> set[str]:
    if not keep_demands:
        return set()
    return {
        "scenario",
        "pipeline_run",
        "demand_allocation_run",
        "allocated_demand",
        "allocated_eff_factor",
        "allocated_vehicle",
    }


def _print_counts(counts: dict[str, int], *, keep_demands: bool, dry_run: bool) -> None:
    action = "Would delete" if dry_run else "Deleted"
    kept_tables = _kept_tables(keep_demands)
    print(f"{action} scenario-related SurroGrid rows:")
    for table_name, count in counts.items():
        suffix = " (kept by --keep-demands)" if table_name in kept_tables else ""
        print(f"  {table_name}: {count}{suffix}")


def _print_files(paths: list[Path], *, dry_run: bool) -> None:
    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} Step 3 files: {len(paths)}")
    for path in paths:
        print(f"  {path.relative_to(REPO_ROOT)}")


def main() -> int:
    args = _build_parser().parse_args()
    dry_run = not args.execute

    db = SurroGridDatabase()
    step3_files = _step3_artifacts_for_scenario(db, args.scenario_key)
    counts = db.delete_scenario_data(
        args.scenario_key,
        keep_demands=args.keep_demands,
        dry_run=dry_run,
        refresh_expansion_views=not args.no_refresh_expansion_views,
    )
    if not dry_run:
        _delete_files(step3_files)

    _print_counts(counts, keep_demands=args.keep_demands, dry_run=dry_run)
    _print_files(step3_files, dry_run=dry_run)
    if dry_run:
        print("Dry-run only. Re-run with --execute to delete these rows and files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
