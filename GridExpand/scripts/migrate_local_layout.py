#!/usr/bin/env python3
"""Move untracked runtime artifacts and large inputs from the old step layout.

Before the package restructure, GridExpand kept runtime artifacts inside the
step folders (``3.urbs/Input``, ``4.powerflow/Output``, ``run_logs``, ...) and
large untracked inputs inside ``2.demand_allocation/gridalloc/data``. This
script moves them to the locations defined by :mod:`gridexpand.paths`
(``work/...`` and ``data/statistics/...``).

Dry-run by default: it only prints the planned moves. ``--execute`` performs
them. Existing targets are never overwritten (they are reported as conflicts)
and nothing is deleted; emptied old folders are left in place. Files tracked by
git are never moved or reported: git removes them when the new layout is
checked out.

    uv run python scripts/migrate_local_layout.py            # show the plan
    uv run python scripts/migrate_local_layout.py --execute  # move
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gridexpand import paths

# Old location (relative to the GridExpand project directory) -> new location.
WORK_MOVES: tuple[tuple[str, Path], ...] = (
    ("1.grid_sampling/gridreadout/results", paths.SAMPLING_RESULTS_DIR),
    ("2.demand_allocation/gridalloc/data/grids", paths.ALLOCATION_GRIDS_DIR),
    ("2.demand_allocation/gridalloc/results", paths.ALLOCATION_RESULTS_DIR),
    ("2.demand_allocation/gridalloc/outputs", paths.ALLOCATION_OUTPUTS_DIR),
    ("2.demand_allocation/gridalloc/logs", paths.ALLOCATION_LOGS_DIR),
    ("3.urbs/Input", paths.OPTIMIZATION_INPUT_DIR),
    ("3.urbs/result", paths.OPTIMIZATION_RESULT_DIR),
    ("3.urbs/logs", paths.OPTIMIZATION_LOGS_DIR),
    ("4.powerflow/Input", paths.POWERFLOW_INPUT_DIR),
    ("4.powerflow/Output", paths.POWERFLOW_OUTPUT_DIR),
    ("4.powerflow/analysis", paths.POWERFLOW_ANALYSIS_DIR),
    ("4.powerflow/plots", paths.POWERFLOW_PLOTS_DIR),
    ("5.postprocessing/output", paths.ANALYSIS_OUTPUT_DIR),
    ("run_logs", paths.RUNS_DIR),
)
OLD_STATISTICS = "2.demand_allocation/gridalloc/data/statistics"
DATA_MOVES: tuple[tuple[str, Path], ...] = tuple(
    (f"{OLD_STATISTICS}/{relative}", paths.STATISTICS_DIR / relative)
    for relative in (
        "inhabited_buildings/elec_lps.h5",
        "general/mobility_profile_pool",
        "general/mobility_profile_pool_old/mobility_demand_pool.csv",
        "general/mobility_profile_pool_old/mobility_availability_pool.csv",
    )
)
OLD_TOP_LEVEL = (
    "1.grid_sampling",
    "2.demand_allocation",
    "3.urbs",
    "4.powerflow",
    "5.postprocessing",
    "common",
    "maintenance",
    "paired_validation",
    "scenario_pipeline",
    "run_logs",
)
PLACEHOLDERS = {".gitkeep"}


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path
    whole_directory: bool = False


def _rebase(target: Path, base: Path, new_base: Path) -> Path:
    return new_base / target.relative_to(base)


def _items(directory: Path, tracked: frozenset[Path] = frozenset()):
    """Yield untracked files and symlinks below ``directory`` without following links."""
    for root, dirnames, filenames in os.walk(directory):
        base = Path(root)
        for name in dirnames:
            if (base / name).is_symlink() and base / name not in tracked:
                yield base / name
        for name in filenames:
            if base / name not in tracked:
                yield base / name


def git_tracked(project_dir: Path) -> frozenset[Path]:
    """Absolute paths git tracks below ``project_dir`` (empty outside a git checkout)."""
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(project_dir), "ls-files", "-z", "--full-name"],
            capture_output=True, check=True,
        )
        top = subprocess.run(
            ["git", "-C", os.fspath(project_dir), "rev-parse", "--show-toplevel"],
            capture_output=True, check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    root = Path(top)
    return frozenset(root / name for name in result.stdout.decode().split("\0") if name)


def plan_moves(
    project_dir: Path, work_dir: Path, data_dir: Path, tracked: frozenset[Path] = frozenset()
) -> tuple[list[Move], list[Move]]:
    """Return (moves, conflicts) for one project directory, leaving git-tracked files alone."""
    moves: list[Move] = []
    conflicts: list[Move] = []
    mapping = [(old, _rebase(new, paths.WORK_DIR, work_dir)) for old, new in WORK_MOVES]
    mapping += [(old, _rebase(new, paths.DATA_DIR, data_dir)) for old, new in DATA_MOVES]
    for old_relative, target in mapping:
        source = project_dir / old_relative
        if not (source.exists() or source.is_symlink()) or source in tracked:
            continue
        if source.is_symlink() or source.is_file():
            move = Move(source, target)
            (conflicts if target.exists() or target.is_symlink() else moves).append(move)
            continue
        if not (target.exists() or target.is_symlink()) and not _needs_file_moves(source, tracked):
            moves.append(Move(source, target, whole_directory=True))
            continue
        for path in sorted(_items(source, tracked)):
            if path.name in PLACEHOLDERS:
                continue
            destination = target / path.relative_to(source)
            move = Move(path, destination)
            (conflicts if destination.exists() or destination.is_symlink() else moves).append(move)
    return moves, conflicts


def _needs_file_moves(directory: Path, tracked: frozenset[Path]) -> bool:
    """True if the directory holds placeholders or tracked files and cannot move as a whole."""
    if any(path.name in PLACEHOLDERS for path in _items(directory)):
        return True
    return any(directory in path.parents for path in tracked)


def leftovers(
    project_dir: Path, moves: list[Move], conflicts: list[Move], tracked: frozenset[Path] = frozenset()
) -> list[Path]:
    """Untracked files in the old top-level folders that no mapping covers."""
    covered = {move.source for move in (*moves, *conflicts)}
    remaining = []
    for top in OLD_TOP_LEVEL:
        root = project_dir / top
        if not root.exists():
            continue
        for path in sorted(_items(root, tracked)):
            if path.name in PLACEHOLDERS:
                continue
            if path in covered or any(parent in covered for parent in path.parents):
                continue
            remaining.append(path)
    return remaining


def execute(moves: list[Move]) -> None:
    for move in moves:
        if move.target.exists() or move.target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {move.target}")
        move.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.fspath(move.source), os.fspath(move.target))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true", help="Move the files (default: dry run).")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=paths.PROJECT_DIR,
        help="GridExpand directory that contains the old step folders (default: this checkout).",
    )
    parser.add_argument("--work-dir", type=Path, help="Target WORK_DIR (default: gridexpand.paths.WORK_DIR).")
    parser.add_argument("--data-dir", type=Path, help="Target DATA_DIR (default: gridexpand.paths.DATA_DIR).")
    args = parser.parse_args(argv)
    project_dir = args.project_dir.resolve()
    default_project = project_dir == paths.PROJECT_DIR
    work_dir = (args.work_dir or (paths.WORK_DIR if default_project else project_dir / "work")).resolve()
    data_dir = (args.data_dir or (paths.DATA_DIR if default_project else project_dir / "data")).resolve()

    tracked = git_tracked(project_dir)
    moves, conflicts = plan_moves(project_dir, work_dir, data_dir, tracked)
    action = "move" if args.execute else "would move"
    for move in moves:
        kind = " (directory)" if move.whole_directory else ""
        print(f"{action}{kind}: {move.source} -> {move.target}")
    for move in conflicts:
        print(f"CONFLICT, target exists, kept both: {move.source} -> {move.target}")
    remaining = leftovers(project_dir, moves, conflicts, tracked)
    for path in remaining:
        print(f"not covered, left in place: {path}")
    old_roots = [project_dir / top for top in OLD_TOP_LEVEL]
    tracked_old = sum(1 for path in tracked if any(root in path.parents for root in old_roots))
    print(
        f"{len(moves)} move(s), {len(conflicts)} conflict(s), "
        f"{len(remaining)} uncovered file(s) in old folders, {tracked_old} git-tracked file(s) left to git."
    )
    if args.execute:
        execute(moves)
        print("Done. Old folders were not removed; delete them once they are empty.")
    elif moves:
        print("Dry run only; re-run with --execute to move the files.")
    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
