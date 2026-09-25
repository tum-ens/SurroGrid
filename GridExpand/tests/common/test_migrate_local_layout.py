"""scripts/migrate_local_layout.py: plan on a temporary old-layout checkout (no moves executed)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from gridexpand.paths import PROJECT_DIR

spec = importlib.util.spec_from_file_location("migrate_local_layout", PROJECT_DIR / "scripts" / "migrate_local_layout.py")
migrate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = migrate  # dataclasses look the module up while the script loads
spec.loader.exec_module(migrate)


def touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_plan_leaves_tracked_files_to_git_and_reports_uncovered(tmp_path, capsys):
    project = tmp_path / "GridExpand"
    tracked = touch(project / "4.powerflow" / "Output" / "README.md")
    untracked = touch(project / "4.powerflow" / "Output" / "grid_1.h5")
    whole = touch(project / "3.urbs" / "result" / "run" / "result.h5")
    uncovered = touch(project / "4.powerflow" / "scratch.txt")
    touch(project / "4.powerflow" / "powerflow.py")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", str(tracked), str(project / "4.powerflow" / "powerflow.py")],
                   check=True)

    tracked_set = migrate.git_tracked(project)
    moves, conflicts = migrate.plan_moves(project, tmp_path / "work", tmp_path / "data", tracked_set)
    sources = {move.source: move for move in moves}
    assert conflicts == []
    assert tracked not in sources and untracked in sources
    assert sources[untracked].target == tmp_path / "work" / "powerflow" / "output" / "grid_1.h5"
    assert sources[whole.parent.parent].whole_directory
    assert migrate.leftovers(project, moves, conflicts, tracked_set) == [uncovered]

    assert migrate.main(["--project-dir", str(project)]) == 0
    out = capsys.readouterr().out
    assert "2 move(s), 0 conflict(s), 1 uncovered file(s) in old folders, 2 git-tracked file(s) left to git." in out
    assert untracked.exists()  # dry run
