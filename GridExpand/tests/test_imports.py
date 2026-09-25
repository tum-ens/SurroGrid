"""Import every gridexpand module and check that CLI help needs no database.

``conftest.py`` points ``GRIDEXPAND_ENV_FILE`` at a database on a closed port,
so any connection attempt during an import or a ``--help`` call fails.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "gridexpand"
VENDORED_PREFIXES = ("gridexpand.allocation.external",)


def _walk(path: Path, prefix: str) -> Iterator[str]:
    """Yield module names below ``path`` without importing any of them."""
    for info in pkgutil.iter_modules([str(path)], prefix):
        if info.name.startswith(VENDORED_PREFIXES):
            continue
        yield info.name
        if info.ispkg:
            yield from _walk(path / info.name.rsplit(".", 1)[-1], info.name + ".")


MODULES = sorted(_walk(PACKAGE_DIR, "gridexpand."))


def _commands() -> list[str]:
    from gridexpand.cli import COMMANDS

    return sorted(COMMANDS)


def test_environment_is_isolated() -> None:
    from gridexpand import paths

    assert paths.ENV_FILE == Path(os.environ["GRIDEXPAND_ENV_FILE"])
    assert "DB_PORT=\"9\"" in paths.ENV_FILE.read_text(encoding="utf-8")


def test_module_list_is_complete() -> None:
    for expected in (
        "gridexpand.cli",
        "gridexpand.paths",
        "gridexpand.db.database",
        "gridexpand.allocation.main",
        "gridexpand.optimization.run_urbs_cluster",
        "gridexpand.optimization.urbs.runfunctions",
        "gridexpand.powerflow.run_pwrflw",
        "gridexpand.scenario.synthetic_ags_runner",
        "gridexpand.analysis.expansion.grid_expansion",
    ):
        assert expected in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)


def test_no_sys_path_or_chdir_hacks() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if "external" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "sys.path" in text or "os.chdir(" in text:
            offenders.append(str(path.relative_to(PACKAGE_DIR)))
    assert offenders == []


def _cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gridexpand", *args],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_cli_help() -> None:
    result = _cli(["--help"])
    assert result.returncode == 0, result.stderr
    for command in _commands():
        assert command in result.stdout


@pytest.mark.parametrize("command", _commands())
def test_command_help_without_database(command: str) -> None:
    result = _cli([command, "--help"])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "usage:" in result.stdout.lower()
    assert "OperationalError" not in output


def test_unknown_command() -> None:
    assert _cli(["no-such-command"]).returncode == 2
