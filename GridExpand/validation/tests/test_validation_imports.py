"""Import every validation module and check that its CLI help needs no database."""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys

import pytest

import validation

MODULES = sorted(f"validation.{info.name}" for info in pkgutil.iter_modules(validation.__path__))


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("command", ["run", "status-quo-test"])
def test_command_help_without_database(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "validation", command, "--help"],
        cwd=validation.ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=300, check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "usage:" in result.stdout.lower() and "OperationalError" not in output
