"""Paired real-vs-synthetic validation of GridExpand (``uv run python -m validation <command>``).

The same post-electrification scenario runs on the real grids of a pylovo alignment bundle and on the synthetic
pylovo grids of the same buildings; power flows and expansion needs are compared. This package imports
``gridexpand``; ``gridexpand`` never imports it. It is not part of the installed package: run it from
``GridExpand/`` (see ``validation/README.md``).
"""

from __future__ import annotations

import os
from pathlib import Path

# GridExpand/, the directory that makes ``import validation`` work.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def export_pythonpath() -> None:
    """Let child processes (``python -m validation.<module>``) import this package from any working directory."""
    current = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
    if str(ROOT) not in current:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *current])
