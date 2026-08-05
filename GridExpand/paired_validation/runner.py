#!/usr/bin/env python3
"""Paired real/synthetic validation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

RUNME_DIR = Path(__file__).resolve().parents[1] / "runme"
if str(RUNME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNME_DIR))

from paired_swf_pipeline_runner import main


if __name__ == "__main__":
    main()
