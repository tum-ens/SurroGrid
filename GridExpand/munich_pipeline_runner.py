#!/usr/bin/env python3
"""Compatibility wrapper for the AGS pipeline runner.

Prefer `ags_pipeline_runner.py` for new runs. This wrapper keeps existing Munich
commands working and defaults `--ags` to 09162000 when the option is omitted.
"""

from __future__ import annotations

import sys

from ags_pipeline_runner import main


if __name__ == "__main__":
    if "--ags" not in sys.argv:
        sys.argv[1:1] = ["--ags", "09162000"]
    raise SystemExit(main())
