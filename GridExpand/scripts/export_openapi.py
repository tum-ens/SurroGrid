#!/usr/bin/env python3
"""Rewrite docs/openapi.json, the committed schema of the GridExpand API.

    uv run python scripts/export_openapi.py
"""

from gridexpand.api.contract import write_snapshot

if __name__ == "__main__":
    print(f"wrote {write_snapshot()}")
