"""The committed API schema matches the app (the GridPlanner UI relies on these routes)."""

from __future__ import annotations

from gridexpand.api.contract import SNAPSHOT, openapi_document, render


def test_openapi_snapshot_is_current(fake_solvers):
    current = render(openapi_document())
    assert SNAPSHOT.exists(), "docs/openapi.json is missing: uv run python scripts/export_openapi.py"
    assert SNAPSHOT.read_text(encoding="utf-8") == current, (
        "The /api routes changed. Review the change, bump API_VERSION if it breaks the UI, then run "
        "`uv run python scripts/export_openapi.py` and commit docs/openapi.json.")
