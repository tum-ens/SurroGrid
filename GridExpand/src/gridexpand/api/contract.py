"""The committed OpenAPI schema of the API (``docs/openapi.json``).

The GridPlanner UI relies on the ``/api`` routes. The snapshot makes every change of them
visible in review: a unit test compares it with the app, and CI checks it for breaking
changes (``oasdiff breaking``) unless ``API_VERSION`` was bumped. Rewrite it with
``uv run python scripts/export_openapi.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gridexpand.api import API_VERSION
from gridexpand.paths import PROJECT_DIR

SNAPSHOT = PROJECT_DIR / "docs" / "openapi.json"


def openapi_document() -> dict[str, Any]:
    """The app's OpenAPI schema without release-specific fields, with ``x-api-version``."""
    import tempfile

    from gridexpand.api.app import create_app
    from gridexpand.api.settings import ServiceSettings

    with tempfile.TemporaryDirectory() as tmp:  # the job manager creates its folders
        schema = create_app(ServiceSettings(state_dir=Path(tmp) / "api", runs_dir=Path(tmp) / "runs")).openapi()
    schema["info"] = {**schema["info"], "version": f"api-{API_VERSION}", "x-api-version": API_VERSION}
    return schema


def render(schema: dict[str, Any]) -> str:
    """Stable text form (sorted keys, two-space indent, trailing newline)."""
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_snapshot(path: Path = SNAPSHOT) -> Path:
    path.write_text(render(openapi_document()), encoding="utf-8")
    return path
