"""The pylovo-ui plugin: manifest and ES modules (no build step)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gridexpand.service import API_VERSION, CSRF_HEADER, PLUGIN_SCHEMA

UI_DIR = Path(__file__).resolve().parents[1] / "ui"

router = APIRouter(tags=["ui"])


def manifest(version: str) -> dict[str, Any]:
    """Plugin manifest; ``entry`` and ``api`` are relative to the plugin base URL."""
    return {
        "schema": PLUGIN_SCHEMA,
        "name": "gridexpand",
        "title": "GridExpand",
        "version": version,
        "api_version": API_VERSION,
        "entry": "ui/plugin.js",
        "api": "api/",
        "csrf_header": CSRF_HEADER,
        "host_api": ">=1",
    }


@router.get("/ui/manifest.json")
def get_manifest(request: Request) -> JSONResponse:
    return JSONResponse(manifest(request.app.version), headers={"Cache-Control": "no-cache"})
