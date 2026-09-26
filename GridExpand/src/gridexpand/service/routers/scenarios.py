"""Scenario files: list, text, and the scenario editor (form, preview, save as a user file, delete)."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from gridexpand.service import scenario_form, scenarios
from gridexpand.service.settings import ServiceSettings

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])

MAX_TEXT = 200_000


def settings_of(request: Request) -> ServiceSettings:
    return request.app.state.settings


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base(settings: ServiceSettings, name: str) -> tuple[Any, str]:
    try:
        path = scenarios.resolve(settings.scenario_dirs, name)
    except KeyError as exc:
        raise HTTPException(404, f"Scenario '{name}' not found") from exc
    return path, path.read_text(encoding="utf-8")


def _busy(request: Request, name: str) -> None:
    """Refuse to change a scenario file that a queued or running job reads."""
    for job in request.app.state.jobs.list():
        if job.active and job.params.get("scenario") == name:
            raise HTTPException(409, f"Job {job.id} ({job.status}) uses {name}; wait until it has finished.")


class PreviewBody(BaseModel):
    """An edited copy of a base scenario: form ``changes`` (dotted keys) on top of ``text`` or the base file."""

    model_config = ConfigDict(extra="forbid")
    base: str = Field(min_length=1, max_length=200, description="file name from GET /api/scenarios")
    changes: dict[str, Any] = Field(default_factory=dict, max_length=200,
                                    description="dotted YAML path -> value (fields of GET …/form)")
    text: str | None = Field(None, max_length=MAX_TEXT, description="edited YAML text (default: the base file)")
    scenario_id: str | None = Field(None, max_length=64, description="sets scenario.id")
    file_name: str | None = Field(None, max_length=120, description="target file name (checked, not written)")
    base_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$",
                                    description="text_sha256 of GET …/form: refuse if the base file changed since")


class SaveBody(PreviewBody):
    """Save the edited copy as ``file_name`` in the user scenario directory."""

    file_name: str = Field(min_length=1, max_length=120)
    scenario_id: str = Field(min_length=1, max_length=64)
    overwrite: bool = Field(False, description="replace an own file of the same name (a backup is kept)")


def _preview(settings: ServiceSettings, body: PreviewBody) -> dict[str, Any]:
    path, base_text = _base(settings, body.base)
    if body.base_sha256 and body.base_sha256 != _sha256(base_text):
        raise HTTPException(409, f"{body.base} changed on disk since it was opened; reload it and apply your edits "
                                 "again.")
    result = scenario_form.preview(base_text, base_name=body.base, changes=body.changes, text=body.text,
                                   scenario_id=body.scenario_id, target_name=body.file_name)
    if result["ok"]:
        result["issues"] += scenarios.identity_issues(settings.scenario_dirs, result["id"], result["scenario_key"],
                                                      exclude=body.file_name)
    result["unchanged"] = result["text"] == base_text
    result["target"] = scenarios.target_status(settings.shipped_scenario_dirs, settings.user_scenario_dir,
                                               body.file_name) if body.file_name else None
    return result


@router.get("")
def list_scenarios(request: Request) -> list[dict[str, Any]]:
    """Scenario YAMLs (``config/scenarios``, extra and user directories) with their key values."""
    settings = settings_of(request)
    return scenarios.list_scenarios(settings.scenario_dirs, settings.user_scenario_dir)


@router.post("/preview")
def preview(body: PreviewBody, request: Request) -> dict[str, Any]:
    """Apply the changes to a copy of the base file and validate it with the loader of the runs.

    Nothing is written. Returns ``{ok, issues: [{level, message, key?, line?}], text, diff (unified),
    changes, values, id, configuration_hash, scenario_key, unchanged, target}``.
    """
    return _preview(settings_of(request), body)


@router.post("", status_code=201)
def save(body: SaveBody, request: Request) -> dict[str, Any]:
    """Save the edited copy as a new file in the user scenario directory.

    Shipped files are never written; replacing an own file needs ``overwrite: true`` and keeps
    ``<name>.bak-<timestamp>``. Returns the new ``GET /api/scenarios`` entry with ``backup``.
    """
    settings = settings_of(request)
    result = _preview(settings, body)
    errors = [issue for issue in result["issues"] if issue["level"] == "error"]
    if errors:
        raise HTTPException(400, {"message": f"The scenario is invalid: {errors[0]['message']}",
                                  "issues": result["issues"]})
    target = result["target"]
    if target["allowed"] and target["exists"]:
        _busy(request, body.file_name)
    try:
        path, backup = scenarios.write_user_scenario(settings.shipped_scenario_dirs, settings.user_scenario_dir,
                                                     body.file_name, result["text"], overwrite=body.overwrite)
    except scenarios.ScenarioFileError as exc:
        raise HTTPException(exc.status, {"message": exc.message, "target": target}) from exc
    return scenarios.describe(body.file_name, path, user=True) | {"backup": backup}


@router.get("/{name}", response_class=PlainTextResponse)
def scenario_text(name: str, request: Request) -> str:
    """The YAML text of one listed scenario file."""
    return _base(settings_of(request), name)[1]


@router.get("/{name}/form")
def scenario_form_state(name: str, request: Request) -> dict[str, Any]:
    """Editable fields of a scenario file with their values and YAML help texts, and its text.

    ``writable``: the file is one of the user's own (it can be replaced or deleted); ``user_dir``
    tells whether new files can be saved; ``proposal`` is the default file name and scenario id.
    """
    settings = settings_of(request)
    path, text = _base(settings, name)
    user = scenarios.is_user_file(path, settings.user_scenario_dir)
    info = scenarios.describe(name, path, user=user)
    user_dir = scenarios.user_dir_status(settings.user_scenario_dir)
    try:
        sections, _ = scenario_form.form_sections(text)
    except Exception as exc:  # noqa: BLE001 - a broken file opens in the YAML view only
        sections = []
        info.setdefault("error", f"{type(exc).__name__}: {exc}")
    return info | {
        "writable": user and user_dir["writable"],
        "text": text,
        "text_sha256": _sha256(text),
        "editable_fields": sections,
        "user_dir": user_dir,
        "proposal": scenarios.proposal(settings.scenario_dirs, name, info.get("id"), base_is_user=user),
    }


@router.delete("/{name}")
def delete(name: str, request: Request) -> dict[str, Any]:
    """Remove one of the user's scenario files (renamed to ``<name>.bak-<timestamp>``, not erased)."""
    settings = settings_of(request)
    _busy(request, name)
    try:
        backup = scenarios.delete_user_scenario(settings.shipped_scenario_dirs, settings.user_scenario_dir, name)
    except scenarios.ScenarioFileError as exc:
        raise HTTPException(exc.status, exc.message) from exc
    return {"deleted": name, "backup": backup}
