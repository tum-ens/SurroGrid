"""Scenario YAML files offered by the service, and the user files of the scenario editor."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gridexpand.scenario.config_loader import (
    load_scenario_config,
    scenario_identity_key,
)

_NAME_RE = re.compile(r"^[\w.-]+\.ya?ml$")
# Names the editor may write: no leading dot or dash, ASCII only, at most 100 characters.
USER_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,94}\.ya?ml$")


class ScenarioFileError(Exception):
    """A refused write or delete in the user scenario directory (``status`` is the HTTP status)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def scenario_files(dirs: Iterable[Path]) -> dict[str, Path]:
    """``file name -> path`` of all scenario YAMLs; the first directory wins on equal names."""
    files: dict[str, Path] = {}
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and _NAME_RE.match(path.name) and path.name not in files:
                files[path.name] = path.resolve()
    return files


def is_user_file(path: Path, user_dir: Path | None) -> bool:
    """Whether ``path`` lies directly in the user scenario directory."""
    return user_dir is not None and Path(path).resolve().parent == Path(user_dir).resolve()


def describe(name: str, path: Path, *, user: bool = False) -> dict[str, Any]:
    """Summary of one scenario file; invalid files are listed with ``valid: false``."""
    info: dict[str, Any] = {"name": name, "path": str(path), "directory": str(path.parent), "user": user}
    try:
        scenario, config_hash = load_scenario_config(path)
    except Exception as exc:  # noqa: BLE001 - templates and broken files are shown, not hidden
        return info | {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    electrification = scenario.electrification
    adoption = {
        tech: {"mode": cfg.adoption_mode, "building_share": cfg.building_share}
        for tech, cfg in (("heat", electrification.heat), ("mobility", electrification.mobility),
                          ("pv_battery", electrification.pv_battery))
    }
    return info | {
        "valid": True,
        "template": name.startswith("00_") or "CHANGE_ME" in scenario.scenario_id,
        "id": scenario.scenario_id,
        "milestone_year": scenario.milestone_year,
        "heat_source": scenario.heat.space_heat_source,
        "adoption": adoption,
        "pv_maximum_fallback_share": scenario.pv.maximum_fallback_share,
        "time_aggregation": scenario.time_aggregation.enabled,
        "configuration_hash": config_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, config_hash),
    }


def list_scenarios(dirs: Iterable[Path], user_dir: Path | None = None) -> list[dict[str, Any]]:
    """Summaries of all scenario files (valid ones first); files of ``user_dir`` have ``user: true``."""
    items = [describe(name, path, user=is_user_file(path, user_dir)) for name, path in scenario_files(dirs).items()]
    return sorted(items, key=lambda item: (not item["valid"], item["name"]))


def resolve(dirs: Iterable[Path], name: str) -> Path:
    """Path of a listed scenario file.

    Raises:
        KeyError: If ``name`` is not one of the listed file names (no paths accepted).
    """
    files = scenario_files(dirs)
    if name not in files:
        raise KeyError(name)
    return files[name]


# --------------------------------------------------------------------------- user directory
def user_dir_status(user_dir: Path) -> dict[str, Any]:
    """``{path, exists, writable, reason}`` of the user scenario directory (created on the first save)."""
    user_dir = Path(user_dir)
    probe = user_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)
    return {"path": str(user_dir), "exists": user_dir.is_dir(), "writable": writable,
            "reason": None if writable else f"{probe} is not writable for the service"}


def target_status(shipped_dirs: Iterable[Path], user_dir: Path, file_name: str | None) -> dict[str, Any]:
    """Whether the editor may save ``file_name`` (``allowed``) and whether that overwrites a user file."""
    status: dict[str, Any] = {"file_name": file_name, "allowed": False, "exists": False, "message": None}
    if not file_name or not USER_NAME_RE.match(file_name):
        status["message"] = ("Use a file name of letters, digits, '_', '.' and '-' that ends with .yaml "
                             "(at most 100 characters).")
    elif file_name in scenario_files(shipped_dirs):
        status["message"] = f"{file_name} is a shipped scenario; choose another file name."
    else:
        path = Path(user_dir) / file_name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            status["message"] = f"{file_name} exists in the user directory and is not a plain file."
        else:
            status.update(allowed=True, exists=path.is_file())
            if status["exists"]:
                status["message"] = f"{file_name} exists in your scenarios: saving replaces it (a backup is kept)."
    return status


def _backup_name(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return backup


def write_user_scenario(shipped_dirs: Iterable[Path], user_dir: Path, file_name: str, text: str,
                        *, overwrite: bool = False) -> tuple[Path, str | None]:
    """Write ``text`` as ``user_dir/file_name`` (atomically); an existing own file is backed up first.

    Returns:
        The written path and the backup file name (``None`` for a new file or an unchanged text).

    Raises:
        ScenarioFileError: Invalid or shipped name (400/409), existing file without ``overwrite`` (409),
            or a user directory the service cannot write (500).
    """
    status = target_status(shipped_dirs, user_dir, file_name)
    if not status["allowed"]:
        raise ScenarioFileError(400 if not USER_NAME_RE.match(file_name or "") else 409, status["message"])
    user_dir = Path(user_dir)
    path = user_dir / file_name
    if status["exists"] and not overwrite:
        raise ScenarioFileError(409, f"{file_name} already exists in your scenarios; confirm to replace it.")
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
        if path.resolve().parent != user_dir.resolve():
            raise ScenarioFileError(400, f"{file_name} resolves outside the user scenario directory.")
        backup = None
        if status["exists"]:
            if path.read_text(encoding="utf-8") == text:
                return path.resolve(), None
            backup_path = _backup_name(path)
            backup_path.write_bytes(path.read_bytes())
            backup = backup_path.name
        tmp = user_dir / f".{file_name}.tmp-{os.getpid()}"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise ScenarioFileError(500, f"Could not write {path}: {exc}") from exc
    return path.resolve(), backup


def delete_user_scenario(shipped_dirs: Iterable[Path], user_dir: Path, file_name: str) -> str:
    """Remove a user scenario from the list by renaming it to ``<name>.bak-<timestamp>``.

    Returns:
        The backup file name.

    Raises:
        ScenarioFileError: Shipped file (403) or no such user file (404).
    """
    if file_name in scenario_files(shipped_dirs):
        raise ScenarioFileError(403, f"{file_name} is a shipped scenario; only your own scenarios can be deleted.")
    path = Path(user_dir) / file_name
    if not USER_NAME_RE.match(file_name) or path.is_symlink() or not path.is_file():
        raise ScenarioFileError(404, f"{file_name} is not one of your scenarios.")
    backup = _backup_name(path)
    try:
        path.replace(backup)
    except OSError as exc:
        raise ScenarioFileError(500, f"Could not delete {path}: {exc}") from exc
    return backup.name


def proposal(dirs: Iterable[Path], base_name: str, base_id: str | None, *, base_is_user: bool) -> dict[str, str]:
    """Default file name and scenario id for saving an edited copy of ``base_name``.

    An own file is proposed as itself (replace it); a shipped one as ``<id>_custom`` (then ``_2``, ``_3``,
    … while that name exists), a template as ``my_scenario``.
    """
    if base_is_user and base_id:
        return {"file_name": base_name, "scenario_id": base_id}
    stem = "my_scenario" if not base_id or "CHANGE_ME" in base_id else f"{base_id}_custom"
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:56]
    names = set(scenario_files(dirs))
    candidate, counter = stem, 2
    while f"{candidate}.yaml" in names or f"{candidate}.yml" in names:
        candidate, counter = f"{stem}_{counter}", counter + 1
    return {"file_name": f"{candidate}.yaml", "scenario_id": candidate}


def identity_issues(dirs: Iterable[Path], scenario_id: str | None, scenario_key: str | None,
                    exclude: str | None) -> list[dict[str, Any]]:
    """Warnings when another listed file has the same scenario id or the same scenario key."""
    issues = []
    if not scenario_id:
        return issues
    for item in list_scenarios(dirs):
        if not item["valid"] or item["name"] == exclude:
            continue
        if item["scenario_key"] == scenario_key:
            issues.append({"level": "warning", "key": "scenario.id",
                           "message": f"Same scenario key as {item['name']}: the content is identical, so both "
                                      "files share their results."})
        elif item["id"] == scenario_id:
            issues.append({"level": "warning", "key": "scenario.id",
                           "message": f"{item['name']} uses the same scenario id; the results stay apart (the key "
                                      "contains the hash), but a distinct id is easier to read."})
    return issues
