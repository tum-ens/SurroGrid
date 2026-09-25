"""Read what a ``gridexpand synthetic`` run writes into its run directory.

The runner (:mod:`gridexpand.scenario.synthetic_ags_runner`) appends one JSON object per
event to ``events.jsonl`` (and prints the same line), keeps one row per grid in
``status.tsv`` and writes each grid's step output to ``logs/candidate_<index>_<file>.log``.
:class:`RunTracker` follows these files incrementally while a job runs; the other
functions turn them into progress, table rows and readable log lines.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENTS_FILE = "events.jsonl"
STATUS_FILE = "status.tsv"
SUMMARY_FILE = "summary.json"
EXPANSION_LOG = "expansion_materialization.log"
LOGS_DIR = "logs"

_CANDIDATE_LOG_RE = re.compile(r"^candidate_(\d+)_")
_EXCEPTION_RE = re.compile(r"^[\w.]*(Error|Exception|Interrupt)\b")


@dataclass
class RunProgress:
    """Progress of one run directory, derived from its events."""

    grids_total: int | None = None
    grids_done: int = 0
    grids_failed: int = 0
    stage: str | None = None
    batch_status: str | None = None
    message: str | None = None
    running: dict[int, str] = field(default_factory=dict)  # candidate index -> stage

    @property
    def finished(self) -> bool:
        return self.batch_status is not None

    def fraction(self) -> float | None:
        """Share of the grids that are finished (0.98 while the expansion is materialized)."""
        if self.finished:
            return 1.0
        if not self.grids_total:
            return None
        return min(0.98, (self.grids_done + self.grids_failed) / self.grids_total)

    def apply(self, event: dict[str, Any]) -> None:
        """Update the progress with one event of ``events.jsonl``."""
        kind = event.get("event")
        index = event.get("candidate_index")
        if kind == "candidates_selected":
            self.grids_total = int(event.get("count") or 0)
        elif kind == "start":
            stage = str(event.get("stage", ""))
            if index is None:
                self.stage = stage
            else:
                self.running[int(index)] = stage
                self.stage = f"grid #{index} · {stage}"
        elif kind in ("candidate_done", "pilot_finish", "candidate_failed_recorded", "candidate_failed_unhandled"):
            if index is not None:
                self.running.pop(int(index), None)
            failed = kind != "candidate_done" and (kind != "pilot_finish" or event.get("status") != "done")
            if failed:
                self.grids_failed += 1
            else:
                self.grids_done += 1
            self.stage = next((f"grid #{i} · {s}" for i, s in self.running.items()), None)
        elif kind == "batch_finish":
            self.batch_status = str(event.get("status") or "unknown")
            self.message = event.get("message") or None
            self.stage = None
            self.running.clear()


class RunTracker:
    """Incrementally read new events and new log lines of one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.progress = RunProgress()
        self._offsets: dict[Path, int] = {}
        self._partial: dict[Path, str] = {}

    def _new_lines(self, path: Path) -> list[str]:
        try:
            size = path.stat().st_size
        except OSError:
            return []
        offset = self._offsets.get(path, 0)
        if size <= offset:
            return []
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(size - offset)
        self._offsets[path] = size
        text = self._partial.pop(path, "") + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        if lines[-1]:
            self._partial[path] = lines[-1]  # incomplete last line: keep for the next poll
        return [line.rstrip("\r") for line in lines[:-1]]

    def poll_events(self) -> list[dict[str, Any]]:
        """New events (already applied to :attr:`progress`)."""
        events = []
        for line in self._new_lines(self.run_dir / EVENTS_FILE):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                self.progress.apply(event)
                events.append(event)
        return events

    def poll_logs(self) -> list[tuple[str, str]]:
        """New ``(source label, line)`` pairs of the step logs of this run."""
        files = sorted((self.run_dir / LOGS_DIR).glob("*.log")) + [self.run_dir / EXPANSION_LOG]
        out: list[tuple[str, str]] = []
        for path in files:
            label = log_label(path)
            out.extend((label, line) for line in self._new_lines(path) if line.strip())
        return out


def log_label(path: Path) -> str:
    """Short source label of a run log file: ``#3``, ``prep`` or ``expansion``."""
    match = _CANDIDATE_LOG_RE.match(path.name)
    if match:
        return f"#{int(match.group(1))}"
    return {"electrification_preparation.log": "prep", EXPANSION_LOG: "expansion"}.get(path.name, path.stem)


def read_progress(run_dir: Path) -> RunProgress:
    """Progress of a (finished or running) run directory from its whole event log."""
    tracker = RunTracker(run_dir)
    tracker.poll_events()
    return tracker.progress


def read_status_rows(run_dir: Path) -> list[dict[str, str]]:
    """Rows of ``status.tsv`` (one per grid), without absolute log paths."""
    path = Path(run_dir) / STATUS_FILE
    if not path.is_file():
        return []
    keep = ("candidate_index", "plz", "kcid", "bcid", "n_buildings", "status", "stage", "started_at",
            "finished_at", "seconds", "timeframe_start", "timeframe_end", "message")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [{key: row.get(key, "") for key in keep} | {"log": Path(row.get("log_file") or "").name}
                for row in csv.DictReader(handle, delimiter="\t")]
    return rows


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    """``summary.json`` of a finished run (``None`` while it runs)."""
    try:
        return json.loads((Path(run_dir) / SUMMARY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_line(text: str) -> str:
    """Display level of one log line: command, success, error, warning or info."""
    stripped = text.strip()
    if stripped.startswith(("Traceback", "✗")) or _EXCEPTION_RE.match(stripped):
        return "error"
    if stripped.startswith("✓"):
        return "success"
    if stripped.startswith(("$ ", "▶")):
        return "command"
    if stripped.startswith("■"):  # a batch that finished with failed grids
        return "warning"
    lowered = stripped[:60].lower()
    if "warning" in lowered or lowered.startswith("warn"):
        return "warning"
    if " error" in lowered or lowered.startswith("error"):
        return "error"
    return "info"


def _grid(event: dict[str, Any]) -> str:
    return f"grid #{event['candidate_index']} · " if event.get("candidate_index") is not None else ""


def format_event(event: dict[str, Any], case: str | None = None) -> str:
    """One readable line for a runner event (the raw JSON stays in ``events.jsonl``)."""
    kind = event.get("event")
    tag = f"[{case}] " if case else ""
    if kind == "batch_start":
        return (f"▶ {tag}start · AGS {event.get('ags')} · pylovo v{event.get('pylovo_version_id')} · "
                f"profiles {event.get('profiles')} · {event.get('timeframe_mode')} · min {event.get('min_buildings')} buildings")
    if kind == "candidates_loaded":
        return f"{tag}{event.get('count')} candidate grid(s) in the AGS"
    if kind == "candidates_selected":
        return f"{tag}{event.get('count')} grid(s) selected"
    if kind == "start":
        return f"{tag}{_grid(event)}{event.get('stage')} …"
    if kind == "finish":
        code = event.get("returncode")
        mark = "✓" if code == 0 else "✗"
        return f"{mark} {tag}{_grid(event)}{event.get('stage')} (exit {code}, {event.get('seconds')} s)"
    if kind in ("candidate_done", "pilot_finish") and event.get("status") == "done":
        return f"✓ {tag}{_grid(event)}done in {event.get('seconds')} s"
    if kind in ("candidate_failed", "candidate_failed_unhandled"):
        return f"✗ {tag}{_grid(event)}failed in {event.get('stage', 'unknown stage')}: {event.get('message', '')}"
    if kind in ("candidate_failed_recorded", "pilot_finish"):
        return f"✗ {tag}{_grid(event)}recorded as failed"
    if kind == "pilot_start":
        return f"{tag}pilot grid #{event.get('candidate_index')} (the batch stops if it fails)"
    if kind == "batch_finish":
        status = event.get("status")
        counts = f"{event.get('completed_count', 0)}/{event.get('candidate_count', 0)} grids"
        mark = {"done": "✓", "failed": "✗"}.get(str(status), "■")
        extra = f" · {event['message']}" if event.get("message") else ""
        return f"{mark} {tag}finished: {status} · {counts} · {event.get('total_seconds')} s{extra}"
    if kind and kind.startswith("electrification_preparation"):
        return f"{tag}electrification assignment {kind.removeprefix('electrification_preparation_')}"
    if kind == "expansion_materialization_failed":
        return f"✗ {tag}expansion materialization failed: {event.get('message')}"
    if kind == "batch_workers_start":
        return f"{tag}{event.get('remaining')} grid(s) with {event.get('workers')} worker(s)"
    compact = {k: v for k, v in event.items() if k not in ("ts", "cmd", "env_extra")}
    return f"{tag}{json.dumps(compact, default=str)[:400]}"


def parse_event_line(line: str) -> dict[str, Any] | None:
    """The event of one runner stdout line, or ``None`` for ordinary output."""
    if not line.startswith("{") or '"event"' not in line:
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None
