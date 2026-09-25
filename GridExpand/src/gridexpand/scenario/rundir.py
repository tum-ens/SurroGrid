"""Run directory of ``gridexpand run``: frozen inputs, identity, state and events.

Layout of ``<run_dir>`` (default ``WORK_DIR/runs/<run.id>``)::

    inputs/run.yaml, inputs/scenario.yaml   frozen copies of the YAMLs
    identity.json                           what a resumed run must share
    state.json                              atomic snapshot for UIs (see RunState)
    events.jsonl                            run-level events (schema 1)
    plan.json                               ordered jobs of the last invocation
    summary.json                            result of the last invocation
    <group>/ …                              one directory per job group

``state.json`` (schema 1) holds ``status``, the current ``stage``, the stage
records, job counts, the running jobs and one entry per job (``job_list``),
keyed by a stable job key; ``gridexpand run --resume`` skips jobs recorded
as done. The file is replaced atomically (write + rename).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gridexpand.common.orchestration import Cancelled, utc_now

SCHEMA = 1
JOB_STATES = ("queued", "running", "done", "failed", "cancelled", "skipped")
FINISHED_JOB_STATES = ("done", "failed", "cancelled", "skipped")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` via a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_revision() -> str | None:
    """Commit of the running code (informational), or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


class IdentityMismatch(ValueError):
    """The run directory belongs to a run with another identity."""


def check_identity(run_dir: Path, identity: dict[str, Any], *, info: dict[str, Any] | None = None) -> None:
    """Record ``identity`` in ``identity.json``, or refuse a directory of another run.

    Raises:
        IdentityMismatch: ``identity.json`` exists and differs.
    """
    path = run_dir / "identity.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8")).get("identity", {})
        mismatches = {
            key: (recorded.get(key), value)
            for key, value in identity.items()
            if recorded.get(key) != json.loads(json.dumps(value, default=str))
        }
        if mismatches:
            raise IdentityMismatch(
                f"{run_dir} belongs to another run. Differences (recorded, requested): "
                f"{mismatches}. Use a new run.id or --run-dir."
            )
        return
    write_json_atomic(path, {"schema": SCHEMA, "identity": identity, "info": info or {}, "created_at": utc_now()})


def freeze_inputs(run_dir: Path, run_yaml: Path, scenario_yaml: Path) -> None:
    """Copy the run and scenario YAMLs into ``<run_dir>/inputs``."""
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_yaml, inputs / "run.yaml")
    shutil.copy2(scenario_yaml, inputs / "scenario.yaml")


class RunState:
    """``state.json`` and ``events.jsonl`` of one run directory (thread-safe).

    Args:
        run_dir: Run directory.
        run_id: ``run.id`` of the YAML.
        pipeline: ``run.pipeline``.
    """

    def __init__(self, run_dir: Path, run_id: str, pipeline: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.pipeline = pipeline
        self.path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self._lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.stages: list[dict[str, Any]] = []
        self.status = "running"
        self.stage: str | None = None
        self.exit_code: int | None = None
        self.started_at = utc_now()
        previous = load_state(self.run_dir)
        if previous:
            for job in previous.get("job_list", []):
                self.jobs[str(job["job"])] = dict(job)

    # jobs -------------------------------------------------------------------------

    def plan(self, jobs: list[dict[str, Any]], *, resume: bool) -> list[str]:
        """Register the jobs of this invocation; returns the keys still to run.

        Jobs of earlier invocations stay in the state. With ``resume`` a job
        already ``done`` keeps its record and is not returned.
        """
        todo = []
        with self._lock:
            for job in jobs:
                key = str(job["job"])
                previous = self.jobs.get(key)
                if resume and previous and previous.get("status") == "done":
                    continue
                self.jobs[key] = {**job, "job": key, "status": "queued", "step": None,
                                  "started_at": None, "finished_at": None, "seconds": None, "message": None}
                todo.append(key)
            self.write()
        return todo

    def job(self, key: str, **updates: Any) -> None:
        """Update one job (``status``, ``step``, ``message``, …) and write the state."""
        with self._lock:
            job = self.jobs.setdefault(key, {"job": key, "status": "queued"})
            status = updates.get("status")
            if status == "running" and not job.get("started_at"):
                job["started_at"] = utc_now()
                job["_t0"] = time.time()
            if status in FINISHED_JOB_STATES:
                job["finished_at"] = utc_now()
                if job.get("_t0") and updates.get("seconds") is None:
                    updates["seconds"] = round(time.time() - job["_t0"], 1)
            job.update(updates)
            self.write()

    def counts(self) -> dict[str, int]:
        with self._lock:
            counts = {state: 0 for state in JOB_STATES}
            for job in self.jobs.values():
                counts[str(job.get("status", "queued"))] = counts.get(str(job.get("status", "queued")), 0) + 1
            counts["total"] = len(self.jobs)
            return counts

    # stages -----------------------------------------------------------------------

    @contextmanager
    def stage_context(self, name: str) -> Iterator[dict[str, Any]]:
        """Record one stage: running, then done / failed / cancelled with its duration."""
        record = {"name": name, "status": "running", "started_at": utc_now(), "finished_at": None, "seconds": None}
        started = time.monotonic()
        with self._lock:
            self.stages = [stage for stage in self.stages if stage["name"] != name] + [record]
            self.stage = name
            self.write()
        self.event(event="stage_start", stage=name)
        try:
            yield record
        except Cancelled:
            self._finish_stage(record, started, "cancelled")
            raise
        except BaseException:
            self._finish_stage(record, started, "failed")
            raise
        self._finish_stage(record, started, record.get("result", "done"))

    def _finish_stage(self, record: dict[str, Any], started: float, status: str) -> None:
        with self._lock:
            record.update(status=status, finished_at=utc_now(), seconds=round(time.monotonic() - started, 1))
            record.pop("result", None)
            self.write()
        self.event(event="stage_finish", stage=record["name"], status=status, seconds=record["seconds"])

    def finish(self, status: str, exit_code: int) -> None:
        with self._lock:
            self.status, self.exit_code, self.stage = status, exit_code, None
            for job in self.jobs.values():
                if status == "cancelled" and job.get("status") in ("queued", "running"):
                    job["status"] = "cancelled"
                elif status == "failed" and job.get("status") == "running":
                    job.update(status="failed", message=job.get("message") or "the run stopped with an error")
            self.write()
        self.event(event="run_finish", status=status, exit_code=exit_code, jobs=self.counts())

    # files ------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = [
                {"job": key, "step": job.get("step"), "since": job.get("started_at")}
                for key, job in self.jobs.items() if job.get("status") == "running"
            ]
            return {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "pipeline": self.pipeline,
                "status": self.status,
                "stage": self.stage,
                "stages": self.stages,
                "jobs": self.counts(),
                "running": running,
                "job_list": [{k: v for k, v in job.items() if not k.startswith("_")} for job in self.jobs.values()],
                "pid": os.getpid(),
                "exit_code": self.exit_code,
                "run_dir": str(self.run_dir),
                "started_at": self.started_at,
                "updated_at": utc_now(),
            }

    def write(self) -> None:
        with self._lock:
            write_json_atomic(self.path, self.snapshot())

    def event(self, **payload: Any) -> None:
        """Append one run-level event (``schema``, ``run_id`` and ``ts`` added)."""
        line = json.dumps({"schema": SCHEMA, "ts": utc_now(), "run_id": self.run_id, **payload},
                          sort_keys=True, default=str)
        with self._lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def load_state(run_dir: Path) -> dict[str, Any] | None:
    """``state.json`` of a run directory, or None."""
    try:
        return json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
