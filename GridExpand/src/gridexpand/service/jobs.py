"""Background jobs: the service runs ``gridexpand`` commands as subprocesses.

Modelled on pylovo-ui's job runner (``pylovo_ui/jobs.py``) so both tools behave alike:
a job does exactly what the same commands do in a terminal, its output is kept in memory
and in ``<state>/jobs/<id>.log``, browsers follow it through Server-Sent Events, and it
can be cancelled. Differences:

- A job is a list of *steps* (one command per model case) that run one after another.
- At most ``max_running`` jobs run at the same time; later jobs wait in a queue.
- Each step runs in its own process group; cancelling sends SIGTERM to the whole group
  (the runner and every step it started), SIGKILL if it is still alive after a grace time.
- A step with a run directory is followed through the runner's ``events.jsonl`` (progress)
  and the step logs in ``logs/`` (shown in the job log with a ``#<grid> │`` prefix).
- Job metadata survives a restart; jobs that were still active are recorded as failed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gridexpand.service.runlog import (
    RunTracker,
    classify_line,
    format_event,
    parse_event_line,
)

MAX_LINES_IN_MEMORY = 50_000
HISTORY_LIMIT = 100
POLL_INTERVAL_S = 0.5
CANCEL_GRACE_S = 20.0
ACTIVE = ("queued", "running")


class JobNotFound(KeyError):
    """No job with this id."""


@dataclass
class JobStep:
    """One command of a job."""

    name: str
    argv: list[str]
    run_dir: str | None = None
    status: str = "pending"  # pending | running | succeeded | failed | cancelled | skipped
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    grids_total: int | None = None
    grids_done: int = 0
    grids_failed: int = 0
    stage: str | None = None
    batch_status: str | None = None
    message: str | None = None

    def fraction(self) -> float:
        if self.status in ("succeeded", "failed", "cancelled", "skipped"):
            return 1.0
        if self.status != "running" or not self.grids_total:
            return 0.0
        return min(0.98, (self.grids_done + self.grids_failed) / self.grids_total)

    def summary(self) -> dict[str, Any]:
        end = self.finished_at or time.time()
        return {
            "name": self.name,
            "command": display_command(self.argv),
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(end - self.started_at, 1) if self.started_at else None,
            "grids_total": self.grids_total,
            "grids_done": self.grids_done,
            "grids_failed": self.grids_failed,
            "stage": self.stage,
            "batch_status": self.batch_status,
            "message": self.message,
        }


@dataclass
class Job:
    """One queued, running or finished job."""

    id: str
    kind: str
    title: str
    steps: list[JobStep]
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    pid: int | None = None
    current_step: int | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"warning": 0, "error": 0})
    lines: list[dict[str, Any]] = field(default_factory=list, repr=False)
    first_seq: int = 0
    next_seq: int = 0
    cancel_requested: bool = False
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _log_path: Path | None = field(default=None, repr=False)
    _trace: dict[str, bool] = field(default_factory=dict, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    @property
    def progress(self) -> float | None:
        if self.status == "succeeded":
            return 1.0
        if self.status == "queued" or not self.steps:
            return None
        return round(sum(step.fraction() for step in self.steps) / len(self.steps), 4)

    @property
    def phase(self) -> str | None:
        if self.status != "running" or self.current_step is None:
            return None
        step = self.steps[self.current_step]
        parts = [step.name]
        if step.grids_total:
            parts.append(f"{step.grids_done + step.grids_failed}/{step.grids_total} grids")
        if step.stage:
            parts.append(step.stage)
        return " · ".join(parts)

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable description without the log lines."""
        end = self.finished_at or time.time()
        index = self.current_step if self.current_step is not None else 0
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "command": display_command(self.steps[index].argv) if self.steps else "",
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(end - self.started_at, 1) if self.started_at else None,
            "exit_code": self.exit_code,
            "error": self.error,
            "progress": self.progress,
            "phase": self.phase,
            "current_step": self.current_step,
            "counts": dict(self.counts),
            "line_count": self.next_seq,
            "steps": [step.summary() for step in self.steps],
        }

    def lines_after(self, seq: int) -> list[dict[str, Any]]:
        """Log lines with a sequence number >= ``seq`` that are still in memory."""
        return self.lines[max(0, seq - self.first_seq):]


def line_level(text: str) -> str:
    """Display level of a stored log line (step output with a ``label │`` prefix is muted)."""
    label, sep, body = text.partition(" │ ")
    if sep and len(label) <= 12:
        level = classify_line(body)
        return "debug" if level == "info" else level
    return classify_line(text)


def display_command(argv: list[str]) -> str:
    """Shell-like, shortened display of a command line."""
    shown = []
    for arg in argv:
        name = os.path.basename(arg)
        if name.startswith("python"):
            shown.append("python")
        elif re.fullmatch(r"[\w./:=@%+,-]+", arg):
            shown.append(arg)
        else:
            shown.append(json.dumps(arg))
    return " ".join(shown)


class JobManager:
    """Queue, run, observe and cancel jobs made of ``gridexpand`` commands.

    Args:
        jobs_dir: Directory for job metadata and logs.
        cwd: Working directory of all commands.
        env: Extra environment variables for all commands.
        max_running: Jobs that may run at the same time.
    """

    def __init__(self, jobs_dir: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None,
                 max_running: int = 1) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = Path(cwd) if cwd else None
        self.env = dict(env or {})
        self.max_running = max(1, int(max_running))
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._lock = threading.RLock()
        self._closing = False
        self._load_history()

    # ------------------------------------------------------------------ queries
    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def counts(self) -> dict[str, int]:
        with self._lock:
            jobs = list(self._jobs.values())
        return {"running": sum(j.status == "running" for j in jobs), "queued": sum(j.status == "queued" for j in jobs)}

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            return self._queue.index(job_id) + 1 if job_id in self._queue else None

    # ------------------------------------------------------------------ control
    @staticmethod
    def new_id() -> str:
        """A fresh job id (to build run directories before :meth:`submit`)."""
        return uuid.uuid4().hex[:10]

    def submit(self, kind: str, title: str, steps: list[JobStep], params: dict[str, Any] | None = None,
               job_id: str | None = None) -> Job:
        """Queue a job; it starts as soon as fewer than ``max_running`` jobs run."""
        if not steps:
            raise ValueError("A job needs at least one step")
        with self._lock:
            if self._closing:
                raise RuntimeError("The service is shutting down")
            job_id = job_id or self.new_id()
            if job_id in self._jobs:
                raise ValueError(f"Job id {job_id} exists")
            job = Job(id=job_id, kind=kind, title=title, steps=list(steps), params=dict(params or {}))
            job._log_path = self.jobs_dir / f"{job.id}.log"
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._append(job, f"Queued: {title}", "info")
            self._save(job)
            self._trim_history()
        self._dispatch()
        return job

    def cancel(self, job_id: str) -> Job:
        """Remove a queued job from the queue or stop a running one."""
        job = self.get(job_id)
        with self._lock:
            if not job.active:
                return job
            job.cancel_requested = True
            if job.status == "queued":
                self._queue.remove(job.id)
                for step in job.steps:
                    step.status = "cancelled"
                self._append(job, "Cancelled before it started", "warning")
                self._finish(job, "cancelled", None)
                return job
            proc = job._proc
        self._append(job, "Cancellation requested: stopping the process group (SIGTERM) …", "warning")
        if proc is not None and proc.poll() is None:
            threading.Thread(target=self._terminate, args=(proc,), name=f"cancel-{job.id}", daemon=True).start()
        return job

    def shutdown(self, wait_s: float = 5.0) -> None:
        """Stop running jobs, drop the queue and record both as cancelled."""
        with self._lock:
            self._closing = True
            queued = [self._jobs[job_id] for job_id in self._queue]
            self._queue.clear()
            running = [job for job in self._jobs.values() if job.status == "running"]
        for job in queued:
            job.cancel_requested = True
            for step in job.steps:
                step.status = "cancelled"
            self._append(job, "The service stopped before the job started", "warning")
            self._finish(job, "cancelled", None)
        for job in running:
            job.cancel_requested = True
            self._append(job, "The service is shutting down: stopping the job", "warning")
            if job._proc is not None and job._proc.poll() is None:
                self._signal(job._proc, signal.SIGTERM)
        deadline = time.time() + wait_s
        while time.time() < deadline and any(job.active for job in running):
            time.sleep(0.1)

    # ------------------------------------------------------------------ internals
    def _dispatch(self) -> None:
        with self._lock:
            if self._closing:
                return
            running = sum(job.status == "running" for job in self._jobs.values())
            while self._queue and running < self.max_running:
                job = self._jobs[self._queue.pop(0)]
                job.status = "running"
                job.started_at = time.time()
                running += 1
                self._save(job)
                threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()

    def _run(self, job: Job) -> None:
        stop_reason = None
        try:
            for index, step in enumerate(job.steps):
                if job.cancel_requested:
                    step.status = "cancelled"
                    continue
                if stop_reason:
                    step.status = "skipped"
                    continue
                job.current_step = index
                code = self._run_step(job, step)
                if job.cancel_requested:
                    continue
                job.exit_code = code if job.exit_code in (None, 0) else job.exit_code
                if code != 0 and step.batch_status != "completed_with_failures":
                    stop_reason = f"step '{step.name}' failed (exit code {code})"
        except Exception as exc:  # noqa: BLE001 - a broken job must not kill the service
            stop_reason = f"internal error: {exc}"
            self._append(job, f"✗ {stop_reason}", "error")
        if job.cancel_requested:
            status = "cancelled"
        elif stop_reason or any(step.status != "succeeded" for step in job.steps):
            status = "failed"
            job.error = stop_reason or "at least one step reported failed grids"
            if stop_reason:
                self._append(job, f"Stopped: {stop_reason}", "error")
        else:
            status = "succeeded"
        self._finish(job, status, job.exit_code)
        self._dispatch()

    def _run_step(self, job: Job, step: JobStep) -> int | None:
        step.status = "running"
        step.started_at = time.time()
        self._save(job)
        tracker = None
        if step.run_dir:
            Path(step.run_dir).mkdir(parents=True, exist_ok=True)
            tracker = RunTracker(Path(step.run_dir))
        env = dict(os.environ)
        env.update(self.env)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "GRIDEXPAND_SERVICE_JOB": job.id})
        self._append(job, f"$ {display_command(step.argv)}", "command")
        try:
            proc = subprocess.Popen(
                step.argv, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            self._append(job, f"✗ Could not start the command: {exc}", "error")
            step.status, step.finished_at = "failed", time.time()
            return None
        job._proc, job.pid = proc, proc.pid
        reader = threading.Thread(target=self._read_output, args=(job, step, proc), daemon=True)
        reader.start()
        while True:
            exited = proc.poll() is not None
            if tracker:
                self._poll_tracker(job, step, tracker)
            if exited:
                break
            time.sleep(POLL_INTERVAL_S)
        reader.join(timeout=10)
        if tracker:
            self._poll_tracker(job, step, tracker)
        code = proc.returncode
        job._proc = None
        step.exit_code = code
        step.finished_at = time.time()
        step.status = "cancelled" if job.cancel_requested else "succeeded" if code == 0 else "failed"
        self._save(job)
        return code

    def _read_output(self, job: Job, step: JobStep, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            event = parse_event_line(line)
            if event is not None:
                text = format_event(event, step.name)
                self._append(job, text, classify_line(text))
            else:
                self._append(job, line, source="stdout")

    def _poll_tracker(self, job: Job, step: JobStep, tracker: RunTracker) -> None:
        tracker.poll_events()
        progress = tracker.progress
        step.grids_total = progress.grids_total
        step.grids_done, step.grids_failed = progress.grids_done, progress.grids_failed
        step.stage, step.batch_status = progress.stage, progress.batch_status
        step.message = progress.message
        for label, line in tracker.poll_logs():
            self._append(job, f"{label} │ {line}", source=label, quiet=True)

    def _append(self, job: Job, text: str, level: str | None = None, *, source: str = "", quiet: bool = False) -> None:
        with self._lock:
            if level is None:
                body = text.split(" │ ", 1)[1] if quiet and " │ " in text else text
                level = classify_line(body)
                if body.startswith("Traceback"):
                    job._trace[source] = True
                elif job._trace.get(source):
                    if body.startswith((" ", "\t")) or not body.strip():
                        level = "trace"
                    else:
                        job._trace[source] = False
                        level = "error"
                elif quiet and level == "info":
                    level = "debug"  # output of the pipeline steps, shown muted
            if level in job.counts:
                job.counts[level] += 1
            job.lines.append({"seq": job.next_seq, "t": round(time.time(), 3), "text": text, "level": level})
            job.next_seq += 1
            if len(job.lines) > MAX_LINES_IN_MEMORY:
                drop = len(job.lines) - MAX_LINES_IN_MEMORY
                del job.lines[:drop]
                job.first_seq += drop
            if job._log_path:
                try:
                    with job._log_path.open("a", encoding="utf-8") as handle:
                        handle.write(text + "\n")
                except OSError:
                    pass

    def _finish(self, job: Job, status: str, code: int | None) -> None:
        job.status = status
        job.exit_code = code
        job.finished_at = time.time()
        job._proc = None
        took = job.finished_at - (job.started_at or job.finished_at)
        label = {"succeeded": "success", "failed": "error", "cancelled": "warning"}[status]
        mark = {"succeeded": "✓", "failed": "✗", "cancelled": "■"}[status]
        self._append(job, f"{mark} Job {status} after {took:.1f} s", label)
        self._save(job)

    def _save(self, job: Job) -> None:
        data = job.summary() | {"argv": [step.argv for step in job.steps],
                                "run_dirs": [step.run_dir for step in job.steps]}
        path = self.jobs_dir / f"{job.id}.json"
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _terminate(self, proc: subprocess.Popen) -> None:
        self._signal(proc, signal.SIGTERM)
        deadline = time.time() + CANCEL_GRACE_S
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        self._signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))

    def _trim_history(self) -> None:
        finished = sorted((job for job in self._jobs.values() if not job.active), key=lambda job: job.created_at)
        for job in finished[: max(0, len(self._jobs) - HISTORY_LIMIT)]:
            self._jobs.pop(job.id, None)

    def _load_history(self) -> None:
        metas = sorted(self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[-HISTORY_LIMIT:]
        for meta_path in metas:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                argvs = meta.get("argv") or []
                run_dirs = meta.get("run_dirs") or [None] * len(argvs)
                steps = []
                for argv, run_dir, info in zip(argvs, run_dirs, meta.get("steps") or [], strict=False):
                    step = JobStep(name=info.get("name", "step"), argv=list(argv), run_dir=run_dir)
                    for key in ("status", "exit_code", "started_at", "finished_at", "grids_total", "grids_done",
                                "grids_failed", "stage", "batch_status", "message"):
                        setattr(step, key, info.get(key, getattr(step, key)))
                    steps.append(step)
                job = Job(id=meta["id"], kind=meta["kind"], title=meta["title"], steps=steps,
                          params=meta.get("params") or {}, status=meta["status"], created_at=meta["created_at"],
                          started_at=meta.get("started_at"), finished_at=meta.get("finished_at"),
                          exit_code=meta.get("exit_code"), error=meta.get("error"),
                          current_step=meta.get("current_step"),
                          counts=meta.get("counts") or {"warning": 0, "error": 0})
                job._log_path = self.jobs_dir / f"{job.id}.log"
                if job._log_path.exists():
                    text = job._log_path.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines()):
                        job.lines.append({"seq": i, "t": None, "text": line, "level": line_level(line)})
                    job.next_seq = len(job.lines)
                if job.active:  # the service stopped while the job was queued or running
                    for step in job.steps:
                        if step.status in ("pending", "running"):
                            step.status = "failed" if step.status == "running" else "cancelled"
                    job.error = "The service stopped before the job finished"
                    job._trace = {}
                    self._jobs[job.id] = job
                    self._append(job, f"✗ {job.error}", "error")
                    job.status, job.finished_at = "failed", job.finished_at or time.time()
                    self._save(job)
                    continue
                self._jobs[job.id] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue
