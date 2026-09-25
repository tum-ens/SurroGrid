"""Run logs, subprocess execution and cancellation shared by the GridExpand runners.

- :class:`StatusLog`: ``events.jsonl`` (append-only), ``status.tsv`` (one row
  per job, rewritten on every update) and ``failed_grids.jsonl`` of one run
  directory.
- :func:`run_step` starts a pipeline step as a child process in its own process
  group and registers it, so :func:`cancel_children` (called by the signal
  handlers of :func:`install_cancel_handlers`) can stop every running step and
  its descendants. After a cancellation :data:`CANCEL` is set and running or
  new steps raise :class:`Cancelled`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from dotenv import dotenv_values

from gridexpand.common.timeframe import read_hdf_metadata
from gridexpand.paths import ENV_FILE, OPTIMIZATION_RESULT_DIR

# Columns of the synthetic runner's status.tsv (read by the service).
SYNTHETIC_STATUS_COLUMNS = (
    "candidate_index",
    "ags",
    "plz",
    "kcid",
    "bcid",
    "n_buildings",
    "bridge_filename",
    "demand_scope",
    "timeframe_mode",
    "horizon_hours",
    "timeframe_start",
    "timeframe_end",
    "status",
    "stage",
    "started_at",
    "finished_at",
    "seconds",
    "step3_cpus",
    "urbs_cluster_concurrency",
    "log_file",
    "message",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StatusLog:
    """Event log and per-job status table of one run directory.

    Args:
        run_dir: Directory of ``events.jsonl``, ``status.tsv``, ``failed_grids.jsonl``.
        resume: Load the rows of an existing ``status.tsv``.
        key_column: Column that identifies a job (``candidate_index`` keys are ints).
        columns: Columns of ``status.tsv`` in order (must contain ``key_column``).
        legacy_key: Returns the key of a loaded row that lacks ``key_column``
            (older files), or None to drop the row.
        listener: Called with every event payload (e.g. to mirror it to a run-level log).
        echo: Print every event line to stdout (the service parses these lines).
    """

    def __init__(
        self,
        run_dir: Path,
        resume: bool = False,
        *,
        key_column: str = "candidate_index",
        columns: tuple[str, ...] = SYNTHETIC_STATUS_COLUMNS,
        legacy_key: Callable[[dict[str, str]], Any] | None = None,
        listener: Callable[[dict[str, Any]], None] | None = None,
        echo: bool = True,
    ) -> None:
        if key_column not in columns:
            raise ValueError(f"status columns must contain {key_column!r}")
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.status_path = run_dir / "status.tsv"
        self.failed_path = run_dir / "failed_grids.jsonl"
        self.key_column = key_column
        self.columns = tuple(columns)
        self.listener = listener
        self.echo = echo
        self.lock = threading.Lock()
        self.rows: dict[Any, dict[str, object]] = {}
        if resume and self.status_path.exists():
            self._load_status(legacy_key)

    def _key(self, value: Any) -> Any:
        return int(value) if self.key_column == "candidate_index" else str(value)

    def _load_status(self, legacy_key: Callable[[dict[str, str]], Any] | None) -> None:
        with self.status_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = row.get(self.key_column)
                if not value and legacy_key is not None:
                    value = legacy_key(row)
                if not value and value != 0:
                    continue
                key = self._key(value)
                self.rows[key] = {**dict(row), self.key_column: key}

    def event(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if self.echo:
                print(line, flush=True)
        if self.listener is not None:
            try:
                self.listener(payload)
            except Exception as exc:  # a broken observer must never fail a grid
                print(f"[{utc_now()}] status listener failed: {exc!r}", file=sys.stderr, flush=True)

    def failed_grid(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.failed_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, key: Any, **updates: object) -> None:
        key = self._key(key)
        with self.lock:
            row = self.rows.setdefault(key, {self.key_column: key})
            row.update(updates)
            self._write_status_locked()

    def status_for(self, key: Any) -> str | None:
        row = self.rows.get(self._key(key))
        if not row:
            return None
        value = row.get("status")
        return str(value) if value else None

    def _write_status_locked(self) -> None:
        lines = ["\t".join(self.columns)]
        for key in sorted(self.rows):
            row = self.rows[key]
            lines.append("\t".join(str(row.get(column, "")) for column in self.columns))
        self.status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_GUROBI_HOME = Path("/opt/gurobi1302/linux64")
DEFAULT_GRB_LICENSE_FILE = Path.home() / "gurobi.lic"


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment for pipeline subprocesses.

    The process environment is inherited (e.g. ``GRIDEXPAND_SOLVER``,
    ``GRIDEXPAND_WORK_DIR``). ``GUROBI_HOME`` and ``GRB_LICENSE_FILE`` come
    from ``.env`` (which wins, as for the database settings) or the process
    environment. If unset, the former defaults ``/opt/gurobi1302/linux64`` and
    ``~/gurobi.lic`` are used when they exist. ``$GUROBI_HOME/bin`` and
    ``/lib`` are prepended to ``PATH`` and ``LD_LIBRARY_PATH``.
    """
    env = os.environ.copy()
    dotenv = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    for key in ("GUROBI_HOME", "GRB_LICENSE_FILE"):
        if dotenv.get(key):
            env[key] = str(dotenv[key])
    if not env.get("GUROBI_HOME") and DEFAULT_GUROBI_HOME.exists():
        env["GUROBI_HOME"] = str(DEFAULT_GUROBI_HOME)
    if not env.get("GRB_LICENSE_FILE") and DEFAULT_GRB_LICENSE_FILE.exists():
        env["GRB_LICENSE_FILE"] = str(DEFAULT_GRB_LICENSE_FILE)
    gurobi_home = env.get("GUROBI_HOME")
    if gurobi_home:
        env["PATH"] = f"{gurobi_home}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{gurobi_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    return env


# Cancellation -------------------------------------------------------------------

CANCEL = threading.Event()
CANCEL_GRACE_S = 15.0
_CHILDREN: dict[int, subprocess.Popen] = {}
_CHILDREN_LOCK = threading.Lock()


class Cancelled(RuntimeError):
    """The run was cancelled (SIGTERM, SIGINT or SIGHUP)."""


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, sig)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def cancel_children(sig: int = signal.SIGTERM, *, grace_s: float = CANCEL_GRACE_S) -> None:
    """Set :data:`CANCEL` and stop every registered step with its descendants.

    Sends ``sig`` to each child's process group; a watchdog sends SIGKILL to
    groups still alive after ``grace_s`` seconds.
    """
    CANCEL.set()
    with _CHILDREN_LOCK:
        children = list(_CHILDREN.values())
    for proc in children:
        _signal_group(proc, sig)

    def _watchdog() -> None:
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if all(proc.poll() is not None for proc in children):
                return
            time.sleep(0.2)
        for proc in children:
            if proc.poll() is None:
                _signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))

    if children:
        threading.Thread(target=_watchdog, name="cancel-watchdog", daemon=True).start()


def install_cancel_handlers() -> None:
    """Make SIGTERM, SIGINT and SIGHUP cancel the run instead of orphaning its steps.

    Call from the main thread of a runner. The first signal stops the child
    steps (their process groups get the same signal) and lets the runner
    record ``cancelled`` jobs; a second SIGINT/SIGTERM kills them at once.
    """

    def handler(signum, _frame) -> None:
        if CANCEL.is_set():
            cancel_children(getattr(signal, "SIGKILL", signal.SIGTERM), grace_s=0.0)
            return
        print(f"[{utc_now()}] received signal {signum}: cancelling running steps", file=sys.stderr, flush=True)
        cancel_children(signum)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, handler)


def run_step(
    cmd: list[str],
    *,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
    echo: bool = False,
    header: str | None = None,
) -> tuple[int, float]:
    """Run one step command, appending its output to ``log_path``.

    Args:
        cmd: argv of the step.
        log_path: Log file (appended).
        env_extra: Extra environment variables (see :func:`command_env`).
        cwd: Working directory (default: inherited).
        echo: Also copy the output to stdout, line by line.
        header: Stage name written into the START/END log lines.

    Returns:
        ``(returncode, seconds)``.

    Raises:
        Cancelled: the run was cancelled before or while the step ran.
    """
    if CANCEL.is_set():
        raise Cancelled(f"cancelled before {header or cmd[0]}")
    label = header or "step"
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] START {label}: {' '.join(cmd)}\n")
        if env_extra:
            log_handle.write(f"[{utc_now()}] ENV {env_extra}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=command_env(env_extra),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if echo else log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        with _CHILDREN_LOCK:
            _CHILDREN[proc.pid] = proc
        try:
            if CANCEL.is_set():  # a signal arrived between the check and the registration
                _signal_group(proc, signal.SIGTERM)
            if echo:
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_handle.write(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            returncode = proc.wait()
        finally:
            with _CHILDREN_LOCK:
                _CHILDREN.pop(proc.pid, None)
        seconds = round(time.monotonic() - started, 1)
        log_handle.write(f"[{utc_now()}] END {label}: rc={returncode} seconds={seconds}\n")
    if CANCEL.is_set() and returncode != 0:
        raise Cancelled(f"{label} cancelled (rc={returncode})")
    return returncode, seconds


def run_command(
    *,
    cmd: list[str],
    log_path: Path,
    status: StatusLog,
    job: Any,
    stage: str,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Run one step of job ``job`` (a ``status`` key) and record start/finish events.

    Raises:
        RuntimeError: the step exited with a non-zero code.
        Cancelled: the run was cancelled.
    """
    status.update(job, stage=stage, status="running", message="")
    key = {status.key_column: job}
    status.event(**key, stage=stage, event="start", cmd=cmd, env_extra=env_extra or {})
    returncode, seconds = run_step(cmd, log_path=log_path, env_extra=env_extra, cwd=cwd, header=stage)
    status.event(**key, stage=stage, event="finish", returncode=returncode, seconds=seconds)
    if returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {returncode}")


def run_batch_command(
    *,
    cmd: list[str],
    log_path: Path,
    status: StatusLog,
    stage: str,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Run one batch-level step (not tied to a job) and record start/finish events."""
    status.event(stage=stage, event="start", cmd=cmd, env_extra=env_extra or {})
    returncode, seconds = run_step(cmd, log_path=log_path, env_extra=env_extra, cwd=cwd, header=stage)
    status.event(stage=stage, event="finish", returncode=returncode, seconds=seconds)
    if returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {returncode}")


def latest_step3_result(input_hdf: Path, result_root: Path = OPTIMIZATION_RESULT_DIR) -> Path:
    """Return the newest Step 3 result for ``input_hdf`` below ``result_root``."""
    try:
        scenario_key = read_hdf_metadata(input_hdf).get("scenario_key")
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
        scenario_key = None
    if scenario_key:
        result_root = result_root / str(scenario_key)
    matches = sorted(
        result_root.glob(f"{input_hdf.stem}_*.h5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No Step 3 result found for {input_hdf.name} in {result_root}"
        )
    return matches[0]
