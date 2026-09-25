"""Shared logging and process execution for GridExpand pipeline runners."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from dotenv import dotenv_values

from gridexpand.common.timeframe import read_hdf_metadata
from gridexpand.paths import ENV_FILE, OPTIMIZATION_RESULT_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StatusLog:
    def __init__(self, run_dir: Path, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.status_path = run_dir / "status.tsv"
        self.failed_path = run_dir / "failed_grids.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[int, dict[str, object]] = {}
        if resume and self.status_path.exists():
            self._load_status()

    def _load_status(self) -> None:
        with self.status_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not row.get("candidate_index"):
                    continue
                self.rows[int(row["candidate_index"])] = dict(row)

    def event(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def failed_grid(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.failed_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, candidate_index: int, **updates: object) -> None:
        with self.lock:
            row = self.rows.setdefault(
                candidate_index, {"candidate_index": candidate_index}
            )
            row.update(updates)
            self._write_status_locked()

    def status_for(self, candidate_index: int) -> str | None:
        row = self.rows.get(candidate_index)
        if not row:
            return None
        value = row.get("status")
        return str(value) if value else None

    def _write_status_locked(self) -> None:
        columns = [
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
        ]
        lines = ["\t".join(columns)]
        for index in sorted(self.rows):
            row = self.rows[index]
            lines.append("\t".join(str(row.get(column, "")) for column in columns))
        self.status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_GUROBI_HOME = Path("/opt/gurobi1302/linux64")
DEFAULT_GRB_LICENSE_FILE = Path.home() / "gurobi.lic"


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment for pipeline subprocesses.

    ``GUROBI_HOME`` and ``GRB_LICENSE_FILE`` come from ``.env`` (which wins, as
    for the database settings) or the process environment. If unset, the
    former defaults ``/opt/gurobi1302/linux64`` and ``~/gurobi.lic`` are used
    when they exist. ``$GUROBI_HOME/bin`` and ``/lib`` are prepended to
    ``PATH`` and ``LD_LIBRARY_PATH``.
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


def run_command(
    *,
    cmd: list[str],
    log_path: Path,
    status: StatusLog,
    candidate_index: int,
    stage: str,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    status.update(candidate_index, stage=stage, status="running", message="")
    status.event(
        candidate_index=candidate_index,
        stage=stage,
        event="start",
        cmd=cmd,
        env_extra=env_extra or {},
    )
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] START {stage}: {' '.join(cmd)}\n")
        if env_extra:
            log_handle.write(f"[{utc_now()}] ENV {env_extra}\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=command_env(env_extra),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        seconds = round(time.monotonic() - started, 1)
        log_handle.write(
            f"[{utc_now()}] END {stage}: rc={completed.returncode} seconds={seconds}\n"
        )
    status.event(
        candidate_index=candidate_index,
        stage=stage,
        event="finish",
        returncode=completed.returncode,
        seconds=seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


def run_batch_command(
    *,
    cmd: list[str],
    log_path: Path,
    status: StatusLog,
    stage: str,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    status.event(stage=stage, event="start", cmd=cmd, env_extra=env_extra or {})
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] START {stage}: {' '.join(cmd)}\n")
        if env_extra:
            log_handle.write(f"[{utc_now()}] ENV {env_extra}\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=command_env(env_extra),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        seconds = round(time.monotonic() - started, 1)
        log_handle.write(
            f"[{utc_now()}] END {stage}: rc={completed.returncode} seconds={seconds}\n"
        )
    status.event(
        stage=stage, event="finish", returncode=completed.returncode, seconds=seconds
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


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
