#!/usr/bin/env python3
"""Run compact electric-only Forchheim synthetic-grid power-flow summaries.

This runner coordinates GridExpand Step 2 demand allocation and Step 4 pre-only
power-flow for the Forchheim PLZ comparison case. It intentionally skips URBS
and stores only compact p99/p01 summary tables in PostgreSQL.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

from sqlalchemy import text


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDEXPAND_DIR = REPO_ROOT / "GridExpand"
DEFAULT_AGS = "9474126"
DEFAULT_PLZ = 91301
DEFAULT_MIN_BUILDINGS = 5
DEFAULT_RUN_NAME = "baseline_static_pre_powerflow_backbone"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run electric-only demand allocation and compact pre-only power-flow "
            "summaries for the Forchheim synthetic grids."
        )
    )
    parser.add_argument("--ags", default=DEFAULT_AGS)
    parser.add_argument("--plz", type=int, default=DEFAULT_PLZ)
    parser.add_argument("--min-buildings", type=int, default=DEFAULT_MIN_BUILDINGS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--step2-cpus", type=int, default=1)
    parser.add_argument("--step4-cpus", type=int, default=1)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-existing", action="store_true", help="Recompute grids that already have compact summaries.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-step4-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    return parser.parse_args()


class StatusLog:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.status_path = run_dir / "status.tsv"
        self.events_path = run_dir / "events.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[int, dict[str, object]] = {}

    def update(self, candidate_index: int, **updates: object) -> None:
        with self.lock:
            row = self.rows.setdefault(candidate_index, {"candidate_index": candidate_index})
            row.update(updates)
            self._write_status_locked()

    def event(self, **payload: object) -> None:
        payload = {"ts": utc_now(), **payload}
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def _write_status_locked(self) -> None:
        columns = [
            "candidate_index",
            "plz",
            "kcid",
            "bcid",
            "n_buildings",
            "bridge_filename",
            "status",
            "stage",
            "started_at",
            "finished_at",
            "seconds",
            "attempt",
            "log_file",
            "message",
        ]
        with self.status_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for index in sorted(self.rows):
                row = self.rows[index]
                writer.writerow({column: row.get(column, "") for column in columns})


def configure_imports() -> None:
    if str(GRIDEXPAND_DIR) not in sys.path:
        sys.path.insert(0, str(GRIDEXPAND_DIR))


def existing_summary_filenames(plz: int, run_name: str) -> set[str]:
    configure_imports()
    from database import SurroGridDatabase

    db = SurroGridDatabase()
    db.ensure_schema()
    query = text(
        """
        SELECT pr.urbs_input_file
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc ON gc.grid_case_id = pr.grid_case_id
        JOIN surrogrid.powerflow_summary pfs ON pfs.powerflow_run_id = pr.powerflow_run_id
        WHERE gc.plz = :plz
          AND pr.run_name = :run_name
          AND pfs.stage = 'pre'
        """
    )
    with db.engine.connect() as conn:
        return set(conn.execute(query, {"plz": plz, "run_name": run_name}).scalars())


def run_command(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    stage: str,
    status: StatusLog,
    candidate_index: int,
    max_retries: int = 1,
    retry_delay: float = 0.0,
) -> None:
    attempt = 1
    while True:
        status.update(candidate_index, stage=stage, attempt=attempt, message="")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] START attempt={attempt} {cwd}: {' '.join(cmd)}\n")
            log_handle.flush()
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_handle.write(f"[{utc_now()}] END rc={completed.returncode}: {' '.join(cmd)}\n")
            log_handle.flush()
        if completed.returncode == 0:
            return

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        transient_deadlock = "DeadlockDetected" in log_text or "deadlock detected" in log_text
        if transient_deadlock and attempt < max_retries:
            status.event(
                event="retry_deadlock",
                candidate_index=candidate_index,
                stage=stage,
                attempt=attempt,
                retry_delay=retry_delay,
            )
            time.sleep(retry_delay * attempt)
            attempt += 1
            continue
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {' '.join(cmd)}")


def run_candidate(candidate: dict[str, Any], args: argparse.Namespace, status: StatusLog) -> dict[str, object]:
    step2_dir = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
    step4_dir = GRIDEXPAND_DIR / "4.powerflow"
    candidate_index = int(candidate["candidate_index"])
    bridge_filename = str(candidate["bridge_filename"])
    log_path = status.run_dir / "logs" / f"{candidate_index:03d}_{bridge_filename}.log"
    started = time.monotonic()
    current_stage = "queued"
    status.update(
        candidate_index,
        plz=candidate.get("plz", ""),
        kcid=candidate.get("kcid", ""),
        bcid=candidate.get("bcid", ""),
        n_buildings=candidate.get("n_buildings", ""),
        bridge_filename=bridge_filename,
        status="running",
        stage=current_stage,
        started_at=utc_now(),
        finished_at="",
        seconds="",
        attempt="",
        log_file=log_path,
        message="",
    )
    try:
        current_stage = "step2_status_quo"
        run_command(
            cmd=[
                "uv",
                "run",
                "--project",
                "..",
                "python",
                "main.py",
                args.ags,
                "--storage",
                "db",
                "--candidate-index",
                str(candidate_index),
                "--min-buildings",
                str(args.min_buildings),
                "--profiles",
                "status_quo",
                "--timeseries-storage",
                "temp",
                "--n_cpu",
                str(args.step2_cpus),
            ],
            cwd=step2_dir,
            log_path=log_path,
            stage=current_stage,
            status=status,
            candidate_index=candidate_index,
        )

        source = step2_dir / "results" / bridge_filename
        target = step4_dir / "Input" / bridge_filename
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

        current_stage = "step4_summary_powerflow"
        run_command(
            cmd=[
                "uv",
                "run",
                "python",
                "run_pwrflw.py",
                bridge_filename,
                "--storage",
                "db",
                "--pre-only",
                "--summary-only",
                "--run-name",
                args.run_name,
                "--n_cpu",
                str(args.step4_cpus),
            ],
            cwd=step4_dir,
            log_path=log_path,
            stage=current_stage,
            status=status,
            candidate_index=candidate_index,
            max_retries=max(1, int(args.max_step4_retries)),
            retry_delay=max(0.0, float(args.retry_delay)),
        )

        seconds = round(time.monotonic() - started, 1)
        status.update(
            candidate_index,
            status="done",
            stage="complete",
            finished_at=utc_now(),
            seconds=seconds,
            message="ok",
        )
        return {"candidate_index": candidate_index, "status": "done", "seconds": seconds}
    except Exception as exc:
        seconds = round(time.monotonic() - started, 1)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] FAILURE in {current_stage}: {exc}\n")
            log_handle.write(traceback.format_exc())
        status.update(
            candidate_index,
            status="failed",
            stage=current_stage,
            finished_at=utc_now(),
            seconds=seconds,
            message=str(exc),
        )
        return {"candidate_index": candidate_index, "status": "failed", "seconds": seconds, "message": str(exc)}


def main() -> int:
    args = parse_args()
    configure_imports()
    from ags_pipeline_runner import get_candidates
    from database import SurroGridDatabase

    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = GRIDEXPAND_DIR / "run_logs" / f"forchheim_summary_only_{stamp}"
    run_dir = run_dir.resolve()
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    status = StatusLog(run_dir)
    started = time.monotonic()

    SurroGridDatabase().ensure_schema()
    candidates = [
        row
        for row in get_candidates(REPO_ROOT, args.ags, args.min_buildings)
        if int(row["plz"]) == int(args.plz)
    ]
    if args.start_index is not None:
        candidates = [row for row in candidates if int(row["candidate_index"]) >= args.start_index]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    existing = set() if args.rerun_existing else existing_summary_filenames(args.plz, args.run_name)
    selected = [row for row in candidates if str(row["bridge_filename"]) not in existing]
    skipped = len(candidates) - len(selected)
    (run_dir / "candidates.json").write_text(json.dumps(candidates, indent=2, default=str), encoding="utf-8")
    status.event(
        event="batch_start",
        ags=args.ags,
        plz=args.plz,
        candidates=len(candidates),
        selected=len(selected),
        skipped_existing=skipped,
        workers=args.workers,
        step2_cpus=args.step2_cpus,
        step4_cpus=args.step4_cpus,
        run_dir=str(run_dir),
    )
    if not selected:
        summary = {
            "status": "done",
            "candidate_count": len(candidates),
            "selected_count": 0,
            "skipped_existing": skipped,
            "completed_count": 0,
            "failure_count": 0,
            "total_seconds": round(time.monotonic() - started, 1),
            "finished_at": utc_now(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        status.event(event="batch_finish", **summary)
        return 0

    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(run_candidate, candidate, args, status): candidate for candidate in selected}
        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"candidate_index": int(candidate["candidate_index"]), "status": "failed", "message": str(exc)}
            if result.get("status") == "done":
                completed.append(result)
                status.event(event="candidate_done", **result)
            else:
                failures.append(result)
                status.event(event="candidate_failed", **result)

    summary = {
        "status": "done" if not failures else "completed_with_failures",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "skipped_existing": skipped,
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "total_seconds": round(time.monotonic() - started, 1),
        "finished_at": utc_now(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status.event(event="batch_finish", **summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
