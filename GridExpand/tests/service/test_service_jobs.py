"""JobManager with stand-in commands: steps, queue, cancel of the process group, history, progress."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gridexpand.service.jobs import JobManager, JobStep


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _alive(pid: int) -> bool:
    """Whether a process exists and is not a zombie."""
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (ProcessLookupError, FileNotFoundError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def wait(job, timeout: float = 30.0):
    deadline = time.time() + timeout
    while job.active and time.time() < deadline:
        time.sleep(0.05)
    assert not job.active, f"job still {job.status}"
    return job


def texts(job) -> list[str]:
    return [line["text"] for line in job.lines]


def test_steps_run_in_order_and_are_logged(tmp_path):
    manager = JobManager(tmp_path / "jobs")
    job = manager.submit("test", "two steps", [JobStep("a", py("print('hello a')")), JobStep("b", py("print('hello b')"))])
    wait(job)
    assert job.status == "succeeded" and job.progress == 1.0 and job.exit_code == 0
    assert [s.status for s in job.steps] == ["succeeded", "succeeded"]
    out = texts(job)
    assert out.index("hello a") < out.index("hello b")
    log = (tmp_path / "jobs" / f"{job.id}.log").read_text(encoding="utf-8")
    assert "hello b" in log and "Job succeeded" in log
    assert json.loads((tmp_path / "jobs" / f"{job.id}.json").read_text())["status"] == "succeeded"


def test_a_failing_step_stops_the_job(tmp_path):
    manager = JobManager(tmp_path / "jobs")
    job = manager.submit("test", "fails", [JobStep("a", py("import sys; print('Traceback (most recent call last):');"
                                                              " print('  File x'); print('ValueError: no'); sys.exit(3)")),
                                           JobStep("b", py("print('never')"))])
    wait(job)
    assert job.status == "failed" and job.exit_code == 3 and "step 'a' failed" in job.error
    assert [s.status for s in job.steps] == ["failed", "skipped"]
    levels = {line["text"]: line["level"] for line in job.lines}
    assert levels["  File x"] == "trace" and levels["ValueError: no"] == "error"
    assert "never" not in texts(job)


def test_queue_runs_one_job_at_a_time(tmp_path):
    manager = JobManager(tmp_path / "jobs", max_running=1)
    first = manager.submit("test", "first", [JobStep("a", py("import time; time.sleep(1)"))])
    second = manager.submit("test", "second", [JobStep("a", py("print('second')"))])
    assert first.status == "running" and second.status == "queued" and manager.queue_position(second.id) == 1
    wait(second)
    assert first.status == "succeeded" and second.status == "succeeded"
    assert first.finished_at <= second.started_at


def test_cancel_stops_the_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            f"open({str(pid_file)!r}, 'w').write(str(child.pid)); print('started', flush=True); time.sleep(120)")
    manager = JobManager(tmp_path / "jobs")
    job = manager.submit("test", "sleeper", [JobStep("a", py(code)), JobStep("b", py("print('never')"))])
    deadline = time.time() + 20
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    child = int(pid_file.read_text())
    queued = manager.submit("test", "waiting", [JobStep("a", py("print('x')"))])
    manager.cancel(queued.id)
    assert queued.status == "cancelled"
    manager.cancel(job.id)
    wait(job)
    assert job.status == "cancelled" and [s.status for s in job.steps] == ["cancelled", "cancelled"]
    deadline = time.time() + 10
    while _alive(child) and time.time() < deadline:
        time.sleep(0.1)
    assert not _alive(child), "the grandchild survived the cancel"


def test_history_survives_a_restart(tmp_path):
    jobs_dir = tmp_path / "jobs"
    manager = JobManager(jobs_dir)
    done = wait(manager.submit("test", "done", [JobStep("a", py("print('kept')"))]))
    # a job that was running when the service died
    meta = json.loads((jobs_dir / f"{done.id}.json").read_text())
    meta.update(id="crashed0001", status="running", title="crashed")
    meta["steps"][0]["status"] = "running"
    (jobs_dir / "crashed0001.json").write_text(json.dumps(meta))
    reloaded = JobManager(jobs_dir)
    assert reloaded.get(done.id).status == "succeeded" and "kept" in texts(reloaded.get(done.id))
    crashed = reloaded.get("crashed0001")
    assert crashed.status == "failed" and crashed.steps[0].status == "failed"
    assert "stopped before the job finished" in crashed.error


def test_run_directory_progress_and_step_logs(tmp_path):
    run_dir = tmp_path / "runs" / "pre"
    events = [{"event": "batch_start", "ags": 1, "pylovo_version_id": "1"}, {"event": "candidates_selected", "count": 2},
              {"event": "start", "candidate_index": 0, "stage": "step2_demand_allocation"},
              {"event": "candidate_done", "candidate_index": 0, "status": "done", "seconds": 1},
              {"event": "candidate_done", "candidate_index": 1, "status": "done", "seconds": 1},
              {"event": "batch_finish", "status": "done", "completed_count": 2, "candidate_count": 2}]
    code = (
        "import json, os, pathlib, sys, time\n"
        f"run = pathlib.Path(sys.argv[1]); (run / 'logs').mkdir(parents=True, exist_ok=True)\n"
        "(run / 'logs' / 'candidate_000_x.h5.log').write_text('step output line\\n')\n"
        f"for e in {events!r}:\n"
        "    line = json.dumps(e)\n"
        "    open(run / 'events.jsonl', 'a').write(line + '\\n'); print(line, flush=True); time.sleep(0.05)\n"
    )
    manager = JobManager(tmp_path / "jobs")
    job = manager.submit("pipeline", "run", [JobStep("pre", [sys.executable, "-c", code, str(run_dir)], run_dir=str(run_dir))])
    wait(job)
    step = job.steps[0]
    assert job.status == "succeeded"
    assert (step.grids_total, step.grids_done, step.grids_failed, step.batch_status) == (2, 2, 0, "done")
    out = texts(job)
    assert "#0 │ step output line" in out
    assert any(t.startswith("✓ [pre] finished: done · 2/2 grids") for t in out)
    assert not any(t.startswith('{"event"') for t in out)  # runner events are shown formatted
    assert {line["level"] for line in job.lines if line["text"].startswith("#0 │")} == {"debug"}
    assert Path(step.run_dir) == run_dir
