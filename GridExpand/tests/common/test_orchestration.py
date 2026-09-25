"""StatusLog, run_step and cancellation of running steps."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from gridexpand.common import orchestration
from gridexpand.common.orchestration import StatusLog, run_command, run_step


@pytest.fixture(autouse=True)
def _reset_cancel():
    orchestration.CANCEL.clear()
    yield
    orchestration.CANCEL.clear()


def test_status_log_keys_listener_and_echo(tmp_path, capsys):
    seen = []
    status = StatusLog(tmp_path, listener=lambda event: seen.append(event["event"]))
    status.update("3", status="running")  # candidate_index keys are ints
    status.event(event="start", candidate_index=3)
    assert 3 in status.rows and seen == ["start"] and '"event": "start"' in capsys.readouterr().out
    broken = StatusLog(tmp_path, echo=False, listener=lambda event: 1 / 0)
    broken.event(event="x")  # a failing observer must not raise
    assert "status listener failed" in capsys.readouterr().err


def test_run_step_logs_and_returns_the_code(tmp_path):
    log = tmp_path / "logs" / "a.log"
    code, seconds = run_step([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], log_path=log,
                             header="demo")
    text = log.read_text()
    assert code == 3 and seconds >= 0 and "START demo" in text and "hello" in text and "rc=3" in text
    status = StatusLog(tmp_path, echo=False)
    with pytest.raises(RuntimeError, match="failed with return code 3"):
        run_command(cmd=[sys.executable, "-c", "raise SystemExit(3)"], log_path=log, status=status, job=1,
                    stage="step")
    assert status.rows[1]["stage"] == "step"


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_cancel_stops_the_step_and_its_children(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    errors = []

    def target():
        try:
            run_step([sys.executable, "-c", code], log_path=tmp_path / "c.log", header="long")
        except orchestration.Cancelled as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    deadline = time.time() + 20
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    orchestration.cancel_children()
    thread.join(timeout=20)
    assert errors and isinstance(errors[0], orchestration.Cancelled)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        # reap if it became our zombie (it is a child of the killed step, not ours)
        time.sleep(0.1)
    else:
        pytest.fail("the step's child process survived the cancellation")
    with pytest.raises(orchestration.Cancelled):
        run_step([sys.executable, "-c", "pass"], log_path=tmp_path / "d.log")  # no new steps after cancel
