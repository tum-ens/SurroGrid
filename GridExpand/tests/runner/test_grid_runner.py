"""Paired runner: job keys (orch B4/B5), result-case rules and adapter commands (no database, no data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from gridexpand.runner import grid_runner as runner
from gridexpand.runner.targets import adapter_for, real, synthetic


def test_job_keys_and_legacy_rows(tmp_path):
    jobs = runner._number_jobs([
        {"target_network": "real_dso_a", "target_grid_id": 12},
        {"target_network": "synthetic", "target_grid_id": 7},
    ])
    assert [(job["job_index"], job["job"]) for job in jobs] == [(0, "real_dso_a:12"), (1, "synthetic:7")]
    # a status.tsv from before job keys: rows keyed by list position, grid only in the log name
    (tmp_path / "status.tsv").write_text(
        "candidate_index\tstatus\tlog_file\n"
        f"0\tdone\t{tmp_path}/logs/real_dso_b_12.log\n"
        f"1\tfailed\t{tmp_path}/logs/synthetic_7.log\n"
    )
    status = runner.open_status(tmp_path, resume=True)
    assert status.status_for("real_dso_b:12") == "done" and status.status_for("synthetic:7") == "failed"
    assert status.status_for("real_dso_b:0") is None
    status.update("real_dso_b:13", target_network="real_dso_b", target_grid_id=13, status="running")
    header = (tmp_path / "status.tsv").read_text().splitlines()[0].split("\t")
    assert header[:3] == ["job", "target_network", "target_grid_id"]  # orch B5: the file names the grid


def _args(tmp_path, **extra):
    base = dict(paired_dataset_id="ds", pylovo_version_id="1", scenario_config=Path("/s.yaml"), target="both",
                provider="dso_a", target_grid_id=None, run_dir=tmp_path, scenario_id="s", scenario_hash="h",
                tsam=False, operating_hours=8760, tsam_periods=None, tsam_hours_per_period=None,
                tsam_extreme_method=None, profile_seed=1, powerflow_grid_scope="full", max_timesteps=None,
                job_subset=None, resume=False)
    return argparse.Namespace(**{**base, **extra})


def test_identity_contains_the_grid_filter(tmp_path):
    identity = runner._run_identity(_args(tmp_path, target_grid_id=80))
    assert identity["target_grid_id"] == 80
    runner._assert_resume_compatible(_args(tmp_path))
    with pytest.raises(ValueError, match="incompatible earlier run"):
        runner._assert_resume_compatible(_args(tmp_path, target_grid_id=80))


def _capture(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "run_command", lambda **kw: calls.append((kw["stage"], kw["job"], kw["cmd"])))
    return calls


def test_real_adapter_commands(tmp_path, monkeypatch):
    calls = _capture(monkeypatch, real)
    args = argparse.Namespace(result_cases=("post-inflex-heuristic", "post-hems-heuristic"), skip_pre=False, plz=99999,
                              powerflow_grid_scope="full", tsam=False, pre_only=False, max_timesteps=None,
                              run_name_prefix="run")
    job = {"target_network": "real_dso_a", "target_grid_id": 80, "job": "real_dso_a:80", "grid_file": "/g/LV_080.xlsx",
           "provider": "dso_a"}
    real.run_powerflows(job=job, args=args, result_hdf=Path("/r.h5"), log_path=tmp_path / "x.log", status=None)
    assert [stage for stage, _, _ in calls] == ["step4_real_dso_a_pre", "step4_real_dso_a_post-inflex-heuristic",
                                                "step4_real_dso_a_post-hems-heuristic"]
    modes = [cmd[cmd.index("--post-demand-mode") + 1] for _, _, cmd in calls]
    labels = [cmd[cmd.index("--scenario-label") + 1] for _, _, cmd in calls]
    assert modes == ["pre-only", "inflex", "flexible"]
    # The pre stage is solved once, by the pre case; result cases solve the post stage only.
    assert ["--post-only" in cmd for _, _, cmd in calls] == [False, True, True]
    assert labels == ["Paired 2045 real_dso_a pre electricity-only", "Paired 2045 real_dso_a heuristic-assets INFLEX",
                      "Paired 2045 real_dso_a heuristic-assets HEMS"]
    assert all("--expect-temporal-method" in cmd and cmd[cmd.index("--grid-file") + 1] == "/g/LV_080.xlsx"
               for _, _, cmd in calls)
    assert {key for _, key, _ in calls} == {"real_dso_a:80"}


def test_synthetic_adapter_commands(tmp_path, monkeypatch):
    calls = _capture(monkeypatch, synthetic)
    monkeypatch.setattr(synthetic, "POWERFLOW_INPUT_DIR", tmp_path / "pf")
    result = tmp_path / "r.h5"
    result.write_bytes(b"x")
    args = argparse.Namespace(result_cases=("post-hems-optimized",), skip_pre=True, powerflow_grid_scope="backbone",
                              step4_cpus=2, max_timesteps=24, tsam=True, pre_only=False, run_name_prefix="run",
                              pylovo_version_id="1", cleanup_intermediates=True)
    job = {"target_network": "synthetic", "target_grid_id": 7, "job": "synthetic:7"}
    synthetic.run_powerflows(job=job, args=args, result_hdf=result, log_path=tmp_path / "x.log", status=None)
    ((stage, _, cmd),) = calls
    assert stage == "step4_synthetic_post-hems-optimized"
    joined = " ".join(cmd)
    for part in ("--grid-case-id 7", "--outputs summary", "--summary-nonconvergence nan",
                 "--summary-grid-scope backbone", "--max-timesteps 24", "--post-demand-mode flexible",
                 "--run-name run_synthetic_post-hems-optimized", "--post-only"):
        assert part in joined
    assert "--pre-only" not in cmd
    assert "--expect-temporal-method" not in cmd and not (tmp_path / "pf" / "r.h5").exists()


def test_synthetic_adapter_jobs(tmp_path):
    pd.DataFrame({
        "target_grid_id": [7, 7, 3, None],
        "synthetic_bridge_filename": ["a.h5", "a.h5", "b.h5", "c.h5"],
    }).to_csv(tmp_path / synthetic.ALLOCATION_PLAN_FILENAME, index=False)
    assert synthetic.load_jobs(tmp_path, None) == [
        {"target_network": "synthetic", "target_grid_id": 3, "bridge_filename": "b.h5"},
        {"target_network": "synthetic", "target_grid_id": 7, "bridge_filename": "a.h5"},
    ]
    assert [job["target_grid_id"] for job in synthetic.load_jobs(tmp_path, 7)] == [7]


def test_real_target_takes_network_and_provider_from_the_plan(tmp_path):
    pd.DataFrame({
        "target_network": ["real_dso_b"] * 3, "provider": ["dso_b"] * 3, "target_grid_id": [7, 7, 9],
        "real_grid_file": ["/g/area-7.json", "/g/area-7.json", "/g/area-9.json"], "postcode": [1, 1, 2],
    }).to_csv(tmp_path / real.ALLOCATION_PLAN_FILENAME, index=False)
    jobs = real.load_jobs(tmp_path, None)
    assert [(j["target_network"], j["provider"], j["target_grid_id"], j["grid_file"]) for j in jobs] == [
        ("real_dso_b", "dso_b", 7, "/g/area-7.json"), ("real_dso_b", "dso_b", 9, "/g/area-9.json")]
    assert real.input_name(jobs[0], "s") == "paired_real_dso_b_7_s.h5"
    assert adapter_for("real_dso_b") is real and adapter_for("real") is real and adapter_for("synthetic") is synthetic


def test_real_input_names_match_the_urbs_input():
    from gridexpand.allocation.scenario_calibration.pipeline.paired_urbs_input import _output_name

    for network in ("real", "real_dso_a"):
        job = {"target_network": network, "target_grid_id": 7}
        assert real.input_name(job, "lab") == _output_name(pd.DataFrame(), network, 7, "lab")
