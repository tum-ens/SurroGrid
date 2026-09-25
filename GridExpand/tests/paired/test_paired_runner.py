"""Paired runner: job keys (orch B4/B5), result-case rules and adapter commands (no database, no data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from gridexpand.paired import runner
from gridexpand.paired.aligned import real_grid_number, select_grid_subset
from gridexpand.paired.sources import swf, synthetic


def test_job_keys_and_legacy_rows(tmp_path):
    jobs = runner._number_jobs([
        {"target_network": "real_swf", "target_grid_id": 12},
        {"target_network": "synthetic", "target_grid_id": 7},
    ])
    assert [(job["job_index"], job["job"]) for job in jobs] == [(0, "real_swf:12"), (1, "synthetic:7")]
    # a status.tsv from before job keys: rows keyed by list position, grid only in the log name
    (tmp_path / "status.tsv").write_text(
        "candidate_index\tstatus\tlog_file\n"
        f"0\tdone\t{tmp_path}/logs/real_uzw_12.log\n"
        f"1\tfailed\t{tmp_path}/logs/synthetic_7.log\n"
    )
    status = runner.open_status(tmp_path, resume=True)
    assert status.status_for("real_uzw:12") == "done" and status.status_for("synthetic:7") == "failed"
    assert status.status_for("real_uzw:0") is None
    status.update("real_uzw:13", target_network="real_uzw", target_grid_id=13, status="running")
    header = (tmp_path / "status.tsv").read_text().splitlines()[0].split("\t")
    assert header[:3] == ["job", "target_network", "target_grid_id"]  # orch B5: the file names the grid


def _args(tmp_path, **extra):
    base = dict(paired_dataset_id="ds", pylovo_version_id="1", scenario_config=Path("/s.yaml"), target="both",
                provider="swf", target_grid_id=None, run_dir=tmp_path, scenario_id="s", scenario_hash="h",
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
    calls = _capture(monkeypatch, swf)
    args = argparse.Namespace(result_cases=("post-inflex-heuristic", "post-hems-heuristic"), skip_pre=False, plz=91301,
                              profile_seed=1, powerflow_grid_scope="full", tsam=False, pre_only=False,
                              grid_data_path=Path("/grids"), max_timesteps=None, run_name_prefix="run")
    job = {"target_network": "real_swf", "target_grid_id": 80, "job": "real_swf:80"}
    swf.run_powerflows(job=job, args=args, result_hdf=Path("/r.h5"), log_path=tmp_path / "x.log", status=None)
    assert [stage for stage, _, _ in calls] == ["step4_real_swf_pre", "step4_real_swf_post-inflex-heuristic",
                                                "step4_real_swf_post-hems-heuristic"]
    modes = [cmd[cmd.index("--post-demand-mode") + 1] for _, _, cmd in calls]
    labels = [cmd[cmd.index("--scenario-label") + 1] for _, _, cmd in calls]
    assert modes == ["pre-only", "inflex", "flexible"]
    assert labels == ["Paired 2045 real_swf pre electricity-only", "Paired 2045 real_swf heuristic-assets INFLEX",
                      "Paired 2045 real_swf heuristic-assets HEMS"]
    assert all("--expect-temporal-method" in cmd and "--grid-data-path" in cmd for _, _, cmd in calls)
    assert {key for _, key, _ in calls} == {"real_swf:80"}


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
                 "--run-name run_synthetic_post-hems-optimized"):
        assert part in joined
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


def test_grid_subset(tmp_path):
    population = tmp_path / "population.json"
    population.write_text(__import__("json").dumps({"metric_cohort": {"uzw": {"successful_components": [
        {"real": ["uzw:area-1"], "synthetic": [101]},
        {"real": ["uzw:area-2", "uzw:area-3"], "synthetic": [102]},
    ]}}}))
    pd.DataFrame({"grid_result_id": [101, 102], "grid_case_id": [11, 12]}).to_csv(
        tmp_path / "paired_registered_synthetic_grids.csv", index=False)
    subset = select_grid_subset(population=population, provider="uzw", paired_dir=tmp_path,
                                grid_subset={"method": "islands", "seed": 1, "real_grids_per_provider": None})
    assert subset["real_uzw"] == [1] and subset["synthetic"] == [11]
    assert real_grid_number("swf", "LV_080__x") == 80
