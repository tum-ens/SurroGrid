"""Pure step-command builders (scenario/commands.py) against the argv the runners used to build by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridexpand.scenario import commands

SC = Path("/s/scenario.yaml")
PY = "py"


def test_allocation_command_matches_the_old_runner_argv():
    argv = commands.allocation_command(
        "9184137", pylovo_version_id="1", candidate_index=3, min_buildings=60, profiles="status_quo",
        demand_scope="all", timeseries_storage="temp", timeframe_mode="max_base_electricity_demand_week",
        model_case="pre", profile_seed=481527, scenario_config=SC,
        electrification_assignment=Path("/r/electrification_assignment.csv"), n_cpu=2,
        case_qualified_output=True, python=PY,
    )
    assert argv == [
        PY, "-m", "gridexpand.allocation.main", "9184137", "--storage", "db", "--pylovo-version-id", "1",
        "--candidate-index", "3", "--min-buildings", "60", "--profiles", "status_quo", "--demand-scope", "all",
        "--mobility-source", "pool", "--timeseries-storage", "temp",
        "--timeframe-mode", "max_base_electricity_demand_week", "--model-case", "pre", "--profile-seed", "481527",
        "--scenario-config", str(SC), "--electrification-assignment", "/r/electrification_assignment.csv",
        "--n_cpu", "2", "--case-qualified-output",
    ]


def test_optimization_command():
    assert commands.optimization_command("x.h5", n_cpu=8, scenario_config=SC, python=PY) == [
        PY, "-m", "gridexpand.optimization.run_urbs_cluster", "x.h5", "--n_cpu", "8", "--scenario-config", str(SC)]
    argv = commands.optimization_command("x.h5", n_cpu=1, scenario_config=SC, cluster_concurrency=2,
                                         solver="appsi_highs", reduce_only=True, python=PY)
    assert argv[-5:] == ["--cluster-concurrency", "2", "--solver", "appsi_highs", "--reduce-only"]


def test_powerflow_command_one_pass_and_paired_summary():
    argv = commands.powerflow_command(
        "in.h5", n_cpu=2, run_name="r_raw", summary_run_name="r_summary", outputs=("raw", "summary"),
        pre_only=True, pylovo_version_id="1", python=PY,
    )
    assert argv == [PY, "-m", "gridexpand.powerflow.run_pwrflw", "in.h5", "--storage", "db", "--pre-only",
                    "--outputs", "raw,summary", "--run-name", "r_raw", "--summary-run-name", "r_summary",
                    "--n_cpu", "2", "--pylovo-version-id", "1"]
    paired = commands.powerflow_command(
        "in.h5", grid_case_id=7, outputs=("summary",), summary_nonconvergence="nan", summary_grid_scope="full",
        n_cpu=2, expect_temporal_method="full_year_no_tsam", post_demand_mode="inflex", run_name="p_x", python=PY,
    )
    joined = " ".join(paired)
    for part in ("--grid-case-id 7", "--outputs summary", "--summary-nonconvergence nan", "--summary-grid-scope full",
                 "--post-demand-mode inflex", "--run-name p_x", "--expect-temporal-method full_year_no_tsam"):
        assert part in joined
    assert "--pre-only" not in paired and "--hh-only" not in paired
    assert "--hh-only" in commands.powerflow_command("in.h5", n_cpu=1, hh_only=True)


def test_expansion_commands():
    argv = commands.expansion_command("run_s", stage="pre", ags="9184137", analysis_key="k_pre", note="n.", python=PY)
    assert argv == [PY, "-m", "gridexpand.analysis.expansion.grid_expansion", "--run-name", "run_s", "--stage", "pre",
                    "--ags", "9184137", "--analysis-key", "k_pre", "--note", "n.", "--replace"]
    real = commands.expansion_command("run_r", stage="post", data_source="real_swf", plz=91301,
                                      exclude_real_lv_ids=(113,), analysis_key="k", python=PY)
    assert real[real.index("--data-source") + 1] == "real_swf" and real[real.index("--plz") + 1] == "91301"
    assert real[real.index("--exclude-real-lv-id") + 1] == "113" and "--ags" not in real
    aligned = commands.aligned_expansion_command("run", providers=["swf", "uzw"], cases=("pre", "post-hems-heuristic"),
                                                 pylovo_version_id="1", python=PY)
    assert aligned[3:] == ["--run-id", "run", "--providers", "swf", "uzw", "--cases", "pre", "post-hems-heuristic",
                           "--pylovo-version-id", "1"]


def test_real_powerflow_command():
    argv = commands.real_powerflow_command(
        plz=91301, lv_id=80, provider="swf", profile_seed=1, urbs_result_hdf=Path("/r.h5"), summary_grid_scope="full",
        post_demand_mode="pre-only", run_name="p_real_swf_pre", scenario_label="Paired 2045 real_swf pre",
        grid_file="LV_080.json", grid_data_path=Path("/grids"), python=PY,
    )
    assert "--grid-file" in argv and "--grid-data-path" not in argv  # an explicit grid file wins
    assert argv[argv.index("--scenario-key") + 1] == "p_real_swf_pre"


def test_paired_runner_command():
    common = dict(
        paired_dataset_id="ds", pylovo_version_id="1", scenario_config=SC, target="both", workers=2, step3_cpus=40,
        step3_cluster_concurrency=1, step4_cpus=2, powerflow_grid_scope="full", profile_seed=481527,
        scenario_label="run", run_name_prefix="run", run_dir=Path("/w/runs/run/heuristic-assets"), python=PY,
    )
    argv = commands.paired_runner_command(
        **common, materialization_case="post-hems-heuristic",
        result_cases=("post-inflex-heuristic", "post-hems-heuristic"), skip_pre=False, resume=True,
    )
    assert argv[:3] == [PY, "-m", "gridexpand.paired.runner"]
    assert argv[argv.index("--result-cases") + 1:argv.index("--result-cases") + 3] == [
        "post-inflex-heuristic", "post-hems-heuristic"]
    assert "--resume" in argv and "--skip-pre" not in argv and argv[-2:] == ["--run-dir", "/w/runs/run/heuristic-assets"]
    pre_only = commands.paired_runner_command(**common, pre_only=True, provider="uzw", max_timesteps=24)
    assert "--pre-only" in pre_only and "--model-case" not in pre_only and "--provider" in pre_only
    with pytest.raises(ValueError):
        commands.paired_runner_command(**common)


def test_preparation_commands():
    paired = commands.paired_preparation_commands(
        ags=9474126, plz=91301, milestone_year=2045, pylovo_version_id="11", min_buildings=5, scenario_config=SC,
        profile_seed=481527, paired_dir=Path("/d"), heat_library=Path("/h.h5"), weather_hdf=Path("/w.h5"),
        reference_year=2009, python=PY,
    )
    assert [stage for stage, _ in paired] == ["prepare_allocation", "prepare_heat_profiles", "prepare_pv_profiles"]
    assert paired[2][1][-2:] == ["--reference-year", "2009"]
    aligned = commands.aligned_preparation_commands(
        provider="uzw", alignment_dir=Path("/a"), population=Path("/p.json"), uzw_grids_dir=Path("/u"),
        pylovo_version_id="1", scenario_config=SC, profile_seed=1, paired_dir=Path("/d"), weather_hdf=Path("/w.h5"),
        heat_sources=Path("/d/heat_sources/tag"), heat_library=Path("/h.h5"), heat_profile_set_id="set",
        heat_workers=6, reference_year=2009, python=PY,
    )
    assert [stage for stage, _ in aligned] == ["allocation", "weather", "heat_regeneration", "heat_readiness_sources",
                                              "heat_library", "heat_readiness_library", "pv_library"]
    assert aligned[0][1][-2:] == ["--uzw-grids-dir", "/u"]
    heat = aligned[2][1]
    assert heat[heat.index("--scenario-config") + 1] == str(SC) and heat[heat.index("--workers") + 1] == "6"
    assert commands.electrification_preparation_command(
        "9184137", min_buildings=60, pylovo_version_id="1", demand_scope="all", mobility_source="pool",
        profile_seed=1, scenario_config=SC, output=Path("/o.csv"), plz=85653, kcid=1, bcid=4, python=PY,
    )[-6:] == ["--plz", "85653", "--kcid", "1", "--bcid", "4"]
