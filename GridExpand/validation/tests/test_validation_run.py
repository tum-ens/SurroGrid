"""python -m validation run: aligned run YAMLs, plans and commands (no database)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gridexpand.paths import RUN_CONFIG_DIR, SCENARIO_CONFIG_DIR
from gridexpand.scenario import run as core_run
from gridexpand.scenario.run_config import with_cases
from validation import CONFIG_DIR, commands
from validation import run as run_module
from validation.run import AlignedRun, aligned_run_from_dict, load_aligned_run

ISLANDS = CONFIG_DIR / "joint_2045_v1_islands.yaml"
SANDBOX = RUN_CONFIG_DIR / "sandbox_example.yaml"
SC = Path("/s/scenario.yaml")
PY = "py"


def context(path: Path, *extra: str, run_dir: Path | None = None):
    argv = [str(path), *extra]
    if run_dir is not None:
        argv += ["--run-dir", str(run_dir)]
    return run_module.load_context(run_module.build_parser().parse_args(argv))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_repository_run_yamls(path):
    run, run_hash = load_aligned_run(path)
    assert run.pipeline == "paired_aligned" and len(run_hash) == 64
    assert run.scenario_path.is_file()


def test_aligned_run(tmp_path):
    run, _ = load_aligned_run(ISLANDS)
    assert isinstance(run, AlignedRun)
    assert [p.provider for p in run.providers] == ["dso_a", "dso_b"] and run.provider("dso_b").workers == 2
    assert run.alignment_dir == Path("~/data/alignment").expanduser().resolve()
    assert run.grid_subset == {"method": "islands", "seed": 20260924, "real_grids_per_provider": None}
    assert run.parallel_providers is True and run.target_for(run.provider("dso_b")) == "both"
    tag = run.providers[0].scenario_tag
    assert len(tag) == 12 and run.providers[0].heat_profile_set_id == f"joint_2045_v1_dso_a_teaser_heat_{tag}"
    raw = yaml.safe_load(ISLANDS.read_text(encoding="utf-8"))
    raw["run"]["scenario"] = str(SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml")
    raw["resources"]["alignment_dir"] = "alignment"  # relative: next to the YAML, not the working directory
    assert aligned_run_from_dict(raw, base_dir=tmp_path).alignment_dir == tmp_path / "alignment"
    assert run.weather_year is None  # PVGIS TMY per provider
    raw["resources"]["weather_year"] = 2016
    assert aligned_run_from_dict(raw, base_dir=tmp_path).weather_year == 2016
    raw["resources"]["weather_year"] = 2016.5
    with pytest.raises(ValueError, match="weather_year"):
        aligned_run_from_dict(raw, base_dir=tmp_path)
    raw["resources"]["weather_year"] = None
    raw["execution"]["parallel_providers"] = "false"
    with pytest.raises(ValueError, match="true or false"):
        aligned_run_from_dict(raw, base_dir=tmp_path)


def test_post_cases_and_optimizer(tmp_path):
    run, _ = load_aligned_run(ISLANDS)
    with pytest.raises(ValueError, match="not allowed"):
        with_cases(run, ("pre",))  # pre is added to the first group; asking for it again is refused
    raw = yaml.safe_load(ISLANDS.read_text(encoding="utf-8"))
    raw["run"]["scenario"] = str(SCENARIO_CONFIG_DIR / "joint_2045_full_year.yaml")
    assert aligned_run_from_dict(raw, base_dir=tmp_path).optimizer is None
    raw["execution"]["optimizer"] = "pypsa"
    assert aligned_run_from_dict(raw, base_dir=tmp_path).optimizer == "pypsa"


def test_aligned_plan(tmp_path):
    ctx = context(ISLANDS, "--provider", "dso_b", run_dir=tmp_path)
    (provider,) = run_module.aligned_providers(ctx)
    jobs = run_module.aligned_jobs(ctx, provider, tmp_path / "dso_b" / "grid_subset.json")
    assert [key for key, _, _ in jobs] == ["dso_b/heuristic-assets"]
    argv = jobs[0][1]
    assert argv[argv.index("--workers") + 1] == "2" and argv[argv.index("--provider") + 1] == "dso_b"
    assert argv[argv.index("--job-subset") + 1] == str(tmp_path / "dso_b" / "grid_subset.json")
    lines = run_module.plan_lines(ctx)
    assert any("grid_subset.json (computed after preparation)" in line for line in lines)
    assert any(line.startswith("[prepare:dso_b:pv_library]") and "--reference-year 2009" in line for line in lines)
    assert any(line.startswith("[prepare:dso_b:allocation]") and "-m validation.aligned_allocation" in line
               for line in lines)
    assert not (tmp_path / "dso_b").exists()  # dry-run planning writes no subset (review-orch B3)
    pre_only = context(ISLANDS, "--pre-only", run_dir=tmp_path)
    keys = [key for p in run_module.aligned_providers(pre_only) for key, _, _ in run_module.aligned_jobs(pre_only, p, None)]
    assert keys == ["dso_a/pre-only", "dso_b/pre-only"]


def test_preparation_and_expansion_commands():
    aligned = commands.aligned_preparation_commands(
        provider="dso_b", alignment_dir=Path("/a"), population=Path("/p.json"), dso_b_grids_dir=Path("/u"),
        pylovo_version_id="1", scenario_config=SC, profile_seed=1, paired_dir=Path("/d"), weather_hdf=Path("/w.h5"),
        heat_sources=Path("/d/heat_sources/tag"), heat_library=Path("/h.h5"), heat_profile_set_id="set",
        heat_workers=6, reference_year=2009, python=PY,
    )
    assert [stage for stage, _ in aligned] == ["allocation", "weather", "heat_regeneration", "heat_readiness_sources",
                                              "heat_library", "heat_readiness_library", "pv_library"]
    assert aligned[0][1][:3] == [PY, "-m", "validation.aligned_allocation"]
    assert aligned[0][1][-2:] == ["--dso-b-grids-dir", "/u"]
    heat = aligned[2][1]
    assert heat[heat.index("--scenario-config") + 1] == str(SC) and heat[heat.index("--workers") + 1] == "6"
    # One weather series per provider: the heat runs use the weather file the PV library and sizing use.
    assert heat[heat.index("--weather-hdf") + 1] == "/w.h5"
    assert aligned[6][1][aligned[6][1].index("--weather-source-hdf") + 1] == "/w.h5"
    assert "--weather-year" not in aligned[1][1]  # PVGIS TMY
    year = commands.aligned_preparation_commands(
        provider="dso_a", alignment_dir=Path("/a"), population=Path("/p.json"), dso_b_grids_dir=None,
        pylovo_version_id="1", scenario_config=SC, profile_seed=1, paired_dir=Path("/d"), weather_hdf=Path("/w.h5"),
        heat_sources=Path("/d/heat_sources/tag"), heat_library=Path("/h.h5"), heat_profile_set_id="set",
        heat_workers=6, reference_year=2009, weather_year=2016, python=PY,
    )
    assert year[1][0] == "weather" and year[1][1][-2:] == ["--weather-year", "2016"]
    expansion = commands.aligned_expansion_command("run", providers=["dso_a", "dso_b"],
                                                   cases=("pre", "post-hems-heuristic"), pylovo_version_id="1",
                                                   python=PY)
    assert expansion[:3] == [PY, "-m", "validation.aligned_expansion"]
    assert expansion[3:] == ["--run-id", "run", "--providers", "dso_a", "dso_b", "--cases", "pre",
                             "post-hems-heuristic", "--pylovo-version-id", "1"]


@pytest.mark.parametrize("argv", [[str(SANDBOX)], [str(SANDBOX), "--provider", "dso_a"], []])
def test_invalid_configuration_exit_code(argv, capsys):
    assert run_module.main(argv) == core_run.EXIT_INVALID
    assert "invalid configuration" in capsys.readouterr().err


def test_gridexpand_run_points_aligned_yamls_here(capsys):
    assert core_run.main([str(ISLANDS)]) == core_run.EXIT_INVALID
    assert "python -m validation run" in capsys.readouterr().err


def test_dry_run_exits_zero(capsys):
    assert run_module.main([str(ISLANDS), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "pipeline paired_aligned" in out and "[execute:dso_a/heuristic-assets]" in out
