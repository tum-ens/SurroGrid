"""``gridexpand run`` with ``grids: files``: configuration and plan (no database)."""

from __future__ import annotations

import pytest

from gridexpand.scenario import run as run_module
from gridexpand.scenario.run_config import RealRun, load_run_config


def write(tmp_path, text: str):
    (tmp_path / "grids").mkdir(exist_ok=True)
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_run_defaults(tmp_path):
    run, _ = load_run_config(write(tmp_path, "run: mine\nscenario: default_2045\ngrids: {files: grids}\n"))
    assert isinstance(run, RealRun) and run.pipeline == "real" and run.grid_dir == tmp_path / "grids"
    assert run.model_cases == ("post-inflex-heuristic", "post-hems-heuristic", "post-hems-optimized")
    assert (run.crs, run.points, run.matching, run.pylovo_version_id) == ("EPSG:25832", None, {}, "none")


@pytest.mark.parametrize("text, message", [
    ("run: x\nscenario: default_2045\ngrids: {files: nowhere}\n", "no directory"),
    ("run: x\nscenario: default_2045\ngrids: {files: {dir: grids, points: p.csv}}\n", "points not found"),
    ("run: x\nscenario: default_2045\ngrids: {files: grids}\nmodel_cases: [pre]\n", "not allowed"),
    ("run: x\nscenario: default_2045\ngrids: {files: grids}\nmatching: {radius: 3}\n", "Unknown matching keys"),
    ("run: x\nscenario: default_2045\ngrids: {pylovo: {version: '1', ags: 1}}\nmatching: {}\n", None),
    ("run: x\nscenario: default_2045\ngrids: {pylovo: {version: '1', ags: 1}}\nmatching: {facade_offset_m: 1}\n",
     "own grid files"),
    ("run: x\nscenario: default_2045\ngrids: {files: grids}\nexecution: {step2_cpus: 2}\n", "Unknown"),
])
def test_real_run_errors(tmp_path, text, message):
    path = write(tmp_path, text)
    if message is None:
        load_run_config(path)
        return
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_run_config(path)


def test_real_plan(tmp_path, capsys):
    path = write(tmp_path, "run: mine\nscenario: default_2045\ngrids: {files: grids}\n"
                           "matching: {nearest_residential_m: 15}\nmodel_cases: [post-hems-heuristic]\n")
    ctx = run_module.load_context(run_module.build_parser().parse_args([str(path), "--run-dir", str(tmp_path / "r")]))
    lines = run_module.plan_lines(ctx)
    stages = [line.split("]")[0].strip("[") for line in lines[2:]]
    assert stages == ["prepare:match", "prepare:dataset", "prepare:weather", "prepare:internal_heat_readiness",
                      "prepare:pv_library", "prepare:check", "execute:heuristic-assets", "postprocess",
                      "postprocess", "postprocess"]
    match = lines[2]
    assert f"--config {path}" in match and "gridexpand.grids.match" in match
    runner = next(line for line in lines if line.startswith("[execute:"))
    assert "--target real" in runner and "--run-name-prefix mine" in runner and "--provider real" in runner
    expansion = [line for line in lines if "grid_expansion" in line]
    assert "--run-name mine_real_pre" in expansion[0] and "--data-source real" in expansion[0]
    assert not (tmp_path / "r").exists()
    assert run_module.main([str(path), "--dry-run"]) == 0
    assert "pipeline real" in capsys.readouterr().out
