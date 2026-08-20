#!/usr/bin/env python3
"""Run one validated scenario case from a compact run YAML."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from scenario_pipeline.config_loader import load_run_config, load_scenario_config
from scenario_pipeline.model_cases import MODEL_CASES, get_model_case


def _write_manifest(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")


def build_command(run, scenario, model_case: str) -> tuple[list[str], Path]:
    case = get_model_case(model_case)
    if model_case not in scenario.model_cases:
        raise ValueError(f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}.")
    if run.pipeline == "scenario":
        workdir = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
        profiles = "status_quo" if case.name == "pre" else "all"
        command = [
            "uv", "run", "--project", "..", "python", "main.py", run.inputfile_id,
            "--storage", run.storage,
            "--n_cpu", str(run.n_cpu),
            "--profiles", profiles,
            "--demand-scope", run.demand_scope,
            "--mobility-source", run.mobility_source,
            "--timeframe-mode", run.timeframe_mode,
            "--model-case", model_case,
            "--scenario-config", str(run.scenario_path),
            "--case-qualified-output",
        ]
        if run.output_directory is not None:
            command.extend(["--output-directory", str(run.output_directory)])
        return command, workdir
    if run.paired_directory is None or run.weather_source_hdf is None:
        raise ValueError("Paired validation requires paired_directory and weather_source_hdf.")
    if run.run_directory is None:
        raise ValueError("Paired validation requires resources.run_directory.")
    workdir = GRIDEXPAND_DIR.parent
    command = [
        "uv", "run", "--project", "GridExpand/2.demand_allocation",
        "python", "GridExpand/paired_validation/runner.py",
        "--repo-root", str(workdir),
        "--plz", run.inputfile_id,
        "--paired-dir", str(run.paired_directory),
        "--weather-source-hdf", str(run.weather_source_hdf),
        "--scenario-config", str(run.scenario_path),
        "--model-case", model_case,
        "--target", str(run.target_network or "both"),
        "--workers", str(run.workers),
        "--step3-cpus", str(run.step3_cpus),
        "--step3-cluster-concurrency", str(run.step3_cluster_concurrency),
        "--step4-cpus", str(run.step4_cpus),
        "--seed", str(run.seed),
        "--scenario-label", run.run_id,
        "--run-name-prefix", run.run_id,
        "--run-dir", str(run.run_directory / model_case),
    ]
    if run.target_grid_id is not None:
        command.extend(["--target-grid-id", str(run.target_grid_id)])
    if run.grid_data_path is not None:
        command.extend(["--grid-data-path", str(run.grid_data_path)])
    if run.heat_profile_library is not None:
        command.extend(["--heat-profile-library", str(run.heat_profile_library)])
    if run.cleanup_intermediates:
        command.append("--cleanup-intermediates")
    if run.resume:
        command.append("--resume")
    return command, workdir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--model-case",
        choices=tuple(MODEL_CASES),
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run, run_hash = load_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    model_case = args.model_case or run.model_case or "post-hems-heuristic"
    command, workdir = build_command(run, scenario, model_case)
    manifest_dir = (
        run.run_directory
        or run.output_directory
        or (GRIDEXPAND_DIR / "run_logs" / "scenario_manifests")
    )
    _write_manifest(
        manifest_dir / f"{run.run_id}_{model_case}.json",
        {
            "run_id": run.run_id,
            "run_hash": run_hash,
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario_hash,
            "model_case": model_case,
            "command": command,
            "working_directory": str(workdir),
            "dry_run": bool(args.dry_run),
        },
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=workdir, check=True)


if __name__ == "__main__":
    main()
