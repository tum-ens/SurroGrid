#!/usr/bin/env python3
"""Run one validated scenario case from a compact run YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = GRIDEXPAND_DIR.parent
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from scenario_pipeline.configuration.loader import load_run_config, load_scenario_config
from scenario_pipeline.core.manifest import write_manifest
from scenario_pipeline.core.model_cases import get_model_case


def build_command(run, scenario, model_case: str) -> tuple[list[str], Path]:
    case = get_model_case(model_case)
    if model_case not in scenario.model_cases:
        raise ValueError(f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}.")
    if run.pipeline == "scenario":
        workdir = GRIDEXPAND_DIR / "2.demand_allocation"
        profiles = "status_quo" if case.name == "pre" else "all"
        command = [
            "uv", "run", "python", "gridalloc/main.py", run.inputfile_id,
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
    if run.target_network is None or run.target_grid_id is None:
        raise ValueError("Paired validation requires target_network and target_grid_id.")
    workdir = GRIDEXPAND_DIR / "2.demand_allocation"
    command = [
        "uv", "run", "python", "-m",
        "gridalloc.src.scenario_calibration.pipeline.paired_urbs_input",
        "--paired-dir", str(run.paired_directory),
        "--target-network", str(run.target_network),
        "--target-grid-id", str(run.target_grid_id),
        "--weather-source-hdf", str(run.weather_source_hdf),
        "--model-case", model_case,
        "--scenario-config", str(run.scenario_path),
        "--scenario-label", f"{scenario.scenario_id}_{model_case}",
    ]
    if run.output_directory is not None:
        command.extend(["--output-dir", str(run.output_directory)])
    return command, workdir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--model-case",
        choices=[
            "pre", "post-inflex-heuristic", "post-hems-optimized",
            "post-hems-heuristic",
        ],
        default="post-hems-heuristic",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run, run_hash = load_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    command, workdir = build_command(run, scenario, args.model_case)
    manifest_dir = run.output_directory or (GRIDEXPAND_DIR / "scenario_pipeline" / "manifests")
    write_manifest(
        manifest_dir / f"{run.run_id}_{args.model_case}.json",
        {
            "run_id": run.run_id,
            "run_hash": run_hash,
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario_hash,
            "model_case": args.model_case,
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
