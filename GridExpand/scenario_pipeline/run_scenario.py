#!/usr/bin/env python3
"""Run one validated scenario or paired-validation configuration."""

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

HEURISTIC_CASES = ("post-inflex-heuristic", "post-hems-heuristic")


def _write_manifest(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")


def _scenario_command(run, scenario, model_case: str) -> tuple[list[str], Path]:
    case = get_model_case(model_case)
    if model_case not in scenario.model_cases:
        raise ValueError(f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}.")
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
    if run.storage == "db":
        command.extend(["--pylovo-version-id", run.pylovo_version_id])
    if run.output_directory is not None:
        command.extend(["--output-directory", str(run.output_directory)])
    return command, workdir


def _paired_command(
    run,
    *,
    materialization_case: str,
    result_cases: tuple[str, ...],
    group_name: str,
    skip_pre: bool,
) -> tuple[list[str], Path]:
    workdir = GRIDEXPAND_DIR.parent
    command = [
        "uv", "run", "--project", "GridExpand/2.demand_allocation",
        "python", "GridExpand/paired_validation/runner.py",
        "--repo-root", str(workdir),
        "--paired-dataset-id", str(run.paired_dataset_id),
        "--pylovo-version-id", run.pylovo_version_id,
        "--scenario-config", str(run.scenario_path),
        "--model-case", materialization_case,
        "--result-cases", *result_cases,
        "--target", str(run.target_network or "both"),
        "--workers", str(run.workers),
        "--step3-cpus", str(run.step3_cpus),
        "--step3-cluster-concurrency", str(run.step3_cluster_concurrency),
        "--step4-cpus", str(run.step4_cpus),
        "--seed", str(run.seed),
        "--scenario-label", run.run_id,
        "--run-name-prefix", run.run_id,
        "--run-dir", str(GRIDEXPAND_DIR / "run_logs" / run.run_id / group_name),
    ]
    if run.target_grid_id is not None:
        command.extend(["--target-grid-id", str(run.target_grid_id)])
    if run.cleanup_intermediates:
        command.append("--cleanup-intermediates")
    if run.resume:
        command.append("--resume")
    if skip_pre:
        command.append("--skip-pre")
    return command, workdir


def build_commands(run, scenario, model_cases: tuple[str, ...]) -> list[tuple[list[str], Path]]:
    for model_case in model_cases:
        if model_case not in scenario.model_cases:
            raise ValueError(
                f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}."
            )
    if run.pipeline == "scenario":
        if len(model_cases) != 1:
            raise ValueError("An ordinary scenario invocation accepts exactly one model case.")
        return [_scenario_command(run, scenario, model_cases[0])]

    commands: list[tuple[list[str], Path]] = []
    requested_heuristic = tuple(case for case in HEURISTIC_CASES if case in model_cases)
    if requested_heuristic:
        commands.append(
            _paired_command(
                run,
                materialization_case="post-hems-heuristic",
                result_cases=requested_heuristic,
                group_name="heuristic-assets",
                skip_pre=False,
            )
        )
    if "post-hems-optimized" in model_cases:
        commands.append(
            _paired_command(
                run,
                materialization_case="post-hems-optimized",
                result_cases=("post-hems-optimized",),
                group_name="post-hems-optimized",
                skip_pre=bool(commands),
            )
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--model-case",
        choices=tuple(MODEL_CASES),
        default=None,
        help="Optional single-case override; otherwise use the run YAML selection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run, run_hash = load_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    if args.model_case is not None:
        model_cases = (args.model_case,)
    elif run.pipeline == "paired_validation":
        model_cases = run.model_cases
    else:
        model_cases = ("post-hems-heuristic",)
    commands = build_commands(run, scenario, model_cases)
    manifest_dir = GRIDEXPAND_DIR / "run_logs" / run.run_id
    _write_manifest(
        manifest_dir / "run_manifest.json",
        {
            "run_id": run.run_id,
            "run_hash": run_hash,
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario_hash,
            "pylovo_version_id": run.pylovo_version_id,
            "model_cases": list(model_cases),
            "commands": [
                {"command": command, "working_directory": str(workdir)}
                for command, workdir in commands
            ],
            "dry_run": bool(args.dry_run),
        },
    )
    for command, workdir in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=workdir, check=True)


if __name__ == "__main__":
    main()
