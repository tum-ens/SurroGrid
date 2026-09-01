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

from scenario_pipeline.config_loader import load_run_config, load_scenario_config  # noqa: E402
from scenario_pipeline.model_cases import MODEL_CASES, get_model_case  # noqa: E402

HEURISTIC_CASES = ("post-inflex-heuristic", "post-hems-heuristic")
GRIDALLOC_DIR = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
PAIRED_DATASET_ROOT = GRIDALLOC_DIR / "outputs" / "scenario_calibration"
HEAT_LIBRARY_ROOT = PAIRED_DATASET_ROOT / "profile_libraries"
WEATHER_RESULT_ROOT = GRIDALLOC_DIR / "results"
POSTPROCESSING_DIR = GRIDEXPAND_DIR / "5.postprocessing"


def _write_manifest(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")


def _paired_artifact_paths(run) -> tuple[Path, Path, Path]:
    paired_dir = PAIRED_DATASET_ROOT / str(run.paired_dataset_id)
    heat_library = HEAT_LIBRARY_ROOT / f"{run.heat_profile_set_id}.h5"
    weather_hdf = WEATHER_RESULT_ROOT / str(run.weather_source_hdf)
    return paired_dir, heat_library, weather_hdf


def _paired_preparation_commands(run, scenario) -> list[tuple[str, list[str], Path]]:
    paired_dir, heat_library, weather_hdf = _paired_artifact_paths(run)
    pv_location_mode = (
        "swf" if scenario.pv.location_mode == "predefined" else "all_buildings"
    )
    common = ["uv", "run", "--project", "..", "python", "-m"]
    return [
        (
            "prepare_allocation",
            common
            + [
                "src.scenario_calibration.allocation.paired_allocation",
                "--ags",
                str(run.ags),
                "--plz",
                str(run.plz),
                "--final-year",
                str(scenario.milestone_year),
                "--pylovo-version-id",
                run.pylovo_version_id,
                "--min-buildings",
                str(run.min_buildings),
                "--pv-location-mode",
                pv_location_mode,
                "--output-dir",
                str(paired_dir),
            ],
            GRIDALLOC_DIR,
        ),
        (
            "prepare_heat_profiles",
            common
            + [
                "src.scenario_calibration.profiles.paired_profile_readiness",
                "--paired-dir",
                str(paired_dir),
                "--heat-profile-library",
                str(heat_library),
            ],
            GRIDALLOC_DIR,
        ),
        (
            "prepare_pv_profiles",
            common
            + [
                "src.scenario_calibration.profiles.pv_profile_library",
                "--roof-catalog",
                str(paired_dir / "paired_roof_sections.csv"),
                "--allocation-plan",
                str(paired_dir / "paired_real_bus_allocation_plan.csv"),
                "--weather-source-hdf",
                str(weather_hdf),
                "--output",
                str(paired_dir / "paired_pv_profile_library.h5"),
                "--reference-year",
                str(scenario.mobility.reference_year),
            ],
            GRIDALLOC_DIR,
        ),
    ]


def _validate_prepared_paired(run) -> dict[str, object]:
    from paired_validation.datasets import resolve_paired_dataset

    dataset = resolve_paired_dataset(
        str(run.paired_dataset_id),
        expected_pylovo_version_id=run.pylovo_version_id,
    )
    metadata = json.loads(
        (dataset.paired_dir / "paired_scenario_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if int(metadata.get("ags", -1)) != int(run.ags):
        raise ValueError("Prepared paired dataset AGS does not match the run YAML.")
    if int(metadata["plz"]) != int(run.plz):
        raise ValueError("Prepared paired dataset PLZ does not match the run YAML.")

    import pandas as pd

    real = pd.read_csv(dataset.paired_dir / "paired_real_bus_allocation_plan.csv")
    synthetic = pd.read_csv(
        dataset.paired_dir / "paired_synthetic_bus_allocation_plan.csv"
    )
    heat = pd.read_csv(dataset.paired_dir / "paired_heat_profile_catalog.csv")
    not_ready = int((~heat["publication_ready"].astype(bool)).sum())
    if not_ready:
        raise ValueError(f"Paired heat readiness failed for {not_ready} buildings.")
    real_buildings = set(real["building_objectid"].astype(str))
    synthetic_buildings = set(synthetic["building_objectid"].astype(str))
    if real_buildings != synthetic_buildings:
        raise ValueError(
            "Prepared real and synthetic plans contain different buildings."
        )
    return {
        "paired_dataset_id": dataset.dataset_id,
        "registered_pylovo_grid_cases": int(metadata["registered_pylovo_grid_cases"]),
        "registered_pylovo_buildings": int(metadata["registered_pylovo_buildings"]),
        "paired_buildings": len(real_buildings),
        "real_grids": int(real["target_grid_id"].nunique()),
        "synthetic_grids": int(synthetic["target_grid_id"].nunique()),
        "heat_profiles": int(heat["building_objectid"].nunique()),
    }


def _expansion_commands(
    run, model_cases: tuple[str, ...]
) -> list[tuple[list[str], Path]]:
    if not run.materialize_expansion or run.target_grid_id is not None:
        return []
    network_pairs = (
        [("synthetic", "synthetic")]
        if run.target_network == "synthetic"
        else [("real_swf", "real_swf")]
        if run.target_network == "real_swf"
        else [("synthetic", "synthetic"), ("real_swf", "real_swf")]
    )
    cases = ("pre", *model_cases)
    commands: list[tuple[list[str], Path]] = []
    for target, data_source in network_pairs:
        source_suffix = "" if target == "synthetic" else "_real"
        for model_case in cases:
            if model_case == "pre":
                analysis_suffix = "pre"
                stage = "pre"
            elif model_case == "post-inflex-heuristic":
                analysis_suffix = "post_no_flex"
                stage = "post"
            elif model_case == "post-hems-heuristic":
                analysis_suffix = "post"
                stage = "post"
            else:
                analysis_suffix = model_case.replace("-", "_")
                stage = "post"
            command = [
                "uv",
                "run",
                "python",
                "-m",
                "expansion.grid_expansion",
                "--run-name",
                f"{run.run_id}_{target}_{model_case}",
                "--data-source",
                data_source,
                "--stage",
                stage,
                "--analysis-key",
                f"{run.run_id}{source_suffix}_{analysis_suffix}",
                "--replace",
            ]
            if target == "synthetic":
                command.extend(["--ags", str(run.ags)])
            else:
                command.extend(["--plz", str(run.plz)])
                for lv_id in run.excluded_real_lv_ids:
                    command.extend(["--exclude-real-lv-id", str(lv_id)])
            commands.append((command, POSTPROCESSING_DIR))
    return commands


def _scenario_command(run, scenario, model_case: str) -> tuple[list[str], Path]:
    case = get_model_case(model_case)
    if model_case not in scenario.model_cases:
        raise ValueError(
            f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}."
        )
    workdir = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
    profiles = "status_quo" if case.name == "pre" else "all"
    command = [
        "uv",
        "run",
        "--project",
        "..",
        "python",
        "main.py",
        run.inputfile_id,
        "--storage",
        run.storage,
        "--n_cpu",
        str(run.n_cpu),
        "--profiles",
        profiles,
        "--demand-scope",
        run.demand_scope,
        "--mobility-source",
        run.mobility_source,
        "--timeframe-mode",
        run.timeframe_mode,
        "--model-case",
        model_case,
        "--scenario-config",
        str(run.scenario_path),
        "--profile-seed",
        str(run.seed),
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
        "uv",
        "run",
        "--project",
        "GridExpand/2.demand_allocation",
        "python",
        "GridExpand/paired_validation/runner.py",
        "--repo-root",
        str(workdir),
        "--paired-dataset-id",
        str(run.paired_dataset_id),
        "--pylovo-version-id",
        run.pylovo_version_id,
        "--scenario-config",
        str(run.scenario_path),
        "--model-case",
        materialization_case,
        "--result-cases",
        *result_cases,
        "--target",
        str(run.target_network or "both"),
        "--workers",
        str(run.workers),
        "--step3-cpus",
        str(run.step3_cpus),
        "--step3-cluster-concurrency",
        str(run.step3_cluster_concurrency),
        "--step4-cpus",
        str(run.step4_cpus),
        "--powerflow-grid-scope",
        run.powerflow_grid_scope,
        "--profile-seed",
        str(run.seed),
        "--scenario-label",
        run.run_id,
        "--run-name-prefix",
        run.run_id,
        "--run-dir",
        str(GRIDEXPAND_DIR / "run_logs" / run.run_id / group_name),
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


def build_commands(
    run, scenario, model_cases: tuple[str, ...]
) -> list[tuple[list[str], Path]]:
    for model_case in model_cases:
        if model_case not in scenario.model_cases:
            raise ValueError(
                f"Model case {model_case!r} is not enabled by {scenario.scenario_id!r}."
            )
    if run.pipeline == "scenario":
        return [
            _scenario_command(run, scenario, model_case) for model_case in model_cases
        ]

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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and validate paired shared artifacts, then stop.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run, run_hash = load_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run.scenario_path)
    if args.prepare_only and run.pipeline != "paired_validation":
        raise ValueError("--prepare-only is only available for paired_validation runs.")
    model_cases = (args.model_case,) if args.model_case is not None else run.model_cases
    execution_commands = build_commands(run, scenario, model_cases)
    preparation = (
        _paired_preparation_commands(run, scenario)
        if run.pipeline == "paired_validation" and not run.resume
        else []
    )
    postprocessing = (
        _expansion_commands(run, model_cases)
        if run.pipeline == "paired_validation" and not args.prepare_only
        else []
    )
    staged_commands = [
        *preparation,
        *(
            ("execute", command, workdir)
            for command, workdir in (() if args.prepare_only else execution_commands)
        ),
        *(
            ("postprocess_expansion", command, workdir)
            for command, workdir in postprocessing
        ),
    ]
    manifest_dir = GRIDEXPAND_DIR / "run_logs" / run.run_id
    manifest_path = manifest_dir / "run_manifest.json"
    manifest = {
        "run_id": run.run_id,
        "run_hash": run_hash,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "pylovo_version_id": run.pylovo_version_id,
        "profile_seed": run.seed,
        "model_cases": list(model_cases),
        "stages": [
            {"stage": stage, "command": command, "working_directory": str(workdir)}
            for stage, command, workdir in staged_commands
        ],
        "prepare_only": bool(args.prepare_only),
        "dry_run": bool(args.dry_run),
    }
    _write_manifest(manifest_path, manifest)

    if args.dry_run:
        for stage, command, _ in staged_commands:
            print(f"[{stage}] {' '.join(command)}")
        return

    for stage, command, workdir in preparation:
        print(f"[{stage}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=workdir, check=True)

    if run.pipeline == "paired_validation":
        readiness = _validate_prepared_paired(run)
        manifest["paired_readiness"] = readiness
        _write_manifest(manifest_path, manifest)
        print(f"[validate] {json.dumps(readiness, sort_keys=True)}", flush=True)
        if args.prepare_only:
            return

    for command, workdir in execution_commands:
        print(f"[execute] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=workdir, check=True)
    for command, workdir in postprocessing:
        print(f"[postprocess_expansion] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=workdir, check=True)


if __name__ == "__main__":
    main()
