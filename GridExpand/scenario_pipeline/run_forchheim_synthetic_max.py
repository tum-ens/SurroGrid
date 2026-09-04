#!/usr/bin/env python3
"""Run all Forchheim 2045 model cases on one residential-dominated synthetic grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
SURROGRID_DIR = GRIDEXPAND_DIR.parent
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402
from common.orchestration import command_env, utc_now  # noqa: E402
from common.timeframe import (  # noqa: E402
    output_filename_for_timeframe,
    read_hdf_metadata,
    scenario_key_for_timeframe,
    scenario_output_directory,
)
from scenario_pipeline.config_loader import (  # noqa: E402
    load_run_config,
    load_scenario_config,
    scenario_identity_key,
)
from scenario_pipeline.synthetic_ags_runner import (  # noqa: E402
    case_qualified_filename,
    get_candidates,
    powerflow_run_name,
    scenario_suffix_from_hdf,
)

DEFAULT_RUN_CONFIG = (
    GRIDEXPAND_DIR
    / "scenario_pipeline"
    / "config"
    / "runs"
    / "forchheim_2045_synthetic.yaml"
)
DEFAULT_OUTPUT_DIR = (
    GRIDEXPAND_DIR / "run_logs" / "forchheim_2045_synthetic_smoke_grid20"
)

SELECTED_GRID = {
    "ags": "9474126",
    "candidate_index": 15,
    "grid_result_id": 20,
    "plz": 91301,
    "kcid": 2,
    "bcid": 6,
    "n_buildings": 48,
    "n_residential_buildings": 37,
    "residential_share": 37 / 48,
    "residential_households": 180,
}

CASE_SPECS = (
    {
        "model_case": "pre",
        "profiles": "status_quo",
        "description": "Reference electricity demand; no urbs optimization.",
        "extra_args": (),
        "powerflow_mode": "summary",
    },
    {
        "model_case": "post-inflex-heuristic",
        "profiles": "all",
        "description": "Shared heuristic capacities with causal rule-based dispatch.",
        "extra_args": ("--no-flex-only",),
        "powerflow_mode": "summary_no_flex",
    },
    {
        "model_case": "post-hems-heuristic",
        "profiles": "all",
        "description": "Shared heuristic capacities with optimized HEMS dispatch.",
        "extra_args": (),
        "powerflow_mode": "summary",
    },
    {
        "model_case": "post-hems-optimized",
        "profiles": "all",
        "description": "Endogenous urbs sizing with optimized HEMS dispatch.",
        "extra_args": (),
        "powerflow_mode": "summary",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step2-cpus", type=int, default=4)
    parser.add_argument("--step3-cpus", type=int, default=8)
    parser.add_argument("--step3-max-cpus", type=int, default=16)
    parser.add_argument("--step4-cpus", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed cases and continue at the first incomplete case.",
    )
    return parser.parse_args()


def selected_candidate(version_id: str) -> dict[str, object]:
    candidates = get_candidates(
        SURROGRID_DIR,
        SELECTED_GRID["ags"],
        min_buildings=5,
        demand_scope="all",
        pylovo_version_id=version_id,
    )
    matches = [
        candidate
        for candidate in candidates
        if int(candidate["candidate_index"]) == SELECTED_GRID["candidate_index"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one candidate at index {SELECTED_GRID['candidate_index']}, found {len(matches)}."
        )
    candidate = matches[0]
    for key in ("grid_result_id", "plz", "kcid", "bcid", "n_buildings"):
        if int(candidate[key]) != int(SELECTED_GRID[key]):
            raise RuntimeError(
                f"Selected-grid contract changed for {key}: "
                f"expected {SELECTED_GRID[key]}, found {candidate[key]}."
            )
    if (
        int(candidate["n_residential_buildings"])
        != SELECTED_GRID["n_residential_buildings"]
    ):
        raise RuntimeError(
            "Selected-grid residential count changed: "
            f"expected {SELECTED_GRID['n_residential_buildings']}, "
            f"found {candidate['n_residential_buildings']}."
        )
    return candidate


def hdf_keys(path: Path) -> list[str]:
    with pd.HDFStore(path, mode="r") as store:
        return sorted(store.keys())


def powerflow_run_id(run_name: str, input_filename: str) -> int:
    db = SurroGridDatabase()
    query = text(
        """
        SELECT powerflow_run_id
        FROM surrogrid.powerflow_run
        WHERE run_name = :run_name
          AND urbs_input_file = :input_filename
        ORDER BY updated_at DESC, powerflow_run_id DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as connection:
        value = connection.execute(
            query, {"run_name": run_name, "input_filename": input_filename}
        ).scalar_one_or_none()
    if value is None:
        raise RuntimeError(
            f"No power-flow run found for run_name={run_name!r}, input={input_filename!r}."
        )
    return int(value)


def archive_case(
    *,
    output_dir: Path,
    candidate: dict[str, object],
    case_spec: dict[str, Any],
    runner_args: argparse.Namespace,
) -> dict[str, Any]:
    model_case = str(case_spec["model_case"])
    case_dir = output_dir / "artifacts" / model_case
    case_dir.mkdir(parents=True, exist_ok=True)

    base_filename = output_filename_for_timeframe(
        str(candidate["bridge_filename"]), "full_year"
    )
    step2_filename = case_qualified_filename(base_filename, runner_args)
    step2_path = (
        GRIDEXPAND_DIR
        / "2.demand_allocation"
        / "gridalloc"
        / "results"
        / runner_args.scenario_key
        / step2_filename
    )
    if not step2_path.exists():
        raise FileNotFoundError(step2_path)

    input_archive = case_dir / "urbs_input.h5"
    shutil.copy2(step2_path, input_archive)
    artifact: dict[str, Any] = {
        "model_case": model_case,
        "description": case_spec["description"],
        "step2_filename": step2_filename,
        "urbs_input_h5": str(input_archive.resolve()),
        "urbs_input_keys": hdf_keys(input_archive),
        "urbs_output_h5": None,
        "urbs_output_keys": [],
    }

    if model_case == "pre":
        powerflow_input_filename = step2_filename
    else:
        suffix = scenario_suffix_from_hdf(step2_path)
        powerflow_input_filename = step2_filename.replace(".h5", f"_{suffix}.h5")
        step3_path = scenario_output_directory(
            GRIDEXPAND_DIR / "3.urbs" / "result", runner_args.scenario_key
        ) / powerflow_input_filename
        if not step3_path.exists():
            raise FileNotFoundError(step3_path)
        output_archive = case_dir / "urbs_output.h5"
        shutil.copy2(step3_path, output_archive)
        artifact.update(
            {
                "step3_filename": powerflow_input_filename,
                "urbs_output_h5": str(output_archive.resolve()),
                "urbs_output_keys": hdf_keys(output_archive),
            }
        )

    run_name = powerflow_run_name(runner_args, str(case_spec["powerflow_mode"]))
    artifact["powerflow_input_filename"] = powerflow_input_filename
    artifact["powerflow_run_name"] = run_name
    artifact["powerflow_run_id"] = powerflow_run_id(run_name, powerflow_input_filename)
    return artifact


def audit_controlled_realization(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail when model cases do not share their controlled physical inputs."""
    cases = {
        str(case["model_case"]): case
        for case in manifest["cases"]
        if case.get("status") == "done"
    }
    expected = {str(spec["model_case"]) for spec in CASE_SPECS}
    if set(cases) != expected:
        raise RuntimeError(
            f"Realization audit requires completed cases {sorted(expected)}; "
            f"found {sorted(cases)}."
        )

    paths = {
        case: Path(values["urbs_input_h5"])
        for case, values in cases.items()
    }
    metadata = {case: read_hdf_metadata(path) for case, path in paths.items()}

    common_keys = (
        "profile_seed",
        "profile_realization_id",
        "profile_realization_contract",
        "profile_hash_base_electricity",
    )
    post_keys = (
        "profile_hash_space_heat",
        "profile_hash_hot_water",
        "profile_hash_heat_pump_cop",
        "profile_hash_mobility_demand",
        "profile_hash_mobility_availability",
    )

    def require_equal(key: str, selected_cases: tuple[str, ...]) -> str:
        values = {case: metadata[case].get(key) for case in selected_cases}
        missing = [case for case, value in values.items() if value in (None, "")]
        if missing:
            raise RuntimeError(f"Missing {key} in model cases {missing}.")
        unique = {str(value) for value in values.values()}
        if len(unique) != 1:
            raise RuntimeError(f"Controlled realization mismatch for {key}: {values}")
        return str(next(iter(values.values())))

    all_cases = tuple(spec["model_case"] for spec in CASE_SPECS)
    post_cases = tuple(case for case in all_cases if case != "pre")
    checked = {
        key: require_equal(key, all_cases)
        for key in common_keys
    }
    checked.update({
        key: require_equal(key, post_cases)
        for key in post_keys
    })

    heuristic_cases = ("post-inflex-heuristic", "post-hems-heuristic")
    plan_keys = (
        "raw_data/asset_plan",
        "raw_data/battery_asset_plan",
        "raw_data/heat_asset_plan",
    )
    plan_hashes: dict[str, str] = {}
    for key in plan_keys:
        frames = {case: pd.read_hdf(paths[case], key=key) for case in heuristic_cases}
        left = frames[heuristic_cases[0]].sort_index(axis=1).reset_index(drop=True)
        right = frames[heuristic_cases[1]].sort_index(axis=1).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(left, right, check_dtype=False)
        except AssertionError as exc:
            raise RuntimeError(
                f"Heuristic asset plans differ for {key}: {exc}"
            ) from exc
        plan_hashes[key] = str(pd.util.hash_pandas_object(left, index=True).sum())

    return {
        "status": "passed",
        "controlled_metadata": checked,
        "identical_heuristic_asset_plans": plan_hashes,
    }


def audit_capacity_plausibility(manifest: dict[str, Any]) -> dict[str, Any]:
    """Write building-level capacity checks and reject physically inconsistent plans."""
    summaries: dict[str, Any] = {}
    for case in manifest["cases"]:
        model_case = str(case["model_case"])
        if model_case == "pre" or case.get("status") != "done":
            continue
        output_path = Path(case["urbs_output_h5"])
        battery = pd.read_hdf(output_path, key="raw_data/battery_asset_plan")
        heat = pd.read_hdf(output_path, key="raw_data/heat_asset_plan")
        cap_pro = pd.read_hdf(output_path, key="urbs_out/MILP/cap_pro")
        cap_sto = pd.read_hdf(output_path, key="urbs_out/MILP/cap_sto_c")
        process_capacity = {
            (int(site), str(process)): float(value)
            for (_, site, process), value in cap_pro.items()
        }
        storage_capacity = {
            (int(site), str(storage)): float(value)
            for (_, site, storage, _), value in cap_sto.items()
        }

        rows: list[dict[str, Any]] = []
        battery_violations = 0
        for _, asset in battery.iterrows():
            site = int(asset["Site"])
            installed = storage_capacity.get((site, "battery_private"), 0.0)
            upper = float(asset["battery_capacity_upper_kwh"])
            htw_upper = float(asset.get("battery_htw_upper_kwh", upper))
            households = pd.to_numeric(
                pd.Series([asset.get("number_of_households")]), errors="coerce"
            ).iloc[0]
            valid = installed <= upper + 1e-6 and upper <= htw_upper + 1e-6
            battery_violations += int(not valid)
            rows.append({
                "asset": "battery",
                "building_objectid": str(asset["building_objectid"]),
                "site": site,
                "installed_capacity": installed,
                "upper_capacity": upper,
                "reference_capacity": float(asset["battery_reference_pv_kwp"]),
                "capacity_per_reference": (
                    installed / float(asset["battery_reference_pv_kwp"])
                    if float(asset["battery_reference_pv_kwp"]) > 0 else 0.0
                ),
                "capacity_per_household": (
                    installed / float(households)
                    if pd.notna(households) and float(households) > 0 else None
                ),
                "at_upper_bound": abs(installed - upper) <= 1e-6 and upper > 0.0,
                "valid": valid,
            })

        buffer_violations = 0
        coverage_violations = 0
        for _, asset in heat.iterrows():
            site = int(asset["Site"])
            hp_kw_el = process_capacity.get((site, "heatpump_air"), 0.0)
            auxiliary_kw_el = process_capacity.get((site, "auxiliary_heater"), 0.0)
            buffer_kwh = storage_capacity.get((site, "heat_storage"), 0.0)
            hp_kw_th = hp_kw_el * float(asset["design_cop"])
            upper_kwh = float(asset["buffer_capacity_upper_kwh_th"])
            upper_litres = float(asset["buffer_volume_l"])
            kwh_per_litre = upper_kwh / upper_litres if upper_litres > 0.0 else 0.0
            buffer_litres = buffer_kwh / kwh_per_litre if kwh_per_litre > 0.0 else 0.0
            litres_per_kw = buffer_litres / hp_kw_th if hp_kw_th > 1e-9 else 0.0
            buffer_valid = (
                buffer_kwh <= upper_kwh + 1e-6
                and (hp_kw_th > 1e-9 or buffer_kwh <= 1e-6)
                and litres_per_kw <= 20.0 + 1e-6
            )
            coverage_valid = bool(asset.get("heat_capacity_peak_coverage_valid", True))
            buffer_violations += int(not buffer_valid)
            coverage_violations += int(not coverage_valid)
            rows.append({
                "asset": "space_heat_buffer",
                "building_objectid": str(asset["building_objectid"]),
                "site": site,
                "installed_capacity": buffer_kwh,
                "upper_capacity": upper_kwh,
                "reference_capacity": hp_kw_th,
                "capacity_per_reference": litres_per_kw,
                "capacity_per_household": (
                    buffer_litres / float(asset["number_of_households"])
                    if pd.notna(asset.get("number_of_households"))
                    and float(asset["number_of_households"]) > 0 else None
                ),
                "buffer_volume_m3": buffer_litres / 1000.0,
                "auxiliary_capacity_kw_el": auxiliary_kw_el,
                "at_upper_bound": abs(buffer_kwh - upper_kwh) <= 1e-6 and upper_kwh > 0.0,
                "valid": buffer_valid and coverage_valid,
            })

        audit = pd.DataFrame(rows)
        audit_path = output_path.parent / "capacity_plausibility_audit.csv"
        audit.to_csv(audit_path, index=False)
        summary = {
            "status": "passed" if not (battery_violations or buffer_violations or coverage_violations) else "failed",
            "audit_csv": str(audit_path.resolve()),
            "battery_candidates": int(len(battery)),
            "battery_positive": int((audit.loc[audit["asset"] == "battery", "installed_capacity"] > 1e-6).sum()),
            "battery_at_upper_bound": int(audit.loc[audit["asset"] == "battery", "at_upper_bound"].sum()),
            "battery_violations": battery_violations,
            "heat_buffer_candidates": int(len(heat)),
            "heat_buffer_at_upper_bound": int(audit.loc[audit["asset"] == "space_heat_buffer", "at_upper_bound"].sum()),
            "heat_buffer_violations": buffer_violations,
            "heat_peak_coverage_violations": coverage_violations,
            "largest_battery_kwh": float(audit.loc[audit["asset"] == "battery", "installed_capacity"].max()),
            "largest_buffer_m3": float(audit.get("buffer_volume_m3", pd.Series(dtype=float)).max()),
        }
        summaries[model_case] = summary
        if summary["status"] != "passed":
            raise RuntimeError(f"Capacity plausibility audit failed for {model_case}: {summary}")
    return {"status": "passed", "cases": summaries}


def main() -> int:
    args = parse_args()
    args.run_config = args.run_config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config, run_hash = load_run_config(args.run_config)
    scenario, scenario_hash = load_scenario_config(run_config.scenario_path)
    if run_config.pipeline != "scenario" or run_config.storage != "db":
        raise ValueError(
            "The smoke runner requires an ordinary DB-backed scenario run YAML."
        )
    candidate = selected_candidate(run_config.pylovo_version_id)

    manifest_path = args.output_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_config_hash") != run_hash:
            raise RuntimeError(
                "Cannot resume smoke artifacts created from a different run YAML. "
                "Start without --resume to regenerate all controlled model cases."
            )
        manifest["status"] = "dry-run" if args.dry_run else "running"
        manifest["finished_at"] = None
    else:
        manifest: dict[str, Any] = {
            "run_id": "forchheim_2045_synthetic_smoke_grid20",
            "started_at": utc_now(),
            "finished_at": None,
            "status": "dry-run" if args.dry_run else "running",
            "run_config": str(args.run_config),
            "run_config_hash": run_hash,
            "scenario_config": str(run_config.scenario_path),
            "pylovo_version_id": run_config.pylovo_version_id,
            "profile_seed": run_config.seed,
            "selected_grid": {**SELECTED_GRID, **candidate},
            "cases": [],
        }

    for case_spec in CASE_SPECS:
        model_case = str(case_spec["model_case"])
        previous = next(
            (case for case in manifest["cases"] if case["model_case"] == model_case),
            None,
        )
        if args.resume and previous and previous.get("status") == "done":
            input_h5 = previous.get("urbs_input_h5")
            if input_h5 and Path(input_h5).exists():
                print(f"Skipping completed case {model_case}.", flush=True)
                continue
        case_run_dir = args.output_dir / "runs" / model_case
        command = [
            "uv",
            "run",
            "--project",
            "GridExpand/2.demand_allocation",
            "python",
            "GridExpand/scenario_pipeline/synthetic_ags_runner.py",
            "--repo-root",
            str(SURROGRID_DIR),
            "--ags",
            SELECTED_GRID["ags"],
            "--pylovo-version-id",
            run_config.pylovo_version_id,
            "--scenario-config",
            str(run_config.scenario_path),
            "--model-case",
            model_case,
            "--profiles",
            str(case_spec["profiles"]),
            "--profile-seed",
            str(run_config.seed),
            "--demand-scope",
            run_config.demand_scope,
            "--timeframe-mode",
            run_config.timeframe_mode,
            "--workers",
            "1",
            "--step2-cpus",
            str(args.step2_cpus),
            "--step3-cpus",
            str(args.step3_cpus),
            "--step3-max-cpus",
            str(args.step3_max_cpus),
            "--step3-cluster-concurrency",
            "1",
            "--step4-cpus",
            str(args.step4_cpus),
            "--step2-timeseries-storage",
            "temp",
            "--powerflow-output",
            "both",
            "--cleanup-intermediates",
            "never",
            "--no-materialize-expansion",
            "--no-pilot-gate",
            "--start-index",
            str(SELECTED_GRID["candidate_index"]),
            "--limit",
            "1",
            "--case-qualified-output",
            "--run-dir",
            str(case_run_dir),
            *case_spec["extra_args"],
        ]
        case_manifest: dict[str, Any] = {
            "model_case": model_case,
            "description": case_spec["description"],
            "command": command,
            "run_dir": str(case_run_dir),
            "status": "dry-run" if args.dry_run else "running",
        }
        if previous is None:
            manifest["cases"].append(case_manifest)
        else:
            manifest["cases"][manifest["cases"].index(previous)] = case_manifest
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(
            command,
            cwd=SURROGRID_DIR,
            env=command_env(),
            check=False,
        )
        case_manifest["returncode"] = completed.returncode
        if completed.returncode != 0:
            case_manifest["status"] = "failed"
            manifest["status"] = "failed"
            manifest["finished_at"] = utc_now()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            return completed.returncode

        runner_args = argparse.Namespace(
            model_case=model_case,
            case_qualified_output=True,
            profiles=case_spec["profiles"],
            demand_scope=run_config.demand_scope,
            timeframe_mode=run_config.timeframe_mode,
            expansion_analysis_prefix=None,
            tsam=True,
            scenario_key=scenario_key_for_timeframe(
                run_config.timeframe_mode,
                base_key=scenario_identity_key(scenario.scenario_id, scenario_hash),
            ),
        )
        case_manifest.update(
            archive_case(
                output_dir=args.output_dir,
                candidate=candidate,
                case_spec=case_spec,
                runner_args=runner_args,
            )
        )
        case_manifest["status"] = "done"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    if not args.dry_run:
        manifest["controlled_realization_audit"] = audit_controlled_realization(manifest)
        manifest["capacity_plausibility_audit"] = audit_capacity_plausibility(manifest)
    manifest["status"] = "dry-run" if args.dry_run else "done"
    manifest["finished_at"] = utc_now()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
