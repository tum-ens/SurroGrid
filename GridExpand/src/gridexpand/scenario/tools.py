"""Small commands around runs: ``gridexpand status``, ``grids`` and ``config check``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gridexpand.paths import RUNS_DIR


# gridexpand status -----------------------------------------------------------------------


def resolve_run_dir(value: str) -> Path:
    """A run directory, or the directory of a run id below ``WORK_DIR/runs``."""
    path = Path(value).expanduser()
    if (path / "state.json").exists() or (path / "summary.json").exists():
        return path.resolve()
    candidate = RUNS_DIR / value
    if (candidate / "state.json").exists():
        return candidate.resolve()
    raise FileNotFoundError(f"No run directory with state.json: {value} (nor {candidate})")


def status_lines(state: dict[str, Any], *, alive: bool) -> list[str]:
    """Readable summary of a ``state.json``."""
    status = state.get("status")
    if status == "running" and not alive:
        status = "running? (the run process is gone: interrupted)"
    jobs = state.get("jobs", {})
    lines = [
        f"run {state.get('run_id')} · {state.get('pipeline')} · {status}"
        + (f" · stage {state['stage']}" if state.get("stage") else "")
        + (f" · exit {state['exit_code']}" if state.get("exit_code") is not None else ""),
        "jobs: " + ", ".join(f"{name} {jobs.get(name, 0)}" for name in
                             ("total", "done", "running", "queued", "failed", "cancelled", "skipped")),
        "stages: " + ", ".join(f"{s['name']} {s['status']}" + (f" ({s['seconds']} s)" if s.get("seconds") else "")
                               for s in state.get("stages", [])),
    ]
    lines += [f"  running {item['job']}: {item.get('step')} since {item.get('since')}"
              for item in state.get("running", [])]
    lines += [f"  failed {job['job']}: {job.get('step') or ''} {job.get('message') or ''}".rstrip()
              for job in state.get("job_list", []) if job.get("status") == "failed"]
    lines.append(f"updated {state.get('updated_at')} · {state.get('run_dir')}")
    return lines


def status_main(argv: list[str] | None = None) -> int:
    from gridexpand.scenario.rundir import load_state, process_alive

    parser = argparse.ArgumentParser(prog="gridexpand status",
                                     description="Show the state of a gridexpand run (reads state.json only).")
    parser.add_argument("run", help="Run directory or run id (WORK_DIR/runs/<run id>).")
    parser.add_argument("--json", action="store_true", help="Print state.json (plus 'alive').")
    args = parser.parse_args(argv)
    try:
        run_dir = resolve_run_dir(args.run)
    except FileNotFoundError as exc:
        print(f"gridexpand status: {exc}", file=sys.stderr)
        return 1
    state = load_state(run_dir)
    if state is None:
        print(f"gridexpand status: {run_dir}/state.json is missing or unreadable", file=sys.stderr)
        return 1
    alive = state.get("status") == "running" and process_alive(state.get("pid"))
    if args.json:
        print(json.dumps({**state, "alive": alive}, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(status_lines(state, alive=alive)))
    return 0


# gridexpand grids ------------------------------------------------------------------------


def grids_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gridexpand grids",
        description="List the candidate grids of an AGS with the runner's numbering (reads pylovo).",
    )
    parser.add_argument("--ags", required=True, help="AGS, e.g. 09184137.")
    parser.add_argument("--pylovo-version-id", required=True)
    parser.add_argument("--plz", type=int, help="Only the grids of this PLZ (numbering stays the AGS's).")
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--demand-scope", choices=("all", "residential"), default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from gridexpand.db import SurroGridDatabase
    from gridexpand.scenario.synthetic_ags_runner import select_region

    db = SurroGridDatabase()
    db.pylovo_version_id = str(args.pylovo_version_id)
    candidates = select_region(
        db.list_grid_candidates(args.ags, min_buildings=args.min_buildings, demand_scope=args.demand_scope),
        plz=args.plz,
    )
    if args.json:
        print(json.dumps(candidates, indent=2, sort_keys=True, default=str))
        return 0
    print(f"{'index':>5}  {'plz':>5}  {'kcid':>4}  {'bcid':>4}  {'buildings':>9}  {'resid.':>6}  bridge file")
    for c in candidates:
        print(f"{c['candidate_index']:>5}  {c['plz']:>5}  {c['kcid']:>4}  {c['bcid']:>4}  "
              f"{c['n_buildings']:>9}  {c['n_residential_buildings']:>6}  {c['bridge_filename']}")
    print(f"{len(candidates)} grid(s)")
    return 0


# gridexpand config check -------------------------------------------------------------------


def check_config_file(path: Path) -> dict[str, Any]:
    """Validate one run or scenario YAML (no database); returns its identity.

    Raises:
        ValueError, FileNotFoundError: invalid file.
    """
    import yaml

    from gridexpand.common.timeframe import scenario_key_for_timeframe
    from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key
    from gridexpand.scenario.model_cases import execution_groups
    from gridexpand.scenario.run_config import load_run_config

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "run" in raw:
        run, run_hash = load_run_config(path)
        scenario, scenario_hash = load_scenario_config(run.scenario_path)
        base_key = scenario_identity_key(scenario.scenario_id, scenario_hash)
        info: dict[str, Any] = {
            "file": str(path),
            "kind": "run",
            "run_id": run.run_id,
            "pipeline": run.pipeline,
            "run_hash": run_hash,
            "scenario": str(run.scenario_path),
            "scenario_id": scenario.scenario_id,
            "scenario_hash": scenario_hash,
            "scenario_key": base_key,
            "pylovo_version_id": run.pylovo_version_id,
            "model_cases": list(run.model_cases),
            "groups": [
                {"name": g.name, "materialization_case": g.materialization_case,
                 "result_cases": list(g.result_cases), "emits_pre": g.emits_pre}
                for g in execution_groups(run.model_cases)
            ],
        }
        if run.pipeline == "synthetic":
            info["pipeline_scenario_key"] = scenario_key_for_timeframe(run.timeframe_mode, base_key=base_key)
        return info
    scenario, scenario_hash = load_scenario_config(path)
    return {
        "file": str(path),
        "kind": "scenario",
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, scenario_hash),
    }


def config_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gridexpand config", description="Configuration tools (no database).")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate run or scenario YAMLs and print their hashes and keys.")
    check.add_argument("paths", nargs="+", type=Path, help="YAML files or directories of YAML files.")
    check.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    files = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.yaml")) if path.is_dir() else [path])
    results, failed = [], 0
    for path in files:
        try:
            results.append({**check_config_file(path), "valid": True})
        except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
            failed += 1
            results.append({"file": str(path), "valid": False, "error": f"{type(exc).__name__}: {exc}"})
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
    else:
        for item in results:
            if not item["valid"]:
                print(f"INVALID {item['file']}: {item['error']}")
            elif item["kind"] == "run":
                groups = ", ".join(f"{g['materialization_case']}->{'+'.join(g['result_cases'])}" for g in item["groups"])
                print(f"ok {item['file']}: run {item['run_id']} ({item['pipeline']}) run_hash {item['run_hash'][:12]} "
                      f"· {item.get('pipeline_scenario_key') or item['scenario_key']} · pylovo {item['pylovo_version_id']}"
                      f" · groups {groups}")
            else:
                print(f"ok {item['file']}: scenario {item['scenario_id']} · {item['scenario_key']}")
    return 3 if failed else 0
