"""Step 3 entry point: run the building optimization for one Step 2 HDF5 file.

Two optimizers solve the same model: ``urbs`` (Pyomo; this urbs version deviates
from urbs-lvds (04 Feb 2025): no grid optimization, 14a/bui-react, uhp,
coordination, curtailment, microgrids, CO2 limits, intertemporal support
timeframes, reactive power or Excel/LP outputs) and ``pypsa`` (PyPSA/linopy,
``gridexpand.optimization.pypsa_model``; full-year chronological inputs only).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gridexpand.common.resource_report import resource_report
from gridexpand.common.timeframe import read_hdf_metadata
from gridexpand.optimization import urbs
from gridexpand.optimization.identity import resolve_input_file, validate_step2_input
from gridexpand.optimization.solver import (
    OPTIMIZER_ENV,
    SOLVER_ENV,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SOLVERS,
    resolve_optimizer_name,
)
from gridexpand.paths import (
    OPTIMIZATION_INPUT_DIR,
    OPTIMIZATION_LOGS_DIR,
    OPTIMIZATION_RESULT_DIR,
)
from gridexpand.scenario.config_loader import load_scenario_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand optimize",
        description="Step 3: urbs optimization of the buildings of one grid (Step 2 HDF5 input).",
    )
    parser.add_argument(
        "inputfile_id",
        help=(
            f"Input file: a path, a file name in {OPTIMIZATION_INPUT_DIR} or the "
            "unique id prefix before the first underscore."
        ),
    )
    parser.add_argument(
        "--optimizer", choices=SUPPORTED_OPTIMIZERS, default=None,
        help=f"Optimizer (default ${OPTIMIZER_ENV}, else urbs).",
    )
    parser.add_argument(
        "--n_cpu", "--partitions", dest="n_cpu", type=int, default=1,
        help=(
            "urbs only: number of building clusters (partitions); each cluster is one "
            "model solved in its own process. It changes the result, it is not a CPU "
            "limit. The pypsa optimizer solves one model per building."
        ),
    )
    parser.add_argument(
        "--cluster-concurrency", type=int, default=None,
        help=(
            "Models solved at the same time (urbs default: $URBS_CLUSTER_CONCURRENCY, "
            "else all clusters; pypsa default: CPUs / solver threads)."
        ),
    )
    parser.add_argument(
        "--solver", choices=SUPPORTED_SOLVERS, default=None,
        help=(
            f"Solver (default ${SOLVER_ENV}, else gurobi). appsi_highs needs no licence "
            "but can return a different optimal vertex / MIP solution."
        ),
    )
    parser.add_argument("--scenario-config", type=Path, required=True,
                        help="Scenario YAML; its hash must match the Step 2 input.")
    parser.add_argument("--tsam", action="store_true", help="Optional enable override; scenario YAML is the default.")
    parser.add_argument("--tsam-periods", type=int, default=None, help="Optional run override for TSAM type weeks.")
    parser.add_argument("--tsam-hours-per-period", type=int, default=None, help="Optional run override for hours per period.")
    parser.add_argument(
        "--tsam-extreme-method",
        choices=["append", "new_cluster_center", "replace_cluster_center"],
        default=None,
        help="How TSAM should include cold and solar extreme weeks.",
    )
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Run preprocessing and TSAM reduction, write reduced_data/tsam outputs, and skip the URBS optimization solve.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_cpu < 1:
        parser.error("--n_cpu must be at least 1.")
    scenario, scenario_hash = load_scenario_config(args.scenario_config)
    if args.reduce_only and not (args.tsam or scenario.time_aggregation.enabled):
        parser.error("--reduce-only requires time_aggregation.enabled in the scenario YAML.")
    args.optimizer = resolve_optimizer_name(args.optimizer)
    if args.optimizer == "pypsa" and (args.tsam or scenario.time_aggregation.enabled):
        parser.error("The pypsa optimizer supports full-year chronological inputs only (no TSAM).")
    args.scenario, args.scenario_hash = scenario, scenario_hash
    return args


def _read_assignment(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_hdf(path, key="raw_data/electrification_assignment")
    except (FileNotFoundError, KeyError):
        return None


def build_global_settings(args, input_file: str, scenario, scenario_hash: str, identity, metadata) -> dict:
    """Run settings; stored as ``urbs_out/reduced_data/global_prop`` (keep keys and order)."""
    time_aggregation = scenario.time_aggregation
    return {
        "input_file": input_file,
        "tsam": bool(args.tsam or time_aggregation.enabled),
        "noTypicalPeriods": args.tsam_periods or time_aggregation.number_of_typical_periods,
        "hoursPerPeriod": args.tsam_hours_per_period or time_aggregation.hours_per_period,
        "tsamExtremePeriodMethod": args.tsam_extreme_method or time_aggregation.extreme_period_method,
        "tsamMethodSettings": {
            "clustering_method": time_aggregation.clustering_method,
            "cluster_representation": time_aggregation.cluster_representation,
            "segmentation": time_aggregation.segmentation,
            "rescale_cluster_periods": time_aggregation.rescale_cluster_periods,
            "feature_weights": time_aggregation.feature_weights,
            "extreme_features": list(time_aggregation.extreme_features),
        },
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": identity.scenario_key,
        "electrification_assignment_hash": metadata.get("electrification_assignment_hash"),
        "reduce_only": args.reduce_only,
        "n_cpu": int(args.n_cpu),
        "hems_session_power_factor": scenario.mobility.hems_session_power_factor,
    }


def main(argv: list[str] | None = None) -> None:
    """Run Step 3 for one input file; see ``gridexpand optimize --help``."""
    with resource_report(include_children=True, name="Urbs Script"):
        args = parse_args(argv)
        scenario, scenario_hash = args.scenario, args.scenario_hash

        input_path = resolve_input_file(OPTIMIZATION_INPUT_DIR, args.inputfile_id)
        metadata = read_hdf_metadata(input_path)
        identity = validate_step2_input(
            metadata, _read_assignment(input_path), scenario, scenario_hash
        )
        global_settings = build_global_settings(
            args, input_path.name, scenario, scenario_hash, identity, metadata
        )

        print("Following global settings are applied:")
        for key, value in global_settings.items():
            print(f"{key:<24} {value}")
        print("\n")

        result_dir = urbs.prepare_result_directory(
            scenario_key=identity.scenario_key, result_root=OPTIMIZATION_RESULT_DIR
        )
        if args.optimizer == "pypsa":
            from gridexpand.optimization.pypsa_model import run_pypsa_opt

            run = run_pypsa_opt
        else:
            run = urbs.run_lvds_opt
        print(f"Optimizer: {args.optimizer}")
        result_path = run(
            os.fspath(input_path),
            result_dir,
            global_settings,
            log_dir=OPTIMIZATION_LOGS_DIR,
            solver=args.solver,
            concurrency=args.cluster_concurrency,
        )
        print(f"Step 3 result: {result_path}")


if __name__ == "__main__":
    main()
