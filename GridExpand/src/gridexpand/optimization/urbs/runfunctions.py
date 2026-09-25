"""Run the urbs building optimization of one Step 2 input: prepare, solve, save."""

import multiprocessing as mp
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
from pyomo.opt import check_optimal_termination

from ..solver import audit_record, make_solver, resolve_solver_name, solver_log_folder
from .features.typeperiod import run_tsam, select_predefined_timesteps
from .identify import get_parallel_building_clusters, identify_mode
from .input import get_cluster_data, read_input_h5
from .model import create_model
from .saveload import (
    HDF_OPTIONS,
    create_result_cache,
    merge_cluster_results,
    save,
    save_reduced_data,
)
from .scenarios import insert_scenario, read_scenario_name

CONCURRENCY_ENV = "URBS_CLUSTER_CONCURRENCY"

# Settings recorded in urbs_out/reduced_data/global_prop after the run settings.
# 'parallel', 'vartariff' and 'power_price_kw' are no longer read by the code; they
# keep the stored settings table unchanged.
_RECORDED_SETTINGS = {"dt": 1, "parallel": True, "vartariff": 0, "power_price_kw": 0}


def prepare_result_directory(input_file=None, script_name=None, scenario_key=None, result_root="result"):
    """Create and return ``result_root/<scenario_key>`` (``result_root`` if no key)."""
    key = str(scenario_key).strip() if scenario_key is not None else ""
    if scenario_key is not None and (
        not key
        or key in {".", ".."}
        or os.path.basename(key) != key
    ):
        raise ValueError(
            f"scenario_key must be one directory-safe path component: {scenario_key!r}"
        )
    result_root = str(result_root)
    result_dir = os.path.join(result_root, key) if key else result_root
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def solve_cluster(data_cluster, global_settings, logfile, solver_name, cluster_index=0):
    """Build and solve one cluster model; return its result cache and solver audit.

    Raises:
        RuntimeError: the solver did not end with an optimal termination (for the
            MIP: optimal within the configured gap).
    """
    start = time.time()
    model = create_model(data_cluster, global_settings)
    if os.path.exists(logfile):
        os.remove(logfile)
    solver = make_solver(solver_name, logfile)
    build_seconds = time.time() - start
    print(f"Model setup {cluster_index} took {build_seconds / 60:.2f} minutes to run!")

    start = time.time()
    result = solver.solve(model, tee=False, report_timing=False)
    solve_seconds = time.time() - start
    if not check_optimal_termination(result):
        raise RuntimeError(
            f"Cluster {cluster_index}: {solver_name} ended with status="
            f"{result.solver.status}, termination={result.solver.termination_condition}; "
            f"see {logfile}."
        )
    print(f"Model solve {cluster_index} took {solve_seconds / 60:.2f} minutes to run")

    sites = sorted(str(site) for site in model.sit)
    audit = {
        "cluster": int(cluster_index),
        "n_sites": len(sites),
        "first_site": sites[0] if sites else "",
        **audit_record(solver_name, solver, result),
        "build_seconds": float(build_seconds),
        "solve_seconds": float(solve_seconds),
    }
    return {"results": create_result_cache(model), "audit": audit}


def _cluster_worker(connection, data_cluster, global_settings, logfile, solver_name, cluster_index):
    try:
        payload = ("ok", solve_cluster(data_cluster, global_settings, logfile, solver_name, cluster_index))
    except BaseException:
        payload = ("error", traceback.format_exc())
    try:
        connection.send(payload)
    finally:
        connection.close()


def solve_partitions(data, global_settings, *, log_dir, scenario_name, solver_name, concurrency=None):
    """Solve every building cluster in its own process and return the outputs in cluster order.

    Args:
        data: prepared urbs input of the whole grid.
        global_settings: run settings (``n_cpu`` is the number of clusters).
        log_dir: directory for one solver log per cluster.
        scenario_name: scenario key (part of the log file names).
        solver_name: see ``gridexpand.optimization.solver``.
        concurrency: clusters solved at the same time; default
            ``$URBS_CLUSTER_CONCURRENCY`` or all clusters.
    """
    clusters = get_parallel_building_clusters(data, global_settings["n_cpu"])
    if concurrency is None:
        concurrency = int(os.getenv(CONCURRENCY_ENV, len(clusters)))
    concurrency = max(1, min(int(concurrency), len(clusters)))
    print(f"Running up to {concurrency} optimization worker(s) concurrently.")
    os.makedirs(log_dir, exist_ok=True)
    stem = global_settings["input_file"][:-3]

    outputs = {}
    running = []

    def collect(index, process, connection):
        try:
            status, payload = connection.recv()
        except EOFError:
            status, payload = None, None
        finally:
            connection.close()
        process.join()
        if status == "error":
            raise RuntimeError(f"Worker {index} failed:\n{payload}")
        if status != "ok":
            raise RuntimeError(
                f"Worker {index} exited without returning a result "
                f"(exitcode={process.exitcode})."
            )
        outputs[index] = payload

    for index, cluster in enumerate(clusters):
        data_cluster = get_cluster_data(data, cluster)
        logfile = os.path.join(str(log_dir), f"{stem}_{scenario_name}_{index}.log")
        receiver, sender = mp.Pipe(duplex=False)
        process = mp.Process(
            target=_cluster_worker,
            args=(sender, data_cluster, global_settings, logfile, solver_name, index),
        )
        process.start()
        sender.close()
        running.append((index, process, receiver))
        if len(running) >= concurrency:
            collect(*running.pop(0))
    for item in running:
        collect(*item)
    return [outputs[index] for index in range(len(clusters))]


def _check_full_year_input(mode, data):
    # A fresh Step-2 input initializes weight_typeperiod to NaN, so mode['tdy']
    # must be False here. Asserting it prevents a stale or hand-edited input from
    # silently reintroducing representative-period storage resets.
    if mode["tdy"]:
        raise ValueError(
            "Time aggregation is disabled but type-period weights are active. "
            "Full-year runs must not carry representative-period weights or "
            "the weekly storage-state constraints they enable."
        )
    occurrences = data["type_period"]["weight_typeperiod"].dropna()
    if not occurrences.empty and not (occurrences == 1).all():
        raise ValueError(
            "Full-year runs require unit timestep occurrence weights; found "
            f"{sorted(occurrences.unique())[:5]}."
        )


def load_and_prepare(input_path, global_settings, solver_name):
    """Read the input, record the run settings and identify the model features.

    Returns:
        ``(data, mode, scenario_name)``; ``global_settings`` gains ``dt``,
        ``solver_name``, the recorded constants and ``timesteps``.
    """
    global_settings.update({"dt": _RECORDED_SETTINGS["dt"], "solver_name": solver_name})
    global_settings.update({key: value for key, value in _RECORDED_SETTINGS.items() if key != "dt"})

    print("Reading and validating input data...")
    data = read_input_h5(input_path)
    max_timestep = int(data["demand"].index.get_level_values("t").max())
    global_settings["timesteps"] = range(0, max_timestep + 1)
    print(f"Using {max_timestep} modeled demand hour(s) plus storage initialization timestep.")

    print("\nReading running modes...")
    scenario_name = read_scenario_name(global_settings, data)
    data = insert_scenario(data, global_settings)
    mode = identify_mode(data)
    print(f"Identified running modes: {mode}")
    if not mode["tsam"]:
        _check_full_year_input(mode, data)
    return data, mode, scenario_name


def apply_temporal_method(data, mode, global_settings):
    """Run TSAM (type periods) or keep the chronological horizon.

    Returns:
        ``(data, tsam_data)``; ``global_settings['timesteps']`` is updated for TSAM.
    """
    if mode["tsam"]:
        print("Running time series aggregation (TSAM)...")
        start = time.time()
        data, global_settings["timesteps"], tsam_data = run_tsam(
            data,
            global_settings["noTypicalPeriods"],
            global_settings["hoursPerPeriod"],
            global_settings.get("tsamExtremePeriodMethod", "replace_cluster_center"),
            global_settings.get("tsamMethodSettings"),
        )
        print(f"TSAM took {(time.time() - start) / 60:.2f} minutes to run!\n")
    else:
        data, tsam_data = select_predefined_timesteps(data, global_settings["timesteps"])
    return data, tsam_data


def temporal_audit(mode, data, global_settings):
    """The ``urbs_out/temporal_method`` record read by Step 4 and the paired checks.

    ``annual_weight`` and ``storage_boundary_policy`` describe the model that is
    solved: with TSAM type periods the model weight is 1 (the type-period weights
    carry the annual scaling) and storages close per period.
    """
    solved_mode = identify_mode(data)
    operating_hours = int(len(global_settings["timesteps"]) - 1)
    return pd.Series(
        {
            "temporal_method": (
                "shared_weather_tsam" if mode["tsam"] else "full_year_no_tsam"
            ),
            "operating_hours": operating_hours,
            "initialization_rows": 1,
            "delta_t_hours": float(global_settings["dt"]),
            "annual_weight": 1.0 if solved_mode["tdy"] else float(8760) / operating_hours,
            "source_reference_year": int(global_settings.get("source_reference_year", 2009)),
            "storage_boundary_policy": (
                "typeperiod_common_initial_state"
                if solved_mode["tdy"]
                else "annual_equality"
            ),
            "ev_boundary_policy": (
                "dedicated_sessions_annual_wrap"
                if mode.get("evs")
                else "legacy_mobility_buffer"
            ),
            "scenario_key": str(global_settings.get("scenario_key", "")),
            "scenario_hash": str(global_settings.get("scenario_hash", "")),
        },
        dtype=object,
    )


def final_result_path(result_dir, input_file, scenario_name):
    """``<result_dir>/<input stem>_<scenario key>.h5``."""
    stem = os.path.splitext(os.path.basename(input_file))[0]
    return Path(result_dir) / f"{stem}_{scenario_name}.h5"


def run_lvds_opt(input_path, result_dir, global_settings, *, log_dir, solver=None, concurrency=None):
    """Run Step 3 for one input and return the path of the result HDF5 file.

    The input is copied to ``<result>.partial``; TSAM tables, the temporal audit,
    the reduced inputs, all cluster results and the solver audit are appended, and
    the file is renamed to its final name only on success (removed on failure).

    Args:
        input_path: Step 2 HDF5 file.
        result_dir: directory of the scenario's results.
        global_settings: run settings (see ``run_urbs_cluster.build_global_settings``).
        log_dir: directory of the solver log files.
        solver: solver name (default ``$GRIDEXPAND_SOLVER`` or ``gurobi``).
        concurrency: clusters solved concurrently (default: all).
    """
    solver_name = resolve_solver_name(solver)
    warnings.filterwarnings("ignore", category=FutureWarning, module="tsam.timeseriesaggregation")
    start = time.time()
    data, mode, scenario_name = load_and_prepare(input_path, global_settings, solver_name)
    print(f"Preprocesssing took {(time.time() - start) / 60:.2f} minutes to run!\n")

    data, tsam_data = apply_temporal_method(data, mode, global_settings)
    audit = temporal_audit(mode, data, global_settings)

    final_path = final_result_path(result_dir, global_settings["input_file"], scenario_name)
    partial_path = final_path.with_name(final_path.name + ".partial")
    try:
        shutil.copyfile(input_path, partial_path)
        with pd.HDFStore(partial_path, mode='a', **HDF_OPTIONS) as store:
            for name, table in tsam_data.items():
                store['urbs_out/tsam/' + name] = table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            with pd.HDFStore(partial_path, mode='a', **HDF_OPTIONS) as store:
                store['urbs_out/temporal_method'] = audit
        print(f"Temporal method: {audit.to_dict()}\n")

        if global_settings.get("reduce_only"):
            print("Reduce-only mode active: saving TSAM/reduced input data and skipping optimization.")
            save_reduced_data(data, partial_path)
        else:
            print("Setting up and running parallel pyomo models...")
            start = time.time()
            outputs = solve_partitions(
                data,
                global_settings,
                log_dir=os.path.join(str(log_dir), solver_log_folder(solver_name)),
                scenario_name=scenario_name,
                solver_name=solver_name,
                concurrency=concurrency,
            )
            print(f"Solving process took {(time.time() - start) / 60:.2f} minutes to run!\n")
            start = time.time()
            results = merge_cluster_results([output["results"] for output in outputs])
            solver_audit = pd.DataFrame([output["audit"] for output in outputs])
            save(data, results, partial_path, solver_audit=solver_audit)
            print(f"Model results saving took {(time.time() - start) / 60:.2f} minutes\n")
        os.replace(partial_path, final_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise
    return final_path
