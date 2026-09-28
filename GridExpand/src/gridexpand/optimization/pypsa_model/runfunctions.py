"""Step 3 with the PyPSA optimizer: partition, parallel solves, result file.

The result file has the layout of the urbs optimizer (``urbs.runfunctions``):
the Step 2 input, ``urbs_out/temporal_method``, ``urbs_out/tsam/kept_timesteps``,
the inputs in ``urbs_out/reduced_data``, the ``results.RESULT_KEYS`` entities in
``urbs_out/MILP`` and one ``urbs_out/solver_audit`` row per model.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import multiprocessing as mp
import os
import shutil
import time
import traceback
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd

from ..solver import resolve_solver_name, solver_log_folder
from ..urbs.input import get_cluster_data
from ..urbs.runfunctions import apply_temporal_method, final_result_path, load_and_prepare, temporal_audit
from ..urbs.saveload import HDF_OPTIONS, merge_cluster_results, save
from .solve import linopy_solver, solve_group

# Buildings per model. The buildings of a grid are independent in the Step 3
# model (no shared constraint), so the partition does not change the optimum; one
# building per model is the fastest partition and applies the MIP gap to every
# building (see docs/optimization.md).
BUILDINGS_PER_MODEL = 1

_GUROBI_ENV = None  # one gurobipy.Env (one licence session) per worker process


def building_groups(data: dict, buildings_per_model: int = BUILDINGS_PER_MODEL) -> list[list]:
    """Consecutive groups of ``buildings_per_model`` buildings, in the input order."""
    if buildings_per_model < 1:
        raise ValueError("buildings_per_model must be at least 1.")
    buildings = list(data["site"].index.get_level_values("Name").unique())
    return [buildings[i:i + buildings_per_model] for i in range(0, len(buildings), buildings_per_model)]


# Default workers: one per 4 CPUs (Gurobi runs 4 threads; HiGHS was not faster with
# more workers either) and one per WORKER_MEMORY_GB of available memory (measured
# peak of a one-building worker: ~1 GB LP, ~1.8 GB MILP).
CPUS_PER_WORKER = 4
WORKER_MEMORY_GB = 2.0


def _available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def default_concurrency(solver_name: str, n_models: int) -> int:
    """Models solved at the same time when ``--cluster-concurrency`` is not given."""
    linopy_solver(solver_name)
    workers = (os.cpu_count() or 1) // CPUS_PER_WORKER
    memory = _available_memory_gb()
    if memory is not None:
        workers = min(workers, int(memory // WORKER_MEMORY_GB))
    return max(1, min(n_models, workers))


def _gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        import gurobipy

        env = gurobipy.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        _GUROBI_ENV = env
    return _GUROBI_ENV


def _init_worker() -> None:
    # PyPSA configures INFO logging; the solver logs go to the log files instead.
    for name in ("linopy", "pypsa", "gurobipy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _release_memory() -> None:
    """Free the solved model now and hand freed heap memory back to the OS.

    Networks and linopy models contain reference cycles, and glibc keeps freed
    memory: without this a worker grows to several GB over a few dozen models.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


def _solve_task(index: int, data: dict, mode: dict, solver_name: str, log_file: str) -> dict:
    """Worker: solve one building group; raises with the worker traceback."""
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
        env = _gurobi_env() if linopy_solver(solver_name) == "gurobi" else None
        output = solve_group(data, mode, solver_name=solver_name, log_file=log_file, env=env, index=index)
        _release_memory()
        audit = output["audit"]
        print(f"Model {index} ({audit['n_sites']} building(s)): build {audit['build_seconds']:.1f} s, "
              f"solve {audit['solve_seconds']:.1f} s, objective {audit['objective']:.2f}", flush=True)
        return output
    except BaseException as exc:
        raise RuntimeError(f"Building group {index} failed:\n{traceback.format_exc()}") from exc


def solve_groups(data: dict, mode: dict, groups: list[list], *, solver_name: str, log_dir: Path,
                 stem: str, scenario_name: str, concurrency: int | None = None) -> list[dict]:
    """Solve every building group in a pool of worker processes; outputs in group order."""
    if concurrency is None:
        concurrency = default_concurrency(solver_name, len(groups))
    concurrency = max(1, min(int(concurrency), len(groups)))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Solving {len(groups)} PyPSA model(s) with up to {concurrency} worker(s).", flush=True)
    outputs: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as pool:
        futures = {
            pool.submit(_solve_task, index, get_cluster_data(data, group), mode, solver_name,
                        os.path.join(str(log_dir), f"{stem}_{scenario_name}_{index}.log")): index
            for index, group in enumerate(groups)
        }
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in pending:
            future.cancel()
        for future in done:
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(str(exc)) from exc
            outputs[futures[future]] = future.result()
    return [outputs[index] for index in range(len(groups))]


def run_pypsa_opt(input_path, result_dir, global_settings, *, log_dir, solver=None, concurrency=None,
                  buildings_per_model: int = BUILDINGS_PER_MODEL):
    """Run Step 3 with the PyPSA optimizer for one input; return the result HDF5 path.

    Same contract as ``urbs.runfunctions.run_lvds_opt``: the input is copied to
    ``<result>.partial``, the outputs are appended and the file is renamed on
    success (removed on failure). ``global_settings['n_cpu']`` (the urbs
    cluster count) is not used: every model holds ``buildings_per_model``
    buildings.
    """
    solver_name = resolve_solver_name(solver)
    linopy_solver(solver_name)
    start = time.time()
    data, mode, scenario_name = load_and_prepare(input_path, global_settings, solver_name)
    if mode["tsam"] or global_settings.get("reduce_only"):
        raise NotImplementedError(
            "The PyPSA optimizer supports the full-year chronological reference only "
            "(no time series aggregation, no --reduce-only)."
        )
    print(f"Preprocessing took {(time.time() - start) / 60:.2f} minutes.", flush=True)
    data, tsam_data = apply_temporal_method(data, mode, global_settings)
    audit = temporal_audit(mode, data, global_settings)

    final_path = final_result_path(result_dir, global_settings["input_file"], scenario_name)
    partial_path = final_path.with_name(final_path.name + ".partial")
    try:
        shutil.copyfile(input_path, partial_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            with pd.HDFStore(partial_path, mode="a", **HDF_OPTIONS) as store:
                for name, table in tsam_data.items():
                    store["urbs_out/tsam/" + name] = table
                store["urbs_out/temporal_method"] = audit
        groups = building_groups(data, buildings_per_model)
        start = time.time()
        outputs = solve_groups(
            data, mode, groups,
            solver_name=solver_name,
            log_dir=Path(log_dir) / solver_log_folder(linopy_solver(solver_name)),
            stem=global_settings["input_file"][:-3],
            scenario_name=scenario_name,
            concurrency=concurrency,
        )
        print(f"Solving took {(time.time() - start) / 60:.2f} minutes.", flush=True)
        start = time.time()
        results = merge_cluster_results([output["results"] for output in outputs])
        solver_audit = pd.DataFrame([output["audit"] for output in outputs])
        save(data, results, partial_path, solver_audit=solver_audit)
        print(f"Saving results took {(time.time() - start) / 60:.2f} minutes.", flush=True)
        os.replace(partial_path, final_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise
    return final_path
