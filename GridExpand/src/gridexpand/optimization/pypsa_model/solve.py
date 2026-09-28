"""Build and solve one PyPSA building model and describe the solve (solver audit)."""

from __future__ import annotations

import json
import time
from importlib.metadata import version

from ..solver import relative_gap, solver_options
from .constraints import add_urbs_constraints
from .building_model import build_building_model
from .results import result_entities

# Step 3 solver names -> linopy solver names. The solver is handed the model as
# matrices (linopy's direct API, no LP file and no names, so polars is not used).
LINOPY_SOLVERS = {"gurobi": "gurobi", "appsi_highs": "highs"}
SOLVER_PACKAGES = {"gurobi": "gurobipy", "highs": "highspy"}
# With a log file the solver output goes there only (logging, not part of the solver options).
QUIET = {"gurobi": {"LogToConsole": 0}, "highs": {"log_to_console": False}}


def linopy_solver(solver_name: str) -> str:
    try:
        return LINOPY_SOLVERS[solver_name]
    except KeyError as exc:
        raise ValueError(f"The PyPSA optimizer does not support solver {solver_name!r}.") from exc


def _bound(model, solver: str, is_mip: bool, objective: float) -> float | None:
    """The solver's best bound (the objective for an LP)."""
    if not is_mip:
        return objective
    solver_model = model.solver_model
    try:
        if solver == "gurobi":
            return float(solver_model.ObjBound)
        return float(solver_model.getInfo().mip_dual_bound)
    except Exception:  # pragma: no cover - diagnostics only
        return None


def solve_network(data: dict, mode: dict, *, solver_name: str, log_file=None, env=None,
                  options: dict | None = None, index: int = 0):
    """Build and solve the model of one building group; return ``(network, parts, info)``.

    Args:
        data: urbs input dict of the group (``urbs.input.get_cluster_data``).
        mode: urbs features of the input (``urbs.identify.identify_mode``).
        solver_name: Step 3 solver name (``gurobi`` or ``appsi_highs``).
        log_file: solver log file.
        env: optional ``gurobipy.Env`` reused across models of one process.
        options: solver options (default: the Step 3 options of the solver).
        index: group number (messages only).

    Raises:
        RuntimeError: the solve did not end optimal (for a MIP: within the gap).
    """
    solver = linopy_solver(solver_name)
    options = dict(solver_options(solver_name) if options is None else options)
    start = time.time()
    network, parts = build_building_model(data, mode)
    network.optimize.create_model(include_objective_constant=False)
    add_urbs_constraints(network, parts)
    build_seconds = time.time() - start

    kwargs = {"solver_name": solver, "io_api": "direct", "set_names": False}
    if log_file is not None:
        kwargs["log_fn"] = str(log_file)
        kwargs.update(QUIET[solver])
    if solver == "gurobi" and env is not None:
        kwargs["env"] = env
    start = time.time()
    status, condition = network.model.solve(**kwargs, **options)
    solve_seconds = time.time() - start
    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Building group {index}: {solver} ended with status={status}, "
            f"termination={condition}; see {log_file}."
        )
    info = {"solver": solver, "options": options, "status": status, "condition": condition,
            "build_seconds": build_seconds, "solve_seconds": solve_seconds}
    return network, parts, info


def solve_group(data: dict, mode: dict, *, solver_name: str, log_file=None, env=None,
                options: dict | None = None, index: int = 0) -> dict:
    """Solve one building group (see ``solve_network``) and return its results and audit.

    Returns:
        ``{"results": {key: Series}, "audit": {...}}``; the audit has the columns of
        urbs' ``urbs_out/solver_audit`` plus ``optimizer`` and ``optimizer_version``.
    """
    import linopy
    import pypsa

    network, parts, info = solve_network(data, mode, solver_name=solver_name, log_file=log_file, env=env,
                                         options=options, index=index)
    solver = info["solver"]
    results = result_entities(network, parts)
    is_mip = network.model.type.startswith("MI")
    objective = float(network.model.objective.value)
    bound = _bound(network.model, solver, is_mip, objective)
    sites = sorted(str(site) for site in parts.processes["site"].unique())
    network.model.solver_model = None  # the solver's copy of the model
    del network
    audit = {
        "cluster": int(index),
        "n_sites": len(sites),
        "first_site": sites[0] if sites else "",
        "solver": solver,
        "solver_interface": "linopy-direct",
        "solver_version": version(SOLVER_PACKAGES[solver]),
        "solver_options": json.dumps(info["options"], sort_keys=True),
        "solver_status": info["status"],
        "termination_condition": info["condition"],
        "objective": objective,
        "best_bound": bound,
        "mip_gap": relative_gap(objective, bound),
        "build_seconds": float(info["build_seconds"]),
        "solve_seconds": float(info["solve_seconds"]),
        "optimizer": "pypsa",
        "optimizer_version": f"pypsa {pypsa.__version__}, linopy {linopy.__version__}",
    }
    return {"results": results, "audit": audit}
