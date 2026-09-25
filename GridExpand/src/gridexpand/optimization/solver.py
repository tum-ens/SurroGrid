"""Solver selection, options and provenance for the Step 3 cluster models.

The default ``gurobi`` is Pyomo's LP-file interface to gurobipy (``_gurobi_file``)
with the options GridExpand has always used. ``appsi_highs`` needs no licence (open
Docker images, the UI demo). The heuristic model case is a degenerate LP and the
optimized case a MIP stopped at a 5 % gap, so another solver (or interface) can
return a different optimal vertex or MIP solution: treat the solver as part of the
scenario and read it from ``urbs_out/solver_audit``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

SOLVER_ENV = "GRIDEXPAND_SOLVER"
DEFAULT_SOLVER = "gurobi"
SUPPORTED_SOLVERS = ("gurobi", "appsi_highs")

# Gurobi parameters, applied in this order (logfile first, as always).
GUROBI_OPTIONS = (
    ("Method", 4),  # deterministic concurrent
    ("MIPFocus", 2),  # optimality focus
    ("MIPGap", 0.05),
    ("Presolve", 2),
    ("Threads", 4),
)
HIGHS_OPTIONS = (("mip_rel_gap", 0.05),)


def resolve_solver_name(requested: str | None = None) -> str:
    """Solver name from the CLI value, else ``$GRIDEXPAND_SOLVER``, else ``gurobi``."""
    name = (requested or os.environ.get(SOLVER_ENV, "") or DEFAULT_SOLVER).strip()
    if name not in SUPPORTED_SOLVERS:
        raise ValueError(
            f"Unsupported Step 3 solver {name!r}; choose one of {SUPPORTED_SOLVERS}."
        )
    return name


def make_solver(name: str, logfile: str):
    """Create the Pyomo solver ``name`` with GridExpand's options and a log file."""
    from pyomo.environ import SolverFactory

    if name == "gurobi":
        solver = SolverFactory("gurobi")
        solver.set_options(f"logfile={logfile}")
        for key, value in GUROBI_OPTIONS:
            solver.set_options(f"{key}={value}")
        return solver
    if name == "appsi_highs":
        solver = SolverFactory("appsi_highs")
        solver.config.logfile = str(logfile)
        solver.options = dict(HIGHS_OPTIONS)
        return solver
    raise ValueError(f"Unsupported Step 3 solver {name!r}.")


def solver_log_folder(name: str) -> str:
    """Sub-folder of the Step 3 log directory (``gurobi`` as before)."""
    return "gurobi" if name == "gurobi" else name


def solver_options(name: str) -> dict[str, Any]:
    """The options that ``make_solver`` sets (without the log file)."""
    return dict(GUROBI_OPTIONS if name == "gurobi" else HIGHS_OPTIONS)


def solver_version(solver) -> str:
    try:
        version = solver.version()
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def relative_gap(objective: float | None, bound: float | None) -> float | None:
    """Gurobi's MIP gap definition ``|objective - bound| / |objective|``."""
    if objective is None or bound is None:
        return None
    return abs(objective - bound) / max(abs(objective), 1e-10)


def audit_record(name: str, solver, result) -> dict[str, Any]:
    """Provenance of one solve: solver, version, options, termination, objective, gap."""
    objective = _finite(getattr(result.problem, "upper_bound", None))
    bound = _finite(getattr(result.problem, "lower_bound", None))
    return {
        "solver": name,
        "solver_interface": type(solver).__name__,
        "solver_version": solver_version(solver),
        "solver_options": json.dumps(solver_options(name), sort_keys=True),
        "solver_status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition),
        "objective": objective,
        "best_bound": bound,
        "mip_gap": relative_gap(objective, bound),
    }


def summarize_audit(audit) -> dict[str, Any]:
    """Compact, deterministic run-level summary of a ``urbs_out/solver_audit`` table."""
    if audit is None or len(audit) == 0:
        return {}
    first = audit.iloc[0]
    gaps = [value for value in audit["mip_gap"] if value is not None and math.isfinite(value)]
    return {
        "optimization_solver": str(first["solver"]),
        "optimization_solver_version": str(first["solver_version"]),
        "optimization_solver_options": json.loads(first["solver_options"]),
        "optimization_partitions": int(len(audit)),
        "optimization_termination": sorted({str(v) for v in audit["termination_condition"]}),
        "optimization_max_mip_gap": max(gaps) if gaps else None,
    }
