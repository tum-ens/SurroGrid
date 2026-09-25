"""Checks of the runtime environment: solvers, large input files, disk space."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from gridexpand.paths import STATISTICS_DIR

SOLVER_CHECK_TTL_S = 600.0
SOLVER_CHECK_TIMEOUT_S = 40.0

# Untracked large inputs of the synthetic pipeline (see GridExpand/.gitignore).
DATA_ASSETS = (
    ("elec_lps.h5", STATISTICS_DIR / "inhabited_buildings" / "elec_lps.h5",
     "household electricity load profiles (Step 2, every case)"),
    ("mobility_demand_pool.csv", STATISTICS_DIR / "general" / "mobility_profile_pool_old" / "mobility_demand_pool.csv",
     "synthetic EV demand pool (Step 2, post cases)"),
    ("mobility_availability_pool.csv",
     STATISTICS_DIR / "general" / "mobility_profile_pool_old" / "mobility_availability_pool.csv",
     "synthetic EV availability pool (Step 2, post cases)"),
    ("mobility_profile_pool/", STATISTICS_DIR / "general" / "mobility_profile_pool",
     "EV session pool v2 (paired validation only)"),
)

# Licence check in a child process: a missing or broken licence must not block the server.
# 2500 variables exceed the size-limited licence bundled with the gurobipy wheel (2000),
# which would otherwise pass the check but fail on every real urbs model.
_GUROBI_CHECK = """
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
model = gp.Model(env=env)
model.addVars(2500, ub=1.0, obj=-1.0)
model.optimize()
print("gurobi", ".".join(map(str, gp.gurobi.version())), "status", model.Status)
"""

_solver_cache: dict[str, Any] = {}
_solver_lock = threading.Lock()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_gurobi() -> dict[str, Any]:
    """Whether a Gurobi licence is usable by the jobs (result cached for 10 minutes)."""
    from gridexpand.common.orchestration import command_env

    with _solver_lock:
        cached = _solver_cache.get("gurobi")
        if cached and time.time() - cached["checked_at"] < SOLVER_CHECK_TTL_S:
            return cached
        info: dict[str, Any] = {"installed": importlib.util.find_spec("gurobipy") is not None,
                                "package_version": _package_version("gurobipy"), "usable": False, "detail": None}
        if info["installed"]:
            try:
                result = subprocess.run([sys.executable, "-c", _GUROBI_CHECK], env=command_env(), capture_output=True,
                                        text=True, timeout=SOLVER_CHECK_TIMEOUT_S, check=False)
                lines = (result.stdout + result.stderr).strip().splitlines()
                info["usable"] = result.returncode == 0 and bool(lines) and lines[-1].startswith("gurobi ")
                info["detail"] = lines[-1][:300] if lines else f"exit code {result.returncode}"
            except subprocess.TimeoutExpired:
                info["detail"] = f"licence check timed out after {SOLVER_CHECK_TIMEOUT_S:.0f} s"
        else:
            info["detail"] = "gurobipy is not installed"
        info["checked_at"] = time.time()
        _solver_cache["gurobi"] = info
        return info


def check_highs() -> dict[str, Any]:
    """Whether HiGHS (``highspy``) is importable."""
    installed = importlib.util.find_spec("highspy") is not None
    return {"installed": installed, "package_version": _package_version("highspy"), "usable": installed}


def solver_status(configured: str) -> dict[str, Any]:
    """Solver availability and whether post cases (which need Step 3) can run."""
    gurobi, highs = check_gurobi(), check_highs()
    name = configured.lower()
    if name.startswith("gurobi"):
        usable, reason = gurobi["usable"], None if gurobi["usable"] else f"Gurobi is not usable: {gurobi['detail']}"
    elif "highs" in name:
        usable, reason = highs["usable"], None if highs["usable"] else "highspy is not installed"
    else:
        usable, reason = False, f"unknown solver '{configured}'"
    return {"configured": configured, "gurobi": gurobi, "highs": highs, "post_cases_supported": usable,
            "post_cases_reason": reason}


def data_assets() -> list[dict[str, Any]]:
    """Presence and size of the untracked large input files."""
    out = []
    for name, path, purpose in DATA_ASSETS:
        present = path.exists()
        size = None
        if present and path.is_file():
            size = round(path.stat().st_size / 1e6, 1)
        elif present:
            size = round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6, 1)
        out.append({"name": name, "path": str(path), "present": present, "size_mb": size, "purpose": purpose})
    return out


def disk_usage(path: Path) -> dict[str, Any]:
    """Free and total space of the file system that holds ``path``."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {"path": str(path), "free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
