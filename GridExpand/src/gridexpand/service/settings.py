"""Runtime settings of the GridExpand service."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from gridexpand.paths import RUNS_DIR, SCENARIO_CONFIG_DIR, WORK_DIR

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18766

# Environment variables read by the service (all optional).
ENV_CORS_ORIGINS = "GRIDEXPAND_UI_CORS_ORIGINS"  # development only: comma-separated origins
ENV_SCENARIO_DIRS = "GRIDEXPAND_SERVICE_SCENARIO_DIRS"  # extra scenario directories (os.pathsep)
ENV_USER_SCENARIO_DIR = "GRIDEXPAND_SERVICE_USER_SCENARIO_DIR"  # where the scenario editor saves files
ENV_SOLVER = "GRIDEXPAND_SOLVER"  # solver of Step 3 (default: gurobi)


def _split(value: str | None, sep: str = ",") -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(sep) if item.strip())


@dataclass(frozen=True)
class ServiceSettings:
    """Everything the service needs to know about its environment.

    Attributes:
        host: Interface the server binds to.
        port: TCP port.
        root_path: Path prefix under which a reverse proxy publishes the service
            (for example ``/gridexpand``); only used for generated URLs (OpenAPI docs).
        allowed_hosts: ``Host`` header values accepted besides ``127.0.0.1:<port>`` and
            ``localhost:<port>`` (for example the proxy's ``127.0.0.1:18780``).
        allow_any_host: Accept every ``Host`` header (only for binding to all interfaces).
        cors_origins: Origins allowed to call the API from another origin (development only).
        scenario_dirs: Directories whose ``*.yaml`` files are offered as scenarios; the user scenario
            directory is always added as the last one (the first directory wins on equal names).
        user_scenario_dir: Directory of the scenarios saved with the scenario editor (created on demand).
        state_dir: Private directory of the service (job metadata and logs).
        runs_dir: Parent of the per-job run directories (``<runs_dir>/<job id>/<case>``).
        python: Interpreter that runs the ``gridexpand`` jobs.
        solver: Solver name for Step 3 (passed to the jobs as ``GRIDEXPAND_SOLVER``).
        max_running_jobs: Pipeline jobs that may run at the same time; the rest wait.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    root_path: str = ""
    allowed_hosts: frozenset[str] = frozenset()
    allow_any_host: bool = False
    cors_origins: tuple[str, ...] = ()
    scenario_dirs: tuple[Path, ...] = (SCENARIO_CONFIG_DIR,)
    user_scenario_dir: Path = WORK_DIR / "scenarios"
    state_dir: Path = WORK_DIR / "service"
    runs_dir: Path = RUNS_DIR
    python: str = field(default_factory=lambda: sys.executable)
    solver: str = "gurobi"
    max_running_jobs: int = 1

    def __post_init__(self) -> None:
        user = Path(self.user_scenario_dir).expanduser().resolve()
        if user == SCENARIO_CONFIG_DIR.resolve():
            raise ValueError(f"The user scenario directory must not be the repository's {SCENARIO_CONFIG_DIR}")
        others = tuple(Path(d) for d in self.scenario_dirs if Path(d).expanduser().resolve() != user)
        object.__setattr__(self, "user_scenario_dir", user)
        object.__setattr__(self, "scenario_dirs", (*others, user))

    @property
    def shipped_scenario_dirs(self) -> tuple[Path, ...]:
        """The scenario directories except the user directory (never written by the service)."""
        return self.scenario_dirs[:-1]

    @property
    def jobs_dir(self) -> Path:
        """Job metadata (``<id>.json``) and logs (``<id>.log``)."""
        return self.state_dir / "jobs"

    def host_allowlist(self) -> set[str]:
        """Accepted ``Host`` header values (lower case, with port)."""
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"}
        return hosts | {host.lower() for host in self.allowed_hosts}

    @classmethod
    def from_env(cls, **overrides: object) -> ServiceSettings:
        """Settings from the environment, updated with explicit (CLI) values."""
        extra_dirs = tuple(Path(p).expanduser().resolve() for p in _split(os.getenv(ENV_SCENARIO_DIRS), os.pathsep))
        values: dict[str, object] = {
            "cors_origins": _split(os.getenv(ENV_CORS_ORIGINS)),
            "scenario_dirs": (SCENARIO_CONFIG_DIR, *extra_dirs),
            "user_scenario_dir": Path(os.getenv(ENV_USER_SCENARIO_DIR, "").strip() or WORK_DIR / "scenarios"),
            "solver": os.getenv(ENV_SOLVER, "").strip() or "gurobi",
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)  # type: ignore[arg-type]
