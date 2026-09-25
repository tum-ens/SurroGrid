"""Command-line entry point: ``gridexpand <command> [args]``.

Each command forwards its remaining arguments to the ``main(argv)`` function of
one module; ``gridexpand <command> --help`` shows that module's options. The
modules are imported only when their command runs, and argument parsing always
happens before any database connection is opened.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module with main(argv), one-line description)
COMMANDS: dict[str, tuple[str, str]] = {
    "run": (
        "gridexpand.scenario.run_scenario",
        "Run one scenario or paired-validation run YAML (config/runs/*.yaml).",
    ),
    "run-aligned": (
        "gridexpand.scenario.run_aligned",
        "Prepare and run the aligned SWF/ÜZW paired comparison of one run YAML.",
    ),
    "synthetic": (
        "gridexpand.scenario.synthetic_ags_runner",
        "Run Steps 2-4 (+ expansion) for the synthetic grids of one AGS.",
    ),
    "allocate": (
        "gridexpand.allocation.main",
        "Step 2: allocate demands for one grid and write the urbs input HDF5.",
    ),
    "optimize": (
        "gridexpand.optimization.run_urbs_cluster",
        "Step 3: run the urbs optimization for one Step 2 HDF5 file.",
    ),
    "powerflow": (
        "gridexpand.powerflow.run_pwrflw",
        "Step 4: run the time-series power flow for one scenario HDF5 file.",
    ),
    "expansion": (
        "gridexpand.analysis.expansion.grid_expansion",
        "Step 5: materialize grid-expansion results from power-flow summaries.",
    ),
    "db": (
        "gridexpand.db.maintenance",
        "Database maintenance: init-schema, migrate, compress, relink-pylovo, delete-scenario.",
    ),
    "serve": (
        "gridexpand.service.cli",
        "Start the web service (job API, results, pylovo-ui plugin); needs the 'service' extra.",
    ),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: gridexpand <command> [args]",
        "",
        "GridExpand pipeline commands (gridexpand <command> --help for details):",
        "",
    ]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in COMMANDS.items()]
    lines += [
        "",
        "Directories: see gridexpand.paths (GRIDEXPAND_DATA_DIR, GRIDEXPAND_WORK_DIR,",
        "GRIDEXPAND_ENV_FILE relocate data, runtime artifacts and the .env file).",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv`` (default ``sys.argv[1:]``) to one command's ``main``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(_usage(), file=sys.stderr)
        print(f"\ngridexpand: unknown command {command!r}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f"gridexpand {command}", *rest]
    result = module.main(rest)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
