"""Command-line entry point: ``uv run python -m validation <command> [args]`` (from ``GridExpand/``).

Each command forwards its remaining arguments to the ``main(argv)`` function of one module; ``<command> --help``
shows that module's options. Argument parsing happens before any database connection is opened.
"""

from __future__ import annotations

import importlib
import sys

from . import export_pythonpath

# command -> (module with main(argv), one-line description)
COMMANDS: dict[str, tuple[str, str]] = {
    "run": (
        "validation.run",
        "Run one paired_aligned run YAML (validation/config/*.yaml): prepare, execute and postprocess.",
    ),
    "status-quo-test": (
        "validation.status_quo",
        "Status-quo power flow of the real grids at alignment; writes the exclusion list per provider.",
    ),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: python -m validation <command> [args]", "", "Validation commands:", ""]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in COMMANDS.items()]
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
        print(f"\npython -m validation: unknown command {command!r}", file=sys.stderr)
        return 2
    export_pythonpath()
    entry = importlib.import_module(COMMANDS[command][0]).main
    sys.argv = [f"python -m validation {command}", *rest]
    return int(entry(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
