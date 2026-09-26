"""``gridexpand api``: start the GridExpand HTTP API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gridexpand.api.settings import DEFAULT_HOST, DEFAULT_PORT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand api",
        description="HTTP API of GridExpand for the GridPlanner UI: pipeline jobs (gridexpand run as "
                    "subprocesses), result queries and scenario files. The UI itself lives in GridPlanner.",
        epilog="The API starts pipeline jobs that write to the database of GridExpand's .env. It binds to "
               "127.0.0.1 by default and accepts only the Host headers 127.0.0.1:<port> and localhost:<port> "
               "(add a reverse proxy's host with --allowed-host). Environment: GRIDEXPAND_SOLVER, "
               "GRIDEXPAND_API_SCENARIO_DIRS, GRIDEXPAND_API_USER_SCENARIO_DIR, "
               "GRIDEXPAND_API_CORS_ORIGINS (development only).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"interface to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    parser.add_argument("--root-path", default="",
                        help="path prefix of a reverse proxy that strips it, e.g. /gridexpand (for the API docs)")
    parser.add_argument("--allowed-host", action="append", default=[], metavar="HOST[:PORT]",
                        help="additional accepted Host header, e.g. 127.0.0.1:18780 of the proxy (repeatable)")
    parser.add_argument("--scenario-dir", action="append", default=[], type=Path, metavar="DIR",
                        help="additional directory with scenario YAMLs (repeatable)")
    parser.add_argument("--user-scenario-dir", type=Path, default=None, metavar="DIR",
                        help="where the scenario editor saves new scenarios (default: "
                             "$GRIDEXPAND_API_USER_SCENARIO_DIR or WORK_DIR/scenarios)")
    parser.add_argument("--max-running-jobs", type=int, default=1,
                        help="pipeline jobs that run at the same time; later ones wait (default: 1)")
    parser.add_argument("--log-level", default="warning",
                        choices=["critical", "error", "warning", "info", "debug"],
                        help="uvicorn log level (default: warning)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, then start uvicorn with the service app."""
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        print("gridexpand api needs the 'api' extra: uv sync --extra api", file=sys.stderr)
        return 1

    from gridexpand.api.app import create_app
    from gridexpand.api.settings import ServiceSettings

    bind_all = args.host in ("0.0.0.0", "::")
    try:
        base = ServiceSettings.from_env(user_scenario_dir=args.user_scenario_dir)
        settings = ServiceSettings.from_env(
            host=args.host, port=args.port, root_path=args.root_path.rstrip("/"),
            allowed_hosts=frozenset(args.allowed_host), allow_any_host=bind_all,
            scenario_dirs=base.shipped_scenario_dirs + tuple(p.expanduser().resolve() for p in args.scenario_dir),
            user_scenario_dir=args.user_scenario_dir,
            max_running_jobs=max(1, args.max_running_jobs),
        )
    except ValueError as exc:
        print(f"gridexpand api: {exc}", file=sys.stderr)
        return 2
    app = create_app(settings)
    if bind_all:
        print("WARNING: the GridExpand service is reachable from other machines and can start pipeline jobs.",
              file=sys.stderr)
    shown = "127.0.0.1" if bind_all else args.host
    print(f"GridExpand service → http://{shown}:{args.port}/  (API docs: /docs, plugin: /ui/manifest.json)",
          flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level,
                                           timeout_graceful_shutdown=3, proxy_headers=False))
    app.state.server = server  # SSE streams end as soon as the server is asked to stop
    server.run()
    return 0
