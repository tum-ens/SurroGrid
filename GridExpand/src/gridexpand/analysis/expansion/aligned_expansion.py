"""Materialize expansion costs for every provider group of one aligned run.

An aligned run (``gridexpand run-aligned``) writes power-flow
summaries named ``{run_id}_{provider}_{network}_{case}`` with network
``real_<provider>`` or ``synthetic``. This entry point materializes the four
groups (SWF real, SWF synthetic, ÜZW real, ÜZW synthetic) under the analysis
keys ``{run_id}_{provider}_{real|synthetic}_{pre|post_inflex|post}``.

Example::

    uv run python -m gridexpand.analysis.expansion.aligned_expansion \\
        --run-id joint_2045_v1_smoke24 --providers swf uzw \\
        --cases pre post-inflex-heuristic post-hems-heuristic

All groups are materialized (one transaction each) and the QGIS views are
refreshed once at the end.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from gridexpand.db import refresh_qgis_views
from gridexpand.db.database import SurroGridDatabase

from . import grid_expansion

# DSO providers of an aligned run.
PROVIDERS = ("swf", "uzw")
# Model case -> (power-flow stage, analysis key suffix); the suffixes are the ones of
# ``gridexpand.run`` (run_scenario) and of ``notebook_workflow.scenario_analysis_keys``.
CASE_STAGES = {
    "pre": ("pre", "pre"),
    "post-inflex-heuristic": ("post", "post_inflex"),
    "post-hems-heuristic": ("post", "post"),
    "post-hems-optimized": ("post", "post_hems_optimized"),
}
DEFAULT_CASES = ("pre", "post-inflex-heuristic", "post-hems-heuristic")


def aligned_groups(
    run_id: str,
    *,
    providers: tuple[str, ...] = PROVIDERS,
    cases: tuple[str, ...] = DEFAULT_CASES,
    pylovo_version_id: str = "1",
    excluded_real_grids: dict[str, tuple[str, ...]] | None = None,
) -> list[list[str]]:
    """Return one ``grid_expansion`` argument list per provider, network and case."""
    excluded_real_grids = excluded_real_grids or {}
    commands = []
    for provider in providers:
        for network in ("real", "synthetic"):
            run_network = f"real_{provider}" if network == "real" else "synthetic"
            for case in cases:
                stage, analysis_suffix = CASE_STAGES[case]
                argv = [
                    "--run-name",
                    f"{run_id}_{provider}_{run_network}_{case}",
                    "--data-source",
                    run_network,
                    "--stage",
                    stage,
                    "--pylovo-version-id",
                    str(pylovo_version_id),
                    "--analysis-key",
                    f"{run_id}_{provider}_{network}_{analysis_suffix}",
                    "--note",
                    f"aligned run {run_id}; provider {provider}",
                    "--replace",
                ]
                # The run name already names the provider, so synthetic grids
                # need no postcode filter.
                if network == "real":
                    for lv_id in excluded_real_grids.get(provider, ()):
                        argv.extend(["--exclude-real-lv-id", str(lv_id)])
                commands.append(argv)
    return commands


def _matching_runs(db: SurroGridDatabase, args: argparse.Namespace) -> int:
    """Number of compact summaries one group would materialize (dry run)."""
    if args.data_source == "synthetic":
        # The run name names the provider; synthetic groups have no postcode filter.
        query = text(
            """
            SELECT COUNT(*)
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            JOIN surrogrid.powerflow_summary pfs USING (powerflow_run_id)
            WHERE pr.run_name = :run_name AND pfs.stage = :stage
              AND gc.pylovo_version_id = :pylovo_version_id
            """
        )
        params = {"pylovo_version_id": args.pylovo_version_id}
    else:
        query = text(
            """
            SELECT COUNT(*)
            FROM surrogrid.real_powerflow_run rpr
            JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
            JOIN surrogrid.real_powerflow_summary rps USING (real_powerflow_run_id)
            WHERE rpr.run_name = :run_name AND rps.stage = :stage
              AND rgc.source = :source
            """
        )
        params = {"source": args.data_source.removeprefix("real_")}
    with db.engine.connect() as conn:
        return int(
            conn.execute(
                query, {"run_name": args.run_name, "stage": args.stage, **params}
            ).scalar_one()
        )


def _parse_exclusions(values: list[str]) -> dict[str, tuple[str, ...]]:
    """``PROVIDER:ID`` options -> ids per provider."""
    excluded: dict[str, list[str]] = {}
    for value in values:
        provider, separator, lv_id = value.partition(":")
        if not separator or provider not in PROVIDERS or not lv_id:
            raise SystemExit(f"--exclude-real-grid expects PROVIDER:ID, got {value!r}.")
        excluded.setdefault(provider, []).append(lv_id)
    return {provider: tuple(ids) for provider, ids in excluded.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS)
    )
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASE_STAGES), default=list(DEFAULT_CASES)
    )
    parser.add_argument("--pylovo-version-id", default="1")
    parser.add_argument(
        "--exclude-real-grid",
        action="append",
        default=[],
        help="PROVIDER:ID real grid kept in coverage but excluded from costing; repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list each group's run name, analysis key and matching summaries.",
    )
    args = parser.parse_args(argv)

    expansion_parser = grid_expansion._build_parser()
    group_args = [
        expansion_parser.parse_args(group_argv)
        for group_argv in aligned_groups(
            args.run_id,
            providers=tuple(args.providers),
            cases=tuple(args.cases),
            pylovo_version_id=args.pylovo_version_id,
            excluded_real_grids=_parse_exclusions(args.exclude_real_grid),
        )
    ]
    db = SurroGridDatabase()
    if args.dry_run:
        for group in group_args:
            print(
                f"{group.analysis_key}: run_name={group.run_name} stage={group.stage} "
                f"summaries={_matching_runs(db, group)}"
            )
        return

    db.ensure_schema()
    failures = []
    for group in group_args:
        print(f"\n=== {group.analysis_key} ({group.run_name}, stage {group.stage})")
        try:
            grid_expansion.materialize(db, group, refresh_views=False)
        except Exception as exc:  # noqa: BLE001 - report every failed group, then fail
            failures.append((group.analysis_key, exc))
            print(f"FAILED: {exc}")
    refresh_qgis_views(db.engine)
    print("QGIS materialized views refreshed.")
    if failures:
        for analysis_key, exc in failures:
            print(f"failed: {analysis_key}: {exc}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} of {len(group_args)} expansion groups failed.")


if __name__ == "__main__":
    main()
