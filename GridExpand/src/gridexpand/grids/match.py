"""``gridexpand match``: match the connection points of given grid files to InfDB buildings.

Reads the grids (pandapower JSON, one file per LV grid), takes their loads as connection points (or the points of
``--points``), reads the buildings around them from InfDB (read-only), applies the rules of
:mod:`gridexpand.grids.matching` and writes to ``--output``:

- ``point_matches.csv``: one row per point with the deciding rule and the supplied building;
- ``building_supply.csv``: one row per supplied building with its grid and bus;
- ``unconnected_buildings.csv``: demand buildings near the grid lines that no point supplies, with the reason;
- ``grid_quality.csv``: per grid, its points by evidence;
- ``matching_report.json``: the configuration, the inputs and the counts per rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from gridexpand.grids import sources
from gridexpand.grids.matching import MatchingConfig, MatchResult, match_points

BuildingReader = Callable[[tuple[float, float, float, float]], tuple[pd.DataFrame, dict[str, Any]]]


def load_config(path: Path | None) -> MatchingConfig:
    """The ``matching:`` block of a YAML file (a run YAML or a small file); defaults without a file."""
    if path is None:
        return MatchingConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "matching" not in raw:
        raise ValueError(f"{path} has no matching: block.")
    return MatchingConfig.from_dict(raw["matching"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand match",
        description="Match the connection points of given LV grids to InfDB buildings (read-only database "
                    "access). Rules and outputs: docs/matching.md.",
    )
    parser.add_argument("grids", nargs="+", type=Path,
                        help="pandapower JSON files or directories of them; one file per LV grid, the file stem "
                             "is the grid id")
    parser.add_argument("--output", type=Path, required=True, help="directory for the CSVs and the report")
    parser.add_argument("--points", type=Path, default=None,
                        help="CSV of connection points (grid_id, bus, x, y; optional point_id, address, remark, "
                             "on_building, exclude, exclude_reason); default: the loads of each grid")
    parser.add_argument("--crs", default=sources.TARGET_CRS,
                        help=f"CRS of the grid and point coordinates (default {sources.TARGET_CRS})")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file with a matching: block (rules and distances); default: the built-in rules")
    return parser


def collect_inputs(grids: list[Path], points_csv: Path | None, crs: str
                   ) -> tuple[pd.DataFrame, dict[str, list]]:
    """Connection points and line geometries of the grids, in EPSG:25832."""
    import pandapower as pp

    nets = {grid_id: pp.from_json(str(path)) for grid_id, path in sources.grid_files(grids).items()}
    lines = {grid_id: sources.lines_from_net(net) for grid_id, net in nets.items()}
    if points_csv is not None:
        points = sources.points_from_csv(points_csv)
        unknown = sorted(set(points["grid_id"]) - set(nets))
        if unknown:
            raise ValueError(f"{points_csv} names grids without a grid file: {unknown[:5]}")
        for grid_id, group in points.groupby("grid_id"):
            missing = sorted(set(group["bus"].astype(int)) - set(nets[grid_id].bus.index.astype(int)))
            if missing:
                raise ValueError(f"{points_csv}: grid {grid_id} has no buses {missing[:5]}.")
    else:
        points = pd.concat([sources.points_from_net(net, grid_id) for grid_id, net in nets.items()],
                           ignore_index=True)
    if points.empty:
        raise ValueError("The grids hold no connection points (no loads and no --points CSV).")
    return sources.to_target_crs(points, lines, crs)


def _box(points: pd.DataFrame, lines: dict[str, list], margin: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = sources.bounds(points, margin)
    for items in lines.values():
        for line in items:
            left, bottom, right, top = line.bounds
            xmin, ymin = min(xmin, left - margin), min(ymin, bottom - margin)
            xmax, ymax = max(xmax, right + margin), max(ymax, top + margin)
    return xmin, ymin, xmax, ymax


def _database_reader() -> BuildingReader:
    from gridexpand.db.engine import get_engine

    def read(box: tuple[float, float, float, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
        return sources.read_buildings(get_engine(), box)

    return read


def run(args: argparse.Namespace, *, read_buildings: BuildingReader | None = None) -> MatchResult:
    """Match and write the outputs; ``read_buildings`` replaces the database (tests)."""
    config = load_config(args.config)
    points, lines = collect_inputs(args.grids, args.points, args.crs)
    if read_buildings is None:
        from gridexpand.db.engine import load_env

        load_env()
        read_buildings = _database_reader()
    buildings, note = read_buildings(_box(points, lines, max(config.candidate_radius_m, config.unconnected_radius_m)))
    if buildings.empty:
        raise ValueError("No InfDB buildings around the connection points (check --crs and the database).")
    result = match_points(points, buildings, config, grid_lines=lines)
    result.report["inputs"] = {"grids": sorted(lines), "points_csv": None if args.points is None else str(args.points),
                               "crs": args.crs, **note}
    write_outputs(result, args.output)
    return result


def write_outputs(result: MatchResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result.points.to_csv(output / "point_matches.csv", index=False)
    result.supply.to_csv(output / "building_supply.csv", index=False)
    result.unconnected.to_csv(output / "unconnected_buildings.csv", index=False)
    result.quality.to_csv(output / "grid_quality.csv", index=False)
    (output / "matching_report.json").write_text(json.dumps(result.report, indent=2, default=str) + "\n",
                                                 encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    report = result.report
    print(f"{report['points']} connection points in {report['grids']} grids: {report['matched_points']} matched, "
          f"{report['supplied_buildings']} buildings supplied.")
    for rule, count in report["rules"].items():
        print(f"  {count:6d}  {rule}")
    if report["unconnected_buildings"]:
        print("Buildings near the lines without a point:",
              ", ".join(f"{count} {reason}" for reason, count in report["unconnected_buildings"].items()))
    print(f"Written to {args.output}")
    return 0
