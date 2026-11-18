"""The buildings of selected pylovo grids and their buses: the supply of a synthetic scenario dataset.

The selection is the one of ``gridexpand synthetic``: the candidate grids of an AGS (numbered by
``(plz, kcid, bcid)``, at least ``min_buildings`` buildings), optionally one postcode or one grid, then a candidate
range (``start_index``, ``limit``). Every selected grid is registered as a ``surrogrid.grid_case`` (its id is the
grid number of the dataset, the runner and the power flow); every building of the grid gets the bus pylovo connects
it to.

Outputs: ``building_supply.csv`` (``building_objectid, grid_id`` = grid case id, ``bus``, the bridge name and the
pylovo keys of the grid) and ``grids.csv`` (one row per selected grid). The dataset builder reads the supply with
``--network synthetic`` (:mod:`gridexpand.grids.dataset`).

``uv run python -m gridexpand.grids.pylovo --version <id> --ags <ags> [--plz ...] --output <dir>``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SUPPLY_COLUMNS = ["building_objectid", "grid_id", "bus", "synthetic_bridge_filename", "synthetic_kcid",
                  "synthetic_bcid", "pylovo_grid_result_id", "candidate_index", "plz"]


def select_grids(engine, *, version: str, ags: int, plz: int | None = None, kcid: int | None = None,
                 bcid: int | None = None, min_buildings: int = 5, start_index: int | None = None,
                 limit: int | None = None) -> list[dict]:
    """The candidate grids of the region, numbered as in ``gridexpand synthetic``."""
    from gridexpand.db.grids import list_grid_candidates

    candidates = list_grid_candidates(engine, ags, min_buildings=min_buildings, pylovo_version_id=str(version))
    if plz is not None:
        candidates = [c for c in candidates if int(c["plz"]) == int(plz)]
    if kcid is not None:
        candidates = [c for c in candidates if (int(c["kcid"]), int(c["bcid"])) == (int(kcid), int(bcid))]
    if start_index is not None:
        candidates = [c for c in candidates if int(c["candidate_index"]) >= int(start_index)]
    if limit is not None:
        candidates = candidates[: int(limit)]
    return candidates


def building_supply(engine, grids: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(supply, grids)``: every building of the grids with its bus; the grids with their grid case ids."""
    from gridexpand.db.grids import get_or_create_grid_cases, read_buildings

    case_ids = get_or_create_grid_cases(engine, grids)
    frames = []
    for grid, case_id in zip(grids, case_ids):
        buildings = read_buildings(engine, grid)
        if buildings["bus"].isna().any():
            raise ValueError(f"pylovo grid {grid['bridge_filename']} has buildings without a bus.")
        frames.append(pd.DataFrame({
            "building_objectid": buildings["objectid"].astype(str),
            "grid_id": int(case_id),
            "bus": pd.to_numeric(buildings["bus"]).astype(int),
            "synthetic_bridge_filename": grid["bridge_filename"],
            "synthetic_kcid": int(grid["kcid"]),
            "synthetic_bcid": int(grid["bcid"]),
            "pylovo_grid_result_id": int(grid["grid_result_id"]),
            "candidate_index": int(grid["candidate_index"]),
            "plz": int(grid["plz"]),
        }))
    supply = pd.concat(frames, ignore_index=True)[SUPPLY_COLUMNS]
    duplicated = supply.loc[supply["building_objectid"].duplicated(), "building_objectid"]
    if not duplicated.empty:
        raise ValueError(f"Buildings in two pylovo grids: {duplicated.tolist()[:5]}")
    table = pd.DataFrame(grids).assign(grid_case_id=case_ids)
    return supply, table


def main(argv: list[str] | None = None) -> int:
    from gridexpand.db.engine import get_engine, load_env

    parser = argparse.ArgumentParser(prog="python -m gridexpand.grids.pylovo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="pylovo version id")
    parser.add_argument("--ags", type=int, required=True)
    parser.add_argument("--plz", type=int)
    parser.add_argument("--kcid", type=int)
    parser.add_argument("--bcid", type=int)
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.kcid is None) != (args.bcid is None):
        parser.error("--kcid and --bcid go together")
    load_env()
    engine = get_engine()
    grids = select_grids(engine, version=args.version, ags=args.ags, plz=args.plz, kcid=args.kcid, bcid=args.bcid,
                         min_buildings=args.min_buildings, start_index=args.start_index, limit=args.limit)
    if not grids:
        raise SystemExit(f"No pylovo v{args.version} candidate grids for AGS {args.ags} (plz={args.plz}, "
                         f"kcid={args.kcid}, bcid={args.bcid}, min_buildings={args.min_buildings}).")
    supply, table = building_supply(engine, grids)
    args.output.mkdir(parents=True, exist_ok=True)
    supply.to_csv(args.output / "building_supply.csv", index=False)
    table.to_csv(args.output / "grids.csv", index=False)
    print(json.dumps({"grids": len(table), "buildings": len(supply)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
