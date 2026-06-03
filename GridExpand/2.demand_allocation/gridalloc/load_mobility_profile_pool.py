#!/usr/bin/env python3
"""Load the mobility profile pool CSV artifact into the SurroGrid schema."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

GRIDALLOC_DIR = Path(__file__).resolve().parent
GRIDEXPAND_DIR = GRIDALLOC_DIR.parents[1]
os.chdir(GRIDALLOC_DIR)
if str(GRIDALLOC_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDALLOC_DIR))
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from config import config
from database import SurroGridDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load mobility profile pool CSVs into SurroGrid DB tables.")
    parser.add_argument("--metadata-csv", default=config.MOBILITY_PROFILE_POOL_METADATA_PATH)
    parser.add_argument("--demand-csv", default=config.MOBILITY_PROFILE_POOL_DEMAND_PATH)
    parser.add_argument("--availability-csv", default=config.MOBILITY_PROFILE_POOL_AVAILABILITY_PATH)
    parser.add_argument("--replace", action="store_true", help="Clear existing DB pool rows before loading CSVs.")
    parser.add_argument("--chunksize", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = SurroGridDatabase()
    db.load_mobility_profile_pool_csv(
        args.metadata_csv,
        args.demand_csv,
        args.availability_csv,
        replace=args.replace,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
