"""``gridexpand expansion summary``: the indicators of one expansion analysis (``overview.expansion_indicators``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand expansion summary",
        description=(
            "Reinforcement need, measures and cost of one expansion analysis: grids with measures, added cable km, "
            "transformer exchanges and kVA, station replacements, load transfers, new substations, cost breakdown."
        ),
    )
    parser.add_argument("analysis_key", nargs="?", help="Expansion analysis key (default: the newest analysis).")
    parser.add_argument(
        "--by-plz", action="store_true",
        help="One row per PLZ, with its population (pylovo.municipal_register) and the cost per inhabitant.",
    )
    parser.add_argument("--csv", type=Path, help="Write the table to this CSV file.")
    return parser


def print_indicators(frame: pd.DataFrame) -> None:
    """Print a total row as ``indicator: value`` lines, a PLZ table as a table."""
    if len(frame) == 1 and "plz" not in frame.columns:
        for column, value in frame.iloc[0].items():
            text = f"{value:,.2f}" if isinstance(value, float) else str(value)
            print(f"{column}: {text}")
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from gridexpand.analysis.expansion.overview import expansion_indicators

    frame = expansion_indicators(args.analysis_key, by_plz=args.by_plz)
    if args.csv is not None:
        frame.to_csv(args.csv, index=False)
        print(f"written: {args.csv}")
    print_indicators(frame)
