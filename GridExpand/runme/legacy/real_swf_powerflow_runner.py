#!/usr/bin/env python3
"""Run compact HH-only backbone real SWF power-flow summaries for the Forchheim comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
STEP4_DIR = REPO_ROOT / "GridExpand" / "4.powerflow"
DEFAULT_RUN_NAME = "real_hybrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real SWF HH-only backbone compact pre-powerflow summaries for PLZ 91301."
    )
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument(
        "--annual-demand-mode",
        choices=["synthetic", "measured"],
        default="synthetic",
        help="Use synthetic sampled annual HH demand or parsed SWF annual kWh values where available.",
    )
    parser.add_argument(
        "--measured-profile-selection",
        choices=["closest", "random_band"],
        default="random_band",
        help="Profile-shape selection used only with --annual-demand-mode measured.",
    )
    parser.add_argument("--measured-profile-band-pct", type=float, default=10.0)
    parser.add_argument("--measured-profile-min-candidates", type=int, default=10)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--lv-id",
        default=None,
        help="Optional single LV id for a pilot run, e.g. 28 or LV_028.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        "uv",
        "run",
        "python",
        "run_real_swf_powerflow.py",
        "--plz",
        str(args.plz),
        "--workers",
        str(args.workers),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
        "--annual-demand-mode",
        args.annual_demand_mode,
        "--measured-profile-selection",
        args.measured_profile_selection,
        "--measured-profile-band-pct",
        str(args.measured_profile_band_pct),
        "--measured-profile-min-candidates",
        str(args.measured_profile_min_candidates),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.lv_id is not None:
        cmd.extend(["--lv-id", str(args.lv_id)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=STEP4_DIR, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
