"""ÜZW real-grid adapter for paired validation (aligned datasets only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.orchestration import StatusLog

from .swf import load_real_jobs, run_real_powerflows

TARGET_NETWORK = "real_uzw"


def load_jobs(paired_dir: Path, target_grid_id: int | None) -> list[dict[str, Any]]:
    return load_real_jobs(paired_dir, target_grid_id, TARGET_NETWORK)


def input_name(job: dict[str, Any], scenario_label: str) -> str:
    return f"paired_real_uzw_area_{int(job['target_grid_id']):04d}_{scenario_label}.h5"


def run_powerflows(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    result_hdf: Path,
    step4_dir: Path,
    log_path: Path,
    status: StatusLog,
) -> None:
    run_real_powerflows(
        job=job, args=args, result_hdf=result_hdf, step4_dir=step4_dir,
        log_path=log_path, status=status,
        target_network=TARGET_NETWORK, provider="uzw",
    )
