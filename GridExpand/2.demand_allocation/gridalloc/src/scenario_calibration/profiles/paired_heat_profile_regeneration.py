"""Regenerate exact Step 2 heat profiles required by a paired scenario."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

import pandas as pd


from ..paths import (
    GRIDALLOC_DIR,
    SYNTHETIC_INPUT_DIR,
    configured_pylovo_version_id,
)

DEFAULT_SYNTHETIC_LIBRARY = SYNTHETIC_INPUT_DIR


def pending_sources(catalog: pd.DataFrame) -> list[str]:
    """Return unique source HDFs containing non-publication heat profiles."""
    required = {"publication_ready", "exact_source_hdf"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Heat profile catalog misses columns: {sorted(missing)}")
    pending = catalog.loc[
        ~catalog["publication_ready"].astype(bool),
        "exact_source_hdf",
    ].dropna()
    return sorted(pending.astype(str).unique())


def required_buses(catalog: pd.DataFrame, source_name: str) -> list[int]:
    rows = catalog[
        catalog["exact_source_hdf"].astype(str).eq(str(source_name))
        & ~catalog["publication_ready"].astype(bool)
    ]
    return sorted(
        pd.to_numeric(rows["exact_source_bus"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )


def validate_exact_profiles(source_hdf: Path, buses: list[int]) -> None:
    """Require heat demand and COP columns for every affected building bus."""
    demand_columns = pd.read_hdf(
        source_hdf,
        key="urbs_in/demand",
        start=0,
        stop=1,
    ).columns
    efficiency_columns = pd.read_hdf(
        source_hdf,
        key="urbs_in/eff_factor",
        start=0,
        stop=1,
    ).columns
    missing = []
    for bus in buses:
        for commodity in ("space_heat", "water_heat"):
            if (bus, commodity) not in demand_columns:
                missing.append(f"demand:{bus}:{commodity}")
        if (bus, "heatpump_air") not in efficiency_columns:
            missing.append(f"eff_factor:{bus}:heatpump_air")
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Regenerated {source_hdf.name} still misses {len(missing)} "
            f"required columns: {preview}"
        )


class _StatusWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.lock = threading.Lock()
        if output_path.exists():
            self.rows = pd.read_csv(output_path).to_dict("records")
        else:
            self.rows: list[dict[str, Any]] = []

    def completed(self) -> set[str]:
        return {
            str(row["source_hdf"]) for row in self.rows if row.get("status") == "done"
        }

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.rows = [
                existing
                for existing in self.rows
                if str(existing["source_hdf"]) != str(row["source_hdf"])
            ]
            self.rows.append(row)
            pd.DataFrame(self.rows).sort_values("source_hdf").to_csv(
                self.output_path,
                index=False,
            )


def _regenerate_one(
    *,
    source_name: str,
    buses: list[int],
    pylovo_version_id: str,
    n_cpu: int,
    result_dir: Path,
    synthetic_library: Path,
    log_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    log_path = log_dir / f"{Path(source_name).stem}.log"
    command = [
        "uv",
        "run",
        "--project",
        "..",
        "python",
        "main.py",
        source_name,
        "--storage",
        "db",
        "--pylovo-version-id",
        str(pylovo_version_id),
        "--profiles",
        "electricity_heat",
        "--demand-scope",
        "all",
        "--mobility-source",
        "pool",
        "--timeseries-storage",
        "temp",
        "--timeframe-mode",
        "full_year",
        "--n_cpu",
        str(n_cpu),
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"COMMAND: {' '.join(command)}\n")
            completed = subprocess.run(
                command,
                cwd=GRIDALLOC_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Step 2 returned {completed.returncode}")
        result_hdf = result_dir / source_name
        if not result_hdf.exists():
            raise FileNotFoundError(f"Missing Step 2 result {result_hdf}")
        validate_exact_profiles(result_hdf, buses)
        target = synthetic_library / source_name
        shutil.copy2(result_hdf, target)
        validate_exact_profiles(target, buses)
        return {
            "source_hdf": source_name,
            "status": "done",
            "required_buses": len(buses),
            "seconds": round(time.monotonic() - started, 1),
            "log_file": str(log_path),
            "error": "",
        }
    except Exception as exc:  # Keep the regional batch running.
        return {
            "source_hdf": source_name,
            "status": "failed",
            "required_buses": len(buses),
            "seconds": round(time.monotonic() - started, 1),
            "log_file": str(log_path),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-cpu", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--synthetic-library",
        type=Path,
        default=DEFAULT_SYNTHETIC_LIBRARY,
    )
    args = parser.parse_args()
    if args.workers < 1 or args.n_cpu < 1:
        parser.error("--workers and --n-cpu must be positive integers")
    pylovo_version_id = configured_pylovo_version_id()

    paired_dir = args.paired_dir.resolve()
    catalog = pd.read_csv(paired_dir / "paired_heat_profile_catalog.csv")
    sources = pending_sources(catalog)
    status = _StatusWriter(paired_dir / "paired_heat_profile_regeneration_status.csv")
    if args.resume:
        sources = [source for source in sources if source not in status.completed()]
    if args.limit is not None:
        sources = sources[: args.limit]

    synthetic_library = args.synthetic_library.resolve()
    synthetic_library.mkdir(parents=True, exist_ok=True)
    result_dir = GRIDALLOC_DIR / "results"
    log_dir = paired_dir / "heat_profile_regeneration_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "batch_start",
                "sources": len(sources),
                "workers": args.workers,
                "n_cpu": args.n_cpu,
                "pylovo_version_id": pylovo_version_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source_name in sources:
            future = pool.submit(
                _regenerate_one,
                source_name=source_name,
                buses=required_buses(catalog, source_name),
                pylovo_version_id=pylovo_version_id,
                n_cpu=args.n_cpu,
                result_dir=result_dir,
                synthetic_library=synthetic_library,
                log_dir=log_dir,
            )
            futures[future] = source_name
        for future in as_completed(futures):
            row = future.result()
            row["finished_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            status.write(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    failures = [row for row in status.rows if row.get("status") == "failed"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
