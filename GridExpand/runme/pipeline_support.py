"""Shared process helpers for GridExpand batch runners."""

from __future__ import annotations

from pathlib import Path

import synthetic_ags_pipeline_runner as _base

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = GRIDEXPAND_DIR / ".env"
REPO_ROOT_DEFAULT = GRIDEXPAND_DIR.parent
StatusLog = _base.StatusLog
utc_now = _base.utc_now


def run_command(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    status: StatusLog,
    job_index: int,
    stage: str,
    env_extra: dict[str, str] | None = None,
) -> None:
    _base.run_command(
        cmd=cmd,
        cwd=cwd,
        log_path=log_path,
        status=status,
        candidate_index=job_index,
        stage=stage,
        env_extra=env_extra,
    )


def run_batch_command(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    status: StatusLog,
    stage: str,
    env_extra: dict[str, str] | None = None,
) -> None:
    _base.run_batch_command(
        cmd=cmd,
        cwd=cwd,
        log_path=log_path,
        status=status,
        stage=stage,
        env_extra=env_extra,
    )


def latest_step3_result(step3_dir: Path, input_hdf: Path) -> Path:
    matches = sorted(
        (step3_dir / "result").glob(f"{input_hdf.stem}_*.h5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No Step 3 result found for {input_hdf.name} in {step3_dir / 'result'}"
        )
    return matches[0]
