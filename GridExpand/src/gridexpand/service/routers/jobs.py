"""Jobs: start pipeline runs, follow their logs (SSE), cancel, read run files."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gridexpand.common.timeframe import TIMEFRAME_MODES
from gridexpand.service import environment, queries, runlog, scenarios
from gridexpand.service.commands import (
    MODEL_CASES,
    POST_CASES,
    PipelineSpec,
    contiguous_range,
    pipeline_steps,
)
from gridexpand.service.jobs import Job, JobManager, JobNotFound
from gridexpand.service.routers.meta import resolve_ags, settings_of

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class PipelineBody(BaseModel):
    """A synthetic pipeline run for the grids of one region."""

    model_config = ConfigDict(extra="forbid")

    ags: int | str | None = Field(None, description="AGS; resolved from the PLZ when omitted")
    plz: int | None = Field(None, ge=1, le=99999, description="only the grids of this PLZ")
    pylovo_version_id: str = Field(pattern=r"^[\w.-]{1,10}$")
    scenario: str = Field(min_length=1, max_length=200, description="file name from GET /api/scenarios")
    model_cases: list[str] = Field(default_factory=lambda: ["pre"], min_length=1)
    timeframe_mode: str = "max_base_electricity_demand_week"
    min_buildings: int = Field(5, ge=1, le=100_000)
    candidate_indexes: list[int] | None = Field(None, description="consecutive candidate numbers of the AGS")
    workers: int = Field(1, ge=1, le=16)
    powerflow_output: str = Field("summary", pattern=r"^(summary|both)$")

    @field_validator("model_cases")
    @classmethod
    def _cases(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(MODEL_CASES))
        if unknown:
            raise ValueError(f"unknown model case(s) {unknown}; allowed: {list(MODEL_CASES)}")
        return [case for case in MODEL_CASES if case in value]  # fixed order, no duplicates

    @field_validator("timeframe_mode")
    @classmethod
    def _timeframe(cls, value: str) -> str:
        if value not in TIMEFRAME_MODES:
            raise ValueError(f"allowed: {list(TIMEFRAME_MODES)}")
        return value


def jobs_of(request: Request) -> JobManager:
    return request.app.state.jobs


def _job(request: Request, job_id: str) -> Job:
    try:
        return jobs_of(request).get(job_id)
    except JobNotFound as exc:
        raise HTTPException(404, "Job not found") from exc


def _summary(request: Request, job: Job) -> dict[str, Any]:
    return job.summary() | {"queue_position": jobs_of(request).queue_position(job.id)}


@router.post("/pipeline", status_code=202)
def start_pipeline(body: PipelineBody, request: Request) -> dict[str, Any]:
    """Queue ``gridexpand synthetic`` for each selected model case (one after another)."""
    settings = settings_of(request)
    try:
        scenario_path = scenarios.resolve(settings.scenario_dirs, body.scenario)
    except KeyError as exc:
        raise HTTPException(404, f"Scenario '{body.scenario}' not found") from exc
    scenario = scenarios.describe(body.scenario, scenario_path)
    if not scenario["valid"]:
        raise HTTPException(400, f"Scenario '{body.scenario}' is invalid: {scenario['error']}")
    if any(case in POST_CASES for case in body.model_cases):
        solvers = environment.solver_status(settings.solver)
        if not solvers["post_cases_supported"]:
            raise HTTPException(409, f"Post cases need Step 3 but the solver is not usable ({solvers['post_cases_reason']}). "
                                     "Run the pre case only, or configure a licence / GRIDEXPAND_SOLVER.")
    ags, _ = resolve_ags(body.ags, body.plz)
    candidates = queries.grid_candidates(ags, body.pylovo_version_id, body.min_buildings)
    selected = [c for c in candidates if body.plz is None or int(c["plz"]) == body.plz]
    if body.candidate_indexes is not None:
        wanted = set(body.candidate_indexes)
        selected = [c for c in selected if int(c["candidate_index"]) in wanted]
    if not selected:
        raise HTTPException(400, "No candidate grids match (check PLZ, pylovo version and minimum buildings)")
    start_index = limit = None
    if len(selected) != len(candidates):
        try:
            start_index, limit = contiguous_range([int(c["candidate_index"]) for c in selected])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    spec = PipelineSpec(ags=ags, pylovo_version_id=body.pylovo_version_id, scenario_config=scenario_path,
                        model_cases=tuple(body.model_cases), timeframe_mode=body.timeframe_mode,
                        min_buildings=body.min_buildings, start_index=start_index, limit=limit,
                        workers=body.workers, powerflow_output=body.powerflow_output)
    manager = jobs_of(request)
    job_id = manager.new_id()
    steps = pipeline_steps(spec, settings.runs_dir / job_id, settings.python)
    region = f"PLZ {body.plz}" if body.plz else f"AGS {ags}"
    title = f"{region} · v{body.pylovo_version_id} · {len(selected)} grid(s) · {', '.join(body.model_cases)}"
    params = body.model_dump() | {
        "ags": ags, "grids": len(selected), "start_index": start_index, "limit": limit,
        "grid_result_ids": [int(c["grid_result_id"]) for c in selected],
        "scenario_key": scenario["scenario_key"], "scenario_id": scenario["id"],
        "run_dir": str(settings.runs_dir / job_id),
    }
    try:
        job = manager.submit("pipeline", title, steps, params, job_id=job_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return _summary(request, job)


@router.get("")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    """All jobs (newest first), including the recent history."""
    return [_summary(request, job) for job in jobs_of(request).list()]


@router.get("/{job_id}")
def get_job(job_id: str, request: Request, tail: int = Query(0, ge=0, le=5000)) -> dict[str, Any]:
    """One job with the per-grid status of every step (and the last ``tail`` log lines)."""
    job = _job(request, job_id)
    data = _summary(request, job)
    data["grids"] = {step.name: runlog.read_status_rows(Path(step.run_dir)) for step in job.steps if step.run_dir}
    data["summaries"] = {step.name: runlog.read_summary(Path(step.run_dir)) for step in job.steps if step.run_dir}
    if tail:
        data["lines"] = job.lines[-tail:]
    return data


@router.get("/{job_id}/log")
def job_log(job_id: str, request: Request, after: int = 0, format: str = "json"):
    """Log lines after a sequence number (``format=text`` downloads the whole log)."""
    job = _job(request, job_id)
    if format == "text":
        text = "\n".join(line["text"] for line in job.lines) + "\n"
        return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="gridexpand-job-{job.id}.log"'})
    return {"job": _summary(request, job), "lines": job.lines_after(after)}


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int = 0) -> StreamingResponse:
    """Server-Sent Events: ``log`` (new lines), ``status`` (summary changes), final ``end``."""
    job = _job(request, job_id)
    last_id = request.headers.get("last-event-id")
    cursor = int(last_id) + 1 if last_id and last_id.isdigit() else after
    server = getattr(request.app.state, "server", None)

    async def events():
        nonlocal cursor
        last_status = None
        idle = 0.0
        yield "retry: 2000\n\n"
        while True:
            if await request.is_disconnected() or (server is not None and server.should_exit):
                return
            lines = job.lines_after(cursor)
            if lines:
                cursor = lines[-1]["seq"] + 1
                for start in range(0, len(lines), 400):
                    chunk = lines[start:start + 400]
                    yield f"id: {chunk[-1]['seq']}\nevent: log\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                idle = 0.0
            summary = _summary(request, job)
            key = (summary["status"], summary["progress"], summary["phase"], summary["counts"]["error"],
                   summary["counts"]["warning"], summary["queue_position"])
            if key != last_status:
                last_status = key
                yield f"event: status\ndata: {json.dumps(summary, default=str)}\n\n"
            if not job.active and not job.lines_after(cursor):
                yield f"event: end\ndata: {json.dumps(summary, default=str)}\n\n"
                return
            await asyncio.sleep(0.25)
            idle += 0.25
            if idle >= 15:
                idle = 0.0
                yield ": keep-alive\n\n"

    async def stream():
        try:
            async for chunk in events():
                yield chunk
        except asyncio.CancelledError:  # client gone or server shutdown
            return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    """Cancel a queued job, or stop a running one (SIGTERM to its process group)."""
    job = _job(request, job_id)
    if not job.active:
        raise HTTPException(409, f"Job is already {job.status}")
    return _summary(request, jobs_of(request).cancel(job_id))


def _run_root(job: Job) -> Path | None:
    run_dir = job.params.get("run_dir")
    return Path(run_dir).resolve() if run_dir else None


@router.get("/{job_id}/files")
def job_files(job_id: str, request: Request) -> list[dict[str, Any]]:
    """Files in the job's run directories (events, status, summaries, per-grid logs)."""
    root = _run_root(_job(request, job_id))
    if root is None or not root.is_dir():
        return []
    return [{"path": str(p.relative_to(root)), "size": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".log", ".json", ".jsonl", ".tsv", ".csv")]


@router.get("/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str, request: Request) -> FileResponse:
    """One file of the job's run directories (text files only)."""
    root = _run_root(_job(request, job_id))
    if root is None:
        raise HTTPException(404, "This job has no run directory")
    target = (root / path).resolve()
    if not target.is_file() or root not in target.parents or target.suffix not in (".log", ".json", ".jsonl", ".tsv", ".csv"):
        raise HTTPException(404, "File not found")
    media = "text/plain" if target.suffix in (".log", ".jsonl", ".tsv") else mimetypes.guess_type(target.name)[0]
    return FileResponse(target, media_type=media or "text/plain", headers={"Cache-Control": "no-store"})
