"""Jobs: start pipeline runs, follow their logs (SSE), cancel, read run files."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gridexpand.common.timeframe import TIMEFRAME_MODES
from gridexpand.service import environment, queries, runlog, scenarios
from gridexpand.paths import PROJECT_DIR, RUNS_DIR
from gridexpand.service.commands import (
    MODEL_CASES,
    POST_CASES,
    PipelineSpec,
    contiguous_range,
    pipeline_steps,
    run_yaml_text,
    terminal_commands,
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
    grid_result_id: int | None = Field(None, ge=1, description="exactly this pylovo grid (ignores min_buildings)")
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


@dataclass(frozen=True)
class PipelinePlan:
    """A validated pipeline request: the run spec and what it selects."""

    spec: PipelineSpec
    scenario: dict[str, Any]
    ags: int
    selected: list[dict[str, Any]]
    title: str


def plan_pipeline(body: PipelineBody, request: Request) -> PipelinePlan:
    """Resolve scenario, solver, AGS and grids of a request (HTTP errors for bad requests)."""
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
    common = dict(ags=ags, pylovo_version_id=body.pylovo_version_id, scenario_config=scenario_path,
                  model_cases=tuple(body.model_cases), timeframe_mode=body.timeframe_mode,
                  workers=body.workers, powerflow_output=body.powerflow_output)
    cases = ", ".join(body.model_cases)
    if body.grid_result_id is not None:
        # One grid by identity: plz/kcid/bcid in the run YAML (no candidate numbering involved).
        selected = [c for c in queries.grid_candidates(ags, body.pylovo_version_id, 1)
                    if int(c["grid_result_id"]) == body.grid_result_id]
        if not selected:
            raise HTTPException(400, f"Grid {body.grid_result_id} is not a grid of AGS {ags} in pylovo v{body.pylovo_version_id}")
        grid = selected[0]
        spec = PipelineSpec(**common, min_buildings=1, plz=int(grid["plz"]), kcid=int(grid["kcid"]), bcid=int(grid["bcid"]))
        title = f"Grid {grid['kcid']}/{grid['bcid']} · PLZ {grid['plz']} · v{body.pylovo_version_id} · {cases}"
        return PipelinePlan(spec, scenario, ags, selected, title)
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
    spec = PipelineSpec(**common, min_buildings=body.min_buildings, start_index=start_index, limit=limit)
    region = f"PLZ {body.plz}" if body.plz else f"AGS {ags}"
    title = f"{region} · v{body.pylovo_version_id} · {len(selected)} grid(s) · {cases}"
    return PipelinePlan(spec, scenario, ags, selected, title)


@router.post("/pipeline", status_code=202)
def start_pipeline(body: PipelineBody, request: Request) -> dict[str, Any]:
    """Queue ``gridexpand run`` for each selected model case (one after another)."""
    settings = settings_of(request)
    plan = plan_pipeline(body, request)
    manager = jobs_of(request)
    job_id = manager.new_id()
    steps = pipeline_steps(plan.spec, settings.runs_dir / job_id, settings.python)
    params = body.model_dump() | {
        "ags": plan.ags, "grids": len(plan.selected), "start_index": plan.spec.start_index, "limit": plan.spec.limit,
        "grid_result_ids": [int(c["grid_result_id"]) for c in plan.selected],
        "scenario_key": plan.scenario["scenario_key"], "scenario_id": plan.scenario["id"],
        "run_dir": str(settings.runs_dir / job_id),
    }
    try:
        job = manager.submit("pipeline", plan.title, steps, params, job_id=job_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return _summary(request, job)


TERMINAL_DIR = "terminal"


def _terminal_dir(request: Request) -> Path:
    return settings_of(request).state_dir / TERMINAL_DIR


@router.post("/terminal", status_code=201)
def prepare_terminal_run(body: PipelineBody, request: Request) -> dict[str, Any]:
    """Write the run YAML of a request for a run in a terminal (``tmux``) instead of a job.

    Returns the YAML, the commands to start and follow it, and a portable copy of the YAML
    (scenario referenced as ``../scenarios/<file>``) for another GridExpand checkout.
    """
    plan = plan_pipeline(body, request)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    region = f"{plan.spec.plz}-{plan.spec.kcid}-{plan.spec.bcid}" if plan.spec.kcid is not None else (
        str(body.plz) if body.plz else str(plan.ags))
    run_id = f"ui_{region}_{plan.scenario['id']}_{stamp}"
    header = (f"{plan.title}\nPrepared by the GridExpand UI for a terminal run ({len(plan.selected)} grid(s)).\n"
              f"Results go to the database of the service; the UI shows them when the run has finished.")
    directory = _terminal_dir(request)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.yaml"
    path.write_text(run_yaml_text(plan.spec, run_id, header), encoding="utf-8")
    portable = run_yaml_text(plan.spec, run_id, header + "\nCopy to config/runs/ of a GridExpand checkout; the scenario "
                             f"file {body.scenario} must be in config/scenarios/.", scenario=f"../scenarios/{body.scenario}")
    in_container = Path("/.dockerenv").exists()
    return {"run_id": run_id, "path": str(path), "title": plan.title, "grids": len(plan.selected),
            "model_cases": list(body.model_cases), "run_yaml": path.read_text(encoding="utf-8"),
            "portable_run_yaml": portable, "scenario": body.scenario, "in_container": in_container,
            "commands": terminal_commands(path, run_id, in_container=in_container, project_dir=PROJECT_DIR),
            "run_dir": str(RUNS_DIR / run_id)}


@router.get("/terminal")
def terminal_runs(request: Request) -> list[dict[str, Any]]:
    """Runs prepared for a terminal (newest first) with the state of their run directory."""
    from gridexpand.scenario.rundir import load_state, process_alive

    out = []
    directory = _terminal_dir(request)
    for path in sorted(directory.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        run_id = path.stem
        state = load_state(RUNS_DIR / run_id)
        title = next((line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines()[:1]
                      if line.startswith("# ")), run_id)
        entry: dict[str, Any] = {"run_id": run_id, "title": title, "path": str(path),
                                 "prepared_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                                 "status": "not started"}
        if state is not None:
            status = state.get("status")
            if status == "running" and not process_alive(state.get("pid")):
                status = "interrupted"
            entry |= {"status": status, "stage": state.get("stage"), "jobs": state.get("jobs", {}),
                      "updated_at": state.get("updated_at"), "exit_code": state.get("exit_code")}
        out.append(entry)
    return out


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
