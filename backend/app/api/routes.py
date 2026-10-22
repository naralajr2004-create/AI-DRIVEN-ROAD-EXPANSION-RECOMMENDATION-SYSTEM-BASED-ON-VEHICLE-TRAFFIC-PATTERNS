from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models import AnalysisRun
from app.schemas import HealthResponse, RunListItem, RunRead, RunSummary
from app.services.jobs import job_service


settings = get_settings()
router = APIRouter(prefix=settings.api_prefix)


def _artifact_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    return f"/artifacts/{path.parent.name}/{path.name}"


def _serialize_run(run: AnalysisRun) -> RunRead:
    metrics = json.loads(run.metrics_json) if run.metrics_json else None
    recommendations = json.loads(run.recommendations_json) if run.recommendations_json else None
    summary = [RunSummary(**item) for item in json.loads(run.summary_json)] if run.summary_json else None
    return RunRead(
        id=run.id,
        original_filename=run.original_filename,
        status=run.status,
        progress=run.progress,
        progress_message=run.progress_message,
        error_message=run.error_message,
        output_video_url=_artifact_url(run.output_video_path),
        report_url=_artifact_url(run.report_path),
        metrics=metrics,
        recommendations=recommendations,
        summary=summary,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@router.get("/runs", response_model=list[RunListItem])
def list_runs() -> list[RunListItem]:
    runs = job_service.list_runs()
    return [
        RunListItem(
            id=run.id,
            original_filename=run.original_filename,
            status=run.status,
            progress=run.progress,
            progress_message=run.progress_message,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        for run in runs
    ]


@router.get("/runs/{run_id}", response_model=RunRead)
def get_run(run_id: str) -> RunRead:
    run = job_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return _serialize_run(run)


@router.post("/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def create_run(file: UploadFile = File(...)) -> RunRead:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only video uploads are supported.")

    safe_filename = Path(file.filename).name
    temp_path = settings.uploads_dir / f"{uuid4().hex}_{safe_filename}"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    run = job_service.create_run(original_filename=safe_filename, source_video_path=temp_path)
    job_service.start(run.id)
    created = job_service.get_run(run.id)
    if created is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Run could not be created.")
    return _serialize_run(created)
