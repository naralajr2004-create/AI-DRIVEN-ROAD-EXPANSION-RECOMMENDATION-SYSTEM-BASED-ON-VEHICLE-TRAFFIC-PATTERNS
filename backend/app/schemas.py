from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    label: str
    value: str


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: str
    progress: float
    progress_message: str
    error_message: str | None
    output_video_url: str | None = None
    report_url: str | None = None
    metrics: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None
    summary: list[RunSummary] | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: str
    progress: float
    progress_message: str
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    app_name: str
