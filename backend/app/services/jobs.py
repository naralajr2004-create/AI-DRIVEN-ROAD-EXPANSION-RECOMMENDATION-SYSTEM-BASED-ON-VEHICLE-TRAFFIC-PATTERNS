from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AnalysisRun
from app.services.analysis.pipeline import AnalysisPipeline


class JobService:
    def __init__(self) -> None:
        self.pipeline: AnalysisPipeline | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _get_pipeline(self) -> AnalysisPipeline:
        with self._lock:
            if self.pipeline is None:
                self.pipeline = AnalysisPipeline()
            return self.pipeline

    def create_run(self, *, original_filename: str, source_video_path: Path) -> AnalysisRun:
        run = AnalysisRun(
            id=uuid4().hex,
            original_filename=original_filename,
            status="queued",
            progress=0.0,
            progress_message="Queued for analysis",
            source_video_path=str(source_video_path),
        )
        with SessionLocal() as session:
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def start(self, run_id: str) -> None:
        thread = threading.Thread(target=self._process_run, args=(run_id,), daemon=True)
        with self._lock:
            self._threads[run_id] = thread
        thread.start()

    def list_runs(self) -> list[AnalysisRun]:
        with SessionLocal() as session:
            return list(session.scalars(select(AnalysisRun).order_by(AnalysisRun.created_at.desc())).all())

    def get_run(self, run_id: str) -> AnalysisRun | None:
        with SessionLocal() as session:
            return session.get(AnalysisRun, run_id)

    def _update_progress(self, run_id: str, progress: float, message: str) -> None:
        with SessionLocal() as session:
            run = session.get(AnalysisRun, run_id)
            if run is None:
                return
            run.progress = round(float(progress), 4)
            run.progress_message = message
            run.updated_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()

    def _process_run(self, run_id: str) -> None:
        with SessionLocal() as session:
            run = session.get(AnalysisRun, run_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            run.progress = 0.01
            run.progress_message = "Preparing analysis"
            session.add(run)
            session.commit()
            source_path = Path(run.source_video_path)

        try:
            result = self._get_pipeline().process_video(
                input_path=source_path,
                run_id=run_id,
                progress_callback=lambda progress, message: self._update_progress(run_id, progress, message),
            )
            with SessionLocal() as session:
                run = session.get(AnalysisRun, run_id)
                if run is None:
                    return
                run.status = "completed"
                run.progress = 1.0
                run.progress_message = "Completed"
                run.output_video_path = str(result["output_path"])
                run.report_path = str(result["report_path"])
                run.metrics_json = json.dumps(result["metrics"])
                run.recommendations_json = json.dumps(result["recommendations"])
                run.summary_json = json.dumps(result["summary"])
                run.completed_at = datetime.now(timezone.utc)
                run.updated_at = datetime.now(timezone.utc)
                session.add(run)
                session.commit()
        except Exception as exc:
            with SessionLocal() as session:
                run = session.get(AnalysisRun, run_id)
                if run is None:
                    return
                run.status = "failed"
                run.progress_message = "Analysis failed"
                run.error_message = str(exc)
                run.completed_at = datetime.now(timezone.utc)
                run.updated_at = datetime.now(timezone.utc)
                session.add(run)
                session.commit()


job_service = JobService()
