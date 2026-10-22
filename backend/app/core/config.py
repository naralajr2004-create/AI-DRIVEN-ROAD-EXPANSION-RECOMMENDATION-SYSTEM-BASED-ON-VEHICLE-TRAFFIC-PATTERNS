from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Road Intelligence Hub"
    api_prefix: str = "/api"
    frontend_origin: str = "http://localhost:5173"

    project_root: Path = Path(__file__).resolve().parents[3]
    storage_dir: Path = Path(__file__).resolve().parents[2] / "storage"
    uploads_dir: Path = Path(__file__).resolve().parents[2] / "storage" / "uploads"
    outputs_dir: Path = Path(__file__).resolve().parents[2] / "storage" / "outputs"
    reports_dir: Path = Path(__file__).resolve().parents[2] / "storage" / "reports"
    models_dir: Path = Path(__file__).resolve().parents[2] / "storage" / "models"
    database_url: str = f"sqlite:///{(Path(__file__).resolve().parents[2] / 'storage' / 'analysis.db').as_posix()}"

    vehicle_detection_model: str = str(Path(__file__).resolve().parents[2] / "storage" / "models" / "yolov8x.pt")
    vehicle_detection_confidence: float = 0.30
    vehicle_detection_iou: float = 0.45
    vehicle_detection_classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])

    hf_token: str | None = None
    hf_vehicle_classifier_repo: str = "AventIQ-AI/ResNet-50-Vehicle-Segment-classification"
    hf_classifier_confidence_threshold: float = 0.45

    roboflow_api_key: str | None = None
    roboflow_model_id: str = "crack-and-pothole-bftyl-3ishu/1"
    roboflow_confidence_threshold: float = 0.40
    roboflow_overlap_threshold: float = 0.30

    max_frame_width: int = 1280
    damage_sample_hz: float = 2.0
    progress_update_every_n_frames: int = 5
    approx_meters_per_pixel: float = 0.08
    min_track_age_seconds: float = 0.4
    min_reliable_speed_kph: float = 5.0
    max_reasonable_speed_kph: float = 95.0

    emission_factor_grams_per_km_car: float = 192.0
    emission_factor_grams_per_km_bus: float = 822.0
    emission_factor_grams_per_km_truck: float = 768.0
    emission_factor_grams_per_km_motorcycle: float = 103.0
    emission_factor_grams_per_km_ambulance: float = 450.0

    def ensure_directories(self) -> None:
        for path in (
            self.storage_dir,
            self.uploads_dir,
            self.outputs_dir,
            self.reports_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
