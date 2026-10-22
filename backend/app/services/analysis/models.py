from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from app.core.config import get_settings


settings = get_settings()


@dataclass
class VehicleDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


@dataclass
class DamageDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


class VehicleDetector:
    VEHICLE_LABELS = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self) -> None:
        self.model = YOLO(settings.vehicle_detection_model)

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        result = self.model.predict(
            source=frame,
            conf=settings.vehicle_detection_confidence,
            iou=settings.vehicle_detection_iou,
            classes=settings.vehicle_detection_classes,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )[0]
        if result.boxes is None:
            return []

        detections: list[VehicleDetection] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in self.VEHICLE_LABELS:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                VehicleDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf.item()),
                    class_id=cls_id,
                    label=self.VEHICLE_LABELS[cls_id],
                )
            )
        return detections


class VehicleClassifier:
    PROJECT_CLASS_MAP = {
        "articulated_truck": "truck",
        "background": "unknown",
        "bicycle": "unknown",
        "bus": "bus",
        "car": "car",
        "motorcycle": "motorcycle",
        "non-motorized_vehicle": "unknown",
        "pedestrian": "unknown",
        "pickup_truck": "truck",
        "single_unit_truck": "truck",
        "work_van": "truck",
    }

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = settings.models_dir / "hf_vehicle_classifier"
        model_path = self._download_asset(model_dir, "pytorch_model.bin")
        classes_path = self._download_asset(model_dir, "classes.txt")

        self.class_names = [line.strip() for line in Path(classes_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_names))
        state_dict = self._load_state_dict(Path(model_path))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def classify(self, crop_bgr: np.ndarray, fallback_label: str) -> tuple[str, float]:
        if crop_bgr.size == 0:
            return fallback_label, 1.0

        pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)

        raw_label = self.class_names[int(idx.item())]
        mapped_label = self.PROJECT_CLASS_MAP.get(raw_label, fallback_label)
        confidence_value = float(confidence.item())

        if mapped_label == "unknown" or confidence_value < settings.hf_classifier_confidence_threshold:
            return fallback_label, confidence_value
        if mapped_label != fallback_label:
            return fallback_label, confidence_value
        return mapped_label, confidence_value

    def _download_asset(self, model_dir: Path, filename: str, force_download: bool = False) -> str:
        return hf_hub_download(
            repo_id=settings.hf_vehicle_classifier_repo,
            filename=filename,
            token=settings.hf_token,
            local_dir=model_dir,
            force_download=force_download,
        )

    def _load_state_dict(self, model_path: Path):
        try:
            return torch.load(model_path, map_location=self.device)
        except Exception:
            if model_path.exists():
                model_path.unlink()
            redownloaded = self._download_asset(model_path.parent, model_path.name, force_download=True)
            return torch.load(redownloaded, map_location=self.device)


class DamageDetector:
    def __init__(self) -> None:
        self.session = requests.Session()

    def is_enabled(self) -> bool:
        return bool(settings.roboflow_api_key)

    def detect(self, frame_bgr: np.ndarray) -> list[DamageDetection]:
        if not self.is_enabled():
            return []

        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return []

        response = self.session.post(
            f"https://detect.roboflow.com/{settings.roboflow_model_id}",
            params={
                "api_key": settings.roboflow_api_key,
                "confidence": int(settings.roboflow_confidence_threshold * 100),
                "overlap": int(settings.roboflow_overlap_threshold * 100),
            },
            files={"file": ("frame.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        detections: list[DamageDetection] = []
        for pred in payload.get("predictions", []):
            x = float(pred["x"])
            y = float(pred["y"])
            width = float(pred["width"])
            height = float(pred["height"])
            label = str(pred.get("class", "damage")).strip().lower().replace("-", " ")
            detections.append(
                DamageDetection(
                    x1=x - width / 2,
                    y1=y - height / 2,
                    x2=x + width / 2,
                    y2=y + height / 2,
                    confidence=float(pred.get("confidence", 0.0)),
                    label=label,
                )
            )
        return detections


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vehicle_detector: VehicleDetector | None = None
        self._vehicle_classifier: VehicleClassifier | None = None
        self._damage_detector: DamageDetector | None = None

    def vehicle_detector(self) -> VehicleDetector:
        with self._lock:
            if self._vehicle_detector is None:
                self._vehicle_detector = VehicleDetector()
            return self._vehicle_detector

    def vehicle_classifier(self) -> VehicleClassifier:
        with self._lock:
            if self._vehicle_classifier is None:
                self._vehicle_classifier = VehicleClassifier()
            return self._vehicle_classifier

    def damage_detector(self) -> DamageDetector:
        with self._lock:
            if self._damage_detector is None:
                self._damage_detector = DamageDetector()
            return self._damage_detector


model_registry = ModelRegistry()
