from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import imageio_ffmpeg
from deep_sort_realtime.deepsort_tracker import DeepSort

from app.core.config import get_settings
from app.services.analysis.models import DamageDetection, VehicleDetection, model_registry
from app.services.analysis.recommendations import build_recommendations, build_summary


settings = get_settings()
ProgressCallback = Callable[[float, str], None]


@dataclass
class TrackSnapshot:
    label: str
    confidence: float
    frames_seen: int = 0
    first_frame: int = 0
    last_frame: int = 0
    counted: bool = False
    centers: list[tuple[int, float, float]] = field(default_factory=list)


class AnalysisPipeline:
    def __init__(self) -> None:
        self.vehicle_detector = model_registry.vehicle_detector()
        self.vehicle_classifier = model_registry.vehicle_classifier()
        self.damage_detector = model_registry.damage_detector()

    def process_video(self, input_path: Path, run_id: str, progress_callback: ProgressCallback) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError("Unable to open the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Unable to read video dimensions.")

        scale = min(1.0, settings.max_frame_width / width)
        output_width = max(1, int(width * scale))
        output_height = max(1, int(height * scale))
        output_path = settings.outputs_dir / f"{run_id}.mp4"
        temp_output_path = settings.outputs_dir / f"{run_id}.raw.mp4"
        report_path = settings.reports_dir / f"{run_id}.json"

        writer = cv2.VideoWriter(
            str(temp_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Unable to create the annotated output video.")

        tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_iou_distance=0.7)
        track_state: dict[str, TrackSnapshot] = {}
        track_class_cache: dict[str, tuple[str, float]] = {}
        unique_vehicle_counts: Counter[str] = Counter()
        visible_vehicle_timeline: list[int] = []
        vehicle_speed_samples: defaultdict[str, list[float]] = defaultdict(list)
        vehicle_emissions_kg: defaultdict[str, float] = defaultdict(float)
        per_frame_flow: list[int] = []

        damage_counts: Counter[str] = Counter()
        damage_frames = 0
        damage_persistent_frames = 0
        severity_counts: Counter[str] = Counter()
        severity_score = 0.0

        processed_frames = 0
        damage_stride = max(1, round(fps / settings.damage_sample_hz))
        previous_visible_track_ids: set[str] = set()
        progress_callback(0.02, "Loading pretrained models")

        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if scale != 1.0:
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

                vehicle_detections = self.vehicle_detector.detect(frame)
                tracks, current_track_ids = self._update_tracks(
                    tracker=tracker,
                    frame=frame,
                    frame_index=frame_index,
                    detections=vehicle_detections,
                    track_state=track_state,
                    track_class_cache=track_class_cache,
                    unique_vehicle_counts=unique_vehicle_counts,
                    fps=fps,
                )

                visible_vehicle_timeline.append(len(current_track_ids))
                new_ids = current_track_ids - previous_visible_track_ids
                per_frame_flow.append(len(new_ids))
                previous_visible_track_ids = current_track_ids

                damage_detections: list[DamageDetection] = []
                if frame_index % damage_stride == 0:
                    try:
                        damage_detections = self.damage_detector.detect(frame)
                    except Exception:
                        damage_detections = []

                    if damage_detections:
                        damage_persistent_frames += 1
                    damage_frames += 1

                    for damage in damage_detections:
                        canonical = self._normalize_damage_label(damage.label)
                        damage_counts[canonical] += 1
                        severity = self._severity_from_bbox(damage, output_width, output_height)
                        severity_counts[severity] += 1
                        severity_score += {"Low": 1.0, "Medium": 2.0, "High": 3.4}[severity]

                self._annotate_frame(
                    frame=frame,
                    tracks=tracks,
                    damage_detections=damage_detections,
                    frame_index=frame_index,
                    unique_vehicle_counts=unique_vehicle_counts,
                    visible_count=len(current_track_ids),
                )
                writer.write(frame)

                processed_frames += 1
                frame_index += 1
                if processed_frames % settings.progress_update_every_n_frames == 0:
                    progress = 0.08 + (processed_frames / max(total_frames, 1)) * 0.84
                    progress_callback(progress, f"Analyzing frame {processed_frames} of {total_frames or '?'}")

            for snapshot in track_state.values():
                if not snapshot.counted:
                    continue
                speed_kph = self._estimate_track_speed_kph(snapshot.centers, fps)
                if speed_kph > 0:
                    vehicle_speed_samples[snapshot.label].append(speed_kph)
                duration_seconds = (snapshot.last_frame - snapshot.first_frame + 1) / fps
                distance_km = max(speed_kph, settings.min_reliable_speed_kph) * duration_seconds / 3600.0
                vehicle_emissions_kg[snapshot.label] += self._emission_factor_grams_per_km(snapshot.label) * distance_km / 1000.0

            metrics = self._build_metrics(
                fps=fps,
                processed_frames=processed_frames,
                width=output_width,
                height=output_height,
                unique_vehicle_counts=unique_vehicle_counts,
                visible_vehicle_timeline=visible_vehicle_timeline,
                per_frame_flow=per_frame_flow,
                damage_counts=damage_counts,
                damage_frames=damage_frames,
                damage_persistent_frames=damage_persistent_frames,
                severity_counts=severity_counts,
                severity_score=severity_score,
                vehicle_speed_samples=vehicle_speed_samples,
                vehicle_emissions_kg=vehicle_emissions_kg,
            )
            recommendations = build_recommendations(metrics)
            summary = build_summary(metrics, recommendations)

            report = {
                "run_id": run_id,
                "metrics": metrics,
                "recommendations": recommendations,
                "summary": summary,
                "artifacts": {
                    "annotated_video": output_path.name,
                    "report": report_path.name,
                },
            }
            writer.release()
            self._transcode_for_web(temp_output_path, output_path)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            progress_callback(1.0, "Analysis completed")

            return {
                "metrics": metrics,
                "recommendations": recommendations,
                "summary": summary,
                "output_path": output_path,
                "report_path": report_path,
            }
        finally:
            capture.release()
            writer.release()
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)

    def _update_tracks(
        self,
        *,
        tracker: DeepSort,
        frame,
        frame_index: int,
        detections: list[VehicleDetection],
        track_state: dict[str, TrackSnapshot],
        track_class_cache: dict[str, tuple[str, float]],
        unique_vehicle_counts: Counter[str],
        fps: float,
    ) -> tuple[list[Any], set[str]]:
        deep_sort_detections = [
            ([det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1], det.confidence, det.label)
            for det in detections
        ]
        tracks = tracker.update_tracks(deep_sort_detections, frame=frame)
        current_track_ids: set[str] = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = str(track.track_id)
            current_track_ids.add(track_id)
            x1, y1, x2, y2 = [max(0, int(v)) for v in track.to_ltrb()]
            x2 = min(x2, frame.shape[1] - 1)
            y2 = min(y2, frame.shape[0] - 1)
            crop = frame[y1:y2, x1:x2]
            fallback_label = str(track.det_class) if track.det_class else "car"

            if track_id not in track_class_cache or track_state.get(track_id, TrackSnapshot("car", 0.0)).frames_seen % 10 == 0:
                track_class_cache[track_id] = self.vehicle_classifier.classify(crop, fallback_label)
            label, confidence = track_class_cache[track_id]

            snapshot = track_state.setdefault(
                track_id,
                TrackSnapshot(
                    label=label,
                    confidence=confidence,
                    first_frame=frame_index,
                    last_frame=frame_index,
                ),
            )
            snapshot.label = label
            snapshot.confidence = confidence
            snapshot.frames_seen += 1
            snapshot.last_frame = frame_index
            snapshot.centers.append((frame_index, (x1 + x2) / 2, (y1 + y2) / 2))

            duration_seconds = (snapshot.last_frame - snapshot.first_frame + 1) / fps
            if not snapshot.counted and duration_seconds >= settings.min_track_age_seconds:
                snapshot.counted = True
                unique_vehicle_counts[label] += 1

        return tracks, current_track_ids

    def _normalize_damage_label(self, raw_label: str) -> str:
        label = raw_label.lower().strip()
        if "pothole" in label:
            return "pothole"
        if "crack" in label:
            return "crack"
        return "surface_damage"

    def _severity_from_bbox(self, damage: DamageDetection, width: int, height: int) -> str:
        area = max(0.0, damage.x2 - damage.x1) * max(0.0, damage.y2 - damage.y1)
        ratio = area / max(width * height, 1)
        if ratio >= 0.03:
            return "High"
        if ratio >= 0.012:
            return "Medium"
        return "Low"

    def _estimate_track_speed_kph(self, centers: list[tuple[int, float, float]], fps: float) -> float:
        if len(centers) < 2:
            return 0.0
        total_distance_px = 0.0
        total_frames = centers[-1][0] - centers[0][0]
        if total_frames <= 0:
            return 0.0
        for (_, x1, y1), (_, x2, y2) in zip(centers, centers[1:]):
            total_distance_px += math.dist((x1, y1), (x2, y2))
        meters = total_distance_px * settings.approx_meters_per_pixel
        seconds = total_frames / fps
        speed_kph = (meters / seconds) * 3.6 if seconds > 0 else 0.0
        if speed_kph < settings.min_reliable_speed_kph:
            return 0.0
        return min(speed_kph, settings.max_reasonable_speed_kph)

    def _emission_factor_grams_per_km(self, label: str) -> float:
        return {
            "car": settings.emission_factor_grams_per_km_car,
            "bus": settings.emission_factor_grams_per_km_bus,
            "truck": settings.emission_factor_grams_per_km_truck,
            "motorcycle": settings.emission_factor_grams_per_km_motorcycle,
            "ambulance": settings.emission_factor_grams_per_km_ambulance,
        }.get(label, settings.emission_factor_grams_per_km_car)

    def _build_metrics(
        self,
        *,
        fps: float,
        processed_frames: int,
        width: int,
        height: int,
        unique_vehicle_counts: Counter[str],
        visible_vehicle_timeline: list[int],
        per_frame_flow: list[int],
        damage_counts: Counter[str],
        damage_frames: int,
        damage_persistent_frames: int,
        severity_counts: Counter[str],
        severity_score: float,
        vehicle_speed_samples: defaultdict[str, list[float]],
        vehicle_emissions_kg: defaultdict[str, float],
    ) -> dict[str, Any]:
        duration_seconds = round(processed_frames / max(fps, 1), 2)
        average_visible = round(sum(visible_vehicle_timeline) / max(len(visible_vehicle_timeline), 1), 2)
        peak_visible = int(max(visible_vehicle_timeline) if visible_vehicle_timeline else 0)
        flow_per_minute = round((sum(per_frame_flow) / max(duration_seconds, 1)) * 60.0, 2)
        slow_moving_ratio = self._slow_moving_ratio(vehicle_speed_samples)
        total_emissions = round(sum(vehicle_emissions_kg.values()), 3)

        return {
            "video": {
                "duration_seconds": duration_seconds,
                "fps": round(fps, 2),
                "processed_frames": processed_frames,
                "resolution": f"{width}x{height}",
            },
            "vehicles": {
                "unique_vehicle_count": int(sum(unique_vehicle_counts.values())),
                "counts_by_type": dict(unique_vehicle_counts),
                "average_speed_kph_by_type": {
                    key: round(sum(values) / max(len(values), 1), 2)
                    for key, values in vehicle_speed_samples.items()
                },
            },
            "congestion": {
                "average_visible_vehicles": average_visible,
                "peak_visible_vehicles": peak_visible,
                "vehicle_flow_per_minute": flow_per_minute,
                "slow_moving_ratio": round(slow_moving_ratio, 3),
            },
            "road_damage": {
                "total_detections": int(sum(damage_counts.values())),
                "counts_by_type": dict(damage_counts),
                "severity_breakdown": dict(severity_counts),
                "persistence_ratio": round(damage_persistent_frames / max(damage_frames, 1), 3),
                "severity_score": round(severity_score, 2),
                "dominant_damage_type": max(damage_counts, key=damage_counts.get) if damage_counts else "none",
            },
            "emissions": {
                "total_kg_co2": total_emissions,
                "kg_co2_by_type": {key: round(value, 3) for key, value in vehicle_emissions_kg.items()},
                "dominant_vehicle_class": max(vehicle_emissions_kg, key=vehicle_emissions_kg.get) if vehicle_emissions_kg else "car",
                "hotspot": total_emissions >= 12.0,
            },
        }

    def _slow_moving_ratio(self, vehicle_speed_samples: defaultdict[str, list[float]]) -> float:
        all_speeds = [speed for values in vehicle_speed_samples.values() for speed in values]
        if not all_speeds:
            return 0.0
        return len([speed for speed in all_speeds if speed <= 20.0]) / len(all_speeds)

    def _annotate_frame(
        self,
        *,
        frame,
        tracks: list[Any],
        damage_detections: list[DamageDetection],
        frame_index: int,
        unique_vehicle_counts: Counter[str],
        visible_count: int,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 18), (460, 170), (10, 17, 31), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

        cv2.putText(frame, "Road Intelligence Live Analysis", (34, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (242, 247, 255), 2)
        cv2.putText(frame, f"Frame {frame_index}", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (166, 182, 202), 2)
        cv2.putText(frame, f"Visible vehicles: {visible_count}", (34, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 220, 175), 2)
        cv2.putText(frame, f"Unique vehicles: {sum(unique_vehicle_counts.values())}", (34, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (118, 160, 255), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = [max(0, int(v)) for v in track.to_ltrb()]
            label = str(track.det_class or "vehicle")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (73, 140, 255), 2)
            cv2.putText(frame, f"{label} #{track.track_id}", (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

        for damage in damage_detections:
            x1, y1, x2, y2 = [max(0, int(v)) for v in (damage.x1, damage.y1, damage.x2, damage.y2)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 108, 83), 2)
            cv2.putText(
                frame,
                f"{damage.label} {damage.confidence:.2f}",
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 245, 241),
                2,
            )

    def _transcode_for_web(self, source_path: Path, output_path: Path) -> None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "fast",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists():
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                "Annotated video transcoding failed. "
                + (stderr[-500:] if stderr else "No FFmpeg error output was captured.")
            )
