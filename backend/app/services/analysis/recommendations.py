from __future__ import annotations

from typing import Any


def classify_priority(score: float, high: float = 75.0, medium: float = 45.0) -> str:
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def build_recommendations(metrics: dict[str, Any]) -> dict[str, Any]:
    congestion = metrics["congestion"]
    damage = metrics["road_damage"]
    emissions = metrics["emissions"]

    expansion_score = round(
        min(
            100.0,
            congestion["average_visible_vehicles"] * 4.8
            + congestion["peak_visible_vehicles"] * 2.5
            + congestion["vehicle_flow_per_minute"] * 0.22
            + congestion["slow_moving_ratio"] * 100 * 0.22,
        ),
        2,
    )
    maintenance_score = round(
        min(
            100.0,
            damage["severity_score"] * 18
            + damage["persistence_ratio"] * 100 * 0.45
            + damage["total_detections"] * 1.3,
        ),
        2,
    )
    environment_score = round(
        min(
            100.0,
            emissions["total_kg_co2"] * 0.9
            + congestion["vehicle_flow_per_minute"] * 0.15
            + (100 if emissions["hotspot"] else 0) * 0.15,
        ),
        2,
    )

    return {
        "road_expansion": {
            "priority": classify_priority(expansion_score),
            "score": expansion_score,
            "reasons": [
                f"Average visible vehicles: {congestion['average_visible_vehicles']}",
                f"Peak visible vehicles: {congestion['peak_visible_vehicles']}",
                f"Vehicle flow per minute: {congestion['vehicle_flow_per_minute']}",
                f"Slow-moving traffic ratio: {round(congestion['slow_moving_ratio'] * 100, 1)}%",
            ],
        },
        "maintenance": {
            "priority": classify_priority(maintenance_score),
            "score": maintenance_score,
            "reasons": [
                f"Damage detections: {damage['total_detections']}",
                f"Dominant damage type: {damage['dominant_damage_type']}",
                f"Damage persistence ratio: {round(damage['persistence_ratio'] * 100, 1)}%",
                f"Damage severity score: {damage['severity_score']}",
            ],
        },
        "environment": {
            "priority": classify_priority(environment_score),
            "score": environment_score,
            "reasons": [
                f"Estimated total emissions: {emissions['total_kg_co2']} kg CO2",
                f"Peak emission class: {emissions['dominant_vehicle_class']}",
                f"Environmental hotspot detected: {'Yes' if emissions['hotspot'] else 'No'}",
            ],
        },
    }


def build_summary(metrics: dict[str, Any], recommendations: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "Vehicles counted", "value": str(metrics["vehicles"]["unique_vehicle_count"])},
        {"label": "Peak congestion", "value": f"{metrics['congestion']['peak_visible_vehicles']} vehicles"},
        {"label": "Road damage alerts", "value": str(metrics["road_damage"]["total_detections"])},
        {"label": "Estimated emissions", "value": f"{metrics['emissions']['total_kg_co2']} kg CO2"},
        {"label": "Expansion priority", "value": recommendations["road_expansion"]["priority"]},
        {"label": "Maintenance priority", "value": recommendations["maintenance"]["priority"]},
    ]
