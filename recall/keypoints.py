"""COCO-17 pose helpers used for custody attribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LEFT_WRIST = 9
RIGHT_WRIST = 10


@dataclass(frozen=True)
class Wrist:
    x: float
    y: float
    confidence: float


def wrists(
    points: Sequence[Sequence[float]], min_confidence: float = 0.3
) -> tuple[Wrist, ...]:
    found = []
    for index in (LEFT_WRIST, RIGHT_WRIST):
        if index >= len(points) or len(points[index]) < 3:
            continue
        x, y, confidence = (float(value) for value in points[index][:3])
        if all(math.isfinite(value) for value in (x, y, confidence)) and confidence >= min_confidence:
            found.append(Wrist(x, y, confidence))
    return tuple(found)


def person_bbox(points: Sequence[Sequence[float]], min_confidence: float = 0.3):
    visible = [
        p for p in points
        if len(p) >= 3
        and all(math.isfinite(float(value)) for value in p[:3])
        and float(p[2]) >= min_confidence
    ]
    if not visible:
        return 0.0, 0.0, 0.0, 0.0
    xs, ys = [float(p[0]) for p in visible], [float(p[1]) for p in visible]
    return min(xs), min(ys), max(xs), max(ys)
