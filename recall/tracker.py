"""Stable person and object tracking for Recall's visual memory."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Literal

import numpy as np

from .masks import area as mask_area
from .masks import centroid as mask_centroid
from .masks import iou as mask_iou

TrackState = Literal["present", "stationary", "carried", "missing"]


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    class_name: str
    score: float
    mask: np.ndarray
    centroid: tuple[float, float] | None = None

    def center(self) -> tuple[float, float]:
        return self.centroid if self.centroid is not None else mask_centroid(self.mask)


@dataclass
class ObjectTrack:
    id: int
    class_id: int
    class_name: str
    mask: np.ndarray
    centroid: tuple[float, float]
    state: TrackState
    first_seen: float
    last_seen: float
    name: str | None = None
    name_pending: bool = False
    stationary_since: float | None = None
    previous_centroid: tuple[float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    score: float = 0.0

    def snapshot(self) -> "ObjectTrack":
        return replace(self, mask=self.mask.copy())


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[float, float, float, float]
    score: float
    wrists: tuple[tuple[float, float, float], ...] = ()


@dataclass
class PersonTrack:
    id: int
    bbox: tuple[float, float, float, float]
    wrists: tuple[tuple[float, float, float], ...]
    first_seen: float
    last_seen: float
    name: str | None = None
    name_pending: bool = False
    missing_frames: int = 0
    score: float = 0.0

    def snapshot(self) -> "PersonTrack":
        return replace(self)


@dataclass(frozen=True)
class FrameState:
    timestamp: float
    frame_width: int
    frame_height: int
    objects: tuple[ObjectTrack, ...]
    persons: tuple[PersonTrack, ...]
    frame_path: str | None = None


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    def __init__(
        self, iou_threshold: float = 0.3, max_missing_frames: int = 15,
        start_id: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_id = max(1, int(start_id))
        self.tracks: dict[int, PersonTrack] = {}

    def update(self, detections: list[PersonDetection], timestamp: float) -> list[PersonTrack]:
        candidates = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                overlap = _bbox_iou(track.bbox, detection.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, index))
        candidates.sort(reverse=True)
        used_tracks, used_detections, matches = set(), set(), {}
        for _, track_id, index in candidates:
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            matches[index] = track_id
        assigned: dict[int, int] = dict(matches)
        for index, detection in enumerate(detections):
            track_id = matches.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = PersonTrack(
                    track_id, detection.bbox, detection.wrists, timestamp, timestamp,
                    score=detection.score,
                )
                assigned[index] = track_id
            else:
                track = self.tracks[track_id]
                track.bbox, track.wrists, track.last_seen = detection.bbox, detection.wrists, timestamp
                track.missing_frames, track.score = 0, detection.score
        active_track_ids = set(assigned.values())
        for track_id, track in list(self.tracks.items()):
            if track_id not in active_track_ids:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing_frames:
                    del self.tracks[track_id]
        return [self.tracks[assigned[i]] for i in range(len(detections))]


class ObjectTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        centroid_ratio: float = 0.05,
        stationary_seconds: float = 3.0,
        missing_seconds: float = 2.0,
        reacquire_ratio: float = 0.08,
        reacquire_seconds: float = 60.0,
        start_id: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.centroid_ratio = centroid_ratio
        self.stationary_seconds = stationary_seconds
        self.missing_seconds = missing_seconds
        self.reacquire_ratio = reacquire_ratio
        self.reacquire_seconds = reacquire_seconds
        self._next_id = max(1, int(start_id))
        self.tracks: dict[int, ObjectTrack] = {}

    def update(
        self, detections: list[ObjectDetection], timestamp: float, frame_width: int
    ) -> list[ObjectTrack]:
        detections = [
            detection for detection in detections
            if mask_area(detection.mask) > 0
            and all(math.isfinite(value) for value in detection.center())
        ]
        centers = [d.center() for d in detections]
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            age = timestamp - track.last_seen
            for index, detection in enumerate(detections):
                if detection.class_id != track.class_id:
                    continue
                distance = math.dist(track.centroid, centers[index])
                overlap = mask_iou(track.mask, detection.mask)
                normal_match = track.state != "missing" and (
                    overlap > self.iou_threshold or distance < self.centroid_ratio * frame_width
                )
                reacquire = track.state == "missing" and age <= self.reacquire_seconds and distance < self.reacquire_ratio * frame_width
                if normal_match or reacquire:
                    score = max(overlap, 1.0 - distance / max(frame_width, 1))
                    candidates.append((score, track_id, index))
        candidates.sort(reverse=True)
        used_tracks, used_detections, matches = set(), set(), {}
        for _, track_id, index in candidates:
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            matches[index] = track_id

        visible = []
        for index, detection in enumerate(detections):
            center = centers[index]
            track_id = matches.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                track = ObjectTrack(
                    track_id, detection.class_id, detection.class_name, detection.mask.copy(),
                    center, "present", timestamp, timestamp, stationary_since=timestamp,
                    score=detection.score,
                )
                self.tracks[track_id] = track
            else:
                track = self.tracks[track_id]
                prior = track.centroid
                velocity = (center[0] - prior[0], center[1] - prior[1])
                motion = math.dist(prior, center)
                if motion < 0.01 * frame_width:
                    if track.stationary_since is None:
                        track.stationary_since = timestamp
                    if timestamp - track.stationary_since >= self.stationary_seconds:
                        track.state = "stationary"
                    elif track.state == "missing":
                        track.state = "present"
                else:
                    track.stationary_since = None
                    if track.state != "carried":
                        track.state = "present"
                track.previous_centroid, track.centroid, track.velocity = prior, center, velocity
                track.mask, track.last_seen, track.score = detection.mask.copy(), timestamp, detection.score
            visible.append(track)

        visible_ids = {track.id for track in visible}
        for track_id, track in list(self.tracks.items()):
            if track_id in visible_ids:
                continue
            if timestamp - track.last_seen >= self.missing_seconds:
                track.state = "missing"
            if timestamp - track.last_seen > self.reacquire_seconds:
                del self.tracks[track_id]
        return visible

    def set_carried(self, track_id: int) -> None:
        if track_id in self.tracks:
            self.tracks[track_id].state = "carried"
            self.tracks[track_id].stationary_since = None

    def frame_state(
        self,
        timestamp: float,
        frame_width: int,
        frame_height: int,
        persons: list[PersonTrack],
        frame_path: str | None = None,
    ) -> FrameState:
        return FrameState(
            timestamp, frame_width, frame_height,
            tuple(track.snapshot() for track in self.tracks.values()),
            tuple(person.snapshot() for person in persons),
            frame_path,
        )
