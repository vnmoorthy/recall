"""Derive semantic room events from consecutive tracked frame states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .masks import point_distance
from .tracker import FrameState, ObjectTrack, PersonTrack


@dataclass(frozen=True)
class Event:
    ts: float
    type: str
    object_id: int | None = None
    person_id: int | None = None
    direction: str | None = None
    frame_path: str | None = None
    crop_path: str | None = None
    id: int | None = None


class EventEngine:
    def __init__(self, wrist_radius: float = 60.0, attribution_seconds: float = 1.5):
        self.wrist_radius = wrist_radius
        self.attribution_seconds = attribution_seconds
        self.previous: FrameState | None = None
        self.wrist_history: deque[tuple[float, int, tuple[tuple[float, float], ...]]] = deque()
        self.carriers: dict[int, int] = {}

    def _remember_wrists(self, state: FrameState) -> None:
        for person in state.persons:
            points = tuple((float(w[0]), float(w[1])) for w in person.wrists if len(w) >= 3 and float(w[2]) >= 0.3)
            if points:
                self.wrist_history.append((state.timestamp, person.id, points))
        cutoff = state.timestamp - self.attribution_seconds
        while self.wrist_history and self.wrist_history[0][0] < cutoff:
            self.wrist_history.popleft()

    def _attributed_person(self, obj: ObjectTrack) -> int | None:
        best = (float("inf"), None)
        for _, person_id, points in self.wrist_history:
            for point in points:
                distance = point_distance(obj.mask, point)
                if distance < best[0]:
                    best = distance, person_id
        return int(best[1]) if best[1] is not None and best[0] <= self.wrist_radius else None

    @staticmethod
    def _direction(obj: ObjectTrack) -> str | None:
        vx, vy = obj.velocity
        if abs(vx) >= abs(vy) and abs(vx) > 0:
            return "right" if vx > 0 else "left"
        if abs(vy) > 0:
            return "toward camera" if vy > 0 else "away"
        return None

    def update(self, state: FrameState) -> list[Event]:
        events: list[Event] = []
        self._remember_wrists(state)
        if self.previous is None:
            events.extend(Event(state.timestamp, "object_appeared", obj.id, frame_path=state.frame_path) for obj in state.objects if obj.state != "missing")
            events.extend(Event(state.timestamp, "person_entered", person_id=p.id, frame_path=state.frame_path) for p in state.persons)
            self.previous = state
            return events

        old_objects = {obj.id: obj for obj in self.previous.objects}
        new_objects = {obj.id: obj for obj in state.objects}
        for retired_id in old_objects.keys() - new_objects.keys():
            self.carriers.pop(retired_id, None)
        old_people = {person.id for person in self.previous.persons}
        new_people = {person.id for person in state.persons}
        events.extend(Event(state.timestamp, "person_entered", person_id=person_id, frame_path=state.frame_path) for person_id in sorted(new_people - old_people))
        events.extend(Event(state.timestamp, "person_left", person_id=person_id, frame_path=state.frame_path) for person_id in sorted(old_people - new_people))

        for object_id, obj in new_objects.items():
            old = old_objects.get(object_id)
            if old is None:
                if obj.state != "missing":
                    events.append(Event(state.timestamp, "object_appeared", object_id, frame_path=state.frame_path))
                continue
            person_id = self._attributed_person(old) or self.carriers.get(object_id)
            if old.state != "missing" and obj.state == "missing":
                direction = self._direction(obj)
                events.append(Event(state.timestamp, "object_missing", object_id, person_id, direction, state.frame_path))
                if person_id is not None:
                    self.carriers[object_id] = person_id
                    events.append(Event(state.timestamp, "picked_up", object_id, person_id, direction, state.frame_path))
            if (
                old.state == "carried"
                or (old.state == "missing" and object_id in self.carriers)
            ) and obj.state in {"present", "stationary"}:
                events.append(Event(state.timestamp, "put_down", object_id, person_id, frame_path=state.frame_path))
                self.carriers.pop(object_id, None)
            shift = math.dist(old.centroid, obj.centroid)
            if obj.state != "missing" and shift > 0.08 * state.frame_width:
                events.append(Event(state.timestamp, "object_moved", object_id, person_id, self._direction(obj), state.frame_path))
                if person_id is not None and old.state != "carried":
                    obj.state = "carried"
                    self.carriers[object_id] = person_id
                    events.append(Event(state.timestamp, "picked_up", object_id, person_id, self._direction(obj), state.frame_path))

        self.previous = state
        return events
