"""Deterministic Recall scene used while compiled vision models are unavailable."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from .events import Event
from .memory import Memory
from .tracker import ObjectTrack, PersonTrack


OBJECTS = [
    (1, "cup", "red ceramic mug", (120, 245), "missing"),
    (2, "laptop", "silver laptop", (405, 220), "missing"),
    (3, "bottle", "green water bottle", (620, 235), "stationary"),
    (4, "cell phone", "black cell phone", (330, 330), "stationary"),
    (5, "book", "blue hardcover book", (535, 340), "stationary"),
]


def _frame(path: Path, title: str, mug=True, laptop=True, person=False):
    image = np.full((450, 800, 3), (232, 235, 232), np.uint8)
    cv2.rectangle(image, (55, 175), (745, 390), (120, 105, 86), -1)
    cv2.rectangle(image, (55, 175), (745, 390), (65, 58, 48), 3)
    if mug:
        cv2.rectangle(image, (95, 230), (150, 310), (35, 35, 205), -1)
        cv2.circle(image, (150, 268), 22, (35, 35, 205), 7)
    if laptop:
        cv2.rectangle(image, (350, 205), (500, 300), (185, 185, 185), -1)
        cv2.rectangle(image, (350, 205), (500, 300), (65, 65, 65), 3)
    cv2.rectangle(image, (590, 210), (635, 320), (30, 130, 60), -1)
    cv2.rectangle(image, (300, 320), (375, 355), (25, 25, 25), -1)
    cv2.rectangle(image, (485, 315), (585, 365), (170, 80, 30), -1)
    if person:
        cv2.circle(image, (685, 95), 30, (145, 175, 205), -1)
        cv2.rectangle(image, (640, 125), (730, 290), (185, 90, 30), -1)
    cv2.putText(image, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def bootstrap(memory: Memory, media_dir: str | Path) -> None:
    if memory.inventory():
        return
    media = Path(media_dir)
    frames, crops = media / "frames", media / "crops"
    frames.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)
    _frame(frames / "scene-inventory.jpg", "12:01 - Five objects on the worktable")
    _frame(frames / "mug-pickup.jpg", "12:04 - Blue-shirt person moves the red mug", mug=False, person=True)
    _frame(frames / "laptop-pickup.jpg", "12:07 - Blue-shirt person takes the laptop", mug=False, laptop=False, person=True)
    _frame(frames / "scene-now.jpg", "Now - Recall continues watching offline", mug=False, laptop=False)
    now = time.time()
    person = PersonTrack(1, (620, 70, 745, 350), (), now - 600, now - 120, "person in blue shirt")
    memory.upsert_person(person, "crops/person-blue-shirt.jpg")
    for object_id, class_name, name, center, state in OBJECTS:
        mask = np.zeros((450, 800), np.uint8)
        cv2.circle(mask, center, 30, 255, -1)
        obj = ObjectTrack(object_id, object_id, class_name, mask, center, state, now - 900, now - 120 if state == "missing" else now, name=name, stationary_since=now - 800)
        memory.upsert_object(obj, "frames/scene-now.jpg", (800, 450))
    events = [
        Event(now - 540, "object_appeared", 1, frame_path="frames/scene-inventory.jpg"),
        Event(now - 540, "object_appeared", 2, frame_path="frames/scene-inventory.jpg"),
        Event(now - 540, "object_appeared", 3, frame_path="frames/scene-inventory.jpg"),
        Event(now - 540, "object_appeared", 4, frame_path="frames/scene-inventory.jpg"),
        Event(now - 540, "object_appeared", 5, frame_path="frames/scene-inventory.jpg"),
        Event(now - 360, "picked_up", 1, 1, "right", "frames/mug-pickup.jpg"),
        Event(now - 358, "object_missing", 1, 1, "right", "frames/mug-pickup.jpg"),
        Event(now - 180, "picked_up", 2, 1, "left", "frames/laptop-pickup.jpg"),
        Event(now - 178, "object_missing", 2, 1, "left", "frames/laptop-pickup.jpg"),
        Event(now - 120, "person_left", person_id=1, frame_path="frames/laptop-pickup.jpg"),
    ]
    for event in events:
        memory.add_event(event)
