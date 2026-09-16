from pathlib import Path

import numpy as np

from recall.events import Event
from recall.memory import Memory
from recall.tracker import ObjectTrack, PersonTrack


def test_memory_inventory_find_events_and_timeline(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    obj = ObjectTrack(1, 41, "cup", np.ones((4, 4), np.uint8), (25, 30), "stationary", 10, 20, "red ceramic mug")
    person = PersonTrack(2, (0, 0, 10, 10), (), 10, 20, "person in blue shirt")
    memory.upsert_object(obj, "crops/mug.jpg")
    memory.upsert_person(person, "crops/person.jpg")
    event_id = memory.add_event(Event(20, "picked_up", 1, 2, "right", "frames/pickup.jpg"))
    assert memory.inventory()[0]["name"] == "red ceramic mug"
    assert memory.find("red mug")[0]["id"] == 1
    assert memory.events_window(20, now=30)[0]["person_name"] == "person in blue shirt"
    assert memory.timeline(1)[0]["id"] == event_id
    memory.close()
