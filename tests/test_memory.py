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


def test_wal_batch_normalizes_centroid_and_reset_restarts_event_ids(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    mask = np.ones((4, 4), np.uint8)
    obj = ObjectTrack(4, 41, "cup", mask, (400, 225), "stationary", 1, 2)
    person = PersonTrack(3, (0, 0, 10, 10), (), 1, 2)
    memory.upsert_tracks((obj,), (person,), (800, 450))
    assert memory.inventory()[0]["last_centroid"] == [0.5, 0.5]
    assert memory._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert memory.add_event(Event(2, "object_appeared", 4)) == 1
    memory.reset()
    assert memory.add_event(Event(3, "object_appeared")) == 1
    memory.close()
