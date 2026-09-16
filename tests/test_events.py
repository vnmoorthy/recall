import numpy as np

from recall.events import EventEngine
from recall.tracker import FrameState, ObjectTrack, PersonTrack


def track(state, center=(50, 50), velocity=(0, 0)):
    mask = np.zeros((120, 200), np.uint8)
    mask[35:65, 35:65] = 255
    return ObjectTrack(1, 1, "laptop", mask, center, state, 0, 1, velocity=velocity)


def person():
    return PersonTrack(7, (10, 0, 100, 115), ((50, 50, 0.9),), 0, 1, "person in blue shirt")


def test_missing_object_is_attributed_to_nearby_wrist_once():
    engine = EventEngine()
    engine.update(FrameState(1.0, 200, 120, (track("present"),), (person(),), "before.jpg"))
    events = engine.update(FrameState(2.0, 200, 120, (track("missing", (80, 50), (30, 0)),), (person(),), "pickup.jpg"))
    picked_up = [event for event in events if event.type == "picked_up"]
    assert len(picked_up) == 1
    assert picked_up[0].person_id == 7
    assert picked_up[0].direction == "right"
    assert any(event.type == "object_missing" for event in events)


def test_empty_scene_does_not_emit_events_repeatedly():
    engine = EventEngine()
    assert engine.update(FrameState(0, 200, 120, (), ())) == []
    assert engine.update(FrameState(60, 200, 120, (), ())) == []


def test_near_wrist_motion_becomes_carried_then_put_down():
    engine = EventEngine()
    engine.update(FrameState(1.0, 200, 120, (track("present"),), (person(),)))
    moving = track("present", (90, 50), (40, 0))
    events = engine.update(FrameState(2.0, 200, 120, (moving,), (person(),)))
    assert any(event.type == "picked_up" for event in events)
    assert moving.state == "carried"
    placed = track("stationary", (90, 50))
    events = engine.update(FrameState(10.0, 200, 120, (placed,), ()))
    put_down = next(event for event in events if event.type == "put_down")
    assert put_down.person_id == 7


def test_missing_pickup_reappearance_is_put_down_by_carrier():
    engine = EventEngine()
    engine.update(FrameState(1, 200, 120, (track("present"),), (person(),)))
    engine.update(FrameState(2, 200, 120, (track("missing"),), (person(),)))
    events = engine.update(FrameState(20, 200, 120, (track("present"),), ()))
    put_down = next(event for event in events if event.type == "put_down")
    assert put_down.person_id == 7
