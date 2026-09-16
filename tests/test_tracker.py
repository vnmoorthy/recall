import numpy as np

from recall.tracker import ObjectDetection, ObjectTracker, PersonDetection, PersonTracker


def object_detection(x, class_id=1, name="cup"):
    mask = np.zeros((100, 200), np.uint8)
    mask[30:60, x : x + 30] = 255
    return ObjectDetection(class_id, name, 0.9, mask)


def test_object_id_stationary_missing_and_reacquired():
    tracker = ObjectTracker()
    visible = tracker.update([object_detection(30)], 0.0, 200)
    assert visible[0].id == 1
    tracker.update([object_detection(31)], 1.0, 200)
    visible = tracker.update([object_detection(31)], 4.1, 200)
    assert visible[0].id == 1
    assert visible[0].state == "stationary"
    tracker.update([], 6.2, 200)
    assert tracker.tracks[1].state == "missing"
    visible = tracker.update([object_detection(34)], 7.0, 200)
    assert visible[0].id == 1
    assert visible[0].state in {"present", "stationary"}


def test_far_same_class_creates_new_object():
    tracker = ObjectTracker()
    tracker.update([object_detection(10)], 0.0, 200)
    visible = tracker.update([object_detection(150)], 0.1, 200)
    assert visible[0].id == 2


def test_people_tracker_keeps_distinct_ids():
    tracker = PersonTracker()
    detections = [PersonDetection((0, 0, 50, 100), 0.9), PersonDetection((100, 0, 150, 100), 0.8)]
    first = tracker.update(detections, 0.0)
    second = tracker.update([PersonDetection((3, 0, 53, 100), 0.9), PersonDetection((97, 0, 147, 100), 0.8)], 0.1)
    assert [p.id for p in first] == [1, 2]
    assert [p.id for p in second] == [1, 2]
    assert all(p.missing_frames == 0 for p in first)


def test_trackers_resume_ids_after_persisted_state():
    objects = ObjectTracker(start_id=12)
    people = PersonTracker(start_id=9)
    assert objects.update([object_detection(30)], 0.0, 200)[0].id == 12
    assert people.update([PersonDetection((0, 0, 50, 100), 0.9)], 0.0)[0].id == 9


def test_zero_timestamp_can_still_become_stationary():
    tracker = ObjectTracker()
    tracker.update([object_detection(30)], 0.0, 200)
    tracker.update([object_detection(30)], 1.0, 200)
    visible = tracker.update([object_detection(30)], 3.1, 200)
    assert visible[0].state == "stationary"


def test_expired_missing_track_releases_its_mask():
    tracker = ObjectTracker(reacquire_seconds=5)
    tracker.update([object_detection(30)], 0.0, 200)
    tracker.update([], 2.1, 200)
    assert tracker.tracks[1].state == "missing"
    tracker.update([], 5.1, 200)
    assert 1 not in tracker.tracks


def test_empty_mask_detection_is_rejected():
    tracker = ObjectTracker()
    empty = ObjectDetection(1, "cup", 0.9, np.zeros((100, 200), np.uint8))
    assert tracker.update([empty], 0.0, 200) == []
