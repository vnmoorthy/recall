import json

import numpy as np

from recall.perception import _crop_for_track, _pose_metadata, _segmentation_metadata
from recall.tracker import FrameState, ObjectTrack


def test_stable_segmentation_id_and_frame_polygon():
    mask = np.zeros((80, 120), np.uint8)
    mask[20:50, 35:75] = 255
    track = ObjectTrack(
        17, 41, "cup", mask, (54.5, 34.5), "stationary", 1.0, 2.0,
        name="red mug", score=0.91,
    )
    state = FrameState(2.0, 120, 80, (track,), ())
    payload = json.loads(_segmentation_metadata(state))
    assert payload["segments"][0]["id"] == "17"
    assert payload["segments"][0]["label"] == "red mug"
    assert len(payload["segments"][0]["mask"]) >= 4
    assert _crop_for_track(np.zeros((80, 120, 3), np.uint8), track).startswith(b"\xff\xd8")


def test_pose_metadata_has_coco_keypoints():
    box = np.array([10, 12, 90, 110, 0.93, 0], np.float32)
    points = np.zeros((17, 3), np.float32)
    points[:, 0], points[:, 1], points[:, 2] = 25, 35, 0.8
    payload = json.loads(_pose_metadata([(box, points)]))
    pose = payload["poses"][0]
    assert pose["id"] == "pose_1"
    assert len(pose["keypoints"]) == 17
    assert pose["keypoints"][9]["name"] == "left_wrist"
