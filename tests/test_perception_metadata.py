import json
import time

import numpy as np

from recall.perception import (
    FrameWriter, PerceptionConfig, TrackWriter, _crop_for_track, _pose_metadata,
    _segmentation_metadata, _tensor_bgr, _validate_config,
)
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
    crop = _crop_for_track(np.zeros((80, 120, 3), np.uint8), track)
    assert crop is not None
    assert crop.shape[0] > 30 and crop.shape[1] > 40


def test_pose_metadata_has_coco_keypoints():
    box = np.array([10, 12, 90, 110, 0.93, 0], np.float32)
    points = np.zeros((17, 3), np.float32)
    points[:, 0], points[:, 1], points[:, 2] = 25, 35, 0.8
    payload = json.loads(_pose_metadata([(box, points)]))
    pose = payload["poses"][0]
    assert pose["id"] == "pose_1"
    assert len(pose["keypoints"]) == 17
    assert pose["keypoints"][9]["name"] == "left_wrist"
    tracked = json.loads(_pose_metadata([(box, points)], [42]))
    assert tracked["poses"][0]["id"] == "42"


def test_pose_metadata_sanitizes_nonfinite_keypoints():
    box = np.array([0, 0, 20, 20, 0.8, 0], np.float32)
    points = np.zeros((17, 3), np.float32)
    points[9] = [np.nan, np.inf, np.nan]
    point = json.loads(_pose_metadata([(box, points)]))["poses"][0]["keypoints"][9]
    assert point == {"name": "left_wrist", "x": 0, "y": 0, "confidence": 0.0}


def test_tensor_bgr_rejects_truncated_or_odd_yuv420():
    class Tensor:
        def __init__(self, width, height, payload):
            self.width, self.height, self.payload = width, height, payload

        def is_nv12(self):
            return True

        def is_i420(self):
            return False

        def copy_payload_bytes(self):
            return self.payload

    for tensor, expected in (
        (Tensor(4, 4, b"\0" * 10), "truncated"),
        (Tensor(3, 4, b"\0" * 18), "dimensions"),
    ):
        try:
            _tensor_bgr(tensor)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("malformed YUV tensor was accepted")


def test_segmentation_metadata_stays_inside_udp_budget():
    mask = np.zeros((80, 120), np.uint8)
    mask[10:70, 10:110] = 255
    tracks = tuple(
        ObjectTrack(i, 41, "cup", mask, (60, 40), "stationary", 1, 2, score=0.9)
        for i in range(1, 801)
    )
    encoded = _segmentation_metadata(FrameState(2, 120, 80, tracks, ()))
    assert len(encoded.encode()) <= 60_000
    assert len(json.loads(encoded)["segments"]) < len(tracks)


def test_frame_writer_returns_public_path_and_drains(tmp_path):
    writer = FrameWriter(tmp_path / "frames")
    ts = float(int(time.time()))
    assert writer.submit(ts, np.zeros((20, 30, 3), np.uint8)) == f"frames/frame-{int(ts)}.jpg"
    writer.close()
    assert (tmp_path / "frames" / f"frame-{int(ts)}.jpg").is_file()


def test_perception_configuration_rejects_invalid_runtime_values():
    cfg = PerceptionConfig("http://camera", "seg", "pose", "labels")
    try:
        _validate_config(cfg, 640, 480, 30)
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("invalid source protocol was accepted")
    invalid_port = PerceptionConfig(
        "rtsp://camera", "seg", "pose", "labels", video_port=70000,
    )
    try:
        _validate_config(invalid_port, 640, 480, 30)
    except ValueError as exc:
        assert "ports" in str(exc)
    else:
        raise AssertionError("invalid Insight port was accepted")


def test_track_writer_drains_without_erasing_vlm_name(tmp_path):
    from recall.memory import Memory

    memory = Memory(tmp_path / "recall.db")
    mask = np.ones((10, 10), np.uint8)
    named = ObjectTrack(1, 41, "cup", mask, (5, 5), "stationary", 1, 2, name="red mug")
    memory.upsert_object(named, "crops/object-1.jpg", (10, 10))
    stale = ObjectTrack(1, 41, "cup", mask, (5, 5), "stationary", 1, 3, name=None)
    writer = TrackWriter(memory)
    writer.submit((stale,), (), (10, 10))
    writer.close()
    item = memory.inventory()[0]
    assert item["name"] == "red mug"
    assert item["crop_path"] == "crops/object-1.jpg"
    memory.close()
