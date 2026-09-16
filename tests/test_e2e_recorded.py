import os
from pathlib import Path
import json
from urllib import request

import pytest


def test_recorded_pickup_has_person_attribution():
    clip = Path(os.environ.get("RECALL_E2E_CLIP", "assets/demo_clips/person-takes-laptop.mp4"))
    seg = Path("models/yolo26m-seg-bf16-b1.tar.gz")
    pose = Path("models/yolo26m-pose-int8-b1.tar.gz")
    if not (clip.is_file() and seg.is_file() and pose.is_file()):
        pytest.skip("requires recorded pickup clip and exact compiled segmentation/pose packages")
    if os.environ.get("RECALL_HARDWARE_E2E") != "1":
        pytest.skip("set RECALL_HARDWARE_E2E=1 while the recorded clip is streaming")
    api = os.environ.get("RECALL_API", "http://10.42.0.232:8090")
    with request.urlopen(f"{api}/events?window=300", timeout=5) as response:
        events = json.load(response)
    attributed = [
        event for event in events
        if event["type"] == "picked_up" and event.get("person_id") is not None
    ]
    assert attributed, "no person-attributed pickup was persisted from the streamed clip"
