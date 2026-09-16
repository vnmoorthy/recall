"""PyNeat perception graph and frame-to-memory processing.

Hardware imports are lazy so tracking, event, memory, API, and UI tests run
without a Modalix runtime or compiled model archives.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import threading
import time

import cv2
import numpy as np

from .events import EventEngine
from .keypoints import wrists
from .masks import project_letterbox_mask
from .memory import Memory
from .namer import Namer
from .tracker import ObjectDetection, ObjectTracker, PersonDetection, PersonTracker


@dataclass(frozen=True)
class PerceptionConfig:
    rtsp_url: str
    seg_model: str
    pose_model: str
    labels_path: str
    insight_host: str = "10.42.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    channel: int = 0
    pose_fps: int = 10
    min_score: float = 0.35
    max_detections: int = 50
    mask_iou: float = 0.30
    centroid_ratio: float = 0.05
    stationary_seconds: float = 3.0
    missing_seconds: float = 2.0
    reacquire_ratio: float = 0.08
    reacquire_seconds: float = 60.0
    wrist_radius_px: float = 60.0
    attribution_seconds: float = 1.5
    frame_retention_seconds: int = 900


class FrameWriter:
    """Bounded JPEG writer with a 15-minute, one-frame-per-second ring."""

    def __init__(self, directory: str | Path, retention_seconds: int = 900):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_seconds
        self.queue: Queue[tuple[float, np.ndarray]] = Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.last_enqueued_second = -1
        self.errors = 0
        self.thread = threading.Thread(target=self._run, name="recall-frame-writer", daemon=True)
        self.thread.start()

    def submit(self, ts: float, frame: np.ndarray) -> str | None:
        second = int(ts)
        filename = f"frame-{second}.jpg"
        path = self.directory / filename
        if second == self.last_enqueued_second:
            return f"frames/{filename}"
        try:
            self.queue.put_nowait((ts, frame.copy()))
            self.last_enqueued_second = second
            return f"frames/{filename}"
        except Full:
            return None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                ts, frame = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                written = cv2.imwrite(
                    str(self.directory / f"frame-{int(ts)}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 84],
                )
            except Exception:
                written = False
            if not written:
                self.errors += 1
            cutoff = ts - self.retention_seconds
            for candidate in self.directory.glob("frame-*.jpg"):
                try:
                    if int(candidate.stem.split("-")[-1]) < cutoff:
                        candidate.unlink()
                except (ValueError, OSError):
                    continue
            self.queue.task_done()


class TrackWriter:
    """Persist latest track state without putting SQLite on the frame path."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self.queue: Queue[tuple[tuple, tuple, tuple[int, int]]] = Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.dropped = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._run, name="recall-track-writer", daemon=True)
        self.thread.start()

    def submit(self, objects, persons, frame_size: tuple[int, int]) -> None:
        lightweight = tuple(replace(obj, mask=np.empty((0, 0), np.uint8)) for obj in objects)
        item = lightweight, tuple(persons), frame_size
        try:
            self.queue.put_nowait(item)
        except Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped += 1
            except Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except Full:
                self.dropped += 1

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                objects, persons, frame_size = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.memory.upsert_tracks(objects, persons, frame_size)
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()


class SceneProcessor:
    def __init__(
        self, memory: Memory, cfg: PerceptionConfig | None = None,
        track_writer: TrackWriter | None = None,
    ):
        self.memory = memory
        self.track_writer = track_writer
        values = cfg or PerceptionConfig("", "", "", "")
        self.objects = ObjectTracker(
            iou_threshold=values.mask_iou,
            centroid_ratio=values.centroid_ratio,
            stationary_seconds=values.stationary_seconds,
            missing_seconds=values.missing_seconds,
            reacquire_ratio=values.reacquire_ratio,
            reacquire_seconds=values.reacquire_seconds,
        )
        self.people = PersonTracker()
        self.events = EventEngine(values.wrist_radius_px, values.attribution_seconds)

    def process(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        object_detections: list[ObjectDetection],
        person_detections: list[PersonDetection],
        frame_path: str | None = None,
    ):
        width, height = frame_size
        people = self.people.update(person_detections, timestamp)
        self.objects.update(object_detections, timestamp, width)
        state = self.objects.frame_state(timestamp, width, height, people, frame_path)
        derived = self.events.update(state)
        for event in derived:
            if event.type == "picked_up" and event.object_id is not None:
                state_track = next((obj for obj in state.objects if obj.id == event.object_id), None)
                if state_track is not None and state_track.state != "missing":
                    self.objects.set_carried(event.object_id)
        if self.track_writer:
            self.track_writer.submit(state.objects, state.persons, frame_size)
        else:
            self.memory.upsert_tracks(state.objects, state.persons, frame_size)
        stored = []
        for event in derived:
            stored.append(event.__class__(**{**event.__dict__, "id": self.memory.add_event(event)}))
        return state, stored


def decode_segments(tensors, frame_size: tuple[int, int], labels: list[str], top_k=50):
    import pyneat

    frame_w, frame_h = frame_size
    detections = []
    for item in pyneat.decode_segmentation(tensors, clamp_to=frame_size, top_k=top_k, strict=False):
        boxes = np.asarray(item.boxes.to_numpy(copy=True)).reshape((-1, 6))
        masks = np.asarray(item.masks.to_numpy(copy=True)).reshape((-1, 160, 160))
        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2, score, class_id = box.tolist()
            if x2 <= x1 or y2 <= y1 or not np.isfinite(box).all():
                continue
            full_mask = project_letterbox_mask(mask.astype(np.uint8), (x1, y1, x2, y2), frame_size)
            class_index = int(class_id)
            if class_index < 0 or not np.any(full_mask):
                continue
            detections.append(ObjectDetection(class_index, labels[class_index] if class_index < len(labels) else f"class_{class_index}", float(score), full_mask))
    return detections


def decode_people(tensors, frame_size: tuple[int, int], top_k=20):
    import pyneat

    people = []
    for item in pyneat.decode_pose(tensors, clamp_to=frame_size, top_k=top_k):
        boxes = np.asarray(item.boxes.to_numpy(copy=True)).reshape((-1, 6))
        points = np.asarray(item.keypoints.to_numpy(copy=True)).reshape((-1, 17, 3))
        for box, pose in zip(boxes, points):
            if box[2] <= box[0] or box[3] <= box[1] or not np.isfinite(box).all():
                continue
            found = tuple((w.x, w.y, w.confidence) for w in wrists(pose))
            people.append(PersonDetection(tuple(float(v) for v in box[:4]), float(box[4]), found))
    return people


def build_neat_graph(cfg: PerceptionConfig, width: int, height: int, fps: int):
    """Build the combined graph; model archives are validated by `pyneat.Model`."""
    import pyneat

    source_options = pyneat.RtspDecodedInputOptions()
    source_options.url = cfg.rtsp_url
    source_options.payload_type = 96
    source_options.insert_queue = True
    source_options.auto_caps_from_stream = True
    source_options.fallback_h264_width = width
    source_options.fallback_h264_height = height
    source_options.fallback_h264_fps = fps
    source_options.decoder_raw_output = True
    source_options.output_caps.enable = True
    source_options.output_caps.format = pyneat.Format.NV12
    source_options.output_caps.width = width
    source_options.output_caps.height = height
    source_options.output_caps.fps = fps
    source = pyneat.groups.rtsp_decoded_input(source_options)

    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    sender_options.host, sender_options.channel = cfg.insight_host, cfg.channel
    sender_options.video_port_base = cfg.video_port
    video = pyneat.Graph("video")
    video.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_options))

    seg_options = pyneat.ModelOptions()
    seg_options.preprocess.kind = pyneat.InputKind.Image
    seg_options.preprocess.enable = pyneat.AutoFlag.On
    seg_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    seg_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    seg_options.decode_type = pyneat.BoxDecodeType.YoloV26Seg
    seg_options.score_threshold, seg_options.top_k = cfg.min_score, cfg.max_detections
    seg_model = pyneat.Model(cfg.seg_model, seg_options)
    seg = pyneat.Graph("segmentation")
    seg.connect(pyneat.nodes.input("seg"), seg_model)
    seg.add(pyneat.nodes.output("segments", pyneat.OutputOptions.every_frame(4)))

    pose_options = pyneat.ModelOptions()
    pose_options.preprocess.kind = pyneat.InputKind.Image
    pose_options.preprocess.enable = pyneat.AutoFlag.On
    pose_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    pose_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    pose_options.decode_type, pose_options.num_classes = pyneat.BoxDecodeType.YoloV26Pose, 1
    pose_options.score_threshold, pose_options.top_k = cfg.min_score, 20
    pose_model = pyneat.Model(cfg.pose_model, pose_options)
    pose = pyneat.Graph("pose")
    pose.connect(pyneat.nodes.input("pose"), pyneat.nodes.video_rate())
    pose.add(pyneat.nodes.caps_raw("NV12", width, height, cfg.pose_fps, pyneat.CapsMemory.Any))
    pose.add(pose_model)
    pose.add(pyneat.nodes.output("poses", pyneat.OutputOptions.every_frame(4)))

    frame = pyneat.Graph("frame")
    frame.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    branch = pyneat.graphs.branch("source", ["video", "seg", "pose", "frame"])
    joined = pyneat.graphs.combine(["frame", "segments"], "segmentation_output", pyneat.CombinePolicy.ByFrame)
    graph = pyneat.Graph("recall")
    graph.connect(source, branch)
    graph.connect(branch, video)
    graph.connect(branch, seg)
    graph.connect(branch, pose)
    graph.connect(branch, frame)
    graph.connect(frame, joined)
    graph.connect(seg, joined)
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = 4
    options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    options.output_memory = pyneat.OutputMemory.Owned
    return graph, options


COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _extract_tensors(sample) -> list:
    import pyneat

    if sample is None:
        return []
    if isinstance(sample, (list, tuple)):
        return list(sample)
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for field in getattr(sample, "fields", []):
        tensors.extend(_extract_tensors(field))
    return tensors


def _find_field(sample, label: str):
    if sample is None:
        return None
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = _find_field(field, label)
        if found is not None:
            return found
    return None


def _joined_field(sample, label: str, index: int):
    import pyneat

    found = _find_field(sample, label)
    if found is not None:
        return found
    fields = list(getattr(sample, "fields", []))
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > index:
        return fields[index]
    raise RuntimeError(f"joined output missing {label}")


def _tensor_bgr(tensor) -> np.ndarray:
    def dimension(name: str) -> int:
        value = getattr(tensor, name)
        return int(value() if callable(value) else value)

    if tensor.is_nv12():
        width, height = dimension("width"), dimension("height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        nv12 = payload[: width * height * 3 // 2].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    if tensor.is_i420():
        width, height = dimension("width"), dimension("height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        i420 = payload[: width * height * 3 // 2].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))
    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    return np.ascontiguousarray(frame)


def _decode_pose_records(tensors, frame_size: tuple[int, int]):
    import pyneat

    detections, records = [], []
    for item in pyneat.decode_pose(tensors, clamp_to=frame_size, top_k=20):
        boxes = np.asarray(item.boxes.to_numpy(copy=True)).reshape((-1, 6))
        points = np.asarray(item.keypoints.to_numpy(copy=True)).reshape((-1, 17, 3))
        for box, pose in zip(boxes, points):
            found = tuple((w.x, w.y, w.confidence) for w in wrists(pose))
            detections.append(
                PersonDetection(tuple(float(v) for v in box[:4]), float(box[4]), found)
            )
            records.append((box, pose))
    return detections, records


def _sample_clock(sample) -> tuple[int, str]:
    pts = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    return pts, frame_id


def _pose_metadata(records, track_ids: list[int] | None = None) -> str:
    poses = []
    for index, (box, points) in enumerate(records, 1):
        poses.append({
            "id": str(track_ids[index - 1]) if track_ids and index <= len(track_ids) else f"pose_{index}",
            "label": "person",
            "confidence": round(float(box[4]), 3),
            "bbox": [
                round(float(box[0])), round(float(box[1])),
                round(max(0.0, float(box[2] - box[0]))),
                round(max(0.0, float(box[3] - box[1]))),
            ],
            "keypoints": [
                {
                    "name": COCO_KEYPOINT_NAMES[k], "x": round(float(x)),
                    "y": round(float(y)), "confidence": round(float(confidence), 3),
                }
                for k, (x, y, confidence) in enumerate(points)
            ],
        })
    return json.dumps({"poses": poses}, separators=(",", ":"))


def _segmentation_metadata(state) -> str:
    segments = []
    for track in state.objects:
        if track.state == "missing" or track.last_seen != state.timestamp:
            continue
        binary = (np.asarray(track.mask) > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(largest, 0.006 * cv2.arcLength(largest, True), True)
        if len(polygon) < 3:
            continue
        x, y, width, height = cv2.boundingRect(largest)
        segments.append({
            "id": str(track.id),
            "label": track.name or track.class_name,
            "confidence": round(float(track.score), 3),
            "bbox": [x, y, width, height],
            "mask_format": "polygon",
            "mask": [[int(point[0][0]), int(point[0][1])] for point in polygon],
        })
    kept = []
    size = len('{"segments":[]}')
    for segment in sorted(segments, key=lambda item: item["confidence"], reverse=True):
        encoded_size = len(json.dumps(segment, separators=(",", ":"))) + 1
        if size + encoded_size > 60_000:
            continue
        kept.append(segment)
        size += encoded_size
    return json.dumps({"segments": kept}, separators=(",", ":"))


def _crop_for_track(frame: np.ndarray, track, margin: float = 0.12) -> np.ndarray | None:
    if hasattr(track, "mask"):
        points = cv2.findNonZero((np.asarray(track.mask) > 0).astype(np.uint8))
        if points is None:
            return None
        x, y, width, height = cv2.boundingRect(points)
    else:
        x1, y1, x2, y2 = track.bbox
        x, y = max(0, int(x1)), max(0, int(y1))
        width, height = int(x2) - x, int(y2) - y
    extra_x, extra_y = int(width * margin), int(height * margin)
    x0, y0 = max(0, x - extra_x), max(0, y - extra_y)
    x1 = min(frame.shape[1], x + max(1, width) + extra_x)
    y1 = min(frame.shape[0], y + max(1, height) + extra_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop.copy())


def _load_config(path: Path) -> tuple[PerceptionConfig, int, int, int]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, models, inference, insight, tracking, memory = (
        raw["source"], raw["models"], raw["inference"], raw["insight"],
        raw["tracking"], raw["memory"],
    )
    cfg = PerceptionConfig(
        rtsp_url=source["url"], seg_model=models["seg"], pose_model=models["pose"],
        labels_path=models["labels"], insight_host=insight["host"],
        video_port=int(insight["video_port"]), metadata_port=int(insight["metadata_port"]),
        channel=int(insight["channel"]), pose_fps=int(inference["pose_fps"]),
        min_score=float(inference["min_score"]),
        max_detections=int(inference["max_detections"]),
        mask_iou=float(tracking["mask_iou"]),
        centroid_ratio=float(tracking["centroid_ratio"]),
        stationary_seconds=float(tracking["stationary_seconds"]),
        missing_seconds=float(tracking["missing_seconds"]),
        reacquire_ratio=float(tracking["reacquire_ratio"]),
        reacquire_seconds=float(tracking["reacquire_seconds"]),
        wrist_radius_px=float(tracking["wrist_radius_px"]),
        attribution_seconds=float(tracking["attribution_seconds"]),
        frame_retention_seconds=int(memory["frame_retention_seconds"]),
    )
    capture = cv2.VideoCapture(cfg.rtsp_url)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or source["width"])
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or source["height"])
    fps = int(round(capture.get(cv2.CAP_PROP_FPS) or source["fps"]))
    capture.release()
    _validate_config(cfg, width, height, fps)
    return cfg, width, height, fps


def _validate_config(cfg: PerceptionConfig, width: int, height: int, fps: int) -> None:
    if not cfg.rtsp_url.startswith("rtsp://"):
        raise ValueError("source URL must use RTSP")
    if min(width, height, fps) <= 0:
        raise ValueError("source width, height, and FPS must be positive")
    if not 0 < cfg.min_score <= 1:
        raise ValueError("min_score must be in (0, 1]")
    if not 1 <= cfg.max_detections <= 200:
        raise ValueError("max_detections must be in [1, 200]")
    if not 1 <= cfg.pose_fps <= fps:
        raise ValueError("pose_fps must be between 1 and source FPS")
    if not 0 <= cfg.channel <= 3:
        raise ValueError("Insight channel must be in [0, 3]")
    for name, value in (
        ("mask_iou", cfg.mask_iou), ("centroid_ratio", cfg.centroid_ratio),
        ("reacquire_ratio", cfg.reacquire_ratio),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    for name, value in (
        ("stationary_seconds", cfg.stationary_seconds),
        ("missing_seconds", cfg.missing_seconds),
        ("reacquire_seconds", cfg.reacquire_seconds),
        ("wrist_radius_px", cfg.wrist_radius_px),
        ("attribution_seconds", cfg.attribution_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.frame_retention_seconds < 60:
        raise ValueError("frame retention must be at least 60 seconds")


def _write_metrics(
    path: Path, seg_count: int, pose_count: int, started: float, tracked: int,
    metadata_errors: int = 0, frame_write_errors: int = 0,
    track_write_drops: int = 0, track_write_errors: int = 0,
):
    elapsed = max(0.001, time.monotonic() - started)
    payload = {
        "updated_at": time.time(), "seg_fps": round(seg_count / elapsed, 1),
        "pose_fps": round(pose_count / elapsed, 1), "objects_tracked": tracked,
        "perception_mode": "hardware", "metadata_errors": metadata_errors,
        "frame_write_errors": frame_write_errors,
        "track_write_drops": track_write_drops,
        "track_write_errors": track_write_errors,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def run_perception(config_path: Path, root: Path) -> int:
    import pyneat

    cfg, width, height, fps = _load_config(config_path)
    for model_path in (cfg.seg_model, cfg.pose_model):
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
    labels = Path(cfg.labels_path).read_text(encoding="utf-8").splitlines()
    memory = Memory(root / "data/recall.db")
    track_writer = TrackWriter(memory)
    processor = SceneProcessor(memory, cfg, track_writer)
    writer = FrameWriter(root / "media/frames", cfg.frame_retention_seconds)
    metrics_path = root / "run/perception-status.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    qwen = Path("/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4/devkit")
    gemma = Path("/media/nvme/llima/models/gemma-4-E2B-it-GPTQ-a16w4/devkit")
    model = qwen.parent.name if qwen.is_dir() else gemma.parent.name if gemma.is_dir() else None
    def persist_name(kind, track, crop_path):
        if kind == "object":
            memory.upsert_object(track, crop_path, (width, height))
        else:
            memory.upsert_person(track, crop_path)

    namer = Namer(
        "http://127.0.0.1:9998", model, root / "logs/vlm.jsonl",
        on_named=persist_name, crop_dir=root / "media/crops",
    ) if model else None

    graph, options = build_neat_graph(cfg, width, height, fps)
    run = graph.build(options)
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host, metadata_options.channel = cfg.insight_host, cfg.channel
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata = pyneat.MetadataSender(metadata_options)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    latest_people = []
    seg_count = pose_count = 0
    metadata_errors = 0
    metric_started = time.monotonic()
    print(f"perception {cfg.rtsp_url} {width}x{height}@{fps} -> {cfg.insight_host}", flush=True)
    try:
        while not stopped:
            while True:
                pose_sample = run.pull("poses", 0)
                if pose_sample is None:
                    break
                latest_people, pose_records = _decode_pose_records(
                    _extract_tensors(pose_sample), (width, height)
                )
                pose_tracks = processor.people.update(latest_people, time.time())
                pts, frame_id = _sample_clock(pose_sample)
                if not metadata.send_metadata(
                    "pose-estimation",
                    _pose_metadata(pose_records, [track.id for track in pose_tracks]),
                    pts,
                    frame_id,
                ):
                    metadata_errors += 1
                pose_count += 1

            sample = run.pull("segmentation_output", 1000)
            if sample is None:
                if time.monotonic() - metric_started >= 5:
                    _write_metrics(
                        metrics_path, seg_count, pose_count, metric_started,
                        len(processor.objects.tracks), metadata_errors, writer.errors,
                        track_writer.dropped, track_writer.errors,
                    )
                    seg_count = pose_count = 0
                    metric_started = time.monotonic()
                continue
            frame_tensors = _extract_tensors(_joined_field(sample, "frame", 0))
            segment_tensors = _extract_tensors(_joined_field(sample, "segments", 1))
            frame = _tensor_bgr(frame_tensors[0])
            timestamp = time.time()
            frame_path = writer.submit(timestamp, frame)
            detections = [
                detection
                for detection in decode_segments(
                    segment_tensors, (width, height), labels, cfg.max_detections
                )
                if detection.class_name != "person"
            ]
            state, _events = processor.process(
                timestamp, (width, height), detections, latest_people, frame_path
            )
            pts, frame_id = _sample_clock(sample)
            if not metadata.send_metadata(
                "segmentation", _segmentation_metadata(state), pts, frame_id
            ):
                metadata_errors += 1
            seg_count += 1

            if namer:
                for track in processor.objects.tracks.values():
                    if track.state == "stationary" and not track.name:
                        crop = _crop_for_track(frame, track)
                        if crop is not None:
                            namer.submit("object", track, crop)
                for track in processor.people.tracks.values():
                    if not track.name and track.missing_frames == 0:
                        crop = _crop_for_track(frame, track)
                        if crop is not None:
                            namer.submit("person", track, crop)
            if time.monotonic() - metric_started >= 5:
                _write_metrics(
                    metrics_path, seg_count, pose_count, metric_started,
                    len(processor.objects.tracks), metadata_errors, writer.errors,
                    track_writer.dropped, track_writer.errors,
                )
                seg_count = pose_count = 0
                metric_started = time.monotonic()
    finally:
        run.close()
        writer.close()
        if namer:
            namer.close()
        track_writer.close()
        memory.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recall Modalix perception worker")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        return run_perception(args.config.resolve(), args.root.resolve())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"perception error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
