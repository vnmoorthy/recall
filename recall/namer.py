"""One-shot object and clothing naming on a bounded VLM worker."""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from urllib import request

import cv2
import numpy as np

from .vlm_lock import exclusive_vlm


@dataclass
class NamingTask:
    kind: str
    track: object
    image: bytes | np.ndarray


class Namer:
    def __init__(
        self, base_url: str, model: str, log_path: str | Path,
        on_named=None, crop_dir: str | Path | None = None,
    ):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_named = on_named
        self.crop_dir = Path(crop_dir) if crop_dir else None
        if self.crop_dir:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.queue: Queue[NamingTask] = Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.submitted: set[tuple[str, int]] = set()
        self._stats_lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.latencies: deque[float] = deque(maxlen=512)
        self.thread = threading.Thread(target=self._run, name="recall-namer", daemon=True)
        self.thread.start()

    def submit(self, kind: str, track, image: bytes | np.ndarray) -> bool:
        key = kind, int(track.id)
        if key in self.submitted:
            return False
        if kind == "object" and track.state != "stationary":
            return False
        try:
            owned = image.copy() if isinstance(image, np.ndarray) else bytes(image)
            self.queue.put_nowait(NamingTask(kind, track, owned))
            self.submitted.add(key)
            track.name_pending = True
            return True
        except Full:
            return False

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=60)

    def stats(self) -> tuple[int, int, int]:
        with self._stats_lock:
            ordered = sorted(self.latencies)
            median = ordered[len(ordered) // 2] if ordered else 0.0
            return self.calls, self.failures, round(median)

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                task = self.queue.get(timeout=0.2)
            except Empty:
                continue
            fallback = task.track.class_name if task.kind == "object" else "person"
            crop_path = None
            try:
                jpeg = self._jpeg(task.image)
                crop_path = self._save_crop(task, jpeg)
                try:
                    task.track.name = self._clean_name(self._request(task, jpeg))
                except Exception:
                    task.track.name = self._clean_name(self._request(task, jpeg))
            except Exception:
                task.track.name = fallback
            finally:
                task.track.name_pending = False
            if self.on_named:
                try:
                    self.on_named(task.kind, task.track, crop_path)
                except Exception:
                    pass
            self.queue.task_done()

    @staticmethod
    def _jpeg(image: bytes | np.ndarray) -> bytes:
        if isinstance(image, bytes):
            return image
        ok, encoded = cv2.imencode(
            ".jpg", np.ascontiguousarray(image), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise RuntimeError("failed to encode naming crop")
        return encoded.tobytes()

    def _save_crop(self, task: NamingTask, jpeg: bytes) -> str | None:
        if self.crop_dir is None:
            return None
        filename = f"{task.kind}-{int(task.track.id)}.jpg"
        (self.crop_dir / filename).write_bytes(jpeg)
        return f"crops/{filename}"

    @staticmethod
    def _clean_name(name) -> str:
        cleaned = " ".join(str(name).split())[:80]
        if not cleaned:
            raise ValueError("VLM returned an empty name")
        return cleaned

    def _request(self, task: NamingTask, jpeg: bytes) -> str:
        prompt = (
            'Name this object in at most 5 words including its color. Answer only JSON {"name":"..."}'
            if task.kind == "object"
            else 'Describe this person clothing in at most 6 words, no identity or protected attributes. Answer only JSON {"name":"..."}'
        )
        payload = {
            "model": self.model, "stream": False, "max_tokens": 24,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}},
                {"type": "text", "text": prompt},
            ]}],
        }
        started = time.perf_counter()
        req = request.Request(f"{self.base_url}/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        response_text = ""
        error = None
        try:
            with exclusive_vlm(timeout=15), request.urlopen(req, timeout=12) as response:
                body = json.load(response)
            response_text = body["choices"][0]["message"]["content"]
            name = json.loads(
                response_text[response_text.find("{") : response_text.rfind("}") + 1]
            )["name"]
            return self._clean_name(name)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency = (time.perf_counter() - started) * 1000
            with self._stats_lock:
                self.calls += 1
                self.failures += int(error is not None)
                self.latencies.append(latency)
            record = {"ts": time.time(), "kind": task.kind, "track_id": task.track.id, "model": self.model, "prompt": prompt, "response": response_text, "error": error, "latency_ms": latency}
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
