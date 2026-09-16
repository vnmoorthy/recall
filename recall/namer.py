"""One-shot object and clothing naming on a bounded VLM worker."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from urllib import request


@dataclass
class NamingTask:
    kind: str
    track: object
    jpeg: bytes


class Namer:
    def __init__(self, base_url: str, model: str, log_path: str | Path, on_named=None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_named = on_named
        self.queue: Queue[NamingTask] = Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.submitted: set[tuple[str, int]] = set()
        self.thread = threading.Thread(target=self._run, name="recall-namer", daemon=True)
        self.thread.start()

    def submit(self, kind: str, track, jpeg: bytes) -> bool:
        key = kind, int(track.id)
        if key in self.submitted:
            return False
        if kind == "object" and track.state != "stationary":
            return False
        try:
            self.queue.put_nowait(NamingTask(kind, track, jpeg))
            self.submitted.add(key)
            track.name_pending = True
            return True
        except Full:
            return False

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.2)
            except Empty:
                continue
            fallback = task.track.class_name if task.kind == "object" else "person"
            try:
                task.track.name = self._request(task)
            except Exception:
                try:
                    task.track.name = self._request(task)
                except Exception:
                    task.track.name = fallback
            task.track.name_pending = False
            if self.on_named:
                self.on_named(task.kind, task.track)
            self.queue.task_done()

    def _request(self, task: NamingTask) -> str:
        prompt = (
            'Name this object in at most 5 words including its color. Answer only JSON {"name":"..."}'
            if task.kind == "object"
            else 'Describe this person clothing in at most 6 words, no identity or protected attributes. Answer only JSON {"name":"..."}'
        )
        payload = {
            "model": self.model, "stream": False, "max_tokens": 24,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(task.jpeg).decode("ascii")}},
                {"type": "text", "text": prompt},
            ]}],
        }
        started = time.perf_counter()
        req = request.Request(f"{self.base_url}/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        response_text = ""
        error = None
        try:
            with request.urlopen(req, timeout=12) as response:
                body = json.load(response)
            response_text = body["choices"][0]["message"]["content"]
            name = json.loads(response_text[response_text.find("{") : response_text.rfind("}") + 1])["name"]
            return str(name).strip()[:80]
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            record = {"ts": time.time(), "kind": task.kind, "track_id": task.track.id, "model": self.model, "prompt": prompt, "response": response_text, "error": error, "latency_ms": (time.perf_counter() - started) * 1000}
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
