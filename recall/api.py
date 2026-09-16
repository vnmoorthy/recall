"""FastAPI surface for Recall memory, questions, summaries, and status."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Literal
import uuid
from urllib import request

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .answerer import Answerer
from .memory import Memory
from .summarizer import Summarizer


@dataclass
class RuntimeStatus:
    started_at: float = field(default_factory=time.time)
    seg_fps: float = 0.0
    pose_fps: float = 0.0
    vlm_calls: int = 0
    vlm_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    resident_model: str = "unavailable"
    objects_tracked: int = 0
    perception_mode: str = "synthetic"
    metrics_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_vlm(self, latency_ms: float) -> None:
        if latency_ms <= 0:
            return
        with self._lock:
            self.vlm_calls += 1
            self.vlm_latencies.append(latency_ms)

    def _process_alive(self, name: str) -> bool:
        if self.metrics_path is None:
            return False
        try:
            pid = int((self.metrics_path.parent / f"{name}.pid").read_text())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            alive = f"-m recall.{name}".encode() in command
            if name == "genai":
                alive = alive and (self.metrics_path.parent / "genai.ready").is_file()
            return alive
        except (OSError, ValueError):
            return False

    def payload(self) -> dict:
        with self._lock:
            ordered = sorted(self.vlm_latencies)
            vlm_calls = self.vlm_calls
        median = ordered[len(ordered) // 2] if ordered else 0.0
        payload = {
            "seg_fps": round(self.seg_fps, 1), "pose_fps": round(self.pose_fps, 1),
            "vlm_calls": vlm_calls, "vlm_ms_p50": round(median),
            "resident_model": self.resident_model, "objects_tracked": self.objects_tracked,
            "uptime": int(time.time() - self.started_at), "perception_mode": self.perception_mode,
            "offline": True,
            "perception_healthy": self._process_alive("preview") if self.perception_mode == "synthetic" else False,
            "genai_healthy": self._process_alive("genai"),
        }
        if self.metrics_path and self.metrics_path.is_file():
            try:
                worker = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                age = time.time() - float(worker.get("updated_at", 0))
                if age <= 10:
                    payload.update({
                        key: worker[key]
                        for key in (
                            "seg_fps", "pose_fps", "objects_tracked", "perception_mode",
                            "metadata_errors", "frame_write_errors",
                            "track_write_drops",
                            "track_write_errors",
                            "naming_vlm_calls", "naming_vlm_failures",
                            "naming_vlm_ms_p50",
                        )
                        if key in worker
                    })
                    payload["perception_healthy"] = True
                    payload["metrics_age_seconds"] = round(max(0.0, age), 1)
            except (OSError, ValueError, TypeError):
                pass
        payload["vlm_calls_total"] = payload["vlm_calls"] + int(payload.get("naming_vlm_calls", 0))
        return payload


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class SummaryBody(BaseModel):
    window: int = Field(default=900, ge=1, le=86400)


class SpeechBody(BaseModel):
    input: str = Field(min_length=1, max_length=2000)
    voice: Literal["kristin"] = "kristin"
    response_format: Literal["wav"] = "wav"


def create_app(memory: Memory, answerer: Answerer, summarizer: Summarizer, status: RuntimeStatus, media_dir="media", speaker=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if speaker:
            speaker.close()
        memory.close()

    app = FastAPI(title="Recall", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/media", StaticFiles(directory=media_dir, check_dir=False), name="media")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "recall", "time": time.time()}

    @app.get("/ready", responses={503: {"description": "A required local component is unavailable"}})
    def ready():
        payload = status.payload()
        tts = speaker.status() if speaker and hasattr(speaker, "status") else "unavailable"
        is_ready = payload["perception_healthy"] and (
            payload["resident_model"] == "fallback rules" or payload["genai_healthy"]
        ) and tts == "piper-ready"
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"ready": is_ready, "perception": payload["perception_healthy"],
                     "genai": payload["genai_healthy"], "tts": tts},
        )

    @app.get("/inventory")
    def inventory():
        return memory.inventory(time.time())

    @app.get("/events")
    def events(
        window: int = Query(default=900, ge=1, le=86400),
        limit: int = Query(default=500, ge=1, le=1000),
    ):
        return memory.events_window(window, limit=limit)

    @app.get("/objects/{object_id}/timeline")
    def timeline(object_id: int, limit: int = Query(default=500, ge=1, le=1000)):
        if object_id <= 0:
            raise HTTPException(400, "object id must be positive")
        if not memory.has_object(object_id):
            raise HTTPException(404, "object not found")
        return memory.timeline(object_id, limit)

    @app.post("/ask")
    def ask(body: AskBody):
        if not body.question.strip():
            raise HTTPException(400, "question is required")
        result = answerer.ask(body.question.strip())
        status.record_vlm(result["vlm_ms"])
        if speaker:
            speaker.speak(result["answer"])
        return result

    @app.post("/ask_audio")
    async def ask_audio(incoming: Request):
        form = await incoming.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(400, "multipart field 'file' is required")
        chunks = []
        total = 0
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > 25 * 1024 * 1024:
                raise HTTPException(413, "audio upload exceeds 25 MiB")
            chunks.append(chunk)
        audio = b"".join(chunks)
        if not audio:
            raise HTTPException(400, "uploaded audio is empty")
        filename = Path(getattr(upload, "filename", "question.wav") or "question.wav").name
        filename = filename.replace('"', "")
        content_type = getattr(upload, "content_type", None) or "application/octet-stream"
        filename = filename.replace("\\", "_").replace("\r", "").replace("\n", "")
        content_type = content_type.replace("\r", "").replace("\n", "")
        boundary = f"recall-{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-small-a16w8\r\n".encode()
            + f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
            + audio + f"\r\n--{boundary}--\r\n".encode()
        )
        req = request.Request("http://127.0.0.1:9998/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})

        def transcribe():
            with request.urlopen(req, timeout=30) as response:
                return __import__("json").load(response)

        try:
            transcript = await run_in_threadpool(transcribe)
            question = transcript.get("text", "")
        except Exception as exc:
            raise HTTPException(503, f"speech recognition unavailable: {exc}") from exc
        if not str(question).strip():
            raise HTTPException(422, "speech recognition returned an empty transcript")
        result = await run_in_threadpool(answerer.ask, str(question).strip())
        result["question"] = str(question).strip()
        status.record_vlm(result["vlm_ms"])
        if speaker:
            speaker.speak(result["answer"])
        return result

    @app.post("/summary")
    def summary(body: SummaryBody):
        result = summarizer.summarize(body.window)
        status.record_vlm(result["vlm_ms"])
        if speaker:
            speaker.speak(result["summary"])
        return result

    @app.post("/v1/audio/speech")
    def speech(body: SpeechBody):
        if speaker is None:
            raise HTTPException(503, "local Piper speech is not installed")
        if not body.input.strip():
            raise HTTPException(400, "speech input is required")
        if body.response_format != "wav":
            raise HTTPException(400, "only WAV output is supported")
        try:
            audio = speaker.synthesize(body.input)
        except Exception as exc:
            raise HTTPException(503, f"speech synthesis unavailable: {exc}") from exc
        return Response(content=audio, media_type="audio/wav")

    @app.get("/status")
    def runtime_status():
        status.objects_tracked = len(memory.inventory())
        payload = status.payload()
        payload["tts"] = speaker.status() if speaker and hasattr(speaker, "status") else "piper-ready" if speaker else "unavailable"
        payload["speech_queue_drops"] = getattr(speaker, "dropped", 0)
        payload["speech_playback_errors"] = getattr(speaker, "errors", 0)
        return payload

    return app
