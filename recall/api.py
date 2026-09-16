"""FastAPI surface for Recall memory, questions, summaries, and status."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from urllib import request

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .answerer import Answerer
from .memory import Memory
from .summarizer import Summarizer


@dataclass
class RuntimeStatus:
    started_at: float = field(default_factory=time.time)
    seg_fps: float = 0.0
    pose_fps: float = 0.0
    vlm_calls: int = 0
    vlm_latencies: list[float] = field(default_factory=list)
    resident_model: str = "unavailable"
    objects_tracked: int = 0
    perception_mode: str = "synthetic"
    metrics_path: Path | None = None

    def payload(self) -> dict:
        ordered = sorted(self.vlm_latencies)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        payload = {
            "seg_fps": round(self.seg_fps, 1), "pose_fps": round(self.pose_fps, 1),
            "vlm_calls": self.vlm_calls, "vlm_ms_p50": round(median),
            "resident_model": self.resident_model, "objects_tracked": self.objects_tracked,
            "uptime": int(time.time() - self.started_at), "perception_mode": self.perception_mode,
            "offline": True,
        }
        if self.metrics_path and self.metrics_path.is_file():
            try:
                worker = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                if time.time() - float(worker.get("updated_at", 0)) <= 10:
                    payload.update({
                        key: worker[key]
                        for key in ("seg_fps", "pose_fps", "objects_tracked", "perception_mode")
                        if key in worker
                    })
            except (OSError, ValueError, TypeError):
                pass
        return payload


class AskBody(BaseModel):
    question: str


class SummaryBody(BaseModel):
    window: int = 900


class SpeechBody(BaseModel):
    input: str
    voice: str = "kristin"
    response_format: str = "wav"


def create_app(memory: Memory, answerer: Answerer, summarizer: Summarizer, status: RuntimeStatus, media_dir="media", speaker=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if speaker:
            speaker.close()

    app = FastAPI(title="Recall", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/media", StaticFiles(directory=media_dir, check_dir=False), name="media")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "recall", "time": time.time()}

    @app.get("/inventory")
    def inventory():
        return memory.inventory(time.time())

    @app.get("/events")
    def events(window: int = 900):
        return memory.events_window(max(1, window))

    @app.get("/objects/{object_id}/timeline")
    def timeline(object_id: int):
        return memory.timeline(object_id)

    @app.post("/ask")
    def ask(body: AskBody):
        if not body.question.strip():
            raise HTTPException(400, "question is required")
        result = answerer.ask(body.question.strip())
        if result["vlm_ms"]:
            status.vlm_calls += 1
            status.vlm_latencies.append(result["vlm_ms"])
        if speaker:
            speaker.speak(result["answer"])
        return result

    @app.post("/ask_audio")
    async def ask_audio(incoming: Request):
        form = await incoming.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "multipart field 'file' is required")
        audio = await upload.read()
        filename = Path(getattr(upload, "filename", "question.wav") or "question.wav").name
        filename = filename.replace('"', "")
        content_type = getattr(upload, "content_type", None) or "application/octet-stream"
        boundary = "recall-audio-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-small-a16w8\r\n".encode()
            + f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
            + audio + f"\r\n--{boundary}--\r\n".encode()
        )
        req = request.Request("http://127.0.0.1:9998/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with request.urlopen(req, timeout=30) as response:
                transcript = __import__("json").load(response)
            question = transcript.get("text", "")
        except Exception as exc:
            raise HTTPException(503, f"speech recognition unavailable: {exc}") from exc
        result = answerer.ask(question)
        result["question"] = question
        if result["vlm_ms"]:
            status.vlm_calls += 1
            status.vlm_latencies.append(result["vlm_ms"])
        if speaker:
            speaker.speak(result["answer"])
        return result

    @app.post("/summary")
    def summary(body: SummaryBody):
        result = summarizer.summarize(max(1, body.window))
        if result["vlm_ms"]:
            status.vlm_calls += 1
            status.vlm_latencies.append(result["vlm_ms"])
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
        payload["tts"] = "piper-local" if speaker else "unavailable"
        return payload

    return app
