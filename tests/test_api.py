import json
import time
import wave
from io import BytesIO

from fastapi.testclient import TestClient

from recall.answerer import Answerer
from recall.api import RuntimeStatus, create_app
from recall.memory import Memory
from recall.summarizer import Summarizer


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)
        return True

    def synthesize(self, text):
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160)
        return output.getvalue()

    def close(self):
        pass

    def status(self):
        return "piper-ready"


def test_status_reads_fresh_worker_metrics(tmp_path):
    metrics = tmp_path / "status.json"
    metrics.write_text(json.dumps({
        "updated_at": time.time(), "seg_fps": 29.8, "pose_fps": 9.9,
        "objects_tracked": 4, "perception_mode": "hardware",
        "naming_vlm_calls": 3, "naming_vlm_failures": 1,
    }))
    status = RuntimeStatus(metrics_path=metrics)
    status.record_vlm(100)
    payload = status.payload()
    assert payload["seg_fps"] == 29.8
    assert payload["pose_fps"] == 9.9
    assert payload["perception_mode"] == "hardware"
    assert payload["vlm_calls_total"] == 4
    assert payload["naming_vlm_failures"] == 1


def test_api_health_question_and_local_speech(tmp_path):
    memory = Memory(tmp_path / "recall.db")
    speaker = FakeSpeaker()
    app = create_app(
        memory, Answerer(memory), Summarizer(memory), RuntimeStatus(),
        tmp_path / "media", speaker,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/status").json()["tts"] == "piper-ready"
        assert client.get("/ready").status_code == 503
        answer = client.post("/ask", json={"question": "Where is my mug?"})
        assert answer.status_code == 200
        assert speaker.spoken
        speech = client.post("/v1/audio/speech", json={"input": "Local speech."})
        assert speech.status_code == 200
        assert speech.headers["content-type"] == "audio/wav"
        with wave.open(BytesIO(speech.content), "rb") as wav:
            assert wav.getframerate() == 16000
        assert client.get("/events?window=0").status_code == 422
        assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
        assert client.post("/summary", json={"window": 86401}).status_code == 422
        assert client.get("/objects/0/timeline").status_code == 400
        assert client.get("/objects/99/timeline").status_code == 404
        assert client.get("/events?limit=1001").status_code == 422
        assert client.post("/v1/audio/speech", json={"input": "hi", "voice": "unknown"}).status_code == 422
        empty = client.post("/ask_audio", files={"file": ("empty.wav", b"", "audio/wav")})
        assert empty.status_code == 400
        not_a_file = client.post("/ask_audio", data={"file": "not-a-file"})
        assert not_a_file.status_code == 400


def test_ready_reports_all_required_components(tmp_path, monkeypatch):
    memory = Memory(tmp_path / "recall.db")
    status = RuntimeStatus(resident_model="fallback rules")
    monkeypatch.setattr(status, "_process_alive", lambda name: name == "preview")
    app = create_app(
        memory, Answerer(memory), Summarizer(memory), status,
        tmp_path / "media", FakeSpeaker(),
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {
            "ready": True, "perception": True, "genai": False, "tts": "piper-ready",
        }
