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


def test_status_reads_fresh_worker_metrics(tmp_path):
    metrics = tmp_path / "status.json"
    metrics.write_text(json.dumps({
        "updated_at": time.time(), "seg_fps": 29.8, "pose_fps": 9.9,
        "objects_tracked": 4, "perception_mode": "hardware",
    }))
    status = RuntimeStatus(metrics_path=metrics)
    payload = status.payload()
    assert payload["seg_fps"] == 29.8
    assert payload["pose_fps"] == 9.9
    assert payload["perception_mode"] == "hardware"


def test_api_health_question_and_local_speech(tmp_path):
    memory = Memory(tmp_path / "recall.db")
    speaker = FakeSpeaker()
    app = create_app(
        memory, Answerer(memory), Summarizer(memory), RuntimeStatus(),
        tmp_path / "media", speaker,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        answer = client.post("/ask", json={"question": "Where is my mug?"})
        assert answer.status_code == 200
        assert speaker.spoken
        speech = client.post("/v1/audio/speech", json={"input": "Local speech."})
        assert speech.status_code == 200
        assert speech.headers["content-type"] == "audio/wav"
        with wave.open(BytesIO(speech.content), "rb") as wav:
            assert wav.getframerate() == 16000
