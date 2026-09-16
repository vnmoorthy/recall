import json
from pathlib import Path

from recall.answerer import Answerer
from recall.demo import bootstrap
from recall.memory import Memory


def test_sample_questions_return_expected_evidence(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    bootstrap(memory, tmp_path / "media")
    questions = json.loads(Path("assets/questions.json").read_text())
    passes = 0
    diagnostics = []
    for item in questions:
        result = Answerer(memory).ask(item["question"])
        event_types = {
            memory.event_by_id(event_id)["type"]
            for event_id in result["event_ids"]
            if memory.event_by_id(event_id)
        }
        ok = item["expected_event_type"] in event_types
        passes += int(ok)
        diagnostics.append((item["question"], item["expected_event_type"], sorted(event_types)))
    assert passes >= 10, diagnostics
