from pathlib import Path

from recall.answerer import Answerer, build_context, clean_model_text, parse_window
from recall.demo import bootstrap
from recall.memory import Memory


class InvalidEvidenceClient:
    def chat(self, system, user, max_tokens):
        return '{"answer":"checked","event_ids":[1,9999,"bad"],"object_ids":[1,9999]}', 5.0


class HallucinatedTimeClient:
    def chat(self, system, user, max_tokens):
        return '{"answer":"The mug moved at 23:59:59.","event_ids":[6],"object_ids":[1]}', 5.0


def test_window_parser():
    assert parse_window("what happened in the last 10 minutes") == 600
    assert parse_window("last 2 hours") == 7200
    assert parse_window("where is the mug") == 900
    assert parse_window("today", now=0) == 1
    assert clean_model_text("First.Second.\n Third.", 100) == "First. Second. Third."


def test_context_and_human_word_fallback(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    bootstrap(memory, tmp_path / "media")
    context = build_context(memory, 900)
    assert any(item["name"] == "red ceramic mug" for item in context["inventory"])
    answer = Answerer(memory).ask("Where is the red mug?")
    assert "red ceramic mug" in answer["answer"]
    assert answer["event_ids"]
    assert answer["snapshots"]


def test_inventory_question_lists_present_objects(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    bootstrap(memory, tmp_path / "media")
    answer = Answerer(memory).ask("What's on the table?")["answer"]
    assert "green water bottle" in answer
    assert "blue hardcover book" in answer


def test_model_cannot_cite_unknown_memory_ids(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    bootstrap(memory, tmp_path / "media")
    result = Answerer(memory, InvalidEvidenceClient()).ask("Where is the mug?")
    assert result["event_ids"] == [1]
    assert result["object_ids"] == [1]


def test_model_cannot_invent_event_times(tmp_path: Path):
    memory = Memory(tmp_path / "recall.db")
    bootstrap(memory, tmp_path / "media")
    result = Answerer(memory, HallucinatedTimeClient()).ask("Where is the mug?")
    assert "23:59:59" not in result["answer"]
    assert result["event_ids"] == [7]


def test_context_caps_historical_inventory():
    class LargeMemory:
        def inventory(self, _now):
            return [
                {
                    "id": index, "name": f"item {index}", "class": "item",
                    "state": "stationary", "last_centroid": [0.5, 0.5],
                    "last_seen": 1,
                }
                for index in range(150)
            ]

        def events_window(self, _window, _now):
            return []

    assert len(build_context(LargeMemory(), 900, now=2)["inventory"]) == 100
