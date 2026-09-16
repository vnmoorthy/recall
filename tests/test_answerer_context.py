from pathlib import Path

from recall.answerer import Answerer, build_context, parse_window
from recall.demo import bootstrap
from recall.memory import Memory


def test_window_parser():
    assert parse_window("what happened in the last 10 minutes") == 600
    assert parse_window("last 2 hours") == 7200
    assert parse_window("where is the mug") == 900


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
