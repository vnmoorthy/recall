"""Factual local-memory summaries and snapshot selection."""

from __future__ import annotations

import json

from .answerer import LocalVLMClient, build_context, parse_json_response
from .memory import Memory

IMPORTANCE = {"picked_up": 5, "object_missing": 4, "put_down": 3, "person_entered": 2, "object_moved": 1}


class Summarizer:
    def __init__(self, memory: Memory, client: LocalVLMClient | None = None):
        self.memory = memory
        self.client = client

    def summarize(self, window: int = 900, now: float | None = None) -> dict:
        context = build_context(self.memory, window, now)
        important = sorted(context["events"], key=lambda event: (IMPORTANCE.get(event["type"], 0), event["time"]), reverse=True)
        chosen = important[:2]
        if self.client and context["events"]:
            try:
                text, latency = self.client.chat(
                    "Summarize only supplied visual-memory facts.",
                    f"Memory JSON: {json.dumps(context, separators=(',', ':'))}\nWrite exactly three factual sentences. Return only JSON {{\"summary\":\"...\"}}.",
                    128,
                )
                summary = str(parse_json_response(text)["summary"])
            except Exception:
                summary, latency = self._fallback(context), 0.0
        else:
            summary, latency = self._fallback(context), 0.0
        snapshots = []
        for event in chosen:
            stored = self.memory.event_by_id(event["id"])
            if stored:
                snapshots.append({"event_id": event["id"], "frame_path": stored.get("frame_path"), "crop_path": stored.get("crop_path")})
        return {"summary": summary, "event_ids": [event["id"] for event in chosen], "snapshots": snapshots, "vlm_ms": latency}

    @staticmethod
    def _fallback(context: dict) -> str:
        events = context["events"]
        if not events:
            return "No significant activity was recorded. The current inventory is unchanged. Recall is continuing to watch the space."
        recent = events[-3:]
        sentences = []
        for event in recent:
            subject = event.get("object") or event.get("person") or "Something"
            action = event["type"].replace("_", " ")
            sentences.append(f"At {event['time']}, {subject} {action}.")
        while len(sentences) < 3:
            sentences.append("No other significant activity was recorded in this window.")
        return " ".join(sentences[:3])
