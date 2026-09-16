"""Question answering over compact local visual-memory context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import re
import time
import threading
from pathlib import Path
from urllib import request

from .memory import Memory
from .vlm_lock import exclusive_vlm


WINDOW_RE = re.compile(r"\blast\s+(\d+)\s*(minute|minutes|hour|hours)\b", re.I)


def clean_model_text(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    return text[:limit].strip()


def parse_window(question: str, now: float | None = None) -> int:
    match = WINDOW_RE.search(question)
    if match:
        amount = int(match.group(1))
        return amount * (3600 if match.group(2).lower().startswith("hour") else 60)
    if re.search(r"\btoday\b", question, re.I):
        current = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc)
        return max(1, current.hour * 3600 + current.minute * 60 + current.second)
    return 15 * 60


def _where(centroid: list[float], frame_width: float = 800.0) -> str:
    x = float(centroid[0])
    ratio = x / max(frame_width, 1.0) if x > 1.0 else x
    return "left" if ratio < 1 / 3 else "right" if ratio > 2 / 3 else "center"


def build_context(memory: Memory, window: int, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    inventory = []
    for item in memory.inventory(now)[:100]:
        inventory.append(
            {
                "id": item["id"],
                "name": item["name"] or item["class"],
                "class": item["class"],
                "state": item["state"],
                "where": _where(item["last_centroid"]),
                "last_seen": datetime.fromtimestamp(item["last_seen"], timezone.utc).strftime("%H:%M:%S"),
            }
        )
    events = []
    for event in reversed(memory.events_window(window, now)):
        events.append(
            {
                "id": event["id"],
                "timestamp": int(event["ts"]),
                "time": datetime.fromtimestamp(event["ts"], timezone.utc).strftime("%H:%M:%S"),
                "type": event["type"],
                "object": event["object_name"] or event["object_class"],
                "person": event["person_name"],
                "direction": event["direction"],
            }
        )
    return {"inventory": inventory, "events": events[-80:]}


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def answer_times_are_grounded(answer: str, context: dict) -> bool:
    cited = set(re.findall(r"\b\d{2}:\d{2}:\d{2}\b", answer))
    known = {item["last_seen"] for item in context["inventory"]}
    known.update(event["time"] for event in context["events"])
    return cited <= known


@dataclass
class LocalVLMClient:
    base_url: str = "http://127.0.0.1:9998"
    model: str = "Qwen3-VL-4B-Instruct-GPTQ-a16w4"
    timeout: float = 8.0
    log_path: Path | None = None
    _log_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, float]:
        payload = {
            "model": self.model,
            "stream": False,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        started = time.perf_counter()
        response_text = ""
        error = None
        req = request.Request(
            f"{self.base_url}/v1/chat/completions",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        try:
            with exclusive_vlm(timeout=self.timeout), request.urlopen(req, timeout=self.timeout) as response:
                body = json.load(response)
            response_text = body["choices"][0]["message"]["content"]
            return response_text, (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if self.log_path:
                elapsed = (time.perf_counter() - started) * 1000.0
                record = {
                    "ts": time.time(), "kind": "text", "model": self.model,
                    "system": system, "prompt": user, "max_tokens": max_tokens,
                    "response": response_text, "error": error, "latency_ms": elapsed,
                }
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class Answerer:
    def __init__(self, memory: Memory, client: LocalVLMClient | None = None):
        self.memory = memory
        self.client = client

    def ask(self, question: str, now: float | None = None) -> dict:
        window = parse_window(question, now)
        context = build_context(self.memory, window, now)
        result: dict
        latency_ms = 0.0
        if self.client is not None:
            prompt = (
                f"Question: {question}\nMemory JSON: {json.dumps(context, separators=(',', ':'))}\n"
                'Return only JSON {"answer":"at most 3 factual sentences with times when relevant",'
                '"event_ids":[integers],"object_ids":[integers]}.'
            )
            try:
                text, latency_ms = self.client.chat(
                    "Answer only from the supplied local visual memory. Never invent identity.",
                    prompt,
                    96,
                )
                result = parse_json_response(text)
                if not answer_times_are_grounded(str(result.get("answer", "")), context):
                    result = self._fallback(question, context)
            except Exception:
                result = self._fallback(question, context)
        else:
            result = self._fallback(question, context)
        raw_event_ids = result.get("event_ids", [])
        raw_object_ids = result.get("object_ids", [])
        if not isinstance(raw_event_ids, list):
            raw_event_ids = []
        if not isinstance(raw_object_ids, list):
            raw_object_ids = []
        event_ids = []
        for value in raw_event_ids[:20]:
            if str(value).isdigit() and self.memory.event_by_id(int(value)):
                if int(value) not in event_ids:
                    event_ids.append(int(value))
        inventory_ids = {item["id"] for item in self.memory.inventory()}
        object_ids = []
        for value in raw_object_ids[:20]:
            if str(value).isdigit() and int(value) in inventory_ids and int(value) not in object_ids:
                object_ids.append(int(value))
        snapshots = []
        snapshot_paths = set()
        for event_id in event_ids:
            event = self.memory.event_by_id(event_id)
            if event and (event.get("frame_path") or event.get("crop_path")):
                key = event.get("frame_path") or event.get("crop_path")
                if key not in snapshot_paths:
                    snapshots.append({"event_id": event_id, "frame_path": event.get("frame_path"), "crop_path": event.get("crop_path")})
                    snapshot_paths.add(key)
        answer = clean_model_text(result.get("answer", ""), 1200)
        if not answer:
            answer = "I do not have enough visual memory to answer that."
        return {
            "answer": answer,
            "event_ids": event_ids,
            "object_ids": object_ids,
            "snapshots": snapshots[:2],
            "window": window,
            "vlm_ms": latency_ms,
        }

    @staticmethod
    def _fallback(question: str, context: dict) -> dict:
        words = question.casefold()
        question_tokens = set(re.findall(r"[a-z0-9]+", words)) - {
            "a", "an", "the", "is", "my", "where", "who", "what", "when", "did", "was",
        }
        ranked = []
        for item in context["inventory"]:
            label = f"{item['name']} {item['class']}".casefold()
            label_tokens = set(re.findall(r"[a-z0-9]+", label))
            overlap = len(question_tokens & label_tokens)
            ranked.append((overlap, SequenceMatcher(None, words, label).ratio(), item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        candidate = ranked[0][2] if ranked and ranked[0][0] > 0 else None
        related = [event for event in context["events"] if candidate and event["object"] and event["object"].casefold() in {candidate["name"].casefold(), candidate["class"].casefold()}]
        if "what" in words and ("table" in words or "here" in words) and context["inventory"]:
            names = ", ".join(item["name"] for item in context["inventory"] if item["state"] != "missing")
            appeared = [event["id"] for event in context["events"] if event["type"] == "object_appeared"]
            return {"answer": f"I can currently see {names}.", "event_ids": appeared[-5:], "object_ids": [item["id"] for item in context["inventory"]]}
        if "missing" in words and not candidate:
            missing = [item for item in context["inventory"] if item["state"] == "missing"]
            names = ", ".join(item["name"] for item in missing) or "none"
            evidence = [event["id"] for event in context["events"] if event["type"] == "object_missing"]
            return {"answer": f"The missing objects are {names}.", "event_ids": evidence, "object_ids": [item["id"] for item in missing]}
        if ("happened" in words or "summarize" in words) and context["events"]:
            recent = context["events"][-6:]
            facts = [f"At {event['time']}, {(event['object'] or event['person'] or 'the room')} {event['type'].replace('_', ' ')}" for event in recent[-3:]]
            return {"answer": ". ".join(facts) + ".", "event_ids": [event["id"] for event in recent], "object_ids": []}
        if ("leave" in words or "left" in words) and not candidate:
            departures = [event for event in context["events"] if event["type"] == "person_left"]
            if departures:
                latest = departures[-1]
                return {"answer": f"{latest['person'] or 'A person'} left at {latest['time']}.", "event_ids": [latest["id"]], "object_ids": []}
        if candidate:
            custody_words = ("who", "move", "moved", "took", "take", "direction")
            custody = [event for event in related if event["type"] == "picked_up"]
            missing = [event for event in related if event["type"] == "object_missing"]
            if any(word in words for word in custody_words) and custody:
                latest = custody[-1]
            elif candidate["state"] == "missing" and missing:
                latest = missing[-1]
            else:
                latest = related[-1] if related else None
            if candidate["state"] == "missing" and latest:
                person = f" by {latest['person']}" if latest.get("person") else ""
                direction = f" toward the {latest['direction']}" if latest.get("direction") else ""
                return {"answer": f"The {candidate['name']} was last seen at {latest['time']}, moved{person}{direction}.", "event_ids": [latest["id"]], "object_ids": [candidate["id"]]}
            return {"answer": f"The {candidate['name']} is {candidate['state']} in the {candidate['where']} of the view.", "event_ids": [latest["id"]] if latest else [], "object_ids": [candidate["id"]]}
        return {"answer": "I do not have enough visual memory to answer that yet.", "event_ids": [], "object_ids": []}
