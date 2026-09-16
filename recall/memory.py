"""SQLite persistence and human-name lookup for Recall."""

from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sqlite3
import threading
import time

from .events import Event
from .tracker import ObjectTrack, PersonTrack


SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
 id INTEGER PRIMARY KEY, class TEXT NOT NULL, name TEXT, state TEXT NOT NULL,
 last_centroid TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, crop_path TEXT
);
CREATE TABLE IF NOT EXISTS persons (
 id INTEGER PRIMARY KEY, name TEXT, first_seen REAL NOT NULL, last_seen REAL NOT NULL, crop_path TEXT
);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, type TEXT NOT NULL,
 object_id INTEGER, person_id INTEGER, direction TEXT, frame_path TEXT, crop_path TEXT
);
CREATE INDEX IF NOT EXISTS events_ts_idx ON events(ts);
"""


class Memory:
    def __init__(self, path: str | Path = "data/recall.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def reset(self) -> None:
        """Clear all memory rows; used only by the isolated synthetic demo database."""
        with self._lock:
            self._db.executescript("DELETE FROM events; DELETE FROM persons; DELETE FROM objects;")
            self._db.commit()

    def upsert_object(self, obj: ObjectTrack, crop_path: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET class=excluded.class, name=excluded.name,
                state=excluded.state, last_centroid=excluded.last_centroid,
                last_seen=excluded.last_seen, crop_path=COALESCE(excluded.crop_path, objects.crop_path)""",
                (obj.id, obj.class_name, obj.name, obj.state, json.dumps(obj.centroid), obj.first_seen, obj.last_seen, crop_path),
            )
            self._db.commit()

    def upsert_person(self, person: PersonTrack, crop_path: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO persons VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, last_seen=excluded.last_seen,
                crop_path=COALESCE(excluded.crop_path, persons.crop_path)""",
                (person.id, person.name, person.first_seen, person.last_seen, crop_path),
            )
            self._db.commit()

    def add_event(self, event: Event) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO events(ts,type,object_id,person_id,direction,frame_path,crop_path) VALUES(?,?,?,?,?,?,?)",
                (event.ts, event.type, event.object_id, event.person_id, event.direction, event.frame_path, event.crop_path),
            )
            self._db.commit()
            return int(cursor.lastrowid)

    def inventory(self, now: float | None = None) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM objects ORDER BY last_seen DESC").fetchall()
        return [self._object_dict(row, now) for row in rows]

    @staticmethod
    def _object_dict(row: sqlite3.Row, now: float | None = None) -> dict:
        item = dict(row)
        item["last_centroid"] = json.loads(item["last_centroid"])
        if now is not None:
            item["seconds_since_seen"] = max(0.0, now - float(item["last_seen"]))
        return item

    def find(self, name_query: str, limit: int = 5) -> list[dict]:
        query = name_query.casefold().strip()
        items = self.inventory()
        for item in items:
            haystack = f"{item.get('name') or ''} {item['class']}".casefold()
            item["match_score"] = SequenceMatcher(None, query, haystack).ratio()
            if query and query in haystack:
                item["match_score"] += 1.0
        return sorted(items, key=lambda item: item["match_score"], reverse=True)[:limit]

    def events_window(self, seconds: float, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute(
                """SELECT e.*, o.name object_name, o.class object_class, p.name person_name
                FROM events e LEFT JOIN objects o ON o.id=e.object_id
                LEFT JOIN persons p ON p.id=e.person_id
                WHERE e.ts >= ? ORDER BY e.ts DESC""",
                (now - seconds,),
            ).fetchall()
        return [dict(row) for row in rows]

    def timeline(self, object_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE object_id=? ORDER BY ts DESC", (object_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def event_by_id(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(row) if row else None
