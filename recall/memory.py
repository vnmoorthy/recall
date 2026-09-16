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
CREATE INDEX IF NOT EXISTS events_object_idx ON events(object_id, ts);
CREATE INDEX IF NOT EXISTS events_person_idx ON events(person_id, ts);
"""


class Memory:
    def __init__(self, path: str | Path = "data/recall.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def reset(self) -> None:
        """Clear all memory rows; used only by the isolated synthetic demo database."""
        with self._lock:
            self._db.executescript(
                "DELETE FROM events; DELETE FROM persons; DELETE FROM objects; "
                "DELETE FROM sqlite_sequence WHERE name='events';"
            )
            self._db.commit()

    @staticmethod
    def _object_values(
        obj: ObjectTrack,
        crop_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        centroid = obj.centroid
        if frame_size and frame_size[0] > 0 and frame_size[1] > 0:
            centroid = (centroid[0] / frame_size[0], centroid[1] / frame_size[1])
        return (
            obj.id, obj.class_name, obj.name, obj.state, json.dumps(centroid),
            obj.first_seen, obj.last_seen, crop_path,
        )

    def upsert_object(
        self,
        obj: ObjectTrack,
        crop_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET class=excluded.class,
                name=COALESCE(excluded.name, objects.name),
                state=excluded.state, last_centroid=excluded.last_centroid,
                last_seen=excluded.last_seen, crop_path=COALESCE(excluded.crop_path, objects.crop_path)""",
                self._object_values(obj, crop_path, frame_size),
            )
            self._db.commit()

    def upsert_tracks(
        self,
        objects: tuple[ObjectTrack, ...],
        persons: tuple[PersonTrack, ...],
        frame_size: tuple[int, int],
    ) -> None:
        """Persist one frame of tracks in a single WAL transaction."""
        with self._lock:
            self._db.executemany(
                """INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET class=excluded.class,
                name=COALESCE(excluded.name, objects.name),
                state=excluded.state, last_centroid=excluded.last_centroid,
                last_seen=excluded.last_seen, crop_path=COALESCE(excluded.crop_path, objects.crop_path)""",
                [self._object_values(obj, frame_size=frame_size) for obj in objects],
            )
            self._db.executemany(
                """INSERT INTO persons VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=COALESCE(excluded.name, persons.name),
                last_seen=excluded.last_seen,
                crop_path=COALESCE(excluded.crop_path, persons.crop_path)""",
                [(p.id, p.name, p.first_seen, p.last_seen, None) for p in persons],
            )
            self._db.commit()

    def upsert_person(self, person: PersonTrack, crop_path: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO persons VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=COALESCE(excluded.name, persons.name),
                last_seen=excluded.last_seen,
                crop_path=COALESCE(excluded.crop_path, persons.crop_path)""",
                (person.id, person.name, person.first_seen, person.last_seen, crop_path),
            )
            self._db.commit()

    def add_event(self, event: Event) -> int:
        return self.add_events([event])[0]

    def add_events(self, events: list[Event]) -> list[int]:
        if not events:
            return []
        with self._lock, self._db:
            ids = []
            for event in events:
                cursor = self._db.execute(
                    "INSERT INTO events(ts,type,object_id,person_id,direction,frame_path,crop_path) VALUES(?,?,?,?,?,?,?)",
                    (event.ts, event.type, event.object_id, event.person_id, event.direction, event.frame_path, event.crop_path),
                )
                ids.append(int(cursor.lastrowid))
            return ids

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

    def events_window(
        self, seconds: float, now: float | None = None, limit: int = 1000
    ) -> list[dict]:
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute(
                """SELECT e.*, o.name object_name, o.class object_class, p.name person_name
                FROM events e LEFT JOIN objects o ON o.id=e.object_id
                LEFT JOIN persons p ON p.id=e.person_id
                WHERE e.ts >= ? ORDER BY e.ts DESC LIMIT ?""",
                (now - seconds, max(1, min(int(limit), 5000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def timeline(self, object_id: int, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT e.*, o.name object_name, o.class object_class,
                p.name person_name FROM events e
                LEFT JOIN objects o ON o.id=e.object_id
                LEFT JOIN persons p ON p.id=e.person_id
                WHERE e.object_id=? ORDER BY e.ts DESC LIMIT ?""",
                (object_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_object(self, object_id: int) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone()
        return row is not None

    def next_track_ids(self) -> tuple[int, int]:
        with self._lock:
            object_id = self._db.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM objects"
            ).fetchone()[0]
            person_id = self._db.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM persons"
            ).fetchone()[0]
        return int(object_id), int(person_id)

    def event_by_id(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(row) if row else None
