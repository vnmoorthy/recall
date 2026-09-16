"""Cross-process serialization for the one resident Modalix VLM."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import time


@contextmanager
def exclusive_vlm(path: str | Path = "/tmp/recall-vlm.lock", timeout: float = 15.0):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="ascii") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for the resident VLM")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
