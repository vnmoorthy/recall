"""Local Piper synthesis and non-blocking DevKit playback."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import tempfile
import threading


class PiperEngine:
    """Load the reviewed Neat GenAI Studio Piper worker by file path."""

    def __init__(self, root: str | Path):
        root = Path(root).resolve()
        python_dir = root / "vendor/neat-genai-studio/src/python"
        module_path = python_dir / "ui/pipertts.py"
        self.model_path = python_dir / "ui/assets/en_US-kristin-medium.onnx"
        worker_python = root / ".venv-pipertts/bin/python"
        if not worker_python.is_file() or not self.model_path.is_file():
            raise FileNotFoundError("the local Piper runtime or English voice is not installed")
        os.environ["PIPERTTS_PYTHON"] = str(worker_python)
        spec = importlib.util.spec_from_file_location("recall_pipertts", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Piper client from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        self._tts_class = module.PiperTTS
        self._tts = None
        self._lock = threading.Lock()
        self.ready = False
        self.error: str | None = None

    def warm(self) -> None:
        with self._lock:
            if self._tts is None:
                try:
                    self._tts = self._tts_class(model_path=self.model_path)
                    self.ready = True
                    self.error = None
                except Exception as exc:
                    self.error = str(exc)
                    raise

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("speech input is required")
        with self._lock:
            if self._tts is None:
                self._tts = self._tts_class(model_path=self.model_path)
                self.ready = True
            return self._tts.synthesize(text.strip()).read()

    def close(self) -> None:
        with self._lock:
            self._module._discard_worker()
            self._tts = None


class Speaker:
    """Synthesize locally and play through ALSA without blocking the API."""

    def __init__(self, engine: PiperEngine, playback: bool = True):
        self.engine = engine
        self.playback = playback
        self.queue: Queue[str] = Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="recall-speaker", daemon=True)
        self.warm_thread = threading.Thread(target=self._warm, name="recall-piper-warm", daemon=True)
        self.warm_thread.start()
        self.thread.start()

    def _warm(self) -> None:
        try:
            self.engine.warm()
        except Exception:
            pass

    def status(self) -> str:
        if self.engine.ready:
            return "piper-ready"
        return "piper-error" if self.engine.error else "piper-warming"

    def synthesize(self, text: str) -> bytes:
        return self.engine.synthesize(text)

    def speak(self, text: str) -> bool:
        try:
            self.queue.put_nowait(text)
            return True
        except Full:
            return False

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        self.warm_thread.join(timeout=30)
        self.engine.close()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                audio = self.synthesize(text)
                if self.playback:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as output:
                        output.write(audio)
                        output.flush()
                        subprocess.run(
                            ["aplay", "-q", output.name],
                            check=False,
                            timeout=60,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            except Exception:
                pass
            finally:
                self.queue.task_done()
