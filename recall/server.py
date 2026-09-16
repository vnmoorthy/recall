"""Recall API process entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .answerer import Answerer, LocalVLMClient
from .actions import PiperEngine, Speaker
from .api import RuntimeStatus, create_app
from .demo import bootstrap
from .memory import Memory
from .summarizer import Summarizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    database = "recall-demo.db" if args.synthetic else "recall.db"
    memory = Memory(root / "data" / database)
    if args.synthetic:
        memory.reset()
        bootstrap(memory, root / "media")
    qwen_path = Path("/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4/devkit")
    gemma_path = Path("/media/nvme/llima/models/gemma-4-E2B-it-GPTQ-a16w4/devkit")
    model = "Qwen3-VL-4B-Instruct-GPTQ-a16w4" if qwen_path.is_dir() else "gemma-4-E2B-it-GPTQ-a16w4" if gemma_path.is_dir() else None
    client = LocalVLMClient(model=model, log_path=root / "logs/vlm.jsonl") if model else None
    status = RuntimeStatus(
        resident_model=model or "fallback rules",
        perception_mode="synthetic" if args.synthetic else "hardware",
        metrics_path=root / "run/perception-status.json",
    )
    try:
        speaker = Speaker(PiperEngine(root))
    except (FileNotFoundError, RuntimeError):
        speaker = None
    app = create_app(
        memory,
        Answerer(memory, client),
        Summarizer(memory, client),
        status,
        root / "media",
        speaker,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
