"""Serve Recall's resident Qwen VLM and Whisper model through PyNeat."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--asr", type=Path, default=Path("/media/nvme/llima/models/whisper-small-a16w8"))
    parser.add_argument("--port", type=int, default=9998)
    args = parser.parse_args()
    import pyneat

    options = pyneat.GenAIServerOptions()
    options.host, options.port = "0.0.0.0", args.port
    server = pyneat.GenAIServer(options)
    vlm = args.vlm
    if vlm is None:
        qwen = Path("/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4")
        gemma = Path("/media/nvme/llima/models/gemma-4-E2B-it-GPTQ-a16w4")
        vlm = qwen if (qwen / "devkit").is_dir() else gemma
    if vlm.is_dir() and (vlm / "devkit").is_dir():
        server.add_model(vlm, vlm.name)
    if args.asr.is_dir() and (args.asr / "devkit").is_dir():
        server.add_model(args.asr, args.asr.name)
    if not server.model_names():
        print("No deployed GenAI model directories are available", flush=True)
        return 2
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    server.start()
    print(f"Recall GenAI models: {', '.join(server.model_names())}", flush=True)
    try:
        while not stop:
            time.sleep(0.5)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
