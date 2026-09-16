#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$ROOT/data" "$ROOT/media/frames" "$ROOT/media/crops"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache-$$"

start_process() {
  local name="$1"
  shift
  local pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pid_file"))"
    return
  fi
  nohup "$@" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$pid_file"
  echo "started $name (pid $!)"
}

PYTHON_BIN="${PYTHON_BIN:-$HOME/pyneat/bin/python3}"
VLM_DIR="/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4/devkit"
FALLBACK_VLM_DIR="/media/nvme/llima/models/gemma-4-E2B-it-GPTQ-a16w4/devkit"
ASR_DIR="/media/nvme/llima/models/whisper-small-a16w8/devkit"
sudo systemctl start simaai-appcomplex.service
if [[ -d "$VLM_DIR" || -d "$FALLBACK_VLM_DIR" ]]; then
  start_process genai "$PYTHON_BIN" -m recall.genai_server
fi

MODE=--synthetic
if [[ -f "$ROOT/models/yolo26m-seg-bf16-b1.tar.gz" && -f "$ROOT/models/yolo26m-pose-int8-b1.tar.gz" ]]; then
  MODE=
fi
if [[ -n "$MODE" ]]; then
  start_process preview "$PYTHON_BIN" -m recall.preview
else
  if [[ -f "$RUN_DIR/preview.pid" ]] && kill -0 "$(cat "$RUN_DIR/preview.pid")" 2>/dev/null; then
    kill "$(cat "$RUN_DIR/preview.pid")"
  fi
  start_process perception "$PYTHON_BIN" -m recall.perception --config "$ROOT/config.yaml" --root "$ROOT"
fi
start_process api "$PYTHON_BIN" -m recall.server --root "$ROOT" --port 8090 $MODE
echo "Recall API: http://$(hostname -I | awk '{print $1}'):8090"
