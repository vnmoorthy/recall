#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESTART=false
if [[ "${1:-}" == "--restart" ]]; then
  RESTART=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--restart]" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$ROOT/data" "$ROOT/media/frames" "$ROOT/media/crops"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache-$$"

start_process() {
  local name="$1"
  shift
  local pid_file="$RUN_DIR/$name.pid"
  local expected="$*"
  local module
  module="$(module_for "$name")"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    local actual
    actual="$(cmdline_for "$pid")"
    if ! $RESTART && [[ "$actual" == "$expected" ]]; then
      echo "$pid" >"$pid_file"
      echo "$name already running (pid $pid)"
      return
    fi
  done < <(pgrep -f -- "[p]ython.*-m recall\.$module([[:space:]]|$)" || true)
  stop_process "$name"
  [[ "$name" == "genai" ]] && rm -f "$RUN_DIR/genai.ready"
  nohup "$@" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$pid_file"
  echo "started $name (pid $!)"
}

cmdline_for() {
  local path="/proc/$1/cmdline"
  [[ -r "$path" ]] || return 0
  tr '\0' ' ' <"$path" | sed 's/ $//'
}

module_for() {
  case "$1" in
    api) echo server ;;
    genai) echo genai_server ;;
    *) echo "$1" ;;
  esac
}

stop_process() {
  local name="$1" pid_file="$RUN_DIR/$1.pid" module
  module="$(module_for "$name")"
  local pids=()
  if [[ -f "$pid_file" ]]; then
    pids+=("$(cat "$pid_file")")
  fi
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(pgrep -f -- "[p]ython.*-m recall\.$module([[:space:]]|$)" || true)
  for pid in $(printf '%s\n' "${pids[@]}" | sort -nu); do
    local actual
    actual="$(cmdline_for "$pid")"
    if kill -0 "$pid" 2>/dev/null && [[ "$actual" == *"-m recall.$module"* ]]; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..100}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  done
  rm -f "$pid_file"
}

PYTHON_BIN="${PYTHON_BIN:-$HOME/pyneat/bin/python3}"
VLM_DIR="/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4/devkit"
FALLBACK_VLM_DIR="/media/nvme/llima/models/gemma-4-E2B-it-GPTQ-a16w4/devkit"
DESIRED_VLM=""
[[ -d "$VLM_DIR" ]] && DESIRED_VLM="${VLM_DIR%/devkit}"
[[ -z "$DESIRED_VLM" && -d "$FALLBACK_VLM_DIR" ]] && DESIRED_VLM="${FALLBACK_VLM_DIR%/devkit}"
if [[ -n "$DESIRED_VLM" ]]; then
  expected_genai="$PYTHON_BIN -m recall.genai_server --vlm $DESIRED_VLM"
  while read -r pid; do
    actual="$(cmdline_for "$pid")"
    [[ "$actual" == "$expected_genai" ]] || RESTART=true
  done < <(pgrep -f -- '[p]ython.*-m recall\.genai_server([[:space:]]|$)' || true)
fi

if $RESTART; then
  stop_process api
  stop_process perception
  stop_process preview
  stop_process genai
  sudo systemctl restart simaai-appcomplex.service
else
  sudo systemctl start simaai-appcomplex.service
fi
if [[ -d "$VLM_DIR" ]]; then
  start_process genai "$PYTHON_BIN" -m recall.genai_server --vlm "${VLM_DIR%/devkit}"
elif [[ -d "$FALLBACK_VLM_DIR" ]]; then
  start_process genai "$PYTHON_BIN" -m recall.genai_server --vlm "${FALLBACK_VLM_DIR%/devkit}"
else
  stop_process genai
fi

if [[ -n "$DESIRED_VLM" ]]; then
  genai_ready=false
  for _ in {1..120}; do
    models="$(curl -fsS --max-time 1 http://127.0.0.1:9998/v1/models 2>/dev/null || true)"
    if [[ "$models" == *"${DESIRED_VLM##*/}"* && "$models" == *"whisper-small-a16w8"* ]]; then
      genai_ready=true
      touch "$RUN_DIR/genai.ready"
      break
    fi
    sleep 0.25
  done
  $genai_ready || echo "warning: GenAI models did not become ready; evidence rules remain available" >&2
fi

MODE=--synthetic
if [[ -f "$ROOT/models/yolo26m-seg-bf16-b1.tar.gz" && -f "$ROOT/models/yolo26m-pose-int8-b1.tar.gz" ]]; then
  MODE=
fi
if [[ -n "$MODE" ]]; then
  stop_process perception
  rm -f "$RUN_DIR/perception-status.json"
  start_process preview "$PYTHON_BIN" -m recall.preview
else
  stop_process preview
  start_process perception "$PYTHON_BIN" -m recall.perception --config "$ROOT/config.yaml" --root "$ROOT"
fi
start_process api "$PYTHON_BIN" -m recall.server --root "$ROOT" --port 8090 $MODE

for _ in {1..60}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8090/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if ! curl -fsS --max-time 1 http://127.0.0.1:8090/health >/dev/null 2>&1; then
  echo "Recall API failed to become healthy; inspect $LOG_DIR/api.log" >&2
  exit 1
fi

PIPER_MODEL="$ROOT/vendor/neat-genai-studio/src/python/ui/assets/en_US-kristin-medium.onnx"
if [[ -f "$PIPER_MODEL" ]]; then
  recall_ready=false
  for _ in {1..120}; do
    if curl -fsS --max-time 1 http://127.0.0.1:8090/ready >/dev/null 2>&1; then
      recall_ready=true
      break
    fi
    sleep 0.25
  done
  $recall_ready || echo "warning: Recall is live but a required component is not ready" >&2
else
  echo "warning: Piper is unavailable; run ./setup_tts_devkit.sh once" >&2
fi
echo "Recall mode: ${MODE:---hardware}"
echo "Recall API: http://$(hostname -I | awk '{print $1}'):8090"
