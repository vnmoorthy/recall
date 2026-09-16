#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_URL="${RECALL_API_URL:-http://127.0.0.1:8090}"
UI_URL="${RECALL_UI_URL:-http://127.0.0.1:8080}"
VIEWER_URL="${RECALL_VIEWER_URL:-https://10.42.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4}"

if [[ "${1:-}" == "--prepare" ]]; then
  "$ROOT/run_devkit.sh" --restart
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--prepare]" >&2
  exit 2
fi

temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT

echo "[1/6] Required components"
ready="$(curl -fsS --max-time 8 "$API_URL/ready")"
READY="$ready" python3 - <<'PY'
import json, os
value = json.loads(os.environ["READY"])
assert value.get("ready") is True, value
print("  perception, GenAI, and Piper ready")
PY

echo "[2/6] Five-object memory"
inventory="$(curl -fsS --max-time 8 "$API_URL/inventory")"
INVENTORY="$inventory" python3 - <<'PY'
import json, os
items = json.loads(os.environ["INVENTORY"])
assert len(items) >= 5, items
names = [item.get("name") or item["class"] for item in items]
print("  " + ", ".join(names[:5]))
PY

ask() {
  local question="$1" response
  response="$(curl -fsS --max-time 25 \
    -H 'Content-Type: application/json' \
    --data "$(QUESTION="$question" python3 -c 'import json,os; print(json.dumps({"question":os.environ["QUESTION"]}))')" \
    "$API_URL/ask")"
  RESPONSE="$response" QUESTION="$question" python3 - <<'PY'
import json, os
result = json.loads(os.environ["RESPONSE"])
assert result.get("answer"), result
assert result.get("event_ids"), result
assert result.get("snapshots"), result
print(f"  Q: {os.environ['QUESTION']}")
print(f"  A: {result['answer']}")
PY
}

echo "[3/6] Evidence-bearing questions"
ask "What's on the table?"
ask "Where is the red mug?"
ask "Who took my laptop?"

echo "[4/6] Ten-minute summary"
summary="$(curl -fsS --max-time 25 -H 'Content-Type: application/json' \
  --data '{"window":600}' "$API_URL/summary")"
SUMMARY="$summary" python3 - <<'PY'
import json, os
result = json.loads(os.environ["SUMMARY"])
assert result.get("summary"), result
assert result.get("event_ids"), result
assert result.get("snapshots"), result
print("  " + result["summary"])
PY

echo "[5/6] Local speech"
curl -fsS --max-time 25 -H 'Content-Type: application/json' \
  --data '{"input":"Recall demo is ready.","response_format":"wav"}' \
  -o "$temporary/ready.wav" "$API_URL/v1/audio/speech"
WAV="$temporary/ready.wav" python3 - <<'PY'
import os, wave
with wave.open(os.environ["WAV"], "rb") as audio:
    assert audio.getnchannels() == 1
    assert audio.getnframes() > 1000
    print(f"  valid mono WAV at {audio.getframerate()} Hz")
PY

echo "[6/6] Presentation surfaces"
test -s "$ROOT/ui/index.html" -a -s "$ROOT/ui/app.js" -a -s "$ROOT/ui/style.css"
if [[ -n "${RECALL_UI_URL:-}" ]]; then
  curl -fsS --max-time 8 -o /dev/null "$UI_URL/"
fi
curl -kfsS --max-time 8 -o /dev/null "$VIEWER_URL"
echo "  Recall UI: $UI_URL"
echo "  Insight:   $VIEWER_URL"
echo "DEMO READY"
