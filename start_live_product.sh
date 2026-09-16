#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVKIT_HOST="${RECALL_DEVKIT_HOST:-10.42.0.232}"
DEVKIT_USER="${RECALL_DEVKIT_USER:-sima}"
API_URL="http://$DEVKIT_HOST:8090"
UI_URL="http://127.0.0.1:8080/app.html"
SSH=(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$DEVKIT_USER@$DEVKIT_HOST")

for archive in yolo26m-seg-bf16-b1.tar.gz yolo26m-pose-int8-b1.tar.gz; do
  if [[ ! -s "$ROOT/models/$archive" ]]; then
    echo "Missing $ROOT/models/$archive" >&2
    echo "Download the exact precompiled package listed in README.md." >&2
    exit 1
  fi
done

echo "[1/4] Checking Modalix DevKit at $DEVKIT_HOST"
if ! "${SSH[@]}" true; then
  cat >&2 <<EOF
DevKit is unreachable. Power it on and connect its direct Ethernet/USB link to
this host, then run this command again:

  cd $ROOT && ./start_live_product.sh
EOF
  exit 1
fi

echo "[2/4] Starting hardware perception and Recall services"
"${SSH[@]}" \
  "cd /workspace/recall && source ~/pyneat/bin/activate && ./run_devkit.sh --restart"

echo "[3/4] Waiting for nonzero segmentation and pose output"
status=""
hardware_ready=false
for _ in {1..120}; do
  status="$(curl -fsS --max-time 3 "$API_URL/status" 2>/dev/null || true)"
  if STATUS="$status" python3 - <<'PY' 2>/dev/null
import json
import os

status = json.loads(os.environ["STATUS"])
assert status.get("perception_mode") == "hardware"
assert float(status.get("seg_fps", 0)) > 0
assert float(status.get("pose_fps", 0)) > 0
PY
  then
    hardware_ready=true
    break
  fi
  sleep 2
done

if ! $hardware_ready; then
  echo "Hardware services started, but live perception did not produce frames." >&2
  [[ -n "$status" ]] && echo "Last status: $status" >&2
  echo "Inspect: ssh $DEVKIT_USER@$DEVKIT_HOST 'tail -n 100 /workspace/recall/logs/perception.log'" >&2
  exit 1
fi

echo "[4/4] Starting the local product UI"
if ! curl -fsS --max-time 2 "$UI_URL" >/dev/null 2>&1; then
  nohup "$ROOT/run_mac.sh" >"$ROOT/logs/ui.log" 2>&1 &
  for _ in {1..20}; do
    curl -fsS --max-time 2 "$UI_URL" >/dev/null 2>&1 && break
    sleep 0.25
  done
fi

echo "$status" | python3 -m json.tool
cat <<EOF

REAL LIVE PRODUCT READY
Recall UI:     $UI_URL
Recall API:    $API_URL/docs
Insight view:  https://127.0.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4
EOF
