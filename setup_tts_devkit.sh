#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STUDIO="$ROOT/vendor/neat-genai-studio/src/python"
VENV="$ROOT/.venv-pipertts"

if [[ ! -d /media/nvme ]]; then
  echo "Run this installer on the Modalix DevKit." >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -r "$STUDIO/requirements-pipertts.txt"
(
  cd "$STUDIO"
  TTS_LANGUAGES=en TTS_OPTIONAL_VOICES= PYTHON="$VENV/bin/python" bash voice_install.sh
)
echo "Piper voice ready: $STUDIO/ui/assets/en_US-kristin-medium.onnx"
