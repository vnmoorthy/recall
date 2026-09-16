#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
module_for() {
  case "$1" in
    api) echo server ;;
    genai) echo genai_server ;;
    *) echo "$1" ;;
  esac
}

for name in api perception preview genai; do
  pid_file="$ROOT/run/$name.pid"
  module="$(module_for "$name")"
  pids=()
  [[ -f "$pid_file" ]] && pids+=("$(cat "$pid_file")")
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(pgrep -f -- "[p]ython.*-m recall\.$module([[:space:]]|$)" || true)
  for pid in $(printf '%s\n' "${pids[@]}" | sort -nu); do
    path="/proc/$pid/cmdline"
    [[ -r "$path" ]] || continue
    actual="$(tr '\0' ' ' <"$path")"
    if [[ "$actual" == *"-m recall.$module"* ]]; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..100}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
      echo "stopped $name (pid $pid)"
    fi
  done
  rm -f "$pid_file"
done
rm -f "$ROOT/run/genai.ready" "$ROOT/run/perception-status.json"
