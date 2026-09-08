#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/codex_resilient.sh <prompt text | prompt-file>

Environment:
  CODEX_WORKDIR     Working directory (default: repository root)
  CODEX_MAX_RETRIES Resume attempts after the initial run (default: 5)
  CODEX_LOG_DIR     Persistent JSONL/final-output directory (default: .codex-runs)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${CODEX_WORKDIR:-$ROOT}"
MAX_RETRIES="${CODEX_MAX_RETRIES:-5}"
LOG_DIR="${CODEX_LOG_DIR:-$ROOT/.codex-runs}"
if [[ ! "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "CODEX_MAX_RETRIES must be a non-negative integer" >&2
  exit 2
fi
if [[ -f "$1" && $# -eq 1 ]]; then
  prompt="$(<"$1")"
else
  prompt="$*"
fi
[[ -n "$prompt" ]] || { echo "prompt must not be empty" >&2; exit 2; }

mkdir -p "$LOG_DIR"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
final_output="$LOG_DIR/$run_id-final.txt"
session_id=""
attempt=0
export TERM="${TERM:-xterm-256color}"
if [[ "$TERM" == "dumb" ]]; then
  export TERM=xterm-256color
fi

extract_session_id() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8", errors="replace") as stream:
    for line in stream:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = event.get("thread_id") or event.get("session_id")
        if value:
            print(value)
            break
PY
}

while :; do
  event_log="$LOG_DIR/$run_id-attempt-$attempt.jsonl"
  echo "[$(date -u +%FT%TZ)] Codex attempt $attempt; log: $event_log"
  set +e
  if [[ -z "$session_id" ]]; then
    codex exec --json -C "$WORKDIR" -o "$final_output" "$prompt" 2>&1 | tee "$event_log"
  else
    codex exec resume --json -o "$final_output" "$session_id" \
      "The previous response stream disconnected. Inspect the existing workspace and continue the original task from the last completed action. Do not redo completed work." \
      2>&1 | tee "$event_log"
  fi
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "[$(date -u +%FT%TZ)] Codex completed; final response: $final_output"
    exit 0
  fi

  if [[ -z "$session_id" ]]; then
    session_id="$(extract_session_id "$event_log")"
  fi
  if (( attempt >= MAX_RETRIES )); then
    echo "Codex failed after $((attempt + 1)) attempts; logs remain in $LOG_DIR" >&2
    exit "$status"
  fi
  attempt=$((attempt + 1))
  delay=$((attempt * 5))
  echo "[$(date -u +%FT%TZ)] stream failed; resuming session ${session_id:-unknown} in ${delay}s" >&2
  sleep "$delay"
done
