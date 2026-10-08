#!/usr/bin/env bash
# Run a training launcher in the background, detached from the terminal, so
# closing the SSH session does not kill training. Output goes to a log file.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/launch_detached.sh <launcher script> <model_key> [launcher args...]"
  echo
  echo "Examples:"
  echo "  bash scripts/launch_detached.sh scripts/run_benchmark.sh qwen3_8b flash_attention_2 0,1,2,3"
  echo "  STOP_AT=\"2026-10-09 18:00\" bash scripts/launch_detached.sh scripts/run_qlora.sh qwen25_72b sdpa 0,1,2,3,4,5,6,7"
  exit 1
fi

LAUNCHER="$1"
shift
MODEL_KEY="$1"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "$LOG_DIR"

export LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "$LAUNCHER" .sh)_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log}"
export NO_TEE=1
PID_FILE="${LOG_FILE%.log}.pid"

setsid nohup bash "$LAUNCHER" "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Started in background (PID ${PID})."
echo "  Follow log : tail -f ${LOG_FILE}"
echo "  Stop+save  : kill -TERM \$(cat ${PID_FILE})"
