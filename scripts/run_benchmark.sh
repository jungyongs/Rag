#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  bash scripts/run_benchmark.sh <model_key> <sdpa|flash_attention_2> <gpu_list> [deepspeed_override]"
  echo
  echo "Examples:"
  echo "  bash scripts/run_benchmark.sh qwen3_8b sdpa 0,1,2,3"
  echo "  bash scripts/run_benchmark.sh qwen25_72b flash_attention_2 0,1,2,3,4,5,6,7"
  echo
  echo "Environment (all optional):"
  echo "  STOP_AT=\"2026-10-09 18:00\"  end of the GPU reservation; save + stop before it"
  echo "  MAX_RETRIES=3               relaunches after a crash (each resumes from the last checkpoint)"
  echo "  RETRY_DELAY=60              seconds between relaunches"
  echo "  LOG_FILE=path               log file (default: logs/<script>_<model>_<attn>_<time>.log)"
  echo
  echo "Stop gracefully (save, then exit): Ctrl+C, or kill -TERM <pid of this script>."
  echo "To survive SSH disconnects use scripts/launch_detached.sh."
  exit 1
fi

MODEL_KEY="$1"
ATTN="$2"
GPU_LIST="$3"
DS_OVERRIDE="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/config.yaml}"
# train_benchmark.py (BF16) or train_qlora.py (4-bit); see run_qlora.sh
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_benchmark.py}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-60}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/${TRAIN_SCRIPT%.py}_${MODEL_KEY}_${ATTN}_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

# Copy all output (including every retry) into the log file.
# launch_detached.sh already redirects into it and sets NO_TEE=1.
# tee ignores Ctrl+C so it keeps logging while training saves on shutdown.
mkdir -p "$(dirname "$LOG_FILE")"
if [[ -z "${NO_TEE:-}" ]]; then
  exec > >(trap '' INT TERM; exec tee -a "$LOG_FILE") 2>&1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS="${#GPU_ARRAY[@]}"

echo "Script      : $TRAIN_SCRIPT"
echo "Model       : $MODEL_KEY"
echo "Attention   : $ATTN"
echo "GPU list    : $GPU_LIST"
echo "GPU count   : $NUM_GPUS"
echo "Stop at     : ${STOP_AT:-none}"
echo "Max retries : $MAX_RETRIES"
echo "Log file    : $LOG_FILE"
echo "Launcher PID: $$"

CMD=(
  torchrun
  --standalone
  --nproc_per_node="$NUM_GPUS"
  "${ROOT}/src/${TRAIN_SCRIPT}"
  --config "$CONFIG"
  --model "$MODEL_KEY"
  --attn_implementation "$ATTN"
)

if [[ -n "$DS_OVERRIDE" ]]; then
  CMD+=(--deepspeed_config "$DS_OVERRIDE")
fi

if [[ -n "${STOP_AT:-}" ]]; then
  CMD+=(--stop_at "$STOP_AT")
  STOP_AT_EPOCH="$(date -d "$STOP_AT" +%s)"
fi

# Graceful stop: signal the training workers directly. Signalling torchrun
# instead makes it SIGKILL the workers ~30 s later, which can cut off a
# large checkpoint mid-save.
STOP_REQUESTED=0
CHILD=""
request_stop() {
  STOP_REQUESTED=1
  echo "[launcher] $1 received: asking training to save and stop..."
  if [[ -n "$CHILD" ]]; then
    pkill -TERM -P "$CHILD" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null || true
  fi
}
trap 'request_stop SIGTERM' TERM
trap 'request_stop SIGINT' INT

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[launcher] Attempt ${attempt} started at $(date '+%F %T')"

  # Run in the background so the traps above fire while training runs.
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "${CMD[@]}" &
  CHILD=$!

  set +e
  wait "$CHILD"
  rc=$?
  # A trapped signal interrupts `wait`; keep waiting until training exits.
  while kill -0 "$CHILD" 2>/dev/null; do
    wait "$CHILD"
    rc=$?
  done
  set -e
  CHILD=""

  if [[ $rc -eq 0 ]]; then
    echo "[launcher] Training exited normally."
    exit 0
  fi
  if [[ $STOP_REQUESTED -eq 1 ]]; then
    echo "[launcher] Stopped on request (exit code ${rc})."
    exit "$rc"
  fi
  if [[ -n "${STOP_AT_EPOCH:-}" && $(date +%s) -ge $STOP_AT_EPOCH ]]; then
    echo "[launcher] Exit code ${rc}, but the reservation (${STOP_AT}) is over; not retrying."
    exit "$rc"
  fi
  if [[ $attempt -gt $MAX_RETRIES ]]; then
    echo "[launcher] Exit code ${rc}; giving up after ${MAX_RETRIES} retries."
    exit "$rc"
  fi

  echo "[launcher] Exit code ${rc}; retrying in ${RETRY_DELAY}s (resumes from the last checkpoint)."
  sleep "$RETRY_DELAY" &
  wait $! || true
  if [[ $STOP_REQUESTED -eq 1 ]]; then
    echo "[launcher] Stopped on request."
    exit "$rc"
  fi
done
