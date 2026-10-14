#!/usr/bin/env bash
# Start the Qwen2.5-72B LoRA run (scripts/run_qwen25_72b_lora_7gpu.sh) once
# the given runs are over. It starts only if
#   1. every given launcher process has exited,
#   2. every given run dir has a complete 100% milestone, and
#   3. all GPUs stay idle (and no training process exists) for 3 checks.
# If a run did not reach 100% it gives up without starting anything.
# Launched by hand, once; it never deletes anything.
#
#   setsid nohup bash scripts/start_72b_lora_after.sh \
#     <launcher .pid file>:<run dir> [<launcher .pid file>:<run dir> ...] \
#     >> logs/start_72b_lora_after.log 2>&1 < /dev/null &
#
# Cancel: kill $(cat logs/start_72b_lora_after.pid)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-$ROOT/scripts/run_qwen25_72b_lora_7gpu.sh}"
POLL="${POLL:-60}"
GPU_CHECK_INTERVAL="${GPU_CHECK_INTERVAL:-20}"

log() { echo "[$(date '+%F %T %Z')] $*"; }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <launcher .pid file>:<run dir> [...]"
  exit 1
fi

pids=()
dirs=()
for arg in "$@"; do
  pid_file="${arg%%:*}"
  run_dir="${arg#*:}"
  pids+=("$(cat "$pid_file")")
  dirs+=("$run_dir")
done

log "Waiting for launchers ${pids[*]} to exit (runs: ${dirs[*]})"
while true; do
  alive=()
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  [[ ${#alive[@]} -eq 0 ]] && break
  sleep "$POLL"
done
log "All launchers have exited."

for run_dir in "${dirs[@]}"; do
  if ! compgen -G "$run_dir/milestone-100pct-step*/_COMPLETE" > /dev/null; then
    log "Not starting the 72B run: no complete 100% milestone in $run_dir (that run did not finish)."
    exit 1
  fi
  log "Finished: $run_dir"
done

ok=0
last=""
while (( ok < 3 )); do
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 > 1000 {printf "%s ", $1}')"
  procs="$(pgrep -f 'train_benchmark.py|train_qlora.py' | tr '\n' ' ' || true)"
  if [[ -z "$busy" && -z "$procs" ]]; then
    ok=$((ok + 1))
  else
    ok=0
    state="busy GPUs [${busy}] training processes [${procs}]"
    [[ "$state" != "$last" ]] && log "Waiting for idle GPUs: $state"
    last="$state"
  fi
  sleep "$GPU_CHECK_INTERVAL"
done

log "Both runs finished and all GPUs are idle: starting Qwen2.5-72B LoRA on 7 GPUs."
exec bash "$RUN_SCRIPT"
