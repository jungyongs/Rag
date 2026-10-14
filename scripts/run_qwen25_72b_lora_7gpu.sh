#!/usr/bin/env bash
# Start the Qwen2.5-72B LoRA run on GPUs 0-6, detached from the terminal.
# Run it by hand when the GPUs are free; nothing calls this automatically.
#
#   bash scripts/run_qwen25_72b_lora_7gpu.sh
#   STOP_AT="2026-10-20 18:00" bash scripts/run_qwen25_72b_lora_7gpu.sh
#
# Settings come from config.yaml (qwen25_72b_lora). It refuses to start
# while any of the GPUs is still in use.
set -euo pipefail

GPUS="0,1,2,3,4,5,6"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/rag}"

busy="$(nvidia-smi -i "$GPUS" --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 > 1000 {printf "GPU %s (%s MiB) ", $1, $2}')"
if [[ -n "$busy" ]]; then
  echo "Not starting: GPUs still in use: $busy"
  echo "Check with nvidia-smi and run this again when they are free."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"
bash scripts/launch_detached.sh scripts/run_benchmark.sh qwen25_72b_lora flash_attention_2 "$GPUS"
