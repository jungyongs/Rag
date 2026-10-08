#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_qlora.sh <model_key> <sdpa|flash_attention_2> <gpu_list> [deepspeed_override]"
  echo
  echo "Examples:"
  echo "  bash scripts/run_qlora.sh qwen3_8b flash_attention_2 0,1,2,3"
  echo "  bash scripts/run_qlora.sh qwen25_72b sdpa 0,1,2,3,4,5,6,7"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_SCRIPT=train_qlora.py bash "${ROOT}/scripts/run_benchmark.sh" "$@"
