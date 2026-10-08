#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_benchmark_sdpa.sh <model_key> <gpu_list> [deepspeed_override]"
  exit 1
fi

MODEL="$1"
GPUS="$2"
DS="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "$DS" ]]; then
  bash "${ROOT}/scripts/run_benchmark.sh" "$MODEL" sdpa "$GPUS" "$DS"
else
  bash "${ROOT}/scripts/run_benchmark.sh" "$MODEL" sdpa "$GPUS"
fi
