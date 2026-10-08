#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  bash scripts/run_benchmark.sh <model_key> <sdpa|flash_attention_2> <gpu_list> [deepspeed_override]"
  echo
  echo "Examples:"
  echo "  bash scripts/run_benchmark.sh qwen3_8b sdpa 0,1,2,3"
  echo "  bash scripts/run_benchmark.sh qwen25_72b flash_attention_2 0,1,2,3,4,5,6,7"
  exit 1
fi

MODEL_KEY="$1"
ATTN="$2"
GPU_LIST="$3"
DS_OVERRIDE="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/config.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS="${#GPU_ARRAY[@]}"

echo "Model       : $MODEL_KEY"
echo "Attention   : $ATTN"
echo "GPU list    : $GPU_LIST"
echo "GPU count   : $NUM_GPUS"

CMD=(
  torchrun
  --standalone
  --nproc_per_node="$NUM_GPUS"
  "${ROOT}/src/train_benchmark.py"
  --config "$CONFIG"
  --model "$MODEL_KEY"
  --attn_implementation "$ATTN"
)

if [[ -n "$DS_OVERRIDE" ]]; then
  CMD+=(--deepspeed_config "$DS_OVERRIDE")
fi

CUDA_VISIBLE_DEVICES="$GPU_LIST" "${CMD[@]}"
