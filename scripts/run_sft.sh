#!/usr/bin/env bash
# SFT on an item-format rx_counsel dataset (see src/train_sft.py).
#   bash scripts/run_sft.sh <data_dir> <output_dir> [gpu_list] [extra train_sft.py args...]
# e.g.
#   bash scripts/run_sft.sh ~/dataset_renewal_1029/rx_counsel_v2_items_ctx50 outputs/ax4_light_7b_sft_v2items 0,1,2,3,4,5,6,7
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/run_sft.sh <data_dir> <output_dir> [gpu_list] [extra args...]"
  exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
GPUS="${3:-0,1,2,3,4,5,6,7}"
shift $(( $# < 3 ? $# : 3 ))

VENV="${VENV:-$HOME/venvs/rag}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
mkdir -p "$OUTPUT_DIR"
# 20k+ token sequences: the logits alone are ~10 GiB, so avoid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29511}" \
  src/train_sft.py --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
