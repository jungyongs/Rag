#!/usr/bin/env bash
# Generate + score one model on an item-format split.
#   bash run_eval.sh <model_dir> <dataset_dir> <split> <out_dir> [gen_gpus] [judge_url]
# e.g.
#   bash run_eval.sh ~/Rag/outputs/ax4_light_7b_sft_v2items/best ~/dataset_renewal_1029/rx_counsel_v2_items_ctx50 test results/sft_best_v2test 0
# With judge_url (an OpenAI-compatible server of A.X-4.0 72B, see below) ② 근거 충실도 is judged by the LLM;
# without it the fast lexical proxy is used.
#   judge server (GPUs 4-7):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/vllm serve ~/Rag/models/AX-4.0 \
#     --tensor-parallel-size 4 --served-model-name ax4-72b --max-model-len 16384 --port 8000
set -euo pipefail
MODEL="$1"; DATA="$2"; SPLIT="$3"; OUT="$4"; GPUS="${5:-0}"; JUDGE="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
CUDA_VISIBLE_DEVICES="$GPUS" VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python "$HERE/gen_items.py" \
  --model "$MODEL" --data "$DATA" --split "$SPLIT" --out "$OUT/preds.jsonl" --tp "$NGPU" 2>&1 | tail -3
if [ -n "$JUDGE" ]; then
  FAITH=(--faith llm --judge-url "$JUDGE" --judge-model ax4-72b)
else
  FAITH=(--faith lexical)
fi
python3 "$HERE/score_items.py" --data "$DATA" --split "$SPLIT" --pred "$OUT/preds.jsonl" "${FAITH[@]}" \
  --out "$OUT/scores.json" --details "$OUT/details.jsonl" | tee "$OUT/scores.txt"
