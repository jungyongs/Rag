#!/usr/bin/env bash
# v2 phclean SFT → evaluation of base / best / last, in one run. Each step leaves a marker and is skipped when
# run again, so a stopped pipeline continues where it left off.
#   0 wait for the v2 phclean download   1 prepare v2 (key order, stray trailing numbers, format check, gold scores)
#   2 train (2 epochs, best + last)      3 judge server (A.X-4.0 72B, GPUs 4-7)
#   4 generate (GPUs 0-3)                5 score (② with the judge)   6 SUMMARY.md, stop the judge
# Start detached: bash launch_pipeline.sh     Follow: tail -f $EXP/pipeline.log
set -euo pipefail

EXP="${EXP:-/home/mtrack/exp_v2items}"
V2="${V2:-/home/mtrack/dataset_renewal_1029/rx_counsel_v2_items_ctx50_fix1029_phclean}"
V3="${V3:-/home/mtrack/dataset_renewal_1029/rx_counsel_v3_items_fix1029_phclean}"
V2_FILES=26550
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG="$(dirname "$EVAL")"
TRAIN_OUT="$RAG/outputs/ax4_light_7b_sft_v2items_phclean"
BASE_MODEL="$RAG/models/AX-4.0-Light"
JUDGE_MODEL="$RAG/models/AX-4.0"
JUDGE_URL=http://localhost:8000/v1,http://localhost:8001/v1   # two judge servers
FAITH_SAMPLE="${FAITH_SAMPLE:-0}"   # ② 근거 충실도를 앞 N개 레코드로만 (0 = 전체)
VLLM=~/venvs/vllm/bin
mkdir -p "$EXP/markers" "$EXP/results" "$EXP/logs" "$(dirname "$V2")/_backup"
log() { echo "[$(date '+%F %T')] $*"; }
done_() { [ -f "$EXP/markers/$1" ]; }
mark() { date '+%F %T' > "$EXP/markers/$1"; }

# 0 ----------------------------------------------------------------------------------------------
if ! done_ 0_download; then
  log "0 waiting for the v2 phclean download ($V2_FILES files)"
  while [ "$(find "$V2" -type f 2>/dev/null | wc -l)" -lt "$V2_FILES" ] || pgrep -f "cp .*fix1029_phclean" > /dev/null; do sleep 60; done
  mark 0_download
fi

# 1 ----------------------------------------------------------------------------------------------
if ! done_ 1_prepare; then
  log "1 prepare v2 phclean"
  ( cd "$V2" && tar czf "$(dirname "$V2")/_backup/$(basename "$V2")_before_prepare.tar.gz" data input_md output_items )
  python3 "$EVAL/check_key_order.py" "$V2" --fix
  python3 "$EVAL/fix_trailing_numbers.py" "$V2"
  python3 "$EVAL/check_format.py" "$V2" | tee "$EXP/logs/v2_format_check.txt"
  grep -q "문제 없음" "$EXP/logs/v2_format_check.txt" || { log "format check failed, stopping"; exit 1; }
  python3 "$EVAL/check_key_order.py" "$V2" | grep -q "순서가 다른 객체: 없음" || { log "key order still differs, stopping"; exit 1; }
  for s in test val; do
    python3 "$EVAL/score_items.py" --data "$V2" --split $s --pred gold > "$EXP/logs/v2_${s}_gold_rules.txt"
    # every rule-based metric is 100 (↓ ones 0) for the gold itself, except the lexical 팝업 근거성 baseline
    if grep -v -E '팝업 근거성|^n |파싱 실패|같은 약 여러 줄' "$EXP/logs/v2_${s}_gold_rules.txt" | grep -v -E ' (100\.0|0\.0|None)$' | grep -q .; then
      log "gold scores are not perfect on v2 $s, stopping"; cat "$EXP/logs/v2_${s}_gold_rules.txt"; exit 1
    fi
  done
  mark 1_prepare
fi

# 2 ----------------------------------------------------------------------------------------------
if ! done_ 2_train; then
  log "2 train → $TRAIN_OUT"
  bash "$RAG/scripts/run_sft.sh" "$V2" "$TRAIN_OUT" 0,1,2,3,4,5,6,7 > "$EXP/logs/train_stdout.log" 2>&1
  [ -f "$TRAIN_OUT/best/BEST.json" ] && [ -f "$TRAIN_OUT/last/LAST.json" ] || { log "training did not save best/last"; exit 1; }
  mark 2_train
fi

MODELS=(base best)
BEST_STEP=$(python3 -c "import json;print(json.load(open('$TRAIN_OUT/best/BEST.json'))['step'])")
LAST_STEP=$(python3 -c "import json;print(json.load(open('$TRAIN_OUT/last/LAST.json'))['step'])")
if [ "$BEST_STEP" != "$LAST_STEP" ]; then MODELS+=(last); else log "best == last (step $BEST_STEP): last is not evaluated"; fi
model_path() { case $1 in base) echo "$BASE_MODEL";; best) echo "$TRAIN_OUT/best";; last) echo "$TRAIN_OUT/last";; esac; }
data_of() { case $1 in v2_test) echo "$V2 test";; v3_real) echo "$V3 real";; esac; }
SPLITS=(v2_test v3_real)

# 3 ----------------------------------------------------------------------------------------------
start_judge() {  # one A.X-4.0 72B server per 4 GPUs (generation is done, so all 8 GPUs are free)
  for spec in "4,5,6,7:8000" "0,1,2,3:8001"; do
    local gpus=${spec%%:*} port=${spec##*:}
    curl -s "http://localhost:$port/v1/models" > /dev/null && continue
    log "3 judge server on GPUs $gpus, port $port"
    CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup "$VLLM/vllm" serve "$JUDGE_MODEL" \
      --tensor-parallel-size 4 --served-model-name ax4-72b --max-model-len 16384 --port $port \
      > "$EXP/logs/judge_server_$port.log" 2>&1 < /dev/null &
    echo $! >> "$EXP/judge.pid"
  done
  for port in 8000 8001; do
    for _ in $(seq 120); do curl -s "http://localhost:$port/v1/models" > /dev/null && break; sleep 15; done
    curl -s "http://localhost:$port/v1/models" > /dev/null || { log "judge server $port did not come up"; exit 1; }
  done
}

# 4 ----------------------------------------------------------------------------------------------
if ! done_ 4_generate; then
  log "4 generate: ${MODELS[*]} × ${SPLITS[*]} on GPUs 0-3"
  JOBS=()
  for m in "${MODELS[@]}"; do for s in "${SPLITS[@]}"; do JOBS+=("$m:$s"); done; done
  run_queue() {  # one GPU runs its jobs one after another
    local gpu=$1; shift
    for job in "$@"; do
      local m=${job%%:*} s=${job##*:}; read -r data split <<< "$(data_of $s)"
      local out="$EXP/results/$m/$s"; mkdir -p "$out"
      local yarn=(); [ "$m" = base ] && yarn=(--yarn 2)
      CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM/python" "$EVAL/gen_items.py" --model "$(model_path $m)" \
        --data "$data" --split "$split" --out "$out/preds.jsonl" "${yarn[@]}" > "$EXP/logs/gen_${m}_${s}.log" 2>&1
      log "  generated $m $s"
    done
  }
  pids=()
  for g in 0 1 2 3; do
    q=(); for i in "${!JOBS[@]}"; do [ $((i % 4)) -eq $g ] && q+=("${JOBS[$i]}"); done
    [ ${#q[@]} -gt 0 ] && { run_queue $g "${q[@]}" & pids+=($!); }
  done
  for p in "${pids[@]}"; do wait "$p"; done
  mark 4_generate
fi

# 5 ----------------------------------------------------------------------------------------------
if ! done_ 5_score; then
  start_judge
  for s in "${SPLITS[@]}"; do
    read -r data split <<< "$(data_of $s)"
    for m in gt "${MODELS[@]}"; do
      out="$EXP/results/$m/$s"; mkdir -p "$out"
      [ -f "$out/scores.json" ] && continue
      pred="$out/preds.jsonl"; [ $m = gt ] && pred=gold
      log "5 score $m $s"
      python3 "$EVAL/score_items.py" --data "$data" --split "$split" --pred "$pred" --faith llm \
        --judge-url "$JUDGE_URL" --judge-model ax4-72b --judge-cache "$EXP/judge_cache.jsonl" --workers 64 --faith-limit "$FAITH_SAMPLE" \
        --out "$out/scores.json" --details "$out/details.jsonl" > "$out/scores.txt" 2> "$EXP/logs/score_${m}_${s}.err"
    done
  done
  mark 5_score
fi

# 6 ----------------------------------------------------------------------------------------------
python3 "$EVAL/make_report.py" "$EXP/results" --models "gt,$(IFS=,; echo "${MODELS[*]}")" > /dev/null
[ -f "$EXP/judge.pid" ] && xargs -r kill < "$EXP/judge.pid" 2>/dev/null || true
pkill -f "vllm serve $JUDGE_MODEL" 2>/dev/null || true
mark 6_report
log "done: $EXP/results/SUMMARY.md"
