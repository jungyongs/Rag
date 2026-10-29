"""Generate item-format (JSON) answers for rx_counsel input_md with vLLM.

CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python gen_items.py \
    --model MODEL_DIR --data DATASET_DIR --split test --out preds.jsonl [--limit N] [--yarn 2]

Same setting as the 2026-10-27 evaluation: one user message (the input_md as is), no system prompt, greedy,
context 32,768 (YaRN x2; SFT checkpoints already carry it, --yarn adds it to a base model).
Output: one {"id", "output", "n_tokens", "finish"} per line. Records already in --out are skipped (resume).
"""
import argparse
import glob
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--data', required=True, help='dataset folder with input_md/<split>/<id>.md')
ap.add_argument('--split', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--tp', type=int, default=1)
ap.add_argument('--max-len', type=int, default=32768)
ap.add_argument('--yarn', type=float, default=0, help='add YaRN rope scaling with this factor (base models)')
args = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402  (after argparse so --help is fast)

paths = sorted(glob.glob(f'{args.data}/input_md/{args.split}/*.md'))
if args.limit:
    paths = paths[:args.limit]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)['id'] for l in open(args.out, encoding='utf-8')}
todo = [p for p in paths if Path(p).stem not in done]
print(f'{len(paths)} records, {len(done)} already in {args.out}, {len(todo)} to generate')

overrides = None
if args.yarn:
    cfg = json.load(open(os.path.join(args.model, 'config.json')))
    orig = cfg.get('max_position_embeddings', 16384)
    overrides = {'rope_scaling': {'rope_type': 'yarn', 'factor': args.yarn, 'original_max_position_embeddings': orig},
                 'max_position_embeddings': int(orig * args.yarn)}
llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_len, dtype='bfloat16',
          gpu_memory_utilization=0.9, hf_overrides=overrides)
tok = llm.get_tokenizer()

convs, params, ids = [], [], []
for p in todo:
    conv = [{'role': 'user', 'content': Path(p).read_text(encoding='utf-8')}]
    n_prompt = len(tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True))
    if n_prompt >= args.max_len - 256:
        print(f'skip {Path(p).stem}: prompt {n_prompt} tokens')
        continue
    convs.append(conv)
    params.append(SamplingParams(temperature=0, max_tokens=args.max_len - n_prompt - 8))
    ids.append(Path(p).stem)

with open(args.out, 'a', encoding='utf-8') as f:
    for rid, o in zip(ids, llm.chat(convs, params)):
        c = o.outputs[0]
        f.write(json.dumps({'id': rid, 'output': c.text, 'n_tokens': len(c.token_ids), 'finish': c.finish_reason},
                           ensure_ascii=False) + '\n')
print(f'wrote {len(ids)} -> {args.out}')
