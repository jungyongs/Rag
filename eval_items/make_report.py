"""Collect results/<model>/<split>/scores.json into SUMMARY_<split>.md per split and SUMMARY.md (all splits),
columns: 정답(gt) and each model; then copy the summaries and the per-model scores to --copy-to (shared, as a backup).

python3 make_report.py RESULTS_DIR [--models gt,base,best,last] [--splits v2_test,v3_real] [--copy-to DIR]
"""
import argparse
import json
import os
import shutil
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('results')
ap.add_argument('--models', default='gt,base,best,last')
ap.add_argument('--splits', default='v2_test,v3_real')
ap.add_argument('--title', default='rx_counsel 항목형 평가 결과')
ap.add_argument('--copy-to', default='/home/mtrack/shared/rx_eval_results_items_v2phclean',
                help="copy SUMMARY*.md and every scores.json / scores.txt here ('' to skip)")
a = ap.parse_args()

NAMES = {'gt': '정답(gt)', 'base': 'base (A.X-4.0-Light)', 'best': 'SFT best (val 최저)', 'last': 'SFT last (2 epoch)'}
SPLITS = {'v2_test': 'v2 test', 'v3_real': 'v3 real'}
HEAD = ['- 생성: vLLM, greedy, user 메시지 하나(input_md 그대로), 시스템 프롬프트 없음, 컨텍스트 32,768 (YaRN×2)',
        '- 채점: rx_eval_items/score_items.py (② 근거 충실도는 A.X-4.0 72B judge). 정답(gt) 열은 정답을 예측으로 넣은 기준선',
        '- 단위: %. ↓ 낮을수록 좋음, 나머지는 높을수록 좋음', '']
lines = [f'# {a.title}', ''] + HEAD
written = []
for split in a.splits.split(','):
    models = [m for m in a.models.split(',') if os.path.exists(f'{a.results}/{m}/{split}/scores.json')]
    if not models:
        continue
    start = len(lines)
    S = {m: json.load(open(f'{a.results}/{m}/{split}/scores.json', encoding='utf-8')) for m in models}
    keys = list(S[models[0]])
    lines += [f'## {SPLITS.get(split, split)}', '', '| 지표 | ' + ' | '.join(NAMES.get(m, m) for m in models) + ' |',
              '|---|' + '---|' * len(models)]
    for k in keys:
        lines.append(f'| {k} | ' + ' | '.join('' if S[m].get(k) is None else str(S[m].get(k)) for m in models) + ' |')
    lines += ['', '생성 통계:', '']
    for m in models:
        p = f'{a.results}/{m}/{split}/preds.jsonl'
        if m == 'gt' or not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p, encoding='utf-8')]
        toks = [r['n_tokens'] for r in rs]
        fin = {}
        for r in rs:
            fin[r['finish']] = fin.get(r['finish'], 0) + 1
        lines.append(f'- {NAMES.get(m, m)}: n {len(rs):,}, 출력 토큰 중앙값 {int(statistics.median(toks)):,}, '
                     f'최대 {max(toks):,}, 종료 {fin}')
    lines.append('')
    part = f'{a.results}/SUMMARY_{split}.md'
    open(part, 'w', encoding='utf-8').write('\n'.join([f'# {a.title} — {SPLITS.get(split, split)}', ''] + HEAD + lines[start:]))
    written.append(part)
open(f'{a.results}/SUMMARY.md', 'w', encoding='utf-8').write('\n'.join(lines))
written.append(f'{a.results}/SUMMARY.md')
print('\n'.join(lines))

if a.copy_to:
    for p in written:
        os.makedirs(a.copy_to, exist_ok=True)
        shutil.copyfile(p, os.path.join(a.copy_to, os.path.basename(p)))
    for m in a.models.split(','):
        for split in a.splits.split(','):
            for f in ('scores.json', 'scores.txt'):
                src = f'{a.results}/{m}/{split}/{f}'
                if os.path.exists(src):
                    os.makedirs(f'{a.copy_to}/{m}/{split}', exist_ok=True)
                    shutil.copyfile(src, f'{a.copy_to}/{m}/{split}/{f}')
    print(f'copied to {a.copy_to}')
