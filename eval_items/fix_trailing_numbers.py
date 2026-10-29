"""phclean에서 약사 복약지도 자료 근거 끝에 잘못 붙은 숫자를 지운다 (예: '…치료에 사용합니다. 00').

대상: 근거유형 '약사 복약지도 자료'의 원문발췌가 문장 끝('다.', '요.') 뒤에 공백 + 숫자 1~3자리로 끝나는 경우만.
(DUR 기준값·허가사항 표처럼 원래 숫자로 끝나는 근거는 건드리지 않는다.)
같은 문구를 data 근거, input_md 근거 줄, 정답(output_items)에서 함께 고친다.

사용: python3 fix_trailing_numbers.py <데이터셋 폴더> [--dry-run]
"""
import collections
import glob
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import dumps  # noqa: E402

DATA = sys.argv[1]
DRY = '--dry-run' in sys.argv
STRAY = re.compile(r'(?<=[다요]\.)\s+\d{1,3}$')

stats = collections.Counter()
fixed_text = {}  # rid -> {old evidence text: new}
for p in sorted(glob.glob(f'{DATA}/data/**/*.jsonl', recursive=True)):
    recs = [json.loads(l) for l in open(p, encoding='utf-8')]
    dirty = False
    for r in recs:
        for e in r['근거']:
            if e['근거유형'] == '약사 복약지도 자료' and STRAY.search(e['원문발췌']):
                new = STRAY.sub('', e['원문발췌'])
                fixed_text.setdefault(r['id'], {})[e['원문발췌']] = new
                stats[repr(e['원문발췌'][-12:])] += 1
                e['원문발췌'] = new
                dirty = True
    if dirty and not DRY:
        with open(p, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

n_md = n_gt = 0
for rid, fx in fixed_text.items():
    mp = glob.glob(f'{DATA}/input_md/*/{rid}.md')[0]
    lines = open(mp, encoding='utf-8').read().split('\n')
    for j, ln in enumerate(lines):
        for old, new in fx.items():
            if ln.startswith('- **[E') and ln.endswith(old):
                lines[j] = ln[:len(ln) - len(old)] + new
                n_md += 1
    gp = mp.replace('/input_md/', '/output_items/')[:-3] + '.json'
    g = json.load(open(gp, encoding='utf-8'))

    def walk(o):
        global n_gt
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str) and STRAY.search(o) and any(old.endswith(o) for old in fx):
            n_gt += 1
            return STRAY.sub('', o)
        return o
    g2 = walk(g)
    if not DRY:
        open(mp, 'w', encoding='utf-8').write('\n'.join(lines))
        open(gp, 'w', encoding='utf-8').write(dumps(g2) + '\n')

print('고친 근거(끝 문구별):', dict(stats))
print(f'레코드 {len(fixed_text)}, 근거 {sum(stats.values())}, md 줄 {n_md}, 정답 문장 {n_gt}' + (' (dry-run)' if DRY else ''))
