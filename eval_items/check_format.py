"""Format checks of an item-format dataset (gold output_items + input_md + data).

python3 check_format.py DATASET_DIR
Checks: gold JSON follows the instruction schema (keys, '_출처' pairs, string/array types, array lengths), every cited
E id is in that record's input_md, the drug cards of 카운터 / 공부 / 복약지도서 are the same and match the prescription
(lines of the same drug merged), 팝업 rules, and the instruction text is identical in every input_md.
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

DATA = sys.argv[1]
SCHEMA_TYPES = {  # field -> 'str' | 'list'
    '카운터': {'복용법': 'str', '빨간박스': 'list', '팝업 한 줄': 'str', '약사 확인': 'list'},
    '공부': {'처방 구성': 'str', '핵심 이슈': 'list', '가이드라인': 'list', '최근 논문 동향': 'list'},
    '공부약별': {k: 'str' for k in ['적응증', '용법·용량', '금기', '경고·주의', '부작용', '상호작용', '약동학·기전', '특수인구', '보관']},
    '복약지도서': {'받은 약': 'str'},
    '복약지도서약별': {'무엇에 쓰는 약': 'str', '복용법': 'str', '꼭 지켜야 할 것': 'list', '생길 수 있는 증상': 'list',
                '바로 연락할 증상': 'list', '보관법': 'list', '생활 수칙': 'list'},
}
problems = collections.Counter()
examples = {}


def bad(kind, where):
    problems[kind] += 1
    examples.setdefault(kind, where)


def check_obj(obj, types, where, ev_ids):
    keys = set(obj)
    want = set(types) | {k + '_출처' for k in types}
    for k in want - keys:
        bad('키 없음', f'{where}: {k}')
    for k in keys - want - {'약별'}:
        bad('지시문에 없는 키', f'{where}: {k}')
    for k, t in types.items():
        if k not in obj or k + '_출처' not in obj:
            continue
        v, s = obj[k], obj[k + '_출처']
        if t == 'str':
            if not isinstance(v, str):
                bad('타입(문자열이어야 함)', f'{where}: {k}')
            if not isinstance(s, list) or not all(isinstance(x, str) for x in s):
                bad('출처 형식(문자열 배열)', f'{where}: {k}_출처')
            flat = s if isinstance(s, list) else []
        else:
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                bad('타입(문자열 배열이어야 함)', f'{where}: {k}')
            if not isinstance(s, list) or not all(isinstance(x, list) for x in s):
                bad('출처 형식(배열의 배열)', f'{where}: {k}_출처')
            elif isinstance(v, list) and len(v) != len(s):
                bad('항목·출처 길이 다름', f'{where}: {k} {len(v)} vs {len(s)}')
            flat = [y for x in s for y in x] if isinstance(s, list) and all(isinstance(x, list) for x in s) else []
        for e in flat:
            if e == '처방전':
                continue
            if not re.fullmatch(r'E\d+', str(e)):
                bad('출처 값이 근거 번호가 아님', f'{where}: {k}_출처 {e}')
            elif e not in ev_ids:
                bad('입력에 없는 근거 번호', f'{where}: {k}_출처 {e}')


base = lambda nm: re.sub(r'\[\d+번\]$', '', nm)  # noqa: E731
recs = {}
for p in glob.glob(f'{DATA}/data/**/*.jsonl', recursive=True):
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        recs[r['id']] = r
instr = collections.Counter()
n = 0
for gp in sorted(glob.glob(f'{DATA}/output_items/*/*.json')):
    rid = Path(gp).stem
    mp = gp.replace('/output_items/', '/input_md/')[:-5] + '.md'
    md = open(mp, encoding='utf-8').read()
    instr[md.split('## 2. 처방전')[0]] += 1
    ev_ids = set(re.findall(r'^- \*\*\[(E\d+)\]\*\*', md, re.M))
    try:
        g = json.load(open(gp, encoding='utf-8'))
    except Exception:  # noqa: BLE001
        bad('JSON 아님', rid)
        continue
    n += 1
    if list(g) != ['카운터', '공부', '복약지도서']:
        bad('최상위 키', f'{rid}: {list(g)}')
    # lines of the same drug (same 보험코드) make one card named after the first line; other codes keep their name
    want_cards, seen = [], set()
    for d in recs[rid]['input']['처방전']['처방약']:
        if d['보험코드'] not in seen:
            seen.add(d['보험코드'])
            want_cards.append(base(d['제품명']) if not any(base(x) == base(d['제품명']) for x in want_cards) else d['제품명'])
    sets = {'카운터': set(g['카운터']), '공부': set(g['공부'].get('약별', {})), '복약지도서': set(g['복약지도서'].get('약별', {}))}
    for k, s in sets.items():
        if s != set(want_cards):
            bad(f'{k} 약 카드가 처방약과 다름', f'{rid}: {sorted(s ^ set(want_cards))}')
    for nm, c in g['카운터'].items():
        check_obj(c, SCHEMA_TYPES['카운터'], f'{rid} 카운터 {nm}', ev_ids)
        rb, pop = c.get('빨간박스') or [], c.get('팝업 한 줄') or ''
        if bool(rb) != bool(pop):
            bad('팝업과 빨간박스 비어 있음이 다름', f'{rid} {nm}')
        if pop and (len(pop) > 45 or len(pop.split(' · ')) > 3):
            bad('팝업 길이·항목 수 초과', f'{rid} {nm}: {pop}')
        if c.get('팝업 한 줄_출처') != []:
            bad('팝업 출처가 비어 있지 않음', f'{rid} {nm}')
    check_obj({k: v for k, v in g['공부'].items() if k != '약별'}, SCHEMA_TYPES['공부'], f'{rid} 공부', ev_ids)
    for nm, c in g['공부'].get('약별', {}).items():
        check_obj(c, SCHEMA_TYPES['공부약별'], f'{rid} 공부 {nm}', ev_ids)
    check_obj({k: v for k, v in g['복약지도서'].items() if k != '약별'}, SCHEMA_TYPES['복약지도서'], f'{rid} 복약지도서', ev_ids)
    for nm, c in g['복약지도서'].get('약별', {}).items():
        check_obj(c, SCHEMA_TYPES['복약지도서약별'], f'{rid} 복약지도서 {nm}', ev_ids)
        if c.get('복용법') and '처방전' not in (c.get('복용법_출처') or []):
            bad('복약지도서 복용법 출처에 처방전 없음', f'{rid} {nm}')
print(f'{DATA}: 정답 {n}개')
print(f'지시문(처방전 앞부분) 종류: {len(instr)}개' + ('' if len(instr) == 1 else ' ← 레코드마다 다름'))
if not problems:
    print('문제 없음')
for k, c in problems.most_common():
    print(f'  {k}: {c}  예) {examples[k]}')
