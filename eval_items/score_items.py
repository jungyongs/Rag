"""Score item-format (JSON) outputs of rx_counsel against output_items/<split>/<id>.json and the record's 정답.

python3 score_items.py --data DATASET_DIR --split test --pred preds.jsonl [--out result.json] [--details per_record.jsonl]
                       [--faith off|llm --judge-url http://host:port/v1 --judge-model ax4-72b]
python3 score_items.py --data DATASET_DIR --split test --pred gold      # 정답을 예측으로 넣은 기준선

PRED  jsonl of {"id", "output"} (gen_items.py), or 'gold'. The JSON object is taken from the output text (```json
      fences, <think> blocks allowed); it must have 카운터 / 공부 / 복약지도서 at the top. An output that does not parse is
      scored as an empty answer (0 in ①–③) and counted in ④. Records without a prediction are reported, not scored.

① 위험 판정   정답 이슈 = 레코드 정답.위험이슈 (약 집합, 구분, 등급). 출력의 구분·등급 = 공부 '핵심 이슈'의 '[구분·등급]'
              태그(공백 무시 완전 일치), 약 = 그 구분을 다룬 '약사 확인'이 달린 카운터 카드. 한 핵심 이슈 줄이 정답 이슈
              여러 개를 덮을 수 있다(정답도 같은 태그를 한 줄로 묶음).
              위험 탐지율 = 정답 이슈 있는 레코드별 (맞은 필드 / 이슈 수×3) 평균, 정확도(약·구분·등급), 전 항목 일치,
              정밀도 = 정답과 짝지어진 핵심 이슈 줄 / 출력 핵심 이슈 줄, 오경보율 = 정답 이슈 없는 레코드 중 핵심 이슈를 낸 비율,
              약사 조치 일치 = 구분을 맞힌 이슈의 '조치:'(핵심 이슈 + 약사 확인)가 그 구분의 핵심 요소를 모두 담음
              (score_rx.action_keys).
② 근거       근거 번호가 붙은 항목: 근거 충실도 = RAGAS (항목을 주장으로 나눠 인용 근거·처방전으로 판정, A.X 72B judge;
              score_rx.faith_llm). '/ 조치:' 부분과 복용법은 빼고 본다. 근거 번호 오류율 = 입력에 없는 번호 / 인용한 번호.
              근거 번호가 없는 항목은 규칙으로 본다:
                처방전 값  복용법(카운터·복약지도서) 정답과 완전 일치, 처방 구성(질환 코드·숫자가 정답과 같음),
                           '같은 약이 N줄로 처방됨'(줄 수·유형이 정답과 같음), 복용법 시각 지어냄
                '자료 없음' 문구  같은 칸이 정답에서도 '자료 없음'이면 맞음 (정답에 근거 있는 내용이 있으면 놓침)
                팝업 한 줄  규칙(빨간박스 있으면 3개·45자 이하, 없으면 빈칸), 근거성(팝업 단어가 같은 카드 빨간박스에 있음)
                그 밖     근거 없는 주장 비율 = 위에 해당하지 않으면서 근거 번호가 없는 항목 / 출력 항목
③ 내용       정답 대비 얼마나 담았나: 항목별 재현율 = 정답 문장(배열이면 하나하나) 중 출력의 같은 카드·항목에 대응하는
              문장(공백 무시 완전 일치 또는 글자 2-gram F1 ≥ 0.8)이 있는 비율. 모드별 평균, 전체 = 모드 평균의 평균.
              대상: 카운터 빨간박스 / 공부 약별 9항목·가이드라인·최근 논문 동향 / 복약지도서 받은 약·약별 6항목
              (핵심 이슈·약사 확인은 ①, 복용법·처방 구성·팝업은 ②에서 본다).
④ 양식       필수 항목 포함률: 카운터·공부·복약지도서별(정답의 약 카드마다 필수 항목, 모드 상위 항목), 전체 = 세 모드의 평균.
              JSON을 못 낸 출력은 포함률 0으로 들어간다.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_rx as sr  # noqa: E402
from popup_check import keyword_misses  # noqa: E402

REQUIRED = {  # mode -> (top-level keys, per-drug keys)
    '카운터': ([], ['복용법', '빨간박스', '팝업 한 줄', '약사 확인']),
    '공부': (['처방 구성', '핵심 이슈', '가이드라인', '약별', '최근 논문 동향'],
           ['적응증', '용법·용량', '금기', '경고·주의', '부작용', '상호작용', '약동학·기전', '특수인구', '보관']),
    '복약지도서': (['받은 약', '약별'],
              ['무엇에 쓰는 약', '복용법', '꼭 지켜야 할 것', '생길 수 있는 증상', '바로 연락할 증상', '보관법', '생활 수칙']),
}
# ③ 재현율 대상 (핵심 이슈·약사 확인은 ①, 복용법·처방 구성·팝업은 ②에서 본다): mode -> (top-level fields, per-drug fields)
RECALL_FIELDS = {
    '카운터': ([], ['빨간박스']),
    '공부': (['가이드라인', '최근 논문 동향'],
           ['적응증', '용법·용량', '금기', '경고·주의', '부작용', '상호작용', '약동학·기전', '특수인구', '보관']),
    '복약지도서': (['받은 약'], ['무엇에 쓰는 약', '꼭 지켜야 할 것', '생길 수 있는 증상', '바로 연락할 증상', '보관법', '생활 수칙']),
}
RECALL_F1 = 0.8
TIME_WORD = re.compile(r'아침|점심|저녁|취침|자기\s*전|식전|식후|공복|오전|오후|잠자리')
TAG = re.compile(r'^\s*\[([^\]]+)\]')
DUP = re.compile(r'같은 약이 (\d+)줄로 처방됨\s*—\s*([^:(/]+)')
NO_DATA = re.compile(r'별도\s*기재\s*없음|해당\s*자료\s*없음|자료를\s*찾지\s*못|안내받으세요|미연결|분류\s*확인\s*필요')
ACTION = re.compile(r'/\s*조치\s*[:：]')


def norm(s):
    return re.sub(r'\s+', '', str(s or ''))


def base(name):
    return re.sub(r'\[\d+번\]$', '', name).strip()


def parse_output(text):
    """The JSON answer in a model output, or None."""
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.S)
    cand = m.group(1) if m else text[text.find('{'):text.rfind('}') + 1]
    try:
        obj = json.loads(cand)
    except Exception:  # noqa: BLE001
        return None
    # a stray {...} inside a Markdown answer is not the answer
    return obj if isinstance(obj, dict) and {'카운터', '공부', '복약지도서'} & set(obj) else None


def cards(obj, mode):
    """{base drug name: card} of 카운터 / 공부.약별 / 복약지도서.약별 ([N번] lines merged into the first)."""
    src = obj.get('카운터') if mode == '카운터' else (obj.get(mode) or {}).get('약별')
    out = {}
    for nm, v in (src or {}).items() if isinstance(src, dict) else []:
        if isinstance(v, dict):
            out.setdefault(base(nm), v)
    return out


def flat_src(src):
    if isinstance(src, list):
        return [y for x in src for y in flat_src(x)]
    return [src] if isinstance(src, str) else []


def items(obj):
    """Every item of an answer: {'mode', 'drug', 'field', 'text', 'ids' (E#), 'rx' (출처 has 처방전)}."""
    out = []

    def add(mode, drug, field, val, src):
        if isinstance(val, list):
            ok = isinstance(src, list) and len(src) == len(val) and all(isinstance(s, list) for s in src)
            for v, s in zip(val, src if ok else [src] * len(val)):
                add(mode, drug, field, v, s)
            return
        if not isinstance(val, str) or not val.strip():
            return
        f = flat_src(src)
        out.append({'mode': mode, 'drug': drug, 'field': field, 'text': val.strip(),
                    'ids': [x for x in f if re.fullmatch(r'E\d+', x)], 'rx': '처방전' in f})

    for nm, c in cards(obj, '카운터').items():
        for k in REQUIRED['카운터'][1]:
            add('카운터', nm, k, c.get(k), c.get(k + '_출처'))
    for mode in ('공부', '복약지도서'):
        top = obj.get(mode) if isinstance(obj.get(mode), dict) else {}
        for k in REQUIRED[mode][0]:
            if k != '약별':
                add(mode, '', k, top.get(k), top.get(k + '_출처'))
        for nm, c in cards(obj, mode).items():
            for k in REQUIRED[mode][1]:
                add(mode, nm, k, c.get(k), c.get(k + '_출처'))
    return out


def judged_text(it):
    """What the faithfulness judge sees: without the '/ 조치:' recommendation."""
    return ACTION.split(it['text'])[0].strip() if it['field'] in ('약사 확인', '핵심 이슈') else it['text']


# ------------------------------------------------------------------ ① 위험 판정
def out_issues(obj):
    res = []
    for s in (obj.get('공부') or {}).get('핵심 이슈') or [] if isinstance(obj.get('공부'), dict) else []:
        m = TAG.match(s or '') if isinstance(s, str) else None
        if m:
            kind, _, grade = m.group(1).partition('·')
            res.append({'구분': kind.strip(), '등급': grade.strip(), 'text': s})
    return res


def check_drugs(obj, kind):
    """Cards whose 약사 확인 names this issue kind (the 'same drug, N lines' note is not an issue)."""
    return {nm for nm, c in cards(obj, '카운터').items()
            if any(isinstance(s, str) and not DUP.search(s) and sr.kind_match(kind, s) for s in c.get('약사 확인') or [])}


def score_issues(rec, obj):
    gold = rec['정답'].get('위험이슈') or []
    outs = out_issues(obj)
    used, res = set(), []
    for g in gold:
        gdrugs = {base(d) for d in g['약']}
        cand = [k for k, o in enumerate(outs) if norm(o['구분']) == norm(g['구분']) and norm(o['등급']) == norm(g['등급'])]
        cand = cand or [k for k, o in enumerate(outs) if norm(o['구분']) == norm(g['구분'])]
        cand = cand or [k for k, o in enumerate(outs) if sr.kind_match(g['구분'], o['구분'])]
        k = min(cand, key=lambda x: (x in used, x)) if cand else None  # an unpaired line first
        if k is not None:
            used.add(k)
        o = outs[k] if k is not None else None
        pdrugs = check_drugs(obj, g['구분'])
        f = {'구분': bool(o) and norm(o['구분']) == norm(g['구분']),
             '등급': bool(o) and norm(o['등급']) == norm(g['등급']),
             '약': bool(o) and gdrugs <= pdrugs}
        acts = ACTION.split(o['text'], 1)[1] if o and ACTION.search(o['text']) else ''
        for nm in gdrugs & pdrugs:
            for s in cards(obj, '카운터')[nm].get('약사 확인') or []:
                if isinstance(s, str) and ACTION.search(s) and sr.kind_match(g['구분'], s):
                    acts += ' ' + ACTION.split(s, 1)[1]
        keys = sr.action_keys(g)
        f['조치'] = (bool(acts.strip()) and bool(keys) and all(re.search(p, acts) for p in keys)) if f['구분'] else None
        res.append(f)
    return res, len(outs), len(used)


# ------------------------------------------------------------------ ② rules for items without E#
def prescription_numbers(text):
    """Disease codes and numbers of a 처방 구성 line."""
    return sorted(re.findall(r'[A-Z]\d{2,4}', text)) + sorted(re.findall(r'\d+(?=\s*(?:종|일|가지))', text))


def rule_items(obj, gold):
    """Verdicts for the items without E#: {kind: [True/False, ...]} and the unsupported count."""
    gi = {}
    for it in items(gold):
        gi.setdefault((it['mode'], it['drug'], it['field']), []).append(it)
    v = {k: [] for k in ('복용법 · 카운터', '복용법 · 복약지도서', '복용법 시각 지어냄', '처방 구성', '같은 약 여러 줄', '자료 없음')}
    unsupported = 0
    for it in items(obj):
        if it['ids'] or it['field'] in ('복용법', '팝업 한 줄'):
            continue  # cited (→ judge), 복용법 and 팝업 are checked below for every gold card
        golds = gi.get((it['mode'], it['drug'], it['field']), [])
        if it['field'] == '처방 구성':
            v['처방 구성'].append(any(prescription_numbers(it['text']) == prescription_numbers(g['text']) for g in golds))
        elif DUP.search(it['text']):
            m = DUP.search(it['text'])
            v['같은 약 여러 줄'].append(any((d := DUP.search(g['text'])) and d.group(1) == m.group(1)
                                       and norm(d.group(2)) == norm(m.group(2)) for g in golds))
        elif NO_DATA.search(it['text']):
            v['자료 없음'].append(bool(golds) and all(not g['ids'] and NO_DATA.search(g['text']) for g in golds))
        else:
            unsupported += 1
    gc, oc = cards(gold, '카운터'), cards(obj, '카운터')
    gl, ol = cards(gold, '복약지도서'), cards(obj, '복약지도서')
    for nm, c in gc.items():
        out = oc.get(nm, {}).get('복용법')
        v['복용법 · 카운터'].append(norm(out) == norm(c['복용법']))
        v['복용법 시각 지어냄'].append(bool(set(TIME_WORD.findall(str(out or ''))) - set(TIME_WORD.findall(c['복용법']))))
    for nm, c in gl.items():
        v['복용법 · 복약지도서'].append(norm(ol.get(nm, {}).get('복용법')) == norm(c['복용법']))
    # 팝업
    rule, grounded = [], []
    for nm, c in gc.items():
        o = oc.get(nm, {})
        p = o.get('팝업 한 줄') if isinstance(o.get('팝업 한 줄'), str) else ''
        orb = [x for x in o.get('빨간박스') or [] if isinstance(x, str)]
        rule.append((not p and not orb) or (bool(p) and bool(orb) and len(p) <= 45 and len(p.split(' · ')) <= 3))
        if p and orb:
            grounded.append(not keyword_misses(p, ' '.join(orb)))
    v['팝업 규칙'] = rule
    v['팝업 근거성'] = grounded
    return v, unsupported


# ------------------------------------------------------------------ per record
def score_record(rec, gold, obj, ctx, args, cache):
    d = {'id': rec['id'], 'parsed': obj is not None}
    obj = obj or {}

    # ④
    req = {}
    for mode, (top, per) in REQUIRED.items():
        have = need = 0
        topobj = obj.get(mode) if isinstance(obj.get(mode), dict) else {}
        need += len(top)
        have += sum(k in topobj for k in top)
        oc = cards(obj, mode)
        for nm in cards(gold, mode):
            need += 1 + len(per)
            if nm in oc:
                have += 1 + sum(k in oc[nm] for k in per)
        req[mode] = (have, need)
    d['req'] = req
    same_cards = all(set(cards(obj, m)) == set(cards(gold, m)) for m in REQUIRED)
    d['format_ok'] = d['parsed'] and same_cards and all(h == n for h, n in req.values())

    # ①
    d['issues'], d['n_out_issues'], d['n_paired'] = score_issues(rec, obj)
    d['has_gold_issue'] = bool(rec['정답'].get('위험이슈'))

    # ②
    its = items(obj)
    cited = [i for it in its for i in it['ids']]
    d['n_items'] = len(its)
    d['cited'], d['bad_cite'] = len(cited), sum(i not in ctx for i in cited)
    d['rules'], d['unsupported'] = rule_items(obj, gold)
    if args.faith == 'llm':
        claims = [{'claim': judged_text(it), 'ids': sorted(set(it['ids']), key=lambda x: int(x[1:])),
                   'where': f"{it['mode']} {it['drug']} {it['field']}".strip()}
                  for it in its if it['ids'] and it['field'] != '복용법' and judged_text(it)]
        d['faith'] = [x for x in sr.faith_llm(rec['id'], claims, ctx, rec, args, cache) if x is not None]
        d['n_claims'] = len(claims)

    # ③ per field: gold sentences that the output has in the same card and field
    d['recall'] = field_recall(gold, obj)
    return d


def matches(g, outs):
    """A gold sentence is covered by an output sentence that equals it (spaces ignored) or has char-bigram F1 >= 0.8."""
    ng = norm(g)
    return any(norm(o) == ng or bigram_f1(g, o) >= RECALL_F1 for o in outs)


def bigram_f1(a, b):
    A, B = sr.bigrams(a), sr.bigrams(b)
    if not A or not B:
        return float(A == B)
    i = len(A & B)
    return 2 * i / (len(A) + len(B))


def as_list(v):
    return [x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and x.strip()]


def field_recall(gold, obj):
    """{(mode, field): (covered, gold sentences)} over RECALL_FIELDS."""
    out = {}
    for mode, (top_fields, drug_fields) in RECALL_FIELDS.items():
        gtop = gold.get(mode) if isinstance(gold.get(mode), dict) else {}
        otop = obj.get(mode) if isinstance(obj.get(mode), dict) else {}
        for k in top_fields:
            gs, os_ = as_list(gtop.get(k)), as_list(otop.get(k))
            out[(mode, k)] = (sum(matches(g, os_) for g in gs), len(gs))
        gcs, ocs = cards(gold, mode), cards(obj, mode)
        for k in drug_fields:
            hit = n = 0
            for nm, c in gcs.items():
                gs, os_ = as_list(c.get(k)), as_list(ocs.get(nm, {}).get(k))
                hit += sum(matches(g, os_) for g in gs)
                n += len(gs)
            out[(mode, k)] = (hit, n)
    return out


# ------------------------------------------------------------------ aggregate
def pct(a, b):
    return round(100 * a / b, 1) if b else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summarize(ds, n_total):
    S = {'n 채점': len(ds), 'n 예측 없음': n_total - len(ds)}
    # ①
    gi = [f for d in ds for f in d['issues']]
    pos = [d for d in ds if d['has_gold_issue']]
    S['① 위험 탐지율'] = mean([sum(f['약'] + f['구분'] + f['등급'] for f in d['issues']) / (3 * len(d['issues'])) for d in pos])
    for k in ('약', '구분', '등급'):
        S[f'① 정확도 · {k}'] = pct(sum(f[k] for f in gi), len(gi))
    S['① 전 항목 일치'] = pct(sum(f['약'] and f['구분'] and f['등급'] for f in gi), len(gi))
    S['① 정밀도'] = pct(sum(d['n_paired'] for d in ds), sum(d['n_out_issues'] for d in ds))
    neg = [d for d in ds if not d['has_gold_issue']]
    S['① 오경보율 ↓'] = pct(sum(d['n_out_issues'] > 0 for d in neg), len(neg))
    acts = [f['조치'] for f in gi if f['조치'] is not None]
    S['① 약사 조치 일치'] = pct(sum(acts), len(acts))
    # ②
    if any('faith' in d for d in ds):
        per = []
        for d in ds:
            st = [x for line in d.get('faith', []) for x in line]
            if st:
                per.append(sum(st) / len(st))
        S['② 근거 충실도 (LLM)'] = mean(per)
    S['② 근거 번호 오류율 ↓'] = pct(sum(d['bad_cite'] for d in ds), sum(d['cited'] for d in ds))
    for k in ('복용법 · 카운터', '복용법 · 복약지도서', '처방 구성', '같은 약 여러 줄', '자료 없음', '팝업 규칙', '팝업 근거성'):
        xs = [x for d in ds for x in d['rules'][k]]
        S[f'② 규칙 · {k}'] = pct(sum(xs), len(xs))
    xs = [x for d in ds for x in d['rules']['복용법 시각 지어냄']]
    S['② 규칙 · 복용법 시각 지어냄 ↓'] = pct(sum(xs), len(xs))
    S['② 근거 없는 주장 ↓'] = pct(sum(d['unsupported'] for d in ds), sum(d['n_items'] for d in ds))
    # ③ per field, the mean of each mode's fields, and the mean of the modes as 전체
    mode_means = []
    for mode, (top_fields, drug_fields) in RECALL_FIELDS.items():
        vals = []
        for k in top_fields + drug_fields:
            v = pct(sum(d['recall'][(mode, k)][0] for d in ds), sum(d['recall'][(mode, k)][1] for d in ds))
            S[f'③ 재현율 · {mode} · {k}'] = v
            if v is not None:
                vals.append(v)
        mode_means.append(round(sum(vals) / len(vals), 1) if vals else None)
        S[f'③ 재현율 · {mode} 평균'] = mode_means[-1]
    S['③ 재현율 · 전체(모드 평균)'] = round(sum(mode_means) / len(mode_means), 1) if None not in mode_means else None
    # ④ per mode, and the mean of the three as 전체
    modes = []
    for mode in REQUIRED:
        modes.append(pct(sum(d['req'][mode][0] for d in ds), sum(d['req'][mode][1] for d in ds)))
        S[f'④ 필수 항목 포함률 · {mode}'] = modes[-1]
    S['④ 필수 항목 포함률 · 전체(모드 평균)'] = round(sum(modes) / len(modes), 1) if None not in modes else None
    S['(참고) JSON 파싱 실패'] = sum(not d['parsed'] for d in ds)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--pred', required=True, help="jsonl of {id, output}, or 'gold'")
    ap.add_argument('--out')
    ap.add_argument('--details')
    ap.add_argument('--faith', default='off', choices=['off', 'llm'])
    ap.add_argument('--judge-url')
    ap.add_argument('--judge-model')
    ap.add_argument('--judge-batch', type=int, default=8)
    ap.add_argument('--verify-batch', type=int, default=10)
    ap.add_argument('--judge-cache', default=str(HERE / 'judge_cache.jsonl'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='score only the first N records (quick checks)')
    ap.add_argument('--faith-limit', type=int, default=0,
                    help='judge ② 근거 충실도 on the first N records only (same records for every model); 0 = all')
    args = ap.parse_args()

    gold_paths = {Path(p).stem: p for p in glob.glob(f'{args.data}/output_items/{args.split}/*.json')}
    recs = {}
    for p in glob.glob(f'{args.data}/data/**/*.jsonl', recursive=True):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r['id'] in gold_paths:
                recs[r['id']] = r
    if args.pred == 'gold':
        preds = {rid: open(p, encoding='utf-8').read() for rid, p in gold_paths.items()}
    else:
        preds = {}
        for line in open(args.pred, encoding='utf-8'):
            x = json.loads(line)
            preds[x['id']] = x.get('output') or x.get('response') or x.get('text') or ''
    ids = sorted(i for i in gold_paths if i in preds)
    if args.limit:
        ids = ids[:args.limit]
        gold_paths = {i: gold_paths[i] for i in ids}

    cache = {}
    if args.faith == 'llm' and os.path.exists(args.judge_cache):
        for line in open(args.judge_cache, encoding='utf-8'):
            k, v = json.loads(line)
            cache[k] = v
    n_cache = len(cache)
    input_md = f'{args.data}/input_md/{args.split}'

    faith_ids = set(ids[:args.faith_limit]) if args.faith_limit else set(ids)

    def one(rid):
        gold = json.load(open(gold_paths[rid], encoding='utf-8'))
        ctx = sr.load_context(recs[rid], input_md)
        a = args if rid in faith_ids else argparse.Namespace(**{**vars(args), 'faith': 'off'})
        return score_record(recs[rid], gold, parse_output(preds[rid]), ctx, a, cache)

    with ThreadPoolExecutor(args.workers if args.faith == 'llm' else 1) as ex:
        ds = list(ex.map(one, ids))
    if args.faith == 'llm':
        with open(args.judge_cache, 'a', encoding='utf-8') as f:
            for k in list(cache)[n_cache:]:
                f.write(json.dumps([k, cache[k]], ensure_ascii=False) + '\n')

    S = summarize(ds, len(gold_paths))
    w = max(len(k) for k in S)
    for k, v in S.items():
        print(f'{k:<{w}}  {v}')
    if args.out:
        json.dump(S, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if args.details:
        with open(args.details, 'w', encoding='utf-8') as f:
            for d in ds:
                d = dict(d, recall={f'{m} · {k}': v for (m, k), v in d['recall'].items()})
                f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
