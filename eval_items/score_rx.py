"""Score model outputs of rx_counsel (Markdown document, sections 0 → ③) against each record's 정답.

python3 -P score_rx.py --gold 'DATA/test/part_*.jsonl' --pred PRED [--ids selection.json:test] [--out result.json]
                       [--details per_record.jsonl] [--faith lexical|llm|off] [--input-md DIR]
                       [--judge-url http://host:port/v1 --judge-model NAME] [--issue-source any|s0|study|counter]

PRED  a folder of <id>.md files, or a jsonl with {"id", "output"} per line ("response" / "pred" / "text" also read).
      <think>…</think> blocks are dropped. Records without a prediction are reported as missing, not scored.
--ids selection.json:test (key of a {split: [ids]} json), a json list, or a text file with one id per line.

Metrics (발표 3-2 '출력 결과 평가 방식', with the rules agreed for the v4 real-prescription set)
 ① 위험 탐지율 ↑   issue level. A 정답 issue is detected when the output reports an issue on the same drug (product or ingredient
                    name), of the same kind (구분) and the same grade class (금기 vs 주의), in 0 요약, ① 공부 모드 and ② 카운터
                    alike (--issue-source all, 지시문 조건 1). Free-form outputs without issue markers are read as a whole.
 ① 오경보율 ↓      record level. Records without 정답 issues where the output reports at least one issue.
 ② 근거 충실도 ↑   RAGAS Faithfulness = supported statements / all statements, over the cited lines ([E#]) of sections 0–③:
                    the judge (1) breaks each line into atomic statements, (2) answers 1/0 per statement given the passages the
                    line cites plus the prescription. Includes the pharmacist judgment (사실), the efficacy evidence row and the
                    plain efficacy one-liner / leaflet use sentence. Not scored here: pharmacist actions (→ ③), risk judgments
                    such as '노인주의 1건' or '비해당' (→ ①), dosing times the prescription does not give (→ ③).
                    lexical: one verdict per line, character-bigram recall (fast proxy only).
                    (the reported numbers used A.X-4.0 72B served by vLLM, e.g.
                    vllm serve skt/A.X-4.0 --tensor-parallel-size 4 --served-model-name ax4-72b,
                    then --faith llm --judge-url http://localhost:8000/v1 --judge-model ax4-72b).
 ③ 종합 ↑          mean of
                    복약 정보 정확도 (drug level): dose (1회 양, 1일 횟수, 일수) equals the prescription, and the ③ timetable equals
                      정답 복용시간표; when the prescription gives no dosing time, the output must not assign one (timetable cells
                      '제공된 정보 없음', no '아침에 …' lines, no time in the card's 복용법 한 줄);
                    처방 정보 정확도 (record level): disease-code names, 처방상태, 의료기관·약국, 임신·수유, 알레르기, 건강기능식품
                      are copied from the input, or '제공된 정보 없음' when the input lacks them (nothing invented);
                    약사 조치 일치율 (detected issues): the issue's block holds every key element of the action fixed for its
                      kind (e.g. 노인주의: 필요성·용량 + 확인 + 이상 증상 안내), wording may differ.
 ④ 세부 항목 포함률 / 양식 통과율 ↑ required items (sections, per-drug 10 items + 포인트, cards, leaflet parts, 필수·금지 문구);
                    포함률 = items present / items required, 통과율 = records with every item present.

Issues are read from the structured markers the training outputs use (0 '- I1: …', ① '**[I1] 구분 — 등급: …**',
② '**[약사 확인] I1 — 구분**'). An output with neither markers nor a 'no issue' statement (e.g. a base model) is read
free-form: a line naming a prescribed drug together with an issue keyword, minus negated lines (미등재, 없음 …).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

SLOTS = ["아침", "점심", "저녁", "자기 전"]
SLOT_RX = [("아침", r"아침"), ("점심", r"점심"), ("저녁", r"저녁"), ("자기 전", r"자기\s*전|취침|잠자리")]
STOP_TEXT = "약사와 상의 없이 복용을 중단하지 마세요"
TOP_MARK = "약사 확인 후 출력"
TEN = [("적응증", r"적응증|효능"), ("용법용량", r"용법|용량"), ("금기", r"금기"), ("경고", r"경고"), ("부작용", r"부작용|이상\s*반응"),
       ("상호작용", r"상호\s*작용"), ("약동학_작용기전", r"약동학|작용\s*기전"), ("보관", r"보관|저장"),
       ("특수인구집단", r"특수\s*인구|임부|고령자|소아"), ("성상", r"성상|모양")]
LEAFLET = [("약국정보", r"약국\s*정보|처방과\s*약국"), ("받으신약", r"받으신\s*약|받은\s*약|약\s*목록"), ("시간표", r"시간표"),
           ("꼭지켜", r"꼭\s*지켜|지켜\s*주세요|주의\s*사항"), ("연락증상", r"연락|증상"), ("보관생활", r"보관|생활"),
           ("약사문의", r"약사에게|물어보|문의")]
SECTION_RX = [("s0", r"입력\s*처방전|^0\s*[.)]"), ("study", r"공부\s*모드"), ("counter", r"카운터"),
              ("patient", r"환자용|복약\s*지도서"), ("check", r"일관성|^4\s*[.)]"), ("sources", r"근거\s*출처|^5\s*[.)]")]
# issue kinds: 정답 구분 → patterns that also accept free wording
KIND_RX = {"병용금기": r"병용\s*금기", "노인주의": r"노인\s*주의|고령자?\s*주의",
           "용량주의": r"용량\s*주의|최대\s*(투여\s*)?(용량|기준량)\S*\s*초과|용량\s*초과|과량",
           "임부금기": r"임부\s*금기|임신\s*금기|임부\s*[12]\s*등급", "연령금기": r"연령\s*금기|소아\s*금기",
           "효능군중복": r"효능군?\s*중복|중복\s*(투여|처방)", "투여기간주의": r"투여\s*기간\s*(주의|초과)",
           "건강기능식품 상호작용": r"건강\s*기능\s*식품", "수유부주의": r"수유부?\s*(주의|금기)"}
ANY_KIND = re.compile("|".join(KIND_RX.values()) + r"|허가\s*사항\s*금기|금기|주의\s*약물|중복")
NEGATED = re.compile(r"미등재|해당\s*(사항\s*)?없|없(음|습니다|다|어요)|아님|아닙니다|않|미해당|이상\s*없|확인되지|점검\s*범위")
NO_ISSUE = re.compile(r"위험\s*이슈(는|가)?\s*(없|0\s*건)|위험\s*이슈\s*집합\s*=\s*`?\{\s*\}|확인된\s*위험\s*이슈\s*없")
CITE = re.compile(r"\[((?:\s*E\d+(?:\s*[~\-–]\s*E?\d+)?\s*[,·、]?)+)\]")
HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------ inputs
def load_ids(spec):
    if not spec:
        return None
    path, _, key = spec.partition(":")
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        return set(data[key] if key else data)
    return {l.strip().rsplit(".", 1)[0] if l.strip().endswith(".md") else l.strip() for l in open(path, encoding="utf-8") if l.strip()}


def load_gold(pattern, ids):
    gold = {}
    for path in sorted(glob.glob(pattern)):
        for rec in read_jsonl(path):
            if ids is None or rec["id"] in ids:
                gold[rec["id"]] = rec
    return gold


def load_pred(path, ids):
    out = {}
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".md") and f[:-3] in ids:
                out[f[:-3]] = open(os.path.join(path, f), encoding="utf-8").read()
    else:
        for p in read_jsonl(path):
            if p.get("id") in ids:
                out[p["id"]] = next(p[k] for k in ("output", "response", "pred", "text") if k in p)
    return {k: re.sub(r"<think>.*?</think>", "", v or "", flags=re.S).strip() for k, v in out.items()}


def load_context(rec, input_md_dir):
    """{E id: passage} as the model saw it: input_md lines (with their group title) when given, else the record's
    근거 + 검색문맥_추가근거 (with source type and file path, which name the DUR list / document)."""
    ctx = {}
    for e in rec.get("근거", []) + rec.get("검색문맥_추가근거", []):
        ctx[e["근거ID"]] = f"{e.get('근거유형', '')} {os.path.basename(e.get('파일', ''))} {e.get('항목', '')} {e.get('원문발췌', '')}"
    if input_md_dir:
        path = os.path.join(input_md_dir, rec["id"] + ".md")
        if os.path.exists(path):
            title = ""
            for line in open(path, encoding="utf-8"):
                if line.startswith("### ") or line.startswith("#### "):
                    title = line.lstrip("#").strip() if line.startswith("#### ") else line.lstrip("#").strip()
                m = re.match(r"- \*\*\[(E\d+)\]\*\*\s*(.+)$", line)
                if m:
                    ctx[m.group(1)] = f"{title} {m.group(2)}"
    return ctx


# ------------------------------------------------------------------ document structure
def split_sections(md):
    """{key: text} for the top-level sections 0 / ① / ② / ③ / 4 / 5 (whatever heading level the document uses)."""
    heads = []
    for m in HEAD.finditer(md):
        title = re.sub(r"^[①②③④⑤]\s*", "", re.sub(r"[*`]", "", m.group(2))).strip()
        key = next((k for k, rx in SECTION_RX if re.search(rx, title)), None)
        heads.append((m.start(), len(m.group(1)), key))
    keyed = [h for h in heads if h[2]]
    if not keyed:
        return {}
    level = min(h[1] for h in keyed)
    bounds, seen = [], set()
    for pos, lv, key in heads:
        if lv <= level:
            bounds.append((pos, key if key and key not in seen else None))
            if key:
                seen.add(key)
    out = {}
    for i, (pos, key) in enumerate(bounds):
        if key:
            out[key] = md[pos:bounds[i + 1][0] if i + 1 < len(bounds) else len(md)]
    return out


def subsections(text):
    """[(heading, body)] split at the headings one level below the section heading."""
    heads = list(HEAD.finditer(text))
    if len(heads) < 2:
        return []
    level = min(len(h.group(1)) for h in heads[1:])
    cut = [h for h in heads[1:] if len(h.group(1)) == level]
    return [(h.group(2), text[h.start():cut[i + 1].start() if i + 1 < len(cut) else len(text)]) for i, h in enumerate(cut)]


def tables(text):
    """Markdown tables as lists of rows (cells stripped); the separator row is dropped."""
    out, cur = [], []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    return out


class Drugs:
    """Prescribed drugs with the names an output may use: product name, product name up to 정/캡슐, ingredient."""

    def __init__(self, rx):
        self.rx = rx
        self.aliases = []
        for d in rx:
            names = {norm(d["제품명"]), norm(d.get("성분명"))}
            m = re.match(r"(.+?(?:정|캡슐|시럽|과립|산))(?=\d|[A-Za-z(]|$)", d["제품명"])
            if m and len(m.group(1)) >= 3:
                names.add(norm(m.group(1)))
            self.aliases.append(sorted((n for n in names if len(n) >= 2), key=len, reverse=True))
        self.by_name = {norm(d["제품명"]): i for i, d in enumerate(rx)}

    def find(self, text):
        t = norm(text)
        return [i for i, names in enumerate(self.aliases) if any(n in t for n in names)]

    def first(self, text):
        t = norm(text)
        # earliest match; at the same position the longer name wins ('소론도정[3번]' over '소론도정' when both are prescribed)
        # still tied: the drug whose full product name it is ('디아미크롱서방정' over '디아미크롱서방정60밀리그램')
        hits = [(t.find(n), -len(n), n != norm(self.rx[i]["제품명"]), i) for i, names in enumerate(self.aliases)
                for n in names if n in t]
        return min(hits)[3] if hits else None

    def index(self, product):
        i = self.by_name.get(norm(product))
        return i if i is not None else self.first(product)


# ------------------------------------------------------------------ ① issues
def grade_class(s):
    s = norm(s)
    if re.search(r"금기|금지|[12]등급", s):
        return "contra"
    if re.search(r"주의|권고|신중|상담", s):
        return "caution"
    return None


def kind_match(kind, text):
    t = norm(text)
    if norm(kind) in t:
        return True
    if kind in KIND_RX:
        return bool(re.search(KIND_RX[kind], text))
    m = re.match(r"허가사항\s*금기\((.+)\)", kind)
    if m:
        word = norm(m.group(1).split()[0])
        return "금기" in t and word in t
    return False


def grade_after(text, kind_text=None):
    """'등급: X' value; with kind_text, the one in the ' · '-separated segment that names that kind."""
    segs = re.split(r"\s·\s", text) if kind_text else [text]
    for seg in segs:
        if kind_text and norm(kind_text) not in norm(seg):
            continue
        m = re.search(r"등급\s*[:：]?\s*([^/,|*\[]+)", seg)
        if m:
            return m.group(1).strip()
    m = re.search(r"등급\s*[:：]?\s*([^/,|*\[]+)", text)
    return m.group(1).strip() if m else None


def issues_structured(sec, drugs):
    """Issues from the markers of the training format, per section: [{src, id, text, grade, drugs}]."""
    out = []
    for line in sec.get("s0", "").splitlines():
        m = re.match(r"^\s*[-*]\s*\**(I\d+)\**\s*[:：.]\s*(.+)$", line)
        if m:
            out.append({"src": "s0", "id": m.group(1), "text": m.group(2), "grade": grade_after(m.group(2)),
                        "drugs": set(drugs.find(m.group(2)))})
    for head, body in subsections(sec.get("study", "")) or [("", sec.get("study", ""))]:
        for sub_head, sub in [(h, b) for h, b in subsections(body)] or [(head, body)]:
            here = drugs.first(sub_head)
            lines = sub.splitlines()
            for j, line in enumerate(lines):
                m = re.search(r"\[(I\d+)\]\s*(.+)", line)
                if not m or "약사 확인" in line:
                    continue
                target = next((r for r in lines[j + 1:j + 14] if re.match(r"\|\s*대상\s*약물", r)), "")
                ds = set(drugs.find(target)) or ({here} if here is not None else set(drugs.find(line)))
                end = next((k for k in range(j + 1, len(lines)) if re.search(r"\[I\d+\]", lines[k]) or lines[k].startswith("#")), len(lines))
                out.append({"src": "study", "id": m.group(1), "text": m.group(2), "grade": grade_after(m.group(2)), "drugs": ds,
                            "block": "\n".join(lines[j:end])})
    for head, card in subsections(sec.get("counter", "")):
        here = drugs.first(head)
        bold = " · ".join(re.findall(r"^\*\*([^*]*등급[^*]*)\*\*\s*$", card, re.M))
        for m in re.finditer(r"\[약사\s*확인\]\**\s*(I\d+)\s*[—–\-:]?\s*([^*\n]+)", card):
            kind = m.group(2).strip()
            out.append({"src": "counter", "id": m.group(1), "text": kind, "grade": grade_after(bold, kind) if bold else None,
                        "drugs": {here} if here is not None else set(drugs.find(card[:200]))})
    return out


LABEL_FIELD = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*\**\s*(적응증|효능|용법|용량|용법\s*·?\s*용량|금기|경고|부작용|이상\s*반응|상호\s*작용|약동학|작용\s*기전|"
                         r"약동학\s*/?\s*작용\s*기전|보관|특수\s*인구\s*집단|성상)\s*\**\s*[:：]")


def issues_free(md, sec, drugs):
    """Free-form reading for outputs without markers: an issue keyword on a line that is not negated, attached to the drug
    named on that line, else on the nearest parent bullet, else in the current heading ('- **아미트리프틸린:**' / '  - 노인 주의: …').
    Label-summary fields of the drug sheet ('금기: …', '상호작용: …') are not issue reports and are skipped."""
    text = "\n".join(sec[k] for k in ("s0", "study", "counter") if k in sec) if sec else md
    out, stack, head = [], [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            head, stack = set(drugs.find(line)), []
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        here = set(drugs.find(line))
        if here:
            stack.append((indent, here))
        if LABEL_FIELD.match(line) or not ANY_KIND.search(line) or NEGATED.search(line):
            continue
        ds = here or (stack[-1][1] if stack else set()) or head
        if ds:
            out.append({"src": "free", "id": None, "text": line, "grade": None, "drugs": ds})
    return out


def issue_matches(g, p, drugs):
    """Same drug (product or ingredient), same kind, same grade class (금기 vs 주의)."""
    gd = {i for i in (drugs.index(n) for n in g["약"]) if i is not None}
    pc = grade_class(p["grade"]) if p["grade"] else grade_class(p["text"])
    return bool(gd & p["drugs"]) and kind_match(g["구분"], p["text"]) and pc == grade_class(g["등급"])


def score_issues(rec, md, sec, drugs, source):
    found = issues_structured(sec, drugs)
    no_issue_said = bool(NO_ISSUE.search(md))
    mode = "structured" if found or no_issue_said else "free"
    if mode == "free":
        found = issues_free(md, sec, drugs)
    elif source not in ("any", "all"):
        found = [p for p in found if p["src"] == source]
    gold = rec["정답"]["위험이슈"]
    hits = []
    for g in gold:
        if mode == "structured" and source == "all":  # 지시문 조건 1: the same issue in 0 요약, ① 공부 모드 and ② 카운터
            ok = all(any(issue_matches(g, p, drugs) for p in found if p["src"] == x) for x in ("s0", "study", "counter"))
        else:
            ok = any(issue_matches(g, p, drugs) for p in found)
        hits.append({"이슈ID": g["이슈ID"], "구분": g["구분"], "등급": g["등급"], "detected": ok})
    pred_ids = {(p["src"], p["id"] or p["text"][:40]) for p in found}
    return {"mode": mode, "gold": len(gold), "hits": hits, "pred_issue_count": len(pred_ids), "any_pred": bool(found)}


# ---- pharmacist action: rule-based (fixed per issue kind, render.py issue_point), checked by its key elements so that
# wording may differ: every element of the kind must appear in the issue's block of the study mode.
DOCTOR = r"처방의|의사|처방\s*의료기관"
ACTION_KEYS = {
    "병용금기": [DOCTOR, r"조회|문의|협의|확인", r"조제\s*(보류|하지\s*않|전)|투약\s*보류|중단|대체"],
    "연령금기": [DOCTOR, r"조회|문의|협의|확인", r"대체|변경|다른\s*약"],
    "임부금기": [DOCTOR, r"공유|알리|조회|확인", r"대체|변경|투약\s*여부|중단"],
    "노인주의": [r"필요성|용량", r"확인|검토|조회", r"이상\s*(증상|반응)|증상|알리|안내"],
    "용량주의": [DOCTOR, r"확인|조회|문의", r"용량|횟수|오기|처방\s*의도"],
    "투여기간주의": [DOCTOR, r"확인|조회|문의", r"기간|일수|장기"],
    "효능군중복": [DOCTOR, r"확인|조회|문의", r"의도|중복|함께|병용"],
    "허가사항 금기": [DOCTOR, r"조회|문의|협의|확인", r"대체|변경|중단"],
}


def action_keys(issue):
    kind = issue["구분"]
    if kind == "수유부주의":
        return ([DOCTOR, r"조회|확인|문의", r"수유\s*중단|대체"] if "금기" in issue["등급"]
                else [r"수유", r"확인|공유|알리", r"필요성|지속|여부"])
    if kind == "건강기능식품 상호작용":
        return ([r"섭취", r"중단|끊|피하"] if issue.get("위험도") == "높음" else [r"섭취|복용", r"주의|상담|줄이|중단"])
    if kind.startswith("허가사항 금기"):
        kind = "허가사항 금기"
    return ACTION_KEYS.get(kind)


def score_actions(rec, sec, md, hits, drugs):
    """For each 정답 issue that ① found: does its study-mode block hold every key element of its kind's action? [True/False]"""
    found = [p for p in issues_structured(sec, drugs) if p["src"] == "study"]
    out = []
    for g, h in zip(rec["정답"]["위험이슈"], hits):
        keys = action_keys(g)
        if not (h["detected"] and keys):
            continue
        blocks = [p["block"] for p in found if issue_matches(g, p, drugs)]
        text = "\n".join(blocks) if blocks else sec.get("study", md)  # free-form outputs: the whole study mode
        out.append(all(re.search(k, text) for k in keys))
    return out


# ------------------------------------------------------------------ ③ medication info
def qty(cell):
    c = cell.replace("½", "0.5")
    if re.search(r"반\s*(알|정|캡슐)", c):
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?|\d+/\d+)\s*(알|정|캡슐|개|포|T\b)", c)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)(?!\s*(분|시|mg|일|회|%))", c) if not re.fullmatch(r"[\s—\-–·x×X]*", c) else None
    if not m:
        return 0.0
    v = m.group(1)
    if "/" in v:
        a, b = v.split("/")
        return float(a) / float(b)
    return float(v)


def slot_of(cell):
    return next((s for s, rx in SLOT_RX if re.search(rx, cell)), None)


def timetable(text, drugs):
    """{drug index: {slot: qty}} from the table whose rows (or columns) are 아침/점심/저녁/자기 전."""
    best = {}
    for t in tables(text):
        head = t[0]
        row_slots = [slot_of(r[0]) for r in t[1:] if r]
        col_slots = [slot_of(c) for c in head]
        merged = any(len([x for x, rx in SLOT_RX if re.search(rx, r[0])]) >= 2 for r in t[1:] if r)  # '아침·점심·저녁·자기 전' row
        got = {}
        if sum(s is not None for s in row_slots) >= 2 or merged:  # rows = slots, columns = drugs
            for ci, h in enumerate(head[1:], 1):
                d = drugs.first(h)
                if d is not None:
                    got[d] = {s: 0.0 for s in SLOTS}
                    for r, s in zip(t[1:], row_slots):
                        if s and ci < len(r):
                            got[d][s] += qty(r[ci])
        elif sum(s is not None for s in col_slots) >= 2:  # rows = drugs, columns = slots
            for r in t[1:]:
                d = drugs.first(r[0]) if r else None
                if d is None:
                    d = drugs.first(" ".join(r[:2]))
                if d is not None:
                    got[d] = {s: 0.0 for s in SLOTS}
                    for ci, s in enumerate(col_slots):
                        if s and ci < len(r):
                            got[d][s] += qty(r[ci])
        if len(got) > len(best):
            best = got
    if not best:  # list form: '- **아침**: 메치론정 0.5정, 덱실란트디알캡슐 1캡슐'
        got = {}
        for line in text.splitlines():
            m = re.match(r"^\s*[-*]?\s*\**\s*(아침|점심|저녁|자기\s*전|취침\s*전)[^:：]*\**\s*[:：]\s*(.+)$", line)
            if not m:  # drug → slots: '- **알레리스정:** 아침, 저녁 식후 30분'
                m = re.match(r"^\s*[-*]\s*\**([^:：*]+?)\**\s*[:：]\**\s*(.+)$", line)
                d = drugs.first(m.group(1)) if m else None
                if d is None:
                    continue
                slots = [x for x, rx in SLOT_RX if re.search(rx, m.group(2))]
                if not slots:
                    continue
                n = re.findall(r"(\d+(?:\.\d+)?)\s*(?:알|정|캡슐|개|포)(?![가-힣])", m.group(2))
                per = float(n[-1]) if n else qty(drugs.rx[d]["1회_투약량"])  # no amount written: the prescribed dose
                got.setdefault(d, {x: 0.0 for x in SLOTS})
                for x in slots:
                    got[d][x] += per
                continue
            slot = slot_of(m.group(1))
            for part in re.split(r"[,，]|\s및\s", m.group(2)):
                d = drugs.first(part)
                if d is not None:  # the dose is the last 'N정/캡슐/알' (product names carry strengths: 낙센에스정500/20밀리그램)
                    n = re.findall(r"(\d+(?:\.\d+)?|반)\s*(?:알|정|캡슐|개|포)(?![가-힣])", part)
                    got.setdefault(d, {x: 0.0 for x in SLOTS})
                    got[d][slot] += (0.5 if n[-1] == "반" else float(n[-1])) if n else 1.0
        if got:
            for d in range(len(drugs.rx)):  # a drug never named in any slot line is listed with no doses
                got.setdefault(d, {x: 0.0 for x in SLOTS})
            best = got
    return best


def expected_timetable(rec, drugs):
    exp = {i: {s: 0.0 for s in SLOTS} for i in range(len(drugs.rx))}
    for slot, items in rec["정답"]["복용시간표"].items():
        for it in items:
            m = re.match(r"(.+?)\s+(\d+(?:\.\d+)?)\s*(정|캡슐|알|개|포|mL|ml)?$", it.strip())
            name, n = (m.group(1), float(m.group(2))) if m else (it, 1.0)
            i = drugs.index(name)
            if i is not None and slot in exp[i]:
                exp[i][slot] += n
    return exp


def dose_rows(text, drugs):
    """{drug index: (1회 양, 1일 횟수, 일수)} from a table with 1회 / 횟수 / 일수 columns."""
    out = {}
    for t in tables(text):
        head = [norm(h) for h in t[0]]
        col = lambda rx: next((i for i, h in enumerate(head) if re.search(rx, h)), None)
        c1, cn, cd = col(r"1회"), col(r"횟수|1일"), col(r"일수|기간")
        if c1 is None or cn is None:
            continue
        for r in t[1:]:
            d = drugs.first(" ".join(r[:3]))
            if d is None or d in out:
                continue
            get = lambda c: r[c] if c is not None and c < len(r) else ""
            nums = lambda s: re.findall(r"\d+(?:\.\d+)?", s)
            times = re.search(r"(\d+)\s*(?:회|번)", get(cn)) or re.search(r"\d+(?:\.\d+)?", get(cn))  # '1일 2회' → 2
            out[d] = (qty(get(c1)), float(times.group(1) if times.lastindex else times.group(0)) if times else None,
                      float(nums(get(cd))[0]) if nums(get(cd)) else None)
    return out


def dose_line(md, drug_idx, drugs):
    for line in md.splitlines():
        if drug_idx in drugs.find(line):
            a = re.search(r"1회\s*(\d+(?:\.\d+)?)\s*(정|캡슐|알)", line)
            b = re.search(r"(?:1일|하루)\s*(\d+)\s*(회|번)", line)
            c = re.search(r"(\d+)\s*일\s*(분|간|동안)", line)
            if a and b:  # 일수 is compared only when written
                return float(a.group(1)), float(b.group(1)), float(c.group(1)) if c else None
    return None


def invented_slots(sec, drugs, i):
    """A dosing time written for drug i in its leaflet entry ('- 아침에 1알 드세요') or card ('복용법 한 줄 | … 아침 식후')."""
    for key, rx in (("patient", r"^\s*[-*]\s*(아침|점심|저녁|자기\s*전|취침)[^\n]*(드세요|복용)"),
                    ("counter", r"^\|\s*\**복용법[^|]*\|[^|\n]*(아침|점심|저녁|자기\s*전|취침)")):
        for h, body in subsections(sec.get(key, "")):
            for _, sub in subsections(body) or [(h, body)]:
                # a note that states the time is not given ('(복용 시각 처방전 미기재 · 허가 용법: 취침 전)') is not an assignment
                plain = re.sub(r"\([^)]*(미기재|처방전에\s*없|정보\s*없)[^)]*\)", "", sub)
                if sub and drugs.first(sub.splitlines()[0]) == i and re.search(rx, plain, re.M):
                    return True
    return False


def score_med(rec, md, sec, drugs):
    exp = expected_timetable(rec, drugs)
    got = timetable(sec["patient"], drugs) if sec.get("patient") else timetable(md, drugs)
    rows = dose_rows(sec.get("s0", md), drugs)
    per = []
    for i, d in enumerate(drugs.rx):
        want = (qty(d["1회_투약량"]), float(d["1일_투여횟수"]), float(d["총_투약일수"]))
        have = rows.get(i) or dose_line(md, i, drugs)
        dose_ok = have is not None and have[0] == want[0] and have[1] == want[1] and (have[2] is None or have[2] == want[2])
        # no dosing time on the prescription (real prescriptions, v4): the output must not assign one ('제공된 정보 없음')
        untimed = bool(re.search(r"미기재|처방전에\s*없", str(d.get("용법", ""))))
        if untimed:
            tt_ok = not any(got.get(i, {}).get(x) for x in SLOTS) and not invented_slots(sec, drugs, i)
        else:
            tt_ok = i in got and all(abs(got[i][x] - exp[i][x]) < 1e-6 for x in SLOTS)
        per.append({"제품명": d["제품명"], "dose_ok": dose_ok, "timetable_ok": tt_ok, "ok": dose_ok and tt_ok})
    info, errs = score_info(rec, md, sec)
    return {"drugs": per, "record_ok": all(p["ok"] for p in per), "info": info, "info_errors": errs}


# ---- prescription information other than the dose: the output must copy it from the input, or say it is not given
NOT_GIVEN = re.compile(r"정보\s*없|자료\s*없|제공되지|미제공|미상|불명|확인되지|미확인|알\s*수\s*없|비공개|미기재|입력되지|처방\s*자료에\s*없")
STATUS_ASSERT = re.compile(r"신규|지속|변경|새로\s*처방된|계속\s*드시던")
INFO_KEYS = {"의료기관": r"^(의료기관|처방\s*병원)$", "조제약국": r"^(조제약국|약국)$", "임신·수유": r"^임신\s*[·/,]?\s*수유$|^임신$",
             "알레르기": r"^알레르기$", "건강기능식품": r"^(복용\s*중인\s*)?건강기능식품$", "질병코드": r"질병\s*코드|^질병$|상병"}


def info_rows(text):
    """(key, value) from 2-column table rows and 'key: value' bullets."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 2:
                out.append((re.sub(r"[*`]", "", c[0]).strip(), c[1]))
        else:
            m = re.match(r"^\s*[-*]\s*\**([^:：*]{1,20}?)\**\s*[:：]\s*\**\s*(.+)$", line)
            if m:
                out.append((m.group(1).strip(), m.group(2)))
    return out


def invented_disease_names(cell, pt):
    names = {c["코드"]: c.get("명칭") for c in pt.get("질병코드") or []}
    bad = []
    for code, name in names.items():
        for m in re.finditer(re.escape(code) + r"\s*[:(（]?\s*([^/|,;)\]]*)", cell):
            tail = re.split(r"\b[A-Z]\d{2,5}\b", m.group(1))[0].strip(" :()")
            if not re.search(r"[가-힣]", tail) or NOT_GIVEN.search(tail) or re.match(r"코드명|명칭", tail):
                continue
            if name and (norm(name)[:5] in norm(tail) or norm(tail)[:5] in norm(name)):
                continue
            bad.append(f"{code}:{tail[:20]}")
    return bad


def score_info(rec, md, sec):
    """{field: True/False} for the fields the output reports (missing fields are ④'s business), plus the errors."""
    rx, pt = rec["input"]["처방전"], rec["input"]["처방전"]["환자"]
    head = sec.get("s0", "") + "\n" + (subsections(sec.get("study", ""))[0][1] if subsections(sec.get("study", "")) else "")
    rows = info_rows(head if sec else md)
    res, errs = {}, []

    def first(field):
        return next((v for k, v in rows if re.search(INFO_KEYS[field], k)), None)

    for field, want in (("의료기관", rx.get("의료기관")), ("조제약국", rx.get("조제약국"))):
        v = first(field)
        if v is not None:
            ok = norm(re.sub(r"\(가상\)", "", want)) in norm(v) if want else bool(NOT_GIVEN.search(v))
            res[field] = ok
            errs += [] if ok else [f"{field}:{v[:25]}"]
    v = first("임신·수유")
    if v is not None:
        p, l = pt.get("임신"), pt.get("수유")
        if pt.get("성별") == "남":
            ok = bool(NOT_GIVEN.search(v) or re.search(r"해당\s*없|남성|아님", v))
        elif p is None and l is None:
            ok = bool(NOT_GIVEN.search(v))
        else:
            ok = bool(re.search(r"임신\s*중|임부|\d+\s*주", v)) == bool(p) and bool(re.search(r"수유\s*중", v)) == bool(l) \
                or (not p and not l and "아님" in v)
        res["임신·수유"] = ok
        errs += [] if ok else [f"임신·수유:{v[:25]}"]
    for field, items in (("알레르기", pt.get("알레르기") or []), ("건강기능식품", pt.get("복용중_건강기능식품") or [])):
        v = first(field)
        if v is None:
            continue
        if items:
            names = [norm(x if isinstance(x, str) else (x.get("이름") or x.get("제품명") or ""))[:4] for x in items]
            ok = all(n in norm(v) for n in names if n)
        elif pt.get("알레르기") is None or "산정특례코드" in pt:  # v4: not part of the source data → must say not given
            ok = bool(NOT_GIVEN.search(v))
        else:  # v3: an empty list was given → '없음/비어 있음' is fine, but no invented item
            ok = bool(NOT_GIVEN.search(v) or re.search(r"없|비어", v))
        res[field] = ok
        errs += [] if ok else [f"{field}:{v[:25]}"]
    cells = [v for k, v in rows if re.search(INFO_KEYS["질병코드"], k)]
    if cells:
        bad = sorted({b for c in cells for b in invented_disease_names(c, pt)})
        res["질병명"] = not bad
        errs += bad
    known = [d.get("처방상태") for d in rx["처방약"] if d.get("처방상태") and not NOT_GIVEN.search(str(d.get("처방상태")))]
    status = re.findall(r"^\|\s*\**처방\s*상태\**\s*\|\s*([^|]+)\|", md, re.M)
    if status and not known:  # status not given: every '처방 상태' cell must say so
        names = sorted({d["제품명"] for d in rx["처방약"]} | {re.sub(r"\[\d+번\]", "", d["제품명"]) for d in rx["처방약"]}, key=len, reverse=True)
        strip = lambda c: re.sub("|".join(map(re.escape, names)), "", c)  # product names such as '이소트릴지속정'
        bad = [c for c in status if STATUS_ASSERT.search(strip(c)) or not NOT_GIVEN.search(c)]
        res["처방상태"] = not bad
        errs += [f"처방상태:{bad[0].strip()[:25]}"] if bad else []
    return res, errs


# ------------------------------------------------------------------ ④ format
def score_format(rec, md, sec, drugs):
    items = {}
    names = [d["제품명"] for d in drugs.rx]
    s0, study, counter, patient = (sec.get(k, "") for k in ("s0", "study", "counter", "patient"))
    items["s0.섹션"] = bool(s0)
    items["s0.약목록"] = bool(s0) and all(i in drugs.find(s0) for i in range(len(names)))
    items["s0.공통분석"] = bool(re.search(r"공통\s*분석|위험\s*이슈", s0))
    items["study.섹션"] = bool(study)
    items["study.A_요약"] = bool(re.search(r"환자.{0,3}처방\s*요약|환자\s*정보", study))
    items["study.E_근거확장"] = bool(re.search(r"근거\s*확장", study))
    per_drug = {}
    for h, body in subsections(study):
        for _, sub in subsections(body) or [(h, body)]:
            d = drugs.first(sub.splitlines()[0]) if sub else None
            if d is not None and d not in per_drug:
                per_drug[d] = sub
    for i, n in enumerate(names):
        body = per_drug.get(i, "")
        items[f"study.C.{n}"] = bool(body)
        for key, rx in TEN:
            items[f"study.C.{n}.{key}"] = bool(re.search(rx, body))
        items[f"study.C.{n}.포인트"] = "포인트" in body
    cards = {}
    for h, body in subsections(counter):
        d = drugs.first(h)
        if d is not None and d not in cards:
            cards[d] = body
    items["counter.섹션"] = bool(counter)
    for i, n in enumerate(names):
        body = cards.get(i, "")
        items[f"counter.카드.{n}"] = bool(body)
        items[f"counter.카드.{n}.효능"] = "효능" in body
        items[f"counter.카드.{n}.복용법"] = bool(re.search(r"복용법|복용\s*방법", body))
        items[f"counter.카드.{n}.근거배지"] = bool(re.search(r"근거\s*배지|근거", body))
    items["counter.공통안내"] = bool(re.search(r"공통\s*안내", counter))
    items["patient.섹션"] = bool(patient)
    heads = " ".join(h for h, _ in subsections(patient))
    for key, rx in LEAFLET:
        items[f"patient.{key}"] = bool(re.search(rx, heads))
    items["patient.약목록"] = bool(patient) and all(i in drugs.find(patient) for i in range(len(names)))
    items["patient.중단금지문구"] = norm(STOP_TEXT) in norm(patient)
    has_issue = bool(rec["정답"]["위험이슈"])
    items["patient.상단표시"] = (norm(TOP_MARK) in norm(patient)) == has_issue
    leak = re.search(r"\[약사\s*확인\]|(?<![A-Za-z0-9])I\d+(?![0-9])", patient.replace(TOP_MARK, ""))
    items["patient.내부정보비노출"] = bool(patient) and not leak
    return items


# ------------------------------------------------------------------ ② faithfulness
# Lines/statements ② does not score for any output, because they are checked elsewhere:
#   pharmacist actions ('| 약사 조치 |', counter '조치:', '처방의에게 조회한다' …): fixed per issue kind → ③ 약사 조치 일치율;
#   risk judgments ('노인주의 1건', '비해당' …) → ①; dosing-time slots when the prescription gives none → ③.
# The efficacy evidence row and the plain efficacy one-liner / leaflet use sentence (cited) ARE scored by ②.
NOT_SCORED_LINE = re.compile(r"^\s*[-*]\s*\**조치\**\s*[:：]|^\s*\|\s*\**약사\s*조치\**\s*\|")
ACTION_STMT = re.compile(r"처방의|조회|협의|조제하지|투약\s*보류|(확인|안내|상담|공유)(한다|해야|하고|하도록|하며|할)|먹는\s*약을\s*(꼭\s*)?알려")
SLOT_STMT = re.compile(r"아침|점심|저녁|자기\s*전|취침|잠자리")
# risk judgments (issue counts/summaries, '비해당', '같은 효능군 약 없음') are scored by ① (위험 탐지율·오경보율), not by ②
JUDGMENT_STMT = re.compile(r"비해당|해당하지\s*않|해당한다|같은\s*효능군|그룹\s*약[이은]?\s*없|위험\s*이슈|\d+\s*건|등급[은이]|위험도[는가]|"
                           r"판정\s*보류|확인(이)?\s*필요|금기(이다|에 해당)")


def cited_ids(inner):
    ids = []
    for part in re.split(r"[,·、]", inner):
        m = re.match(r"\s*E(\d+)(?:\s*[~\-–]\s*E?(\d+))?", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            ids += [f"E{k}" for k in range(a, min(b, a + 200) + 1)]
    return ids


def claims_of(sec, md):
    text = "\n".join(sec[k] for k in ("s0", "study", "counter", "patient") if k in sec) if sec else md
    out = []
    lines = text.splitlines()
    where = ""  # nearest heading: which drug section / card / leaflet entry the line belongs to
    for j, line in enumerate(lines):
        h = HEAD.match(line)
        if h:
            where = re.sub(r"[*`]", "", h.group(2)).strip()
        if not CITE.search(line) or re.match(r"\s*근거\s*배지", line) or NOT_SCORED_LINE.search(CITE.sub("", line)):
            continue
        ids = [i for m in CITE.finditer(line) for i in cited_ids(m.group(1))]
        if re.fullmatch(r"\s*\*\*[^*]+\*\*\s*", CITE.sub("", line)):
            continue  # a cited heading ('**어떻게 먹나요?** [E40]'): the uncited lines under it come from the prescription (③), not ②
        claim = re.sub(r"[*`>#|]", " ", CITE.sub(" ", line))
        claim = re.sub(r"\s+", " ", claim).strip(" -·:")
        if len(norm(claim)) >= 4 and ids:
            out.append({"claim": claim, "ids": sorted(set(ids), key=lambda x: int(x[1:])), "where": where})
    return out


def bigrams(s):
    s = re.sub(r"[^0-9a-z가-힣]", "", norm(s))
    return {s[i:i + 2] for i in range(len(s) - 1)}


def rx_text(rec):
    return json.dumps(rec["input"]["처방전"], ensure_ascii=False)


def faith_lexical(claims, ctx, rec, th):
    """One verdict per cited line: [[0|1], ...]."""
    base = bigrams(rx_text(rec))
    out = []
    for c in claims:
        if any(i not in ctx for i in c["ids"]):
            out.append([0])
            continue
        cb = bigrams(c["claim"])
        sup = base | set().union(*(bigrams(ctx[i]) for i in c["ids"]))
        out.append([int(bool(cb) and len(cb & sup) / len(cb) >= th)])
    return out


# RAGAS Faithfulness, two steps: (1) break each cited line into atomic statements, (2) judge every statement against the
# passages that line cites (+ the prescription). Score = supported statements / all statements.
DECOMPOSE_PROMPT = """아래는 약국 복약지도 문서에서 근거 번호를 인용한 문장들입니다.
각 문장을, 문장 하나만 보고 참·거짓을 판단할 수 있는 단순한 주장들로 나누세요.
- 대명사나 생략된 대상('이 약', '성상', '하루 2회' 등)은 그 문장의 [위치]에 나온 약 이름으로 채워 넣습니다.
  [위치]에도 약 이름이 없으면 대상을 지어내지 말고 생략된 채로 둡니다. 다른 문장의 약 이름을 가져오지 않습니다.
- 표의 행은 '항목: 내용'을 하나의 문장으로 보고 나눕니다.
- 문장에 없는 내용을 더하지 않습니다. 인용·발췌는 문장에 나온 범위 안에서만 나누고 내용을 바꾸지 않습니다.
- 사실을 담지 않은 것(제목·항목명만 있는 것, '다음과 같다' 같은 문구)은 주장으로 만들지 않습니다. 모두 그렇다면 빈 배열로 둡니다.
출력은 JSON 객체 하나입니다. 키는 문장 번호, 값은 주장 문자열 배열입니다. 예: {{"1": ["...", "..."], "2": ["..."]}}

{items}"""

VERIFY_PROMPT = """당신은 약학 문서의 근거 충실도(faithfulness)를 판정하는 심사자입니다.
각 주장이 그 주장에 딸린 [근거]와 [처방전]에서 추론될 수 있는지 판정하세요.
- 근거의 내용을 쉬운 말로 바꾸거나 요약한 것은 뜻이 같으면 추론 가능(1)입니다(예: '호흡억제' → '숨쉬기가 약하고 느려질 때',
  '백색의 원형 정제' → '흰색의 동그란 알약').
- '처방 내 같은 효능군 약 없음', '조합이 병용금기 목록에 없음'처럼 처방 약 목록과 근거를 대조해야 하는 주장은
  [처방전]의 처방약 목록과 근거를 맞춰 보고 판정합니다.
- 근거가 '…'로 끝나는 잘린 발췌이면 보이는 부분까지만 대조합니다.
- 처방 내용(약 이름, 1회 양, 1일 횟수, 투약 일수, 환자 나이·성별·질병코드 등)에 관한 주장은 [처방전]과 대조합니다.
  [처방전]에 그대로 있으면 인용 근거에 없어도 추론 가능(1)입니다.
- 환자용 안내('~할 때 바로 연락하세요', '~하지 마세요', '~에 두세요')는 근거에 같은 이상반응·주의·보관 내용이 있으면
  추론 가능(1)입니다. 이상반응을 일반인이 알아볼 증상으로 바꾼 것도 같습니다(예: '위장관 출혈' → '대변이 검게 나오거나
  피를 토할 때', '황달' → '눈 흰자나 피부가 노랗게 변할 때', '혈관부종·아나필락시스' → '얼굴·입술·혀가 붓거나 숨쉬기 힘들 때').
- 주장이 근거 발췌(표를 풀어 쓴 것 포함)의 내용을 그대로 옮긴 것이면 추론 가능(1)입니다.
- 근거에 없는 사실·수치·대상이 들어 있거나 근거와 어긋나면 추론 불가(0)입니다.
- 근거에 DUR 목록·허가사항 등재 사실이 있으면, 그 목록에 따른 이슈 종류와 문서의 표기 규칙인 등급(금기/주의)·위험도(높음/중간)
  표기는 추론 가능(1)으로 봅니다.
출력은 JSON 객체 하나입니다. 키는 주장 번호, 값은 1 또는 0이며 모든 주장 번호를 빠짐없이 씁니다. 예: {{"1": 1, "2": 0, "3": 1}}

[처방전]
{rx}

{items}"""


def judge_call(a, prompt, timeout=900):
    body = json.dumps({"model": a.judge_model, "temperature": 0, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    url = random.choice(a.judge_url.split(","))  # several judge servers: requests are spread over them
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json",
                                 "Authorization": "Bearer " + os.environ.get("JUDGE_API_KEY", "EMPTY")})
    text = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def ask(a, rid, prompt, parse, n):
    """Call the judge up to 3 times until parse(text) gives n results; None when it never does."""
    for _ in range(3):
        try:
            res = parse(judge_call(a, prompt), n)
        except Exception as e:  # noqa: BLE001
            print(f"judge error {rid}: {e}", file=sys.stderr)
            res = None
        if res is not None:
            return res
    print(f"judge gave no usable answer for {rid} ({n} items) after 3 tries", file=sys.stderr)
    return None


def json_objects(text):
    """All JSON objects in the text merged (judges sometimes answer '{"1": …}\n{"2": …}'); None when there is none."""
    obj, dec, pos, found = {}, json.JSONDecoder(), 0, False
    while (pos := text.find("{", pos)) >= 0:
        try:
            part, end = dec.raw_decode(text, pos)
        except ValueError:
            pos += 1
            continue
        if isinstance(part, dict):  # keys may come as "1" or "주장 1" / "문장 1"
            found = True
            obj.update({(m.group(0) if (m := re.search(r"\d+", str(k))) else str(k)): v for k, v in part.items()})
        pos = end
    return obj if found else None


def parse_statements(text, n):
    obj = json_objects(text)
    if obj is None:
        return None
    if not obj:  # '{}': none of the lines states a fact (e.g. bare source names with a citation)
        return [[] for _ in range(n)]
    out = [obj.get(str(k)) for k in range(1, n + 1)]
    if any(not isinstance(x, list) for x in out):
        return None
    return [[str(s).strip() for s in x if str(s).strip()] for x in out]


def parse_verdicts(text, n):
    """[0|1|None] * n from {"1": 1, ...}; items the judge skipped stay None (judged on a later run). None if none usable."""
    obj = json_objects(text) or {}
    out = []
    for k in range(1, n + 1):
        v = obj.get(str(k))
        out.append(int(v) if v in (0, 1, "0", "1", True, False) else None)
    return out if any(v is not None for v in out) else None


def key_of(*parts):
    return hashlib.sha1("\t".join(parts).encode()).hexdigest()


# prompt versions are part of the cache keys, so a changed prompt never reuses old answers
D_VER = hashlib.sha1(DECOMPOSE_PROMPT.encode()).hexdigest()[:8]
V_VER = hashlib.sha1(VERIFY_PROMPT.encode()).hexdigest()[:8]


def faith_llm(rid, claims, ctx, rec, a, cache):
    """Verdicts per cited line, one per atomic statement: [[1, 0, ...], ...]; None = not judged."""
    out = [None] * len(claims)
    # (1) statements
    stmts = [None] * len(claims)
    todo = []
    for k, c in enumerate(claims):
        if any(i not in ctx for i in c["ids"]):
            out[k] = [0]  # cites an id that is not in the context
            continue
        dk = key_of("D", D_VER, rid, c.get("where", ""), c["claim"])  # per record and position: the same line means a different drug elsewhere
        if dk in cache:
            stmts[k] = cache[dk]
        else:
            todo.append((k, dk))
    def fits(k, st):  # statements must come from this line (a batched answer can shift onto a neighbouring line)
        src = bigrams(claims[k]["claim"] + " " + claims[k].get("where", ""))
        got = set().union(*(bigrams(x) for x in st)) if st else set()
        return not got or len(got & src) / len(got) >= 0.5

    def decompose(chunk):
        items = "\n".join(f"{n}. [위치: {claims[k].get('where') or '-'}] {claims[k]['claim']}" for n, (k, _) in enumerate(chunk, 1))
        res = ask(a, rid, DECOMPOSE_PROMPT.format(items=items), parse_statements, len(chunk))
        bad = []
        for n, (k, dk) in enumerate(chunk):
            if res is None:
                continue
            if fits(k, res[n]):
                stmts[k] = cache[dk] = res[n]
            else:
                bad.append((k, dk))
        return bad

    redo = []
    for s in range(0, len(todo), a.judge_batch):
        redo += decompose(todo[s:s + a.judge_batch])
    for one in redo:  # one line per call
        decompose([one])
    untimed = any(re.search(r"미기재|처방전에\s*없", str(d.get("용법", ""))) for d in rec["input"]["처방전"]["처방약"])
    for k, st in enumerate(stmts):
        if st:
            stmts[k] = [x for x in st if not ACTION_STMT.search(x) and not JUDGMENT_STMT.search(x)
                        and not (untimed and SLOT_STMT.search(x))]
    # (2) verdicts
    todo = []
    for k, c in enumerate(claims):
        if out[k] is not None or stmts[k] is None:
            continue
        if not stmts[k]:
            out[k] = []  # heading-only line: no statement to count
            continue
        vk = [key_of("V", V_VER, rid, st, ",".join(c["ids"])) for st in stmts[k]]
        if all(x in cache for x in vk):
            out[k] = [cache[x] for x in vk]
        else:
            todo.append((k, vk))
    budget = a.verify_batch  # statements per verify call
    s = 0
    while s < len(todo):
        chunk, size = [], 0
        while s < len(todo) and (not chunk or size + len(stmts[todo[s][0]]) <= budget):
            chunk.append(todo[s])
            size += len(stmts[todo[s][0]])
            s += 1
        blocks, n = [], 0
        for k, _ in chunk:
            ev = "\n".join(f"  [{i}] {ctx[i][:1500]}" for i in claims[k]["ids"])
            lines = []
            for st in stmts[k]:
                n += 1
                lines.append(f"주장 {n}: {st}")
            blocks.append(f"[근거]\n{ev}\n" + "\n".join(lines))
        res = ask(a, rid, VERIFY_PROMPT.format(rx=rx_text(rec), items="\n\n".join(blocks)), parse_verdicts, n)
        if res is None:
            continue
        pos = 0
        for k, vk in chunk:
            vs = res[pos:pos + len(vk)]
            pos += len(vk)
            for x, v in zip(vk, vs):
                if v is not None:
                    cache[x] = v
            if None not in vs:
                out[k] = vs
    return out


# ---- plain efficacy one-liner: compared with the reference one-liner of the product (--efficacy-gt), not with evidence
EFF_PROMPT = """약국 복약지도 문서의 '효능 한 줄'을 정답과 비교합니다.
각 항목의 [모델]이 [정답]과 같은 맥락의 효능(무엇을 치료하거나 어디에 쓰는 약인지)을 말하는지 판정하세요.
- 표현·말투·길이 차이는 무시합니다(예: '간질 발작을 막는 약' = '뇌전증(간질) 발작을 치료하는 약').
- [정답]에 효능이 여러 개일 때 대표 효능 하나라도 같은 뜻으로 말하면 1입니다.
- 약 계열명 같은 덧붙인 설명(예: '(스타틴)', '(벤조디아제핀계)')은 판정에 반영하지 않습니다.
- 대표 용도가 [정답]과 다른 질환·용도이거나, 효능을 말하지 않으면 0입니다.
출력은 JSON 객체 하나입니다. 키는 항목 번호, 값은 1 또는 0이며 모든 번호를 씁니다. 예: {{"1": 1, "2": 0}}

{items}"""
EFF_VER = hashlib.sha1(EFF_PROMPT.encode()).hexdigest()[:8]


def efficacy_lines(sec, drugs):
    """{drug index: one-liner} from the counter card '효능 한 줄' row, else the leaflet sentence under '어떤 약인가요?',
    else a '효능/용도' key-value line in the drug's counter or leaflet block."""
    out = {}
    for key in ("counter", "patient"):
        for h, body in subsections(sec.get(key, "")):
            for _, sub in subsections(body) or [(h, body)]:
                d = drugs.first(sub.splitlines()[0]) if sub else None
                if d is None or d in out:
                    continue
                m = re.search(r"^\|\s*\**효능\s*한\s*줄\**\s*\|\s*(.+?)\s*\|\s*$", sub, re.M)
                if not m:
                    m = re.search(r"어떤\s*약인가요\?\**\s*\n+\s*([^\n#|]+)", sub)
                if not m:
                    m = re.search(r"^\s*[-*|]?\s*\**(?:효능|용도)[^:：|]{0,6}\**\s*[:：|]\s*\**\s*(.+?)\s*\|?\s*$", sub, re.M)
                if m:
                    out[d] = CITE.sub("", m.group(1)).strip(" *|")
    return out


def score_efficacy(rid, rec, sec, drugs, gt, a, cache):
    """[1/0/None] per prescribed drug that has a reference one-liner; None = judge gave no answer."""
    got = efficacy_lines(sec, drugs)
    out, todo = [], []
    for i, d in enumerate(drugs.rx):
        ref = gt.get(str(d.get("품목기준코드")))
        if not ref:
            continue
        mine = got.get(i)
        if not mine:
            out.append(0)  # no efficacy line for this drug
        elif norm(mine) in (norm(ref["card"]), norm(ref["patient"])):
            out.append(1)
        elif not a.judge_url:
            out.append(None)
        else:
            k = key_of("F", EFF_VER, ref["card"], mine)
            if k in cache:
                out.append(cache[k])
            else:
                out.append(None)
                todo.append((len(out) - 1, k, ref, mine))
    if todo:
        items = "\n\n".join(f"{n}. [정답] {ref['card']} / {ref['patient']}\n   [모델] {mine}"
                             for n, (_, _, ref, mine) in enumerate(todo, 1))
        res = ask(a, rid, EFF_PROMPT.format(items=items), parse_verdicts, len(todo))
        for n, (pos, k, _, _) in enumerate(todo):
            if res is not None and res[n] is not None:
                out[pos] = cache[k] = res[n]
    return out


# ------------------------------------------------------------------ main
def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", required=True, help="glob of record jsonl files (with input / 정답 / 근거)")
    ap.add_argument("--pred", required=True, help="folder of <id>.md or jsonl of {id, output}")
    ap.add_argument("--ids", help="selection.json:test | ids.json | ids.txt")
    ap.add_argument("--input-md", help="folder of the input documents (<id>.md) to read the [E#] passages from")
    ap.add_argument("--issue-source", default="all", choices=["all", "any", "s0", "study", "counter"],
                    help="all: an issue counts only when 0 요약, ① and ② all report it (default)")
    ap.add_argument("--faith", default="lexical", choices=["lexical", "llm", "off"])
    ap.add_argument("--lex-th", type=float, default=0.5)
    ap.add_argument("--judge-url", help="OpenAI-compatible base URL(s), comma-separated to spread requests over several servers")
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-batch", type=int, default=8, help="cited lines per decomposition call")
    ap.add_argument("--verify-batch", type=int, default=10, help="statements per verification call")
    ap.add_argument("--judge-cache", default="judge_cache.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--efficacy-gt", help="json {품목기준코드: {card, patient}}: reference efficacy one-liners for ③")
    ap.add_argument("--out")
    ap.add_argument("--details")
    a = ap.parse_args()
    if a.faith == "llm" and not (a.judge_url and a.judge_model):
        ap.error("--faith llm needs --judge-url and --judge-model")

    ids = load_ids(a.ids)
    gold = load_gold(a.gold, ids)
    if ids is not None and len(gold) < len(ids):
        print(f"warning: {len(ids) - len(gold)} ids not found in --gold", file=sys.stderr)
    pred = load_pred(a.pred, set(gold))
    if ids is None:  # without --ids the predictions define the evaluation set
        gold = {k: gold[k] for k in pred}
    missing = sorted(set(gold) - set(pred))
    print(f"records: gold {len(gold)} · predictions {len(pred)} · missing {len(missing)}", file=sys.stderr)

    cache = {}
    if a.faith == "llm" and os.path.exists(a.judge_cache):
        cache = {r["k"]: r["v"] for r in read_jsonl(a.judge_cache)}
    n_cache = len(cache)
    eff_gt = json.load(open(a.efficacy_gt, encoding="utf-8")) if a.efficacy_gt else None

    def one(rid):
        rec, md = gold[rid], pred[rid]
        drugs = Drugs(rec["input"]["처방전"]["처방약"])
        sec = split_sections(md)
        r = {"id": rid, "issues": score_issues(rec, md, sec, drugs, a.issue_source), "med": score_med(rec, md, sec, drugs)}
        r["actions"] = score_actions(rec, sec, md, r["issues"]["hits"], drugs)
        if eff_gt:
            r["efficacy"] = score_efficacy(rid, rec, sec, drugs, eff_gt, a, cache)
        items = score_format(rec, md, sec, drugs)
        r["format"] = {"present": sum(items.values()), "required": len(items), "missing": [k for k, v in items.items() if not v]}
        names = sorted((d["제품명"] for d in drugs.rx), key=len, reverse=True)
        r["format"]["missing_kinds"] = sorted({re.sub("|".join(map(re.escape, names)), "<약>", k) for k in r["format"]["missing"]})
        if a.faith != "off":
            claims = claims_of(sec, md)
            ctx = load_context(rec, a.input_md)
            if a.faith == "lexical":
                v = faith_lexical(claims, ctx, rec, a.lex_th)
            else:
                v = faith_llm(rid, claims, ctx, rec, a, cache)
                if None in v:  # second pass asks only for what the judge skipped (the rest comes from the cache)
                    v = faith_llm(rid, claims, ctx, rec, a, cache)
            r["faith"] = {"lines": len(claims), "lines_judged": sum(x is not None for x in v),
                          "statements": sum(len(x) for x in v if x is not None),
                          "supported": sum(sum(x) for x in v if x is not None),
                          "unsupported": [c["claim"][:160] for c, x in zip(claims, v) if x and 0 in x][:20]}
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, sorted(pred)))

    if a.faith == "llm" and len(cache) > n_cache:
        with open(a.judge_cache, "w", encoding="utf-8") as f:
            for k, v in cache.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")

    # ① issues
    gi = [h for r in res for h in r["issues"]["hits"]]
    neg = [r for r in res if r["issues"]["gold"] == 0]
    by_kind = defaultdict(lambda: [0, 0])
    for h in gi:
        by_kind[h["구분"]][0] += h["detected"]
        by_kind[h["구분"]][1] += 1
    buckets = {}
    for name, lo, hi in [("1~2개 (단순 처방)", 1, 2), ("3개 이상 (복합 처방)", 3, 99)]:
        rs = [r for r in res if lo <= r["issues"]["gold"] <= hi]
        hs = [h for r in rs for h in r["issues"]["hits"]]
        buckets[name] = {"처방 수": len(rs), "이슈 수": len(hs), "재현율": pct(sum(h["detected"] for h in hs), len(hs))}
    # ③ med
    dr = [d for r in res for d in r["med"]["drugs"]]
    # ② faith
    fc = sum(r.get("faith", {}).get("statements", 0) for r in res)
    fs = sum(r.get("faith", {}).get("supported", 0) for r in res)
    # ④ format
    fp = sum(r["format"]["present"] for r in res)
    fr = sum(r["format"]["required"] for r in res)
    miss_items = Counter(k for r in res for k in r["format"]["missing_kinds"])  # records missing each item

    summary = {
        "records": len(res), "missing_predictions": len(missing),
        "① 위험 탐지율": pct(sum(h["detected"] for h in gi), len(gi)), "① 정답 이슈 수": len(gi),
        "① 오경보율": pct(sum(r["issues"]["any_pred"] for r in neg), len(neg)), "① 비위험 기록 수": len(neg),
        "② 근거 충실도": pct(fs, fc) if a.faith != "off" else None, "② 판정된 주장 수": fc,
        "② 판정 못 한 문장 수": sum(r.get("faith", {}).get("lines", 0) - r.get("faith", {}).get("lines_judged", 0) for r in res),
        "③ 복약 정보 정확도(기록)": pct(sum(r["med"]["record_ok"] for r in res), len(res)),
        "③ 처방 정보 정확도(기록)": pct(sum(all(r["med"]["info"].values()) for r in res), len(res)),
        "③ 처방 정보 항목별": {f: pct(sum(r["med"]["info"][f] for r in res if f in r["med"]["info"]),
                                   sum(f in r["med"]["info"] for r in res))
                             for f in ("질병명", "처방상태", "의료기관", "조제약국", "임신·수유", "알레르기", "건강기능식품")},
        "③ 종합(하위 항목 평균)": None,
        "③ 약사 조치 일치율(탐지된 이슈)": pct(sum(sum(r["actions"]) for r in res), sum(len(r["actions"]) for r in res)),
        "③ 효능 한 줄 일치율(약)": pct(sum(x for r in res for x in r.get("efficacy", []) if x),
                                    sum(x is not None for r in res for x in r.get("efficacy", []))) if eff_gt else None,
        "③ 효능 한 줄 판정 못 함": sum(x is None for r in res for x in r.get("efficacy", [])),
        "③ 약사 조치 판정 이슈 수": sum(len(r["actions"]) for r in res),
        "③ 복약 정보 정확도(약)": pct(sum(d["ok"] for d in dr), len(dr)),
        "③ 용량 일치(약)": pct(sum(d["dose_ok"] for d in dr), len(dr)),
        "③ 시간표 일치(약)": pct(sum(bool(d["timetable_ok"]) for d in dr if d["timetable_ok"] is not None),
                                sum(d["timetable_ok"] is not None for d in dr)),
        "③ 시간표 비교 약 수": sum(d["timetable_ok"] is not None for d in dr),
        "④ 세부 항목 포함률": pct(fp, fr),
        "④ 양식 통과율": pct(sum(r["format"]["present"] == r["format"]["required"] for r in res), len(res)),
        "처방 복잡도별 위험 재현율": buckets,
        "구분별 위험 탐지율": {k: {"탐지": v[0], "전체": v[1], "%": pct(*v)} for k, v in sorted(by_kind.items(), key=lambda x: -x[1][1])},
        "이슈 판독 방식": dict(Counter(r["issues"]["mode"] for r in res)),
        "누락 항목 상위": dict(miss_items.most_common(15)),
        "settings": {"issue_source": a.issue_source, "faith": a.faith, "lex_th": a.lex_th if a.faith == "lexical" else None,
                     "judge_model": a.judge_model if a.faith == "llm" else None},
    }
    parts = [summary["③ 복약 정보 정확도(약)"], summary["③ 처방 정보 정확도(기록)"], summary["③ 약사 조치 일치율(탐지된 이슈)"]]
    parts = [x for x in parts if x is not None]
    summary["③ 종합(하위 항목 평균)"] = round(sum(parts) / len(parts), 1) if parts else None
    keys = ["① 위험 탐지율", "① 오경보율", "② 근거 충실도", "③ 종합(하위 항목 평균)", "③ 복약 정보 정확도(기록)", "③ 복약 정보 정확도(약)", "③ 처방 정보 정확도(기록)", "③ 약사 조치 일치율(탐지된 이슈)", "④ 세부 항목 포함률", "④ 양식 통과율"]
    print(f"\n{'지표':<24}{'값':>8}")
    for k in keys:
        v = summary[k]
        print(f"{k:<24}{'-' if v is None else f'{v:.1f}%':>8}")
    print("\n처방 복잡도별 위험 재현율")
    for k, v in buckets.items():
        print(f"  {k:<18} 처방 {v['처방 수']:>4}  이슈 {v['이슈 수']:>4}  재현율 {'-' if v['재현율'] is None else v['재현율']}")
    if a.out:
        json.dump(summary, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.details:
        with open(a.details, "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
