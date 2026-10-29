"""팝업 한 줄 근거성 검사: 팝업의 핵심 단어가 빨간박스 문장에 있는지 (popup_gen/validate.py에서 가져옴)."""
import re

# 요약 표현에 흔히 붙는 말: 원문에 그대로 없어도 됨
GENERIC = {'주의', '시', '연락', '금지', '피하기', '복용', '확인', '상의', '필요', '즉시', '바로',
           '등', '후', '전', '중', '때', '가능', '권장', '알리기', '알림', '삼가기', '함께', '병용',
           '섭취', '사용', '이상', '발생', '나타나면', '의사', '약사', '피함', '말기', '않기', '지양'}
# 요약에서 흔히 바꿔 쓰는 말: 오른쪽 중 하나가 원문에 있으면 통과
SYNONYMS = {'음주': ['술'], '취침': ['자기', '잠'], '상담': ['상의'], '연락': ['상의', '알리'],
            '동시': ['함께', '같이'], '지속': ['계속', '가라앉지', '지속'], '건조': ['마른', '말린'],
            '토혈': ['피를 토'], '중단': ['중지', '끊'], '관계없': ['상관없', '무관'],
            '관계없이': ['상관없이', '무관'], '없으면': ['없는', '없을'], '없도록': ['않도록', '없게'],
            '미리': ['사전', '전에', '미리'], '통째': ['그대로', '통째'], '통째로': ['그대로', '통째'],
            '못하면': ['없는', '없을', '못하'], '흡입': ['사용', '흡입']}


def squash(s):
    return re.sub(r'\s+', '', s)


def onset_vowel(ch):
    """한글 음절의 초성·중성만 (받침 무시): 누/눌, 삼/삼킵 비교용"""
    code = ord(ch) - 0xAC00
    return code // 28 if 0 <= code < 11172 else ch


def onset(ch):
    code = ord(ch) - 0xAC00
    return code // 588 if 0 <= code < 11172 else ch


def stem_in(tok, src):
    if tok[:2] in src:
        return True
    # 명사형 어미(씻기·삼킴 등): 첫 음절이 원문에 그대로 있으면 통과
    if re.search(r'(기|음|임|함|김|킴|림|름)$', tok) and tok[0] in src:
        return True
    # 활용형(누르기/눌러, 삼킴/삼킵): 첫 음절 초성·중성 + 둘째 음절 초성이 같으면 통과
    return any(onset_vowel(src[p]) == onset_vowel(tok[0]) and onset(src[p + 1]) == onset(tok[1])
               for p in range(len(src) - 1))


def keyword_misses(text, source):
    src = squash(source)
    misses = []
    for tok in re.split(r'[\s·,/()]+', text):
        tok = squash(re.sub(r'(은|는|이|가|을|를|과|와|에|의|로|으로)$', '', tok))
        if len(tok) < 2 or tok in GENERIC:
            continue
        if stem_in(tok, src):
            continue
        if any(squash(s) in src for s in SYNONYMS.get(tok, [])):
            continue
        misses.append(tok)
    return misses
