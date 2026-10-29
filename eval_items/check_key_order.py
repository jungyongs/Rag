"""Compare the key order of output_items/*.json with the JSON schema in the instruction (input_md).

python3 check_key_order.py DATASET_DIR [--fix]
The schema block is read from one input_md file ('```' fenced JSON after '**출력 형식**'). For each object level
(top, 카운터 card, 공부, 공부 약별, 복약지도서, 복약지도서 약별) the gold keys must follow the schema order.
--fix rewrites the gold files with keys in schema order (values unchanged; keys not in the schema keep their place
after the schema keys).
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
FIX = '--fix' in sys.argv

md = open(sorted(glob.glob(f'{DATA}/input_md/*/*.md'))[0], encoding='utf-8').read()
block = re.search(r'\*\*출력 형식\*\*.*?```\s*\n(.*?)\n```', md, re.S).group(1)


def schema_tree(text):
    """{path: [keys in order]} from the indented schema text."""
    tree = collections.defaultdict(list)
    stack = []  # (indent, path)
    for line in text.split('\n'):
        m = re.match(r'^(\s*)"([^"]+)":\s*(\{)?', line)
        if not m:
            continue
        ind, key, opens = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1] if stack else ()
        k = '*' if key.startswith('<') else key  # '<제품명>' = any drug
        if k not in tree[parent]:
            tree[parent].append(k)
        if opens:
            stack.append((ind, parent + (k,)))
    return tree


SCHEMA = schema_tree(block)


def walk(obj, path, bad, fix):
    want = SCHEMA.get(path)
    if want is None or want == ['*']:
        keys = list(obj)
        if want == ['*']:
            for k in keys:
                obj[k] = walk(obj[k], path + ('*',), bad, fix) if isinstance(obj[k], dict) else obj[k]
        return obj
    keys = list(obj)
    order = [k for k in want if k in obj] + [k for k in keys if k not in want]
    if keys != order:
        bad[path] += 1
    new = {k: obj[k] for k in order} if fix else obj
    for k, v in new.items():
        if isinstance(v, dict):
            new[k] = walk(v, path + (k,), bad, fix)
    return new


bad = collections.Counter()
n = 0
for p in sorted(glob.glob(f'{DATA}/output_items/*/*.json')):
    t = open(p, encoding='utf-8').read()
    g = json.loads(t)
    before = sum(bad.values())
    new = walk(g, (), bad, FIX)
    if FIX and sum(bad.values()) > before:
        assert new == g  # same content
        open(p, 'w', encoding='utf-8').write(dumps(new) + '\n')
    n += 1
print(f'{DATA}: 정답 {n}개 파일')
print('지시문 순서:', {'/'.join(k) or '(최상위)': v for k, v in SCHEMA.items()})
print('순서가 다른 객체:', {'/'.join(k) or '(최상위)': c for k, c in bad.items()} or '없음', '(고침)' if FIX and bad else '')
