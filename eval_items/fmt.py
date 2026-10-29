"""항목형 정답 JSON을 원본과 같은 형식으로 쓴다: 객체만 줄바꿈·들여쓰기, 배열·문자열은 한 줄."""
import json


def dumps(obj, indent=0):
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        pad = ' ' * (indent + 2)
        body = ',\n'.join(f'{pad}{json.dumps(k, ensure_ascii=False)}: {dumps(v, indent + 2)}' for k, v in obj.items())
        return '{\n' + body + '\n' + ' ' * indent + '}'
    return json.dumps(obj, ensure_ascii=False)
