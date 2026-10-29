# eval_items — 항목형(JSON) rx_counsel 생성·채점

`input_md/<split>/<id>.md`를 넣어 모델이 낸 JSON 답을 `output_items/<split>/<id>.json` 정답과 비교한다.

## 파일
| 파일 | 하는 일 |
|---|---|
| `gen_items.py` | vLLM 생성 (user 메시지 하나, 시스템 프롬프트 없음, greedy, 32,768, base 모델은 `--yarn 2`) |
| `score_items.py` | 채점 ①~④ (아래). `--pred gold`면 정답 자체를 넣은 기준선 |
| `score_rx.py` | 2026-10-27 평가(`rx_eval/score_rx.py`)의 사본. LLM judge(RAGAS)·이슈 종류·조치 기준을 가져다 쓴다 |
| `popup_check.py` | 팝업 한 줄 근거성 검사 (단어가 빨간박스에 있는지) |
| `make_report.py` | `results/<model>/<split>/scores.json` → `SUMMARY_<split>.md`, `SUMMARY.md` (+ shared 복제) |
| `run_eval.sh` | 모델 하나를 생성 + 채점 |
| `run_pipeline.sh`, `launch_pipeline.sh` | 데이터 준비 → SFT(`../scripts/run_sft.sh`) → base·best·last 생성 → 채점 → 요약을 한 번에 (단계별 재시작 가능) |
| `check_format.py`, `check_key_order.py` | 데이터 검사: 지시문 스키마·출처·약 카드 / 키 순서(`--fix`로 지시문 순서로 맞춤) |
| `fix_trailing_numbers.py` | 약사 복약지도 자료 근거 끝에 잘못 붙은 숫자('…사용합니다. 00') 정리 |
| `fmt.py` | 정답 JSON을 원래 형식(객체만 줄바꿈)으로 쓰기 |

## 채점 지표 (`score_items.py`)
- **① 위험 판정**: 정답 = 레코드 `정답.위험이슈`. 출력의 구분·등급 = 공부 '핵심 이슈' `[구분·등급]` 태그, 약 = 그 구분의 '약사 확인'이 달린 카드.
  위험 탐지율(필드 부분 점수), 정확도(약·구분·등급), 전 항목 일치, 정밀도, 오경보율, 약사 조치 일치.
- **② 근거**: 근거 번호가 붙은 항목은 A.X-4.0 72B judge로 충실도(`--faith llm`), 입력에 없는 근거 번호 비율.
  근거 번호가 없는 항목은 규칙: 복용법(처방과 완전 일치), 처방 구성, '같은 약이 N줄로 처방됨', '자료 없음' 문구, 팝업 규칙·근거성, 근거 없는 주장 비율.
- **③ 내용**: 항목별 재현율 = 정답 문장 중 출력의 같은 카드·항목에 대응 문장(공백 무시 완전 일치 또는 글자 2-gram F1 ≥ 0.8)이 있는 비율. 모드 평균, 전체 = 모드 평균의 평균.
- **④ 양식**: 필수 항목 포함률(카운터·공부·복약지도서, 전체 = 평균). JSON을 못 낸 출력은 빈 답으로 채점(①~③ 0점).

## 실행 예
```bash
# judge 서버 (A.X-4.0 72B)
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/vllm serve ../models/AX-4.0 \
  --tensor-parallel-size 4 --served-model-name ax4-72b --max-model-len 16384 --port 8000
# 모델 하나 평가
bash run_eval.sh <model_dir> <dataset_dir> test results/<name> 0 http://localhost:8000/v1
# 정답 기준선 (규칙 지표는 모두 100이어야 함)
python3 score_items.py --data <dataset_dir> --split test --pred gold
# 전체 파이프라인 (V2, V3, EXP 환경 변수로 경로 지정)
bash launch_pipeline.sh
```
