# MBTI 연계 심리상담 챗봇: RAG 데이터셋

사례(`case`) 문서와 정보(`info`) 문서로 구성한 RAG 코퍼스와 검색 함수, 평가 스크립트다.

파인튜닝 데이터(ESConv, Counsel-Chat, AI Hub 감성 대화 말뭉치)는 RAG와 효과를 분리해 평가하기 위해 포함하지 않는다.

## 진행 현황

| 항목 | 상태 | 비고 |
|---|---|---|
| 웰니스 사례(`case`) 코퍼스 | 완료 | 299문서, 552청크 |
| 포털 정보(`info`) 코퍼스 | 완료 | 28문서, 297청크 |
| 위기 연락처 파일 | 완료 | 109, 1577-0199 |
| 임베딩, Chroma 인덱스, `retrieve` | 완료 | `case`와 `info` 고정 배분 |
| 평가 질문 30개 | 초안 | 정답 라벨 검토 필요 |
| Recall@3, Recall@5 | 측정 | 각 0.867, `info`만 0.316, 0.368 |
| Gemini 연결 시연 데모 | 작동 | 샘플 4개만 점검 |
| 포털 자료 재구성 답변 이용 조건 | 미확정 | 공공누리 제4유형(변경 금지) 근거 확인 필요 |
| AI Hub 재배포 조건 | 미확인 | 약관 직접 확인 필요 |

다음 할 일:

1. `eval/questions.jsonl` 정답 라벨 검토
2. `info` 검색 개선 (유사도가 낮으면 제외하는 기준 등)
3. 위기 탐지 규칙 보강
4. 포털 자료와 AI Hub 데이터 이용 조건 확인

이 저장소에는 원문 데이터가 없다. `data/raw/wellness/`에 AI Hub 파일을 넣고 아래 실행 방법을 따르면 다시 만들 수 있다.

## 데이터 출처와 이용 조건

| 데이터 | 출처 | 이용 조건 |
|---|---|---|
| 웰니스 대화 스크립트 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=267) | AI Hub 이용약관을 따른다. 공개 페이지에서 세부 조건을 확인하지 못했으므로 재배포와 외부 공개 전에 약관을 직접 확인할 것 |
| 정신건강 정보 문서 | [국가정신건강정보포털](https://www.mentalhealth.go.kr) (국립정신건강센터) | 사용한 28개 페이지 모두 공공누리 제4유형: 출처표시, 상업적 이용 금지, 변경 금지 |
| 위기 연락처 | `data/crisis_contacts.json` | RAG 검색 대상이 아니며 규칙 기반 출력용 |

### 포털 자료 이용 시 주의

- 2026-11-01 기준 `robots.txt`는 404로 제한 규칙이 없다. 수집 간격은 1.2초이며, 원본 HTML은 `data/raw/portal/`에 저장했다.
- 프로젝트 담당자가 사용 가능한 자료라고 확인해 수집을 진행했다(2026-11-01). 이용 허가 근거 문서는 이 저장소에 없다.
- [저작권 정책](https://www.mentalhealth.go.kr/portal/board/policy/selectCpyrhtInfo.do)은 공공누리 유형별 조건 준수와 출처 표시를 요구한다. 답변에 `source`, `source_url`을 표시하고 상업적으로 이용하지 않는다.
- 제4유형은 변경을 금지한다. 코퍼스는 본문 문장을 고치지 않고 발췌하지만, 챗봇이 내용을 요약하거나 재구성해 답변하는 것까지 허용되는지는 별도로 확인해야 한다.
- 공공누리 표시가 없는 칼럼 3개(nttId 604, 607, 620)는 사전 협의 대상이라 코퍼스에서 제외했다. 원본 HTML만 남아 있다.

## 코퍼스 현황

| 구분 | 문서 수 | 청크 수 |
|---|---:|---:|
| `case` | 299 | 552 |
| `info` | 28 | 297 |
| 합계 | 327 | 849 |

청크 길이: `case` 261~769자(평균 495자), `info` 183~797자(평균 575자). 300자 미만 청크는 6개다.

### 카테고리 분포

`case` (`category`의 대분류 기준):

| 대분류 | 문서 수 | 청크 수 |
|---|---:|---:|
| 배경 | 121 | 216 |
| 감정 | 76 | 166 |
| 증상 | 83 | 143 |
| 치료이력 | 6 | 9 |
| 자가치료 | 4 | 6 |
| 상태 | 3 | 5 |
| 내원이유 | 3 | 4 |
| 현재상태 | 2 | 2 |
| 원인 | 1 | 1 |

`info`의 페이지별 category 지정은 `scripts/collect_portal.py`에 있다.

### 처리 규칙

**웰니스 (`case`)**: `웰니스_대화_스크립트_데이터셋.xlsx`(5,231행, 359개 카테고리)를 사용한다. `02)웰니스_대화_스크립트_데이터셋.xlsx`(19,769행, 19개 의도)는 답변이 템플릿이고 위기 답변의 연락처가 오래되어 쓰지 않았으며, 두 파일 모두 `data/raw/wellness/`에 보관한다.

- 제외 141행: `감정/자살충동`, `감정/살인욕구`, `증상/자살시도`, `증상/자해`. 부적절한 답변이 검색되는 것을 막기 위해서이며, 발화는 위기 탐지 규칙 참고용으로 `data/processed/excluded_crisis_utterances.jsonl`에 보관한다.
- 제외 237행: `모호함`, `일반대화`, `부가설명`. 상담 의도가 없는 맞장구와 잡담이다.
- 카테고리 안의 중복 발화는 공백과 문장부호를 무시하고 제거한다.
- 300자에 못 미치는 카테고리는 상위 카테고리로 합치고, 2단계에서도 부족하면 `대분류/기타`로 합친다.
- 상담자 답변은 해당 카테고리의 모든 청크에 반복해 넣는다.

**포털 (`info`)**: 질환 페이지 18개와 칼럼 10개를 사용한다.

- 질환 페이지의 소제목(정의, 원인, 증상, 스스로 돕는 법 등)을 단위로 묶고, 칼럼은 줄 단위로 나눈다.
- `참고문헌`, `지원체계` 소제목은 제외한다. 오래된 연락처(1393, 129)가 답변에 쓰이지 않게 하기 위해서다.
- `자살과 자살예방` 페이지는 위기 대응이 규칙 기반이므로 수집하지 않았다.
- 40자가 넘는 중복 문단은 첫 등장만 남긴다.
- 원문 줄바꿈이 문장 중간에 있는 경우가 있어, 청크가 문장 중간에서 끝나는 경우가 일부 있다.

## 평가

`eval/questions.jsonl`의 30개 질문은 **검토 전 초안**(`status: draft`)이다. 정답은 검색 결과를 보기 전에 지정했고, 지정한 카테고리와 페이지의 모든 청크를 정답으로 둔다. 19개 질문에는 `info` 페이지가 정답으로 함께 지정되어 있다.

| 지표 | k=3 | k=5 |
|---|---:|---:|
| Recall@k (정답 청크가 하나라도 포함된 질문 비율) | 0.867 (26/30) | 0.867 (26/30) |
| `case` Recall@k | 0.833 (25/30) | 0.833 (25/30) |
| `info` Recall@k (`info` 정답이 지정된 질문만) | 0.316 (6/19) | 0.368 (7/19) |
| 커버리지@k (정답 청크 중 포함된 비율의 평균) | 0.193 | 0.261 |

- `retrieve`는 `case`와 `info`를 따로 검색해 고정 배분한다. k=3이면 2+1개, k=5이면 3+2개다. `n_info`로 바꿀 수 있다.
- 고정 배분 전에는 평가 질문의 상위 5개 결과 150개 중 `info`가 6개뿐이었다. 1인칭 고민 문장이 내담자 발화로 된 `case`에 훨씬 가깝기 때문이다.
- `info`가 항상 포함되지만 관련 없는 페이지일 수 있다. `info` Recall이 낮고, 유사도가 낮아도 그대로 반환하므로 답변 생성 시 주의가 필요하다.
- 실패한 질문은 q03, q04, q13, q25다.
- 커버리지는 정답 청크가 많은 질문에서 낮게 나오므로 Recall과 함께 해석한다.

## 실행 방법

Python 3.14, CUDA 12.8 환경에서 확인했다.

```bash
python -m venv ~/.venvs/mbti-rag
~/.venvs/mbti-rag/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
~/.venvs/mbti-rag/Scripts/python -m pip install -r requirements.txt
```

```bash
~/.venvs/mbti-rag/Scripts/python scripts/build_wellness_corpus.py
~/.venvs/mbti-rag/Scripts/python scripts/collect_portal.py
~/.venvs/mbti-rag/Scripts/python scripts/build_portal_corpus.py
~/.venvs/mbti-rag/Scripts/python scripts/merge_corpus.py
~/.venvs/mbti-rag/Scripts/python scripts/build_index.py
~/.venvs/mbti-rag/Scripts/python eval/make_questions.py
~/.venvs/mbti-rag/Scripts/python eval/evaluate.py
```

`collect_portal.py`는 이미 저장된 HTML은 다시 받지 않는다. Windows에서 한글 출력이 깨지면 `PYTHONIOENCODING=utf-8`을 설정한다.

검색:

```python
from rag import retrieve

for chunk in retrieve("시험 때문에 잠이 안 와요", k=5):
    print(chunk["id"], chunk["score"], chunk["doc_type"], chunk["category"])

retrieve("시험 때문에 잠이 안 와요", k=5, n_info=0)  # case만 검색
```

### 시연 데모

```bash
~/.venvs/mbti-rag/Scripts/python -m uvicorn demo.app:app --port 8000
```

http://localhost:8000 에서 검색 결과를 볼 수 있다. 시작 시 모델을 불러오므로 첫 화면까지 시간이 걸린다. 서버를 띄우는 환경에 `GEMINI_API_KEY`가 있으면 Gemini로 답변을 생성하고, 없으면 검색 전용으로 동작한다. 기본 모델은 `gemini-3.5-flash-lite`이며 `GEMINI_MODEL`로 바꿀 수 있다.

Gemini 무료 등급은 요청 한도가 있고 입력이 Google 제품 개선에 쓰일 수 있으므로 샘플 문장만 넣고 실제 상담 내용은 넣지 않는다.

위기 표현은 `demo/app.py`의 정규식으로 잡아 검색과 LLM 없이 `data/crisis_contacts.json`의 연락처를 보여준다. 시연용 최소 규칙이라 놓치는 표현이 있다.

Chroma 인덱스는 기본적으로 `~/.cache/mbti-rag/chroma`에 저장한다. OneDrive 폴더에 SQLite를 두면 동기화로 잠금 문제가 생길 수 있어서다. 위치는 `RAG_CHROMA_DIR` 환경 변수로 바꿀 수 있다.

코퍼스를 다시 만들면 청크 구성이 바뀔 수 있으므로 `build_index.py`와 `make_questions.py`도 다시 실행한다.
