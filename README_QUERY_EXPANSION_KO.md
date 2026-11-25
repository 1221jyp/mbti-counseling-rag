# Query Expansion + RRF 비교 실험

기존 프로젝트 루트에 ZIP을 풀면 `scripts/evaluate_query_expansion.py`가 추가됩니다. 기존 `scripts/textbook_v2.py`, `scripts/evaluate_reranker.py`, V2 JSONL 및 벡터 DB가 필요합니다. 기존 데이터, 인덱스, 결과 파일은 변경하지 않습니다. 새 패키지나 모델 다운로드는 필요하지 않습니다(기존 모델 캐시가 유지된 경우).

## 실행 (VS Code PowerShell, .venv 활성화)

```powershell
python scripts/evaluate_query_expansion.py --question-id v02 --candidates 15 --k 5
python scripts/evaluate_query_expansion.py --candidates 15 --k 5
```

CPU가 메모리 부족을 겪으면 `--batch-size 1`을 붙입니다. 재정렬 없이 검색 방식만 빠르게 비교하려면 `--skip-rerank`를 붙입니다. `--questions`로 나중에 검수한 라벨 또는 독립 평가 파일을 지정할 수 있습니다.

## 비교 방식

- `dense`: 원래 질문의 벡터 검색.
- `dense_rerank`: 원래 질문으로 찾은 15개 후보를 기존 CrossEncoder로 재정렬.
- `expanded_single`: 질문에 가장 높은 규칙 매칭 수를 가진 분야의 한·영 검색어를 덧붙여 단일 검색. 동점이면 고정 규칙 순서. 매칭이 없으면 원래 질문으로 검색.
- `expansion_rrf`: 원래 질문 + 최대 3개의 분야별 확장 질문을 각각 검색. 각 검색 후보 15개를 중복 제거한 후 가중 RRF로 통합해 15개 선정.
- `expansion_rrf_rerank`: RRF로 선정한 15개를 원래 질문과의 관련성으로 재정렬.

모든 방식에서 최종 평가 K는 5입니다. 확장 방식은 추가 검색 호출을 하므로 계산량은 동일하지 않습니다. `candidate_summary`는 최종 후보 풀 15개 기준, `summary`는 상위 5개 기준입니다. RRF 점수는 원래 검색 순위에 가중치 2, 확장 검색 순위에는 1을 주고 각각 weight/(60+rank)를 합산합니다. 이는 표준 RRF의 가중 변형이며 설정값은 검증된 최적값이 아닙니다. 검색어는 질문 문자열에서 일반 심리교육 용어를 매칭하는 규칙 기반이며 평가 정답을 검색 입력으로 사용하지 않습니다. LLM 생성 확장이 아닙니다.

결과는 `data/processed/textbook_expansion_c15_k5_<UTC timestamp>.json`으로 저장됩니다. 기존 파일은 덮어쓰지 않습니다. `mean_seconds`에서 dense는 기본 검색, expanded_single_and_rrf_extra는 추가 검색 및 통합, *_rerank_extra는 추가 재정렬 시간입니다. 모델 로딩 시간은 제외되지만 첫 질문의 임베딩 모델 초기화 비용이 dense 시간에 포함될 수 있습니다. 비교 실험에서 두 재정렬 방식을 모두 실행하므로 실제 단일 파이프라인보다 총 실행 시간이 길어집니다.

## 한계

기존 21개 질문과 V1 정답 라벨을 사용하므로 추가 41개 청크가 미평가 상태입니다. 새 청크를 무조건 오답으로 취급할 수 없습니다. 규칙 및 RRF 파라미터는 개발 단계 가설이며 신규 질문에서 독립 검증이 필요합니다. 로컬 검색 성능이며 임상 안전성이나 상담 답변 정확도 평가가 아닙니다. 원문은 로컬 모델에서만 처리하고 보고서에는 문서 ID와 수치만 저장합니다. 데이터 재배포 및 외부 LLM 전송은 별도 이용 조건을 확인해야 합니다.

문법 및 순수 로직 테스트는 완료했습니다. 사용자 PC의 실제 모델/DB를 통한 결과는 아직 측정되지 않았습니다.
