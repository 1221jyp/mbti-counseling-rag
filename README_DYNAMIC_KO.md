# 통합 검색 배분 실험 (로컬 연구 전용)

기존 101 교재 + 541 Wellness DB를 읽기 전용으로 사용한다. 기존 `rag/integrated_retrieve.py`, `scripts/textbook_v2.py`, `rag/retrieve.py`, `eval/questions.jsonl`, `eval/textbook_questions_v2_holdout70.jsonl`, `data/processed/wellness_unflagged.jsonl`이 필요하다. ZIP을 프로젝트 최상위에 풀고 가상환경에서 실행한다. 원본 상담 데이터/벡터 DB를 ZIP에 포함하지 않았다.

```powershell
python -m unittest discover -s tests -p "test_dynamic_retrieval.py" -v
python scripts/evaluate_integrated.py --question-id q01 --permission-confirmed
python scripts/evaluate_integrated.py --permission-confirmed
```

권한 플래그는 AI Hub/교재의 로컬 이용 조건과 개인정보 및 상담 안전성 검토를 확인한 경우에만 사용한다. 약물/위기 키워드 미검출은 임상 안전성 인증이 아니다. 실제 상담에는 적용하지 않는다.

- 교재 및 상담 사례 각각 Top-15 후보를 가져와 기존 BGE Reranker로 1회씩 정렬한다. 배분별 모델 재실행은 하지 않는다. 모델은 프로세스 내부에서 재사용하며 워밍업 후 시간을 기록한다.
- 고정 배분 5:0 ~ 0:5는 오프라인 비교 전용이다. 0:5는 실제 답변 생성에 사용하지 않는다.
- 동적 배분은 교재 기준 후보 0.01/0.05/0.1/0.2, 사례 기준 0.2/0.5/0.7의 12개 조합을 통합 개발 11문항에서만 평가한다. 양쪽 Hit 우선, 그룹 Recall 평균, 고정 분모 P@5 순으로 선택한다. 점수는 관련성 신호이지 정답 확률이나 위험도 점수가 아니다. 최대 사례 2개, 최대 전체 5개. 교재 기준 통과 문서가 없으면 빈 결과를 반환한다. 기준은 그룹별 적용하며 서로 점수 비교하지 않는다.
- 통합 20문항(개발 11, 내부 확인 9)은 기존 팀 `eval/questions.jsonl`의 일부를 재사용한다. 상담 ID 라벨은 실제 541개 인덱스와 교집합을 취한다. 교재 라벨은 질문에 관련된 심리교육 자료를 연결한 초안이며 완전하거나 전문가 검수된 정답은 아니다. 70개 교재 질문은 이미 검색 기술 선택에 사용했으므로 새 독립 테스트가 아니라 회귀 평가이다. 사례 정답이 없으므로 사례 관련성은 평가하지 않는다.
- 그룹별 hit/recall/mrr은 각 그룹의 반환 문서만을 기준으로 한다. `precision_returned`는 반환 문서 수를 분모로 하며 빈 결과는 0이다. `both_hit`는 두 그룹 모두 정답 라벨을 하나 이상 포함하는 질문 비율이다. `joint_p5`는 두 그룹 정답 합계를 고정 분모 5로 나눈 값이다. `abstained`는 빈 결과 비율이다. 결합 MRR은 그룹 순서의 영향을 피하기 위해 사용하지 않는다.
- 결과는 `data/processed/integrated_eval_*.json`에 저장되며 원문/질문 텍스트를 저장하지 않는다. 타이밍은 양쪽 후보 검색·재정렬 공통 작업의 측정치이며 각 고정 비율의 독립 실행시간이 아니다. 테스트 결과에 맞춰 임계값을 반복 조정하면 추가 독립 평가가 필요하다.
- `rag.dynamic_retrieve.retrieve_dynamic`은 선택한 임계값을 명시적으로 전달하는 실험 API이다. 기존 챗봇, 모델 연결, 위기 대응, GitHub 코드는 변경하지 않는다.
