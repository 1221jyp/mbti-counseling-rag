# 교재 RAG V2 평가 데이터 보완 (2026-11-25)

기존 V2 101개 청크, ChromaDB, 평가 코드, 기존 21개 질문 파일은 변경하지 않습니다.

## 파일
- `eval/textbook_questions_v2_reviewed.jsonl`: 기존 21개 질문의 초안 관련 문서 라벨 재작성 (97개 질문-문서 연결).
- `eval/textbook_questions_v2_holdout70.jsonl`: 7개 분야별 10개, 신규 합성 질문 70개 (172개 연결).
- `docs/textbook_v2_quality_review.json`: 101개 청크의 메타데이터·페이지 범위·영어 원문 대조·중복 검토 기록.
- `scripts/validate_textbook_eval_sets.py`: V2 해시와 평가 형식·문서 ID 검증.

## 실행 (PowerShell)
프로젝트 최상위에 압축 해제 후 기존 가상환경을 사용합니다. 추가 패키지나 재임베딩은 필요하지 않습니다.

```powershell
python scripts/validate_textbook_eval_sets.py
python scripts/evaluate_hybrid.py --questions eval/textbook_questions_v2_reviewed.jsonl --candidates 15 --k 5 --output data/processed/textbook_eval_reviewed21.json
```

개발용 질문 결과를 바탕으로 검색 설정을 확정한 다음, 새 질문 평가를 실행합니다. 검색 설정을 새 질문의 정답에 맞춰 재조정하지 마세요.

```powershell
python scripts/evaluate_hybrid.py --questions eval/textbook_questions_v2_holdout70.jsonl --candidates 15 --k 5 --output data/processed/textbook_eval_holdout70.json
```

출력 경로가 이미 있으면 기존 평가 파일을 보존하기 위해 새 파일명을 사용합니다. `dense_rerank`와 `hybrid_rrf`를 우선 비교하고 후보 성능·MRR·검색 시간도 확인합니다. 기본 평가 스크립트는 다른 방식도 함께 측정합니다.

## 관련성 라벨 기준
질문에 직접 답하거나 답변에 필요한 실질적 심리교육·지원 정보가 있는 청크를 포함했습니다. 단순 주제 유사성만으로는 포함하지 않습니다. 라벨의 연결 개수는 고유 문서 수와 다릅니다. 보완 라벨은 기존 라벨을 단순히 합친 것이 아니라 질문의 의미에 따라 일부 추가·수정했습니다. 모든 관련 문서를 완전하게 찾아냈다고 보장하지 않습니다.

- 자료 본문과 출처를 참고해 작성한 AI 보조 초안이며, 실제 상담 기록이나 독립 외부 평가가 아닙니다.
- 70개 질문은 기존 21개와 문장 중복 없이 별도 작성했지만 일부 개념은 중첩됩니다. 독립 전문가 평가와 실제 사용 사례 검증은 별도로 필요합니다.
- 91 × 101 모든 문서-질문 조합에 대한 복수 평가자 판정은 하지 않았습니다. 따라서 미라벨 문서가 반드시 무관하지 않으며 Precision/Recall은 초안 기준입니다.
- 101개 청크의 메타데이터와 페이지 범위는 확인했고 OpenStax 75개 영어 본문은 지정 PDF 페이지에서 공백, 구두점, 대소문자 등 정규화 후 일치했습니다. WHO 26개는 한국어 편집 요약으로 동일한 문자열 대조가 불가능합니다. 원문 해당 페이지 확인과 별개로 문장별 번역 충실도와 임상 타당성은 전문가 검토가 필요합니다.
- OpenStax 원문에는 학술적 일반화나 약물, 진단, 사례 설명이 있습니다. 개별 사용자에게 그대로 적용하면 안 됩니다. 실제 상담 응답의 정확성이나 안전성을 이번 검색 평가가 증명하지 않습니다.
- 정확한 문자열 중복 0건, 토큰 집합 Jaccard 0.65 이상 유사쌍 0건은 어휘 기반 검사일 뿐 의미 중복이 없다는 뜻은 아닙니다.
- 이번 평가 파일은 로컬에서만 이용하고, 원문 및 제한 데이터의 외부 전송·공개 재배포는 해당 이용허가 범위를 확인하세요.
