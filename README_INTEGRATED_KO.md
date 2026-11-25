# 로컬 통합 검색 실험

기존 V2 교재 101개와 rag_corpus(case/info) 또는 별도 counseling_cases 컬렉션을 읽기 전용으로 검색한다. 기존 DB/원본 파일을 수정하거나 재임베딩하지 않는다. GitHub와 기존 demo/app.py는 변경하지 않는다.

```powershell
python scripts/test_integrated_retrieval.py inventory
python -m unittest discover -s tests -p "test_integrated_retrieval.py" -v
# 허용된 로컬 RAG 사용 범위 확인 후
python scripts/test_integrated_retrieval.py search --permission-confirmed
# 기존 Wellness 등 사례 DB의 로컬 사용권 및 개인정보/상담 안전성 검수 완료 후에만
python scripts/test_integrated_retrieval.py search --case-source legacy --case-slots 2 --permission-confirmed
```

`inventory`는 실제 컬렉션별 저장 개수와 메타데이터의 출처/유형별 집계만 표시한다. legacy가 없으면 조장에게 기존 DB 구축 현황 및 원본 확보 여부를 확인한다. Counsel-Chat의 포함 여부는 실제 메타데이터로 판단하며 별도 추가하지 않는다. 별도 검수된 `counseling_cases` DB가 있다면 `--case-source private`를 사용할 수 있다. 포털은 재구성 이용 조건 미확정으로 기본 제외한다. 로컬 검색 범위를 확인한 경우에만 `--portal-slots 1`로 실험 가능하다. `--mode fast`는 reranker 없이 검색한다.

기본 교재 단독 모드는 기존 15→5 설정을 유지한다. 사례 통합 모드는 교재 15개와 사례 최대 15개 후보를 각 그룹에서 재정렬하고 최종 교재 3개, 사례 최대 2개를 선택한다. 사례가 부족하면 교재로 채운다. 이 비율은 검증 전 초기값이며 통합 성능을 따로 평가해야 한다. 각 그룹의 점수는 직접 비교하지 않는다. 사례는 심리교육 근거가 아니며 답변에 그대로 재사용하지 않는다. 포털/사례/HF 권리 확인 전 외부 API 호출 금지. HF Q&A는 원출처 권리 미확정으로 이번 실험에서 제외한다.

모델 다운로드 외에는 외부 요청 없이 로컬 추론을 수행한다. CLI는 문서 본문과 질문을 출력하지 않는다. 통합 모듈은 본문을 메모리에서 반환하므로 외부 LLM/로그로 전달하지 않는다. 사용 권리와 개인정보/안전성 검수는 자동 확인할 수 없다. 실제 상담 서비스가 아니라 연구용 검색 단계이다. 위기 탐지, prompt builder 및 생성 모델 연동은 아직 포함하지 않는다. 실제 시스템에서는 위기 처리 및 안전성 검사를 검색/생성 앞에 구현해야 한다.
