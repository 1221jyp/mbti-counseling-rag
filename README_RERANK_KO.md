# V2 Reranking 실험 (로컬)

기존 프로젝트 루트에 ZIP을 풀고 scripts/evaluate_reranker.py를 추가한다. 기존 DB와 파일을 수정하지 않는다.

- BGE-M3 + V2 ChromaDB에서 후보 15개 확보
- 동일한 후보에서 dense 상위 5개와 BAAI/bge-reranker-v2-m3 상위 5개 비교
- CPU 기본, batch size 2, max_length 512 (긴 입력은 잘릴 수 있음)
- Hugging Face 모델 파일 다운로드 이후 로컬 추론; 교재 텍스트를 외부 API로 전송하지 않음
- V2 기존 이용 허가 범위 안에서만 실행

```powershell
python scripts/evaluate_reranker.py --question-id v02 --candidates 15 --k 5
python scripts/evaluate_reranker.py --candidates 15 --k 5
```

CPU 메모리가 부족하면 `--batch-size 1`을 추가한다. 모델 다운로드 및 로딩에 저장 공간/메모리가 필요하다. 기존 `sentence-transformers`를 사용하므로 추가 설치는 기본적으로 필요하지 않다.

결과는 `data/processed/textbook_rerank_c15_k5_<UTC시간>.json`에 저장한다. 기존 평가 파일은 덮어쓰지 않는다. `summary.dense`, `summary.reranked`, `delta`로 성능을 비교하며 `candidate_summary`는 후보 전체의 정답 포함률/재현율이다. `mean_rerank_seconds`는 질문당 추가 재정렬 시간이다. 첫 dense 측정에 임베딩 모델 로딩 시간이 포함될 수 있다.

기존 21문항 초안 라벨 기준이며 신규 41개 청크는 미판정이다. 점수 상승은 보장되지 않고 최종 임상적 안전성 평가가 아니다. 새 문서 라벨 검수 및 독립 질문 검증은 후속 작업이다. 챗봇 연동 및 GitHub 업로드는 하지 않는다.
