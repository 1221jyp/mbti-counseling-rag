# Hybrid Search 실험

기존 프로젝트 루트에 압축 해제. 기존 V2 DB, JSONL, `scripts/textbook_v2.py`, `scripts/evaluate_reranker.py` 필요. 추가 패키지 설치 및 재임베딩 불필요. 기존 파일을 수정하지 않는다.

```powershell
python scripts/evaluate_hybrid.py --question-id v02 --candidates 15 --k 5
python scripts/evaluate_hybrid.py --candidates 15 --k 5
```

CPU 메모리 부족 시 `--batch-size 1`, Reranker 없이 빠르게 확인하려면 `--skip-rerank` 사용. 보고서는 `data/processed/textbook_hybrid_c15_k5_<UTC timestamp>.json`에 별도 저장된다.

비교: dense, bm25, hybrid_rrf, dense_rerank, hybrid_rerank. BM25(k1=1.5,b=0.75)는 한글 어절 내 음절 bigram과 영어 단어를 사용한다. 번역이나 형태소 분석은 하지 않으며 한국어-영어 간 어휘 일치에 한계가 있다. 점수가 0인 BM25 결과는 반환하지 않으며 Precision의 분모는 고정 K다. RRF는 상수 60, dense 가중치 2, BM25 가중치 1인 초기 설정이다. 모델은 기존 CrossEncoder를 그대로 사용한다. RRF 후보 수와 최종 K를 동일하게 유지한다. 후보/최종 지표, 반환 개수, 시간, rerank 대비 차이를 기록한다. 임베딩 워밍업은 별도 측정한다. 외부 LLM API는 호출하지 않지만 모델 파일 로딩 시 HF Hub 통신이 발생할 수 있다.

기존 21개 정답 라벨은 V2 추가 청크를 충분히 반영하지 않는다. 성능 비교는 개발용이며 임상 검증이 아니다. `--questions`로 별도의 검수된 평가 JSONL을 지정할 수 있다. 기존 데이터 이용 허가 범위 내 로컬 실험으로 제한한다.
