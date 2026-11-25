# SIR-Lab Mental Health Knowledge — RAG Integration (Experimental)

데이터: https://huggingface.co/datasets/SIR-Lab/Mental_Health_Knowledge_Dataset

## 목적 및 범위
- 기존 `case` / `info` 검색을 그대로 유지하고, 검수한 심리교육 Q&A를 opt-in으로 검색한다.
- 선택 주제: 스트레스, 감정 조절, CBT, 대인관계, 불안·우울·수면, 자존감·자기수용, 전문 도움 기준.
- 출처는 여러 심리학 교재를 기반으로 한 **영어 2차 가공 자료**. 개별 Q&A의 원출처는 보증되지 않는다.
- 데이터 카드의 표시 라이선스는 CC BY-NC-SA 4.0. 비상업적 연구·교육, 비임상 목적이며 원교재 저작권/출처 확인이 별도로 필요하다.
- 내용의 의학적 정확성, 최신성, 한국어 검색 적합성은 검증되지 않았다.

## Windows VS Code PowerShell 실행
저장소 루트에서:
```powershell
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
python -m unittest discover -s tests -p "test_hf_knowledge.py" -v
python scripts/build_hf_knowledge_corpus.py --download
```

생성 파일: `data/processed/hf_knowledge_candidates.jsonl` (검수 전 후보)
- 다운로드 파일명 `Knowleadge_dataset.json`은 원래 저장소 표기를 따른다.
- 실데이터 포맷이 달라 다운로드/파싱 오류가 발생하면 원본 저장소의 Files 화면에서 이름과 구조를 확인한다.

검수 기준: 원본 라이선스, 정확성, 주제 적합성, 자해·위기·약물 등 고위험 항목, 개인정보, 중복(기존 포털/Counsel-Chat 포함), 문화·번역 정확성.
승인할 후보의 `id`만 `data/processed/hf_approved_ids.txt`에 한 줄씩 기록한다.

```powershell
python scripts/build_hf_knowledge_corpus.py --download --approved-ids data/processed/hf_approved_ids.txt --output data/processed/hf_knowledge_approved.jsonl
python scripts/build_hf_knowledge_index.py
```

별도 인덱스 기본 위치:
`%USERPROFILE%\.cache\mbti-rag\hf_knowledge_chroma`
컬렉션: `hf_knowledge_info`. 기존 DB(`rag_corpus`)는 건드리지 않는다.
실험 DB가 이미 존재하면 확인 후 `python scripts/build_hf_knowledge_index.py --replace`로 해당 컬렉션만 재구축한다.

## 검색 비교
```python
from rag import retrieve
question = "시험 스트레스 때문에 자꾸 실패할 것 같아요."
before = retrieve(question, k=5)
after = retrieve(question, k=5, n_expert=1)
for chunk in after:
    print(chunk["id"], chunk["score"], chunk["category"], chunk["source_url"])
```

- 기본 `n_expert=0`은 기존 결과 개수 및 구성을 유지한다.
- `n_expert=1`은 `k`개 중 1개를 별도 전문지식 컬렉션에서 검색한다.
- 검수 DB가 아직 없으면 `n_expert>0` 검색은 파일 없음 오류를 발생시킨다.
- 기존 데모는 변경하지 않았으므로, 이 옵션을 직접 호출하거나 별도로 연결해야 한다.

## 평가 원칙
- 기존 `eval/questions.jsonl`은 변경하지 않는다. 새 데이터의 정답 청크 ID가 없기 때문에 기존 파일만으로 개선 효과를 판단할 수 없다.
- 원본 기준선과 추가 검색을 같은 한국어 질문으로 비교하고, 질의별 관련성 라벨을 사람이 검토한다.
- Precision@5, MRR, 주제별 검색 성공률, 부적절한 조언·출처 누락률 등을 별도로 보고한다.
- 검색된 지식은 임상적 진단/치료 지시로 사용하지 않는다.
- 원문/정제 데이터와 DB는 `.gitignore`의 `data/raw/`, `data/processed/` 규칙에 따라 GitHub에 올리지 않는다. 외부 API로 보내지 않는다.
