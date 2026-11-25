# WHO / OpenStax 교재 기반 RAG — 최종 로컬 구축 안내

## 범위 및 권리
- WHO 2026 매뉴얼: 한국어 심리교육 요약 20개, CC BY-NC-SA 3.0 IGO. 공식 WHO 번역/승인이 아님.
- OpenStax Psychology 2e (2026 PDF): 28개 PDF 페이지에서 선별한 영어 발췌문 40개, 한국어 편집 제목 및 원문 페이지·절·출처 표기. CC BY-NC-SA 4.0. Access for free at openstax.org.
- 사용자가 OpenStax의 허가를 받았다고 보고했으나 승인서 원문과 허가 범위는 확인하지 않았음. 서면 허가가 로컬 RAG 추출·임베딩을 포함할 때만 실행. 허가서, 원본·가공 본문 및 DB는 비공개로 관리. 공개 재배포 및 외부 LLM API 전송은 허가 범위 확인 전 금지.
- 소스 기반 선별은 전문가 임상 검증이 아니다. 진단 및 치료에 사용하지 않으며, MBTI는 의학적 판단 근거가 아니다.

## 데이터와 인덱스
| 출처 | 선별 수 | 로컬 파일 | 컬렉션 |
|---|---:|---|---|
| WHO | 20 | `data/processed/who_self_help_selected.jsonl` | `who_self_help_2026` |
| OpenStax | 40 | `data/processed/openstax_selected.jsonl` | `openstax_psychology2e_2026` |
| SIR-Lab | 최대 5 | `data/processed/hf_knowledge_approved.jsonl` | `hf_knowledge_info` |

교재 JSONL 2개는 별도 전달한 로컬 패키지에서 그대로 복사한다. GitHub에 코퍼스 본문이 없다. 파일을 다시 저장하거나 인코딩/줄바꿈을 바꾸면 SHA-256 검사가 실패할 수 있다. 새 컬렉션은 `rag_corpus`를 변경하지 않는다. SIR-Lab은 원본 ID 일치와 상위 원출처 이용조건이 확인된 경우에만 선택적으로 추가한다.

## Windows / VS Code PowerShell
```powershell
git fetch origin
git switch feature/hf-mental-health-knowledge-rag
git pull origin feature/hf-mental-health-knowledge-rag
python -m pip install -r requirements.txt
python scripts/build_who_knowledge_index.py --validate-only
python scripts/build_openstax_knowledge_index.py --validate-only
python scripts/build_who_knowledge_index.py --replace
python scripts/build_openstax_knowledge_index.py --permission-confirmed --replace
python eval/evaluate_textbooks.py --k 5
```

기존 DB가 로컬에 구성되어 있는 경우에만 통합 호출:
```python
from rag import retrieve
results = retrieve('시험 스트레스 때문에 불안해요', k=5, n_textbook=1, n_openstax=2)
for r in results:
    print(r['id'], r['score'], r.get('source_url'), r.get('printed_page_start'))
```

`n_textbook`는 WHO, `n_openstax`는 OpenStax, `n_expert`는 SIR-Lab 슬롯. 합계는 k 이하. 모두 0인 기본 동작은 기존 case/info 검색과 동일하다. 각 컬렉션의 고정 슬롯 결과를 합쳐 cosine 점수로 정렬하므로, 전역 top-k나 임상적 관련성 보장은 아니다.

## 평가 및 팀 통합
- `eval/textbook_questions.jsonl`: 주제별 3개, 총 21개 신규 한국어 질문. 정답 청크는 출처 기반 초안으로, 독립적 사람 검토 필요.
- `eval/evaluate_textbooks.py`: WHO 단독, OpenStax 단독, 두 교재 통합의 Hit@K, Precision@K, Recall@K, MRR@K를 계산한다. `data/processed/textbook_eval.json`에 저장. 두 교재 정답이 없는 단독 모드는 해당 질문을 제외한다.
- 기존 `eval/questions.jsonl`을 그대로 유지한다. 기존 코퍼스와 확장 코퍼스 비교 시 공통 질문에 모든 코퍼스의 관련성 라벨을 추가해야 한다. 새 평가만으로 개선을 주장하지 않는다.
- 사용자 실제 상담 발화와 외부 API 전송은 이용조건 및 개인정보 검토 후에만 연결. 기존 데모와 파인튜닝 모델은 자동 변경하지 않음.

## 상태
40개 OpenStax + 20개 WHO 로컬 JSONL 생성. PDF 판본 및 파일 무결성 검사와 출처 메타데이터 확인. 모델 다운로드/실제 임베딩/Chroma 생성 및 검색 성능은 현 환경에서 실행하지 못했으므로 로컬 실행 결과로 검증 필요.
