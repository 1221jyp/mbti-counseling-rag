# 교재 RAG V2 확장 (로컬 연구용)

V1 WHO 20개 + OpenStax 40개 보존. 추가 WHO 6개, OpenStax 35개. 총 101개.
추가 자료는 제공받은 PDF에서 선택한 원문 문단 및 한국어 요약이다. 임상 검증이나 치료 효과 검증은 하지 않았다.
OpenStax PDF의 7~11쪽 목차와 선택 페이지, WHO PDF의 17, 76, 93, 94, 98쪽을 확인했다. 원문 페이지와 저작권 정보를 각 레코드에 기록했다.
WHO 추가 자료는 기존 자기 도움 기술을 보완하는 요약이고 공식 번역은 아니다. 약물 및 위기 대응 프로토콜은 추가하지 않았다.
기존 V1 파일과 코드를 수정하지 않는다. OpenStax 재사용 허가가 V2의 추가 선별/임베딩까지 포함하는지 확인한 경우에만 build를 실행한다. 비상업 학술 연구용, 공개 업로드 및 외부 API 전송 금지(별도 권한 확인 전).

## 사용
ZIP을 프로젝트 루트에 풀어 기존 파일은 보존하고 새 파일 3개를 추가한다.

```powershell
python scripts/textbook_v2.py validate
python scripts/textbook_v2.py build --permission-confirmed
python scripts/textbook_v2.py evaluate --k 5
```

별도 벡터 DB: `~/.cache/mbti-rag/textbook_v2_chroma`. V1 DB는 그대로 유지된다.
평가는 기존 21문항 정답 ID만 사용한다. 새 청크는 정답 여부가 미판정이므로 기존 정답 유지 여부를 측정하는 임시 평가이며 실제 V2 성능을 확정할 수 없다. 추후 신규 청크의 관련성 라벨을 보완하고 독립 평가를 수행한다.
기존 챗봇은 아직 V2 DB에 연결되지 않는다. 로컬 파일/정적 검증 완료; 임베딩 및 검색 평가 실행은 사용자의 환경에서 필요하다.
