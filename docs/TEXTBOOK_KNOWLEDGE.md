# WHO / OpenStax 교재 기반 심리교육 RAG (2026-11-25)

## WHO — 로컬 비상업적 연구용

- 업로드 PDF: Psychological self-help interventions (2026), generic field-trial version 1.0, 156 PDF pages.
- License: CC BY-NC-SA 3.0 IGO. Source: https://www.who.int/publications/i/item/9789240120785
- 20 Korean educational adaptations of selected concepts from Chapters 1, 5 and 6, with section, PDF page, printed page, attribution and translation disclaimer.
- Counts: stress management 4; emotion regulation 4; CBT-related behavioral skills 4; relationships 2; anxiety/depression/sleep 1 (warning signs, not sleep); self-esteem/acceptance 2; professional help 3.
- Step-by-Step emphasizes behavioral activation; Doing What Matters is ACT-based, not classic CBT. Do not represent this educational adaptation as a clinically tested chatbot or WHO endorsement.
- Original PDFs, adapted text, and vector DB remain local; the approved JSONL is supplied separately.
- Source review is not clinical expert review. No treatment protocols, risk assessment forms, or staff scripts are indexed.

## Local execution

Copy the unchanged supplied file `who_self_help_selected.jsonl` into `data/processed/`. Its SHA-256 is verified by the indexer. The file is a noncommercial adaptation under CC BY-NC-SA 3.0 IGO.

```powershell
git fetch origin
git switch feature/hf-mental-health-knowledge-rag
git pull origin feature/hf-mental-health-knowledge-rag
python -m pip install -r requirements.txt
python scripts/build_who_knowledge_index.py --validate-only
python scripts/build_who_knowledge_index.py --replace
```

Collection: `who_self_help_2026`; directory: `%USERPROFILE%\.cache\mbti-rag\who_knowledge_chroma` (override: `RAG_WHO_KNOWLEDGE_DIR`). The existing case/info and HF knowledge indexes are preserved.

```python
from rag.retrieve import embed
from rag.who_knowledge import search_who
print(search_who(embed(['기분이 가라앉아서 아무것도 하기 싫어요']), 3))
```

With the original case/info index available:

```python
from rag import retrieve
results = retrieve('시험 스트레스 때문에 불안해요', k=5, n_textbook=2)
```

`n_textbook=0` is default; the existing demo is unchanged. Use `n_expert=1` only if the separate HF index is also present and rights have been checked. Keep research local, do not send retrieved copyrighted or sensitive text to external model APIs without clearance.

## OpenStax — pending authorization

Uploaded 2026 edition PDF: 755 pages, CC BY-NC-SA 4.0. Twenty relevant section review ranges are recorded in [openstax_selection.json](openstax_selection.json). The current online edition requests written permission for generative-AI ingestion (https://openstax.org/books/psychology-2e/pages/preface). Accordingly this implementation does not extract or embed OpenStax content until permission for this use is clarified. The textbook selection is not a RAG corpus.

## Validation status

WHO corpus structure, record count, SHA-256 and source metadata checked; actual BGE-M3 encoding and retrieval performance remain to be tested on the local machine. The GitHub index/search code has not yet been run end-to-end. New evaluation questions and relevance labels are required; do not claim improvement from record count alone.
