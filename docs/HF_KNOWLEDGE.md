# SIR-Lab Mental Health Knowledge — strict source-crosschecked RAG

Original dataset: https://huggingface.co/datasets/SIR-Lab/Mental_Health_Knowledge_Dataset

**As of 2026-11-25, only five exact Q&A pairs have been conceptually cross-checked against OpenStax Psychology 2e.** The remaining pairs are excluded by default. This is not expert clinical validation or verification of every medical claim.

## Selection policy

- Source: derived English Q&A from multiple textbooks; the original item-level citations are missing.
- Selection manifest: `docs/hf_knowledge_verified_manifest.json` contains exact Q&A IDs, official evidence URLs and relevant textbook sections. It contains no raw Q&A.
- Verification level: `concept_source_correlated_not_clinically_validated`.
- Evidence matched: problem-focused coping and emotion-focused coping (OpenStax 14.4); social support (OpenStax 14.4); CBT for stress (OpenStax 16.2); job burnout (OpenStax 14.2).
- The original dataset's displayed license is CC BY-NC-SA 4.0; individual source textbook terms and attribution still require checking.
- All other Q&A are removed from the *embedding output*, not deleted from the original source repository.

## Windows VS Code PowerShell

Run from repository root, on branch `feature/hf-mental-health-knowledge-rag`:

```powershell
git fetch origin
git switch feature/hf-mental-health-knowledge-rag
git pull origin feature/hf-mental-health-knowledge-rag
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
python -m unittest discover -s tests -p "test_hf_knowledge.py" -v
```

### Step 1: download and filter to source-crosschecked items

```powershell
python scripts/build_hf_knowledge_corpus.py --download
```

- Output: `data/processed/hf_knowledge_approved.jsonl`, containing only exact hash-matched, source-crosschecked Q&A.
- Output statistics report how many entries were removed from embedding.
- If the underlying source has changed so that no evidence-list IDs match, the script fails without modifying the output.
- Each included record has `evidence_url`, `evidence_section`, `review_status`, `verification_level`, and `text_sha256`.

### Step 2: embed with BGE-M3 and index

```powershell
python scripts/build_hf_knowledge_index.py
```

If you built the old experimental collection previously, replace **only that experimental collection** (removing unverified entries):

```powershell
python scripts/build_hf_knowledge_index.py --replace
```

Collection: `hf_knowledge_info`; default path: `%USERPROFILE%\.cache\mbti-rag\hf_knowledge_chroma`.
The pre-existing `rag_corpus` collection and case/info corpus are not changed.
The indexer rejects any record not present in the evidence manifest, changed Q&A IDs or text digest, missing citations, and incorrect verification status.
The staging collection is prepared before replacing the old experiment collection.

### Step 3: verify stored count and retrieval

```powershell
python -c "import chromadb; from pathlib import Path; c=chromadb.PersistentClient(path=str(Path.home()/'.cache/mbti-rag/hf_knowledge_chroma')); print(c.get_collection('hf_knowledge_info').count())"
python -c "from rag.retrieve import embed; from rag.hf_knowledge import search_knowledge; print(*search_knowledge(embed(['시험 스트레스에 대처하려면?']), 3), sep='\n')"
```

With the existing local case/info DB present:

```python
from rag import retrieve
baseline = retrieve('시험 스트레스가 심해요', k=5)
with_expert = retrieve('시험 스트레스가 심해요', k=5, n_expert=1)
```

By default `n_expert=0` preserves the original retrieval logic and does not enable new knowledge in the demo automatically.

## Limitations

- Cross-checking general definitions against OpenStax is **not** clinician review or validation of individualized advice.
- It does not prove dataset-wide reliability, Korean retrieval quality, model safety, absence of demographic bias, or clinical efficacy.
- To expand beyond these five entries, a qualified human reviewer must compare each exact answer with an authoritative passage and document evidence in the manifest.
- Never push raw/processed Q&A or embeddings to public GitHub. Use only for permitted non-commercial, non-clinical research; respect source permissions and third-party privacy.

## Rights limitation (2026 OpenStax online edition)

Current OpenStax Psychology 2e web pages explicitly state that ingesting the book into generative AI offerings may require OpenStax permission. This integration stores **no OpenStax full text**, only URLs as reference evidence, but some HF Q&A may derive from earlier editions of OpenStax. The HF dataset license tag alone does not establish that all upstream rights are cleared. Confirm the source edition and AI/RAG rights before deployment or broader redistribution. Until then keep this strictly an offline, non-commercial academic experiment.
