"""Compare dense retrieval and local cross-encoder reranking on the existing V2 DB."""
import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from statistics import mean

from textbook_v2 import ROOT, DATA, DB, load

MODEL = 'BAAI/bge-reranker-v2-m3'


def metrics(ids, gold, k):
    ids = ids[:k]
    hits = sum(i in gold for i in ids)
    return {'hit': float(hits > 0), 'precision': hits / k,
            'recall': hits / len(gold),
            'mrr': next((1 / j for j, i in enumerate(ids, 1) if i in gold), 0.0)}


def rerank(candidates, scores):
    if len(scores) != len(candidates):
        raise RuntimeError('Unexpected reranker score count')
    ranked = []
    for candidate, score in zip(candidates, scores):
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeError('Non-finite reranker score')
        ranked.append({**candidate, 'reranker_score': score})
    # Python's stable sort preserves dense order for ties.
    return sorted(ranked, key=lambda row: row['reranker_score'], reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=int, default=15)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--question-id', help='Run one evaluation question as a smoke test')
    p.add_argument('--questions', type=Path, default=ROOT/'eval/textbook_questions.jsonl')
    p.add_argument('--output', type=Path, help='New report path (will not overwrite)')
    a = p.parse_args()
    if not (1 <= a.k <= a.candidates and a.batch_size >= 1 and a.max_length >= 1):
        p.error('Require 1 <= k <= candidates, positive batch size and max length')
    rows, digest = load()
    if a.candidates > len(rows):
        p.error('Candidate count exceeds corpus size')
    expected_ids = {r['id'] for r in rows}
    qs = [json.loads(l) for l in a.questions.read_text(encoding='utf-8').splitlines() if l.strip()]
    if a.question_id:
        qs = [q for q in qs if q['id'] == a.question_id]
    if not qs or len({q['id'] for q in qs}) != len(qs):
        p.error('No matching questions or duplicate question IDs')
    for q in qs:
        gold = set(q['relevant_chunk_ids'])
        if not gold or not gold <= expected_ids or not q['question'].strip():
            p.error('Invalid gold labels or question: ' + q['id'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = a.output or DATA/f'textbook_rerank_c{a.candidates}_k{a.k}_{stamp}.json'
    if output.exists():
        p.error('Output already exists; select a new path')

    import chromadb
    import torch
    import sentence_transformers
    from rag.retrieve import embed
    from sentence_transformers import CrossEncoder
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA not available; use --device cpu')
    coll = chromadb.PersistentClient(path=str(DB)).get_collection('textbook_v2_'+digest[:16])
    if (coll.metadata or {}).get('corpus_sha256') != digest or set(coll.get(include=[])['ids']) != expected_ids:
        raise RuntimeError('V2 DB mismatch. Build the complete V2 index first.')
    print(f'Validated {len(rows)} V2 records; {len(qs)} questions', flush=True)
    print(f'Loading {MODEL} locally on {a.device}', flush=True)
    start = perf_counter()
    model = CrossEncoder(MODEL, device=a.device, max_length=a.max_length, trust_remote_code=False)
    model_load_seconds = perf_counter() - start
    details = []
    for idx, q in enumerate(qs, 1):
        gold = set(q['relevant_chunk_ids'])
        t = perf_counter()
        result = coll.query(query_embeddings=embed([q['question']]), n_results=a.candidates,
                            include=['documents', 'distances'])
        ids, docs, distances = (result[key][0] for key in ('ids', 'documents', 'distances'))
        if len(ids) != a.candidates or len(set(ids)) != len(ids) or not len(ids)==len(docs)==len(distances):
            raise RuntimeError('Incomplete or duplicate candidate list: '+q['id'])
        candidates = [{'id': uid, 'text': doc, 'dense_score': 1-float(dist), 'dense_rank': i}
                      for i, (uid, doc, dist) in enumerate(zip(ids, docs, distances), 1)]
        dense_seconds = perf_counter() - t
        t = perf_counter()
        scores = model.predict([(q['question'], c['text']) for c in candidates],
                               batch_size=a.batch_size, show_progress_bar=False, convert_to_numpy=True)
        ranked = rerank(candidates, scores.reshape(-1))
        rerank_seconds = perf_counter() - t
        reordered = [r['id'] for r in ranked]
        # Only IDs/scores saved; no raw copyrighted text or user queries in reports.
        detail = {'id': q['id'], 'category': q['category'],
                  'candidate_metrics': metrics(ids, gold, a.candidates),
                  'dense': {'ids': ids[:a.k], 'metrics': metrics(ids, gold, a.k)},
                  'reranked': {'ids': reordered[:a.k], 'metrics': metrics(reordered, gold, a.k)},
                  'dense_seconds': dense_seconds, 'rerank_seconds': rerank_seconds,
                  'candidate_scores': [{k:v for k,v in r.items() if k != 'text'} for r in ranked],
                  'new_unjudged_dense_ids': [i for i in ids[:a.k] if '-v2-' in i and i not in gold],
                  'new_unjudged_reranked_ids': [i for i in reordered[:a.k] if '-v2-' in i and i not in gold]}
        details.append(detail)
        print(f'[{idx}/{len(qs)}] {q["id"]}: dense hit={detail["dense"]["metrics"]["hit"]:.0f}, rerank hit={detail["reranked"]["metrics"]["hit"]:.0f}, rerank={rerank_seconds:.2f}s', flush=True)
    summary = {mode: {key: mean(d[mode]['metrics'][key] for d in details)
                      for key in ('hit','precision','recall','mrr')} for mode in ('dense','reranked')}
    candidate_summary = {key: mean(d['candidate_metrics'][key] for d in details) for key in ('hit','precision','recall','mrr')}
    delta = {key: summary['reranked'][key]-summary['dense'][key] for key in summary['dense']}
    report = {'status':'draft_relevance_labels_not_clinical_validation',
              'label_file': str(a.questions), 'corpus_digest': digest, 'n_questions': len(qs),
              'candidates': a.candidates, 'k': a.k, 'model': MODEL, 'device': a.device,
              'max_length': a.max_length, 'batch_size': a.batch_size,
              'versions': {'torch': str(torch.__version__), 'sentence_transformers': sentence_transformers.__version__, 'chromadb': chromadb.__version__},
              'model_load_seconds': model_load_seconds,
              'mean_dense_seconds': mean(d['dense_seconds'] for d in details),
              'mean_rerank_seconds': mean(d['rerank_seconds'] for d in details),
              'candidate_summary': candidate_summary, 'summary': summary, 'delta': delta, 'details': details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'candidate_summary': candidate_summary, 'summary': summary, 'delta': delta,
                      'mean_dense_seconds': report['mean_dense_seconds'], 'mean_rerank_seconds': report['mean_rerank_seconds']}, indent=2))
    print('Report:', output)


if __name__ == '__main__':
    main()
