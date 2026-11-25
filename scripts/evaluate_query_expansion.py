"""Local V2 ablation: dense, rerank, query expansion, RRF and combined retrieval."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from pathlib import Path
from collections import defaultdict

from textbook_v2 import ROOT, DATA, DB, load
from evaluate_reranker import MODEL, metrics, rerank

# Broad educational vocabulary; no evaluation IDs, categories or gold labels.
RULES = (
    ('stress', ('스트레스','압박','부담','과로','지쳤','긴장','시험'), '스트레스 평가 대처 전략 stress appraisal coping'),
    ('emotion', ('감정','화가','분노','기분','짜증','울컥','슬픔','감정적'), '감정 인식 감정 조절 정서 수용 emotion regulation emotional awareness acceptance'),
    ('thoughts', ('생각','실패','최악','단정','걱정','완벽','부정적','후회'), '자동적 사고 인지 왜곡 인지 재구성 cognitive distortions automatic thoughts cognitive restructuring'),
    ('relationships', ('친구','가족','부모','연인','관계','동료','갈등','사람들','외로','소통'), '사회적 지지 의사소통 대인관계 social support interpersonal communication'),
    ('anxiety', ('불안','두려','무서','초조','공포','떨려','긴장'), '불안 대처 불확실성 anxiety coping grounding'),
    ('sleep', ('잠','수면','불면','새벽','피곤'), '수면 습관 불면 스트레스 sleep habits insomnia'),
    ('low_mood', ('우울','무기력','의욕','아무것도','즐겁지','기운'), '기분과 활동 작은 행동 behavioral activation mood activity'),
    ('self', ('자존감','자신감','자기','나 자신','내 자신','내가 싫','못난','비교','가치','부족','능력'), '자기개념 자기효능감 자기수용 self concept self efficacy self acceptance'),
    ('help', ('상담','전문가','치료','병원','도움','심리상담','정신건강'), '전문가 도움 기준 심리교육 professional support help seeking psychotherapy'),
)


def expand(question, max_expansions=3):
    q = question.casefold()
    matches = []
    for i, (name, words, terms) in enumerate(RULES):
        hits = sum(word.casefold() in q for word in words)
        if hits:
            matches.append((-hits, i, name, terms))
    matches.sort()
    return [(name, question + '\n관련 검색어: ' + terms) for _, _, name, terms in matches[:max_expansions]]


def fuse(rankings, limit, constant=60, original_weight=2.0):
    scores = defaultdict(float)
    original_rank = {r['id']: i for i, r in enumerate(rankings[0], 1)}
    items = {}
    for j, ranking in enumerate(rankings):
        seen = set()
        for i, row in enumerate(ranking, 1):
            uid = row['id']
            if uid in seen:
                raise ValueError('Duplicate ID in a ranking')
            seen.add(uid)
            items.setdefault(uid, row)
            scores[uid] += (original_weight if j == 0 else 1.0) / (constant + i)
    ids = sorted(items, key=lambda uid: (-scores[uid], original_rank.get(uid, math.inf), uid))[:limit]
    return [{**items[uid], 'rrf_score': scores[uid]} for uid in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=int, default=15)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--max-expansions', type=int, default=3)
    p.add_argument('--rrf-constant', type=int, default=60)
    p.add_argument('--original-weight', type=float, default=2.0)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--question-id')
    p.add_argument('--questions', type=Path, default=ROOT/'eval/textbook_questions.jsonl')
    p.add_argument('--output', type=Path)
    p.add_argument('--skip-rerank', action='store_true', help='Fast retrieval-only test')
    a = p.parse_args()
    if not (1 <= a.k <= a.candidates and a.max_expansions >= 1 and a.rrf_constant >= 1 and math.isfinite(a.original_weight) and a.original_weight > 0 and a.batch_size >= 1 and a.max_length >= 1):
        p.error('Invalid parameters')
    rows, digest = load()
    expected = {r['id'] for r in rows}
    if a.candidates > len(rows):
        p.error('Candidate count exceeds corpus size')
    qs = [json.loads(l) for l in a.questions.read_text(encoding='utf-8').splitlines() if l.strip()]
    if a.question_id:
        qs = [q for q in qs if q['id'] == a.question_id]
    if not qs or len({q['id'] for q in qs}) != len(qs):
        p.error('No matching questions or duplicate IDs')
    for q in qs:
        if not q['question'].strip() or not q['relevant_chunk_ids'] or not set(q['relevant_chunk_ids']) <= expected:
            p.error('Invalid evaluation question or labels: ' + q['id'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = a.output or DATA/f'textbook_expansion_c{a.candidates}_k{a.k}_{stamp}.json'
    if output.exists():
        p.error('Output exists; choose a new path')
    import chromadb
    import torch
    from rag.retrieve import embed
    coll = chromadb.PersistentClient(path=str(DB)).get_collection('textbook_v2_'+digest[:16])
    if (coll.metadata or {}).get('corpus_sha256') != digest or set(coll.get(include=[])['ids']) != expected:
        raise RuntimeError('V2 DB mismatch; build the complete V2 index first')
    print(f'Validated {len(rows)} V2 records; {len(qs)} questions', flush=True)
    model = None
    if not a.skip_rerank:
        from sentence_transformers import CrossEncoder
        if a.device == 'cuda' and not torch.cuda.is_available():
            p.error('CUDA unavailable')
        print(f'Loading {MODEL} locally on {a.device}', flush=True)
        model = CrossEncoder(MODEL, device=a.device, max_length=a.max_length, trust_remote_code=False)

    def search(queries):
        raw = coll.query(query_embeddings=embed(queries), n_results=a.candidates, include=['documents','distances'])
        out = []
        for ids, docs, distances in zip(raw['ids'], raw['documents'], raw['distances']):
            if len(ids) != a.candidates or len(set(ids)) != len(ids) or not len(ids)==len(docs)==len(distances):
                raise RuntimeError('Incomplete or duplicate candidate list')
            out.append([{'id': uid, 'text': doc, 'dense_score': 1-float(dist)} for uid, doc, dist in zip(ids, docs, distances)])
        if len(out) != len(queries):
            raise RuntimeError('Unexpected query count')
        return out

    def rank(question, candidates):
        scores = model.predict([(question, c['text']) for c in candidates], batch_size=a.batch_size, show_progress_bar=False, convert_to_numpy=True)
        return rerank(candidates, scores.reshape(-1))

    details = []
    for i, q in enumerate(qs, 1):
        question = q['question']
        t = perf_counter()
        base = search([question])[0]
        dense_seconds = perf_counter()-t
        t = perf_counter()
        variants = expand(question, a.max_expansions)
        extra = search([query for _, query in variants]) if variants else []
        single = extra[0] if extra else base
        fused = fuse([base]+extra, a.candidates, a.rrf_constant, a.original_weight)
        expansion_seconds = perf_counter()-t
        modes = {'dense': base, 'expanded_single': single, 'expansion_rrf': fused}
        times = {'dense': dense_seconds, 'expanded_single_and_rrf_extra': expansion_seconds}
        if model is not None:
            t = perf_counter()
            modes['dense_rerank'] = rank(question, base)
            times['dense_rerank_extra'] = perf_counter()-t
            t = perf_counter()
            modes['expansion_rrf_rerank'] = rank(question, fused)
            times['expansion_rrf_rerank_extra'] = perf_counter()-t
        # Gold labels only consulted for evaluation after ranking is finished.
        gold = set(q['relevant_chunk_ids'])
        result = {}
        for mode, ranking in modes.items():
            ids = [r['id'] for r in ranking]
            result[mode] = {'ids': ids[:a.k], 'candidate_ids': ids, 'metrics': metrics(ids, gold, a.k),
                            'candidate_metrics': metrics(ids, gold, a.candidates),
                            'new_unjudged_ids': [uid for uid in ids[:a.k] if '-v2-' in uid and uid not in gold]}
        details.append({'id': q['id'], 'category': q['category'], 'expansion_rules': [name for name,_ in variants], 'seconds': times, 'results': result})
        print(f'[{i}/{len(qs)}] {q["id"]} rules={",".join(name for name,_ in variants) or "none"} '+ ' '.join(f'{m}={r["metrics"]["hit"]:.0f}' for m,r in result.items()), flush=True)
    modes = list(details[0]['results'])
    summary = {mode: {key: mean(d['results'][mode]['metrics'][key] for d in details) for key in ('hit','precision','recall','mrr')} for mode in modes}
    candidate_summary = {mode: {key: mean(d['results'][mode]['candidate_metrics'][key] for d in details) for key in ('hit','precision','recall','mrr')} for mode in modes}
    deltas = {mode: {key: val-summary['dense'][key] for key,val in vals.items()} for mode,vals in summary.items() if mode!='dense'}
    report = {'status':'draft_old_labels_not_clinical_validation', 'corpus_digest':digest, 'questions_sha256':hashlib.sha256(a.questions.read_bytes()).hexdigest(), 'label_file':str(a.questions), 'n_questions':len(details), 'k':a.k, 'candidates':a.candidates, 'max_expansions':a.max_expansions, 'rrf_constant':a.rrf_constant, 'original_weight':a.original_weight, 'reranker_model':MODEL if model is not None else None, 'device':a.device, 'batch_size':a.batch_size, 'max_length':a.max_length, 'summary':summary, 'candidate_summary':candidate_summary, 'delta_vs_dense':deltas, 'mean_seconds':{key:mean(d['seconds'][key] for d in details) for key in details[0]['seconds']}, 'details':details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({key:report[key] for key in ('summary','candidate_summary','delta_vs_dense','mean_seconds')},indent=2))
    print('Report:',output)

if __name__=='__main__':
    main()
