"""Compare local dense, BM25, hybrid RRF and cross-encoder retrieval on V2."""
import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from textbook_v2 import ROOT, DATA, DB, load
from evaluate_reranker import MODEL, metrics, rerank


def tokenize(text):
    """Korean syllable bigrams within words, English alphanumeric words."""
    text = unicodedata.normalize('NFKC', text).casefold()
    for token in re.findall(r'[가-힣]+|[a-z0-9]+', text):
        if re.fullmatch(r'[가-힣]+', token):
            if len(token) == 1:
                yield 'ko:' + token
            else:
                for i in range(len(token)-1):
                    yield 'ko:' + token[i:i+2]
        else:
            yield 'en:' + token


class BM25:
    def __init__(self, rows, k1=1.5, b=0.75):
        self.rows = {r['id']: r for r in rows}
        if not self.rows or len(self.rows) != len(rows):
            raise ValueError('Empty corpus or duplicate IDs')
        self.k1, self.b = k1, b
        self.freq = {uid: Counter(tokenize(r['text'])) for uid, r in self.rows.items()}
        self.length = {uid: sum(c.values()) for uid, c in self.freq.items()}
        self.avgdl = mean(self.length.values()) or 1.0
        df = Counter(term for freq in self.freq.values() for term in freq)
        n = len(rows)
        self.idf = {term: math.log1p((n-count+0.5)/(count+0.5)) for term, count in df.items()}

    def search(self, question, limit):
        terms = set(tokenize(question))
        scored = []
        for uid, freq in self.freq.items():
            norm = self.k1 * (1-self.b + self.b*self.length[uid]/self.avgdl)
            score = sum(self.idf[t]*freq[t]*(self.k1+1)/(freq[t]+norm)
                        for t in sorted(terms) if freq[t])
            # Do not include arbitrary zero-overlap documents.
            if score > 0:
                scored.append({'id': uid, 'text': self.rows[uid]['text'], 'bm25_score': score})
        return sorted(scored, key=lambda r: (-r['bm25_score'], r['id']))[:limit]


def fuse(dense, sparse, limit, constant=60, dense_weight=2.0):
    scores = defaultdict(float)
    items = {}
    dense_rank = {r['id']: i for i, r in enumerate(dense, 1)}
    for ranking, weight in ((dense, dense_weight), (sparse, 1.0)):
        seen = set()
        for i, row in enumerate(ranking, 1):
            uid = row['id']
            if uid in seen:
                raise ValueError('Duplicate ranking ID')
            seen.add(uid)
            items[uid] = {**items.get(uid, {}), **row}
            scores[uid] += weight/(constant+i)
    ids = sorted(items, key=lambda uid: (-scores[uid], dense_rank.get(uid, math.inf), uid))[:limit]
    return [{**items[uid], 'rrf_score': scores[uid]} for uid in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=int, default=15)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--rrf-constant', type=int, default=60)
    p.add_argument('--dense-weight', type=float, default=2.0)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--question-id')
    p.add_argument('--questions', type=Path, default=ROOT/'eval/textbook_questions.jsonl')
    p.add_argument('--output', type=Path)
    p.add_argument('--skip-rerank', action='store_true')
    a = p.parse_args()
    if not (1 <= a.k <= a.candidates and a.rrf_constant >= 1 and math.isfinite(a.dense_weight) and a.dense_weight > 0 and a.batch_size >= 1 and a.max_length >= 1):
        p.error('Invalid parameters')
    rows, digest = load()
    expected = {r['id'] for r in rows}
    if a.candidates > len(rows):
        p.error('Candidate count exceeds corpus size')
    question_bytes = a.questions.read_bytes()
    qs = [json.loads(l) for l in question_bytes.decode('utf-8-sig').splitlines() if l.strip()]
    if a.question_id:
        qs = [q for q in qs if q['id'] == a.question_id]
    if not qs or len({q['id'] for q in qs}) != len(qs):
        p.error('No matching questions or duplicate IDs')
    for q in qs:
        if not q['question'].strip() or not q['relevant_chunk_ids'] or not set(q['relevant_chunk_ids']) <= expected:
            p.error('Invalid question or gold labels: '+q['id'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = a.output or DATA/f'textbook_hybrid_c{a.candidates}_k{a.k}_{stamp}.json'
    if output.exists():
        p.error('Output already exists')
    import chromadb
    import torch
    from rag.retrieve import embed
    coll = chromadb.PersistentClient(path=str(DB)).get_collection('textbook_v2_'+digest[:16])
    if (coll.metadata or {}).get('corpus_sha256') != digest or set(coll.get(include=[])['ids']) != expected:
        raise RuntimeError('V2 DB mismatch; build complete V2 index first')
    t = perf_counter()
    bm25 = BM25(rows)
    bm25_build_seconds = perf_counter()-t
    print(f'Validated {len(rows)} V2 records; {len(qs)} questions', flush=True)
    model = None
    if not a.skip_rerank:
        from sentence_transformers import CrossEncoder
        if a.device == 'cuda' and not torch.cuda.is_available():
            p.error('CUDA unavailable')
        print(f'Loading {MODEL} locally on {a.device}', flush=True)
        model = CrossEncoder(MODEL, device=a.device, max_length=a.max_length, trust_remote_code=False)
    # Warm up embedding outside per-query timing.
    t = perf_counter()
    embed(['retrieval warmup'])
    warmup_seconds = perf_counter()-t
    def rank(question, candidates):
        scores = model.predict([(question,r['text']) for r in candidates], batch_size=a.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
        return rerank(candidates, scores.reshape(-1))
    details = []
    for i,q in enumerate(qs,1):
        question = q['question']
        t = perf_counter()
        raw = coll.query(query_embeddings=embed([question]), n_results=a.candidates, include=['documents','distances'])
        ids, docs, dist = (raw[key][0] for key in ('ids','documents','distances'))
        if len(ids) != a.candidates or len(set(ids)) != len(ids) or not len(ids)==len(docs)==len(dist):
            raise RuntimeError('Invalid dense candidate list')
        dense = [{'id':uid,'text':doc,'dense_score':1-float(d)} for uid,doc,d in zip(ids,docs,dist)]
        dense_time = perf_counter()-t
        t = perf_counter()
        sparse = bm25.search(question,a.candidates)
        sparse_time = perf_counter()-t
        t = perf_counter()
        hybrid = fuse(dense,sparse,a.candidates,a.rrf_constant,a.dense_weight)
        fusion_time = perf_counter()-t
        modes = {'dense':dense,'bm25':sparse,'hybrid_rrf':hybrid}
        times = {'dense':dense_time,'bm25':sparse_time,'rrf_extra':fusion_time,'hybrid_total':dense_time+sparse_time+fusion_time}
        if model is not None:
            t=perf_counter(); modes['dense_rerank']=rank(question,dense); times['dense_rerank_extra']=perf_counter()-t
            t=perf_counter(); modes['hybrid_rerank']=rank(question,hybrid); times['hybrid_rerank_extra']=perf_counter()-t
            times['dense_rerank_total']=dense_time+times['dense_rerank_extra']
            times['hybrid_rerank_total']=times['hybrid_total']+times['hybrid_rerank_extra']
        # Relevance labels never used to choose or rank candidates.
        gold=set(q['relevant_chunk_ids'])
        results={}
        for name,ranking in modes.items():
            ranked_ids=[r['id'] for r in ranking]
            results[name]={'ids':ranked_ids[:a.k],'candidate_ids':ranked_ids,'returned_count':min(a.k,len(ranked_ids)),
                           'metrics':metrics(ranked_ids,gold,a.k),'candidate_metrics':metrics(ranked_ids,gold,a.candidates)}
        details.append({'id':q['id'],'category':q['category'],'seconds':times,'results':results})
        print(f'[{i}/{len(qs)}] {q["id"]} '+ ' '.join(f'{name}={r["metrics"]["hit"]:.0f}' for name,r in results.items()),flush=True)
    modes=list(details[0]['results'])
    def aggregate(key):
        return {name:{metric:mean(d['results'][name][key][metric] for d in details) for metric in ('hit','precision','recall','mrr')} for name in modes}
    summary=aggregate('metrics')
    report={'status':'draft_relevance_labels_not_clinical_validation','corpus_digest':digest,'questions_sha256':hashlib.sha256(question_bytes).hexdigest(),
            'label_file':str(a.questions),'n_questions':len(qs),'k':a.k,'candidates':a.candidates,'rrf_constant':a.rrf_constant,'dense_weight':a.dense_weight,
            'bm25':{'k1':1.5,'b':0.75,'tokenizer':'Hangul syllable bigrams and English alphanumeric words'},
            'reranker_model':MODEL if model is not None else None,'device':a.device,'batch_size':a.batch_size,'max_length':a.max_length,
            'embedding_warmup_seconds':warmup_seconds,'bm25_build_seconds':bm25_build_seconds,'summary':summary,'candidate_summary':aggregate('candidate_metrics'),
            'mean_seconds':{key:mean(d['seconds'][key] for d in details) for key in details[0]['seconds']},
            'mean_returned':{name:mean(d['results'][name]['returned_count'] for d in details) for name in modes},
            'delta_vs_dense_rerank':{key:summary['hybrid_rerank'][key]-summary['dense_rerank'][key] for key in summary['dense_rerank']} if model is not None else None,
            'details':details}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({key:report[key] for key in ('summary','candidate_summary','mean_seconds','mean_returned','delta_vs_dense_rerank')},indent=2))
    print('Report:',output)

if __name__=='__main__':
    main()
