"""Local-only retrieval experiment. Never sends documents to an LLM service."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import math
import os

MODEL = 'BAAI/bge-reranker-v2-m3'
CACHE = Path.home() / '.cache' / 'mbti-rag'
BASE_DB = Path(os.environ.get('RAG_CHROMA_DIR', str(CACHE/'chroma')))
PRIVATE_DB = Path(os.environ.get('RAG_COUNSELING_DIR', str(CACHE/'counseling_chroma')))


def existing_collection(path, name):
    # Avoid creating an empty database when the path is missing.
    if not (Path(path)/'chroma.sqlite3').is_file():
        raise FileNotFoundError(f'Chroma DB missing: {path}')
    import chromadb
    coll = chromadb.PersistentClient(path=str(path)).get_collection(name)
    return coll


@lru_cache(maxsize=1)
def textbook_collection():
    from scripts.textbook_v2 import DB, load
    rows, digest = load()
    coll = existing_collection(DB, 'textbook_v2_'+digest[:16])
    if ((coll.metadata or {}).get('corpus_sha256') != digest or
            set(coll.get(include=[])['ids']) != {r['id'] for r in rows}):
        raise RuntimeError('V2 manifest/DB mismatch; run textbook_v2.py validate/build first')
    return coll


def inventory_collection(coll):
    counts = Counter()
    for offset in range(0, coll.count(), 500):
        result = coll.get(limit=500, offset=offset, include=['metadatas'])
        for meta in result['metadatas']:
            meta = meta or {}
            counts[(str(meta.get('doc_type','unknown')), str(meta.get('source','unknown')))] += 1
    return {'count': coll.count(), 'groups': [dict(doc_type=t, source=s, count=n) for (t,s),n in sorted(counts.items())]}


def inventory():
    result = {}
    for label, opener in [('textbook_v2', textbook_collection),
                          ('legacy', lambda: existing_collection(BASE_DB,'rag_corpus')),
                          ('private_cases', lambda: existing_collection(PRIVATE_DB,'counseling_cases'))]:
        try:
            result[label] = {'status':'available', **inventory_collection(opener())}
        except Exception as exc:
            result[label] = {'status':'unavailable', 'reason':f'{type(exc).__name__}: {exc}'}
    return result


def candidates(coll, embedding, limit, group, doc_type=None):
    kw = {'where': {'doc_type':doc_type}} if doc_type else {}
    count = len(coll.get(include=[], **kw)['ids']) if doc_type else coll.count()
    if count == 0 or limit == 0:
        return []
    out = coll.query(query_embeddings=embedding, n_results=min(limit,count),
                     include=['documents','metadatas','distances'], **kw)
    rows=[]
    for uid, text, meta, dist in zip(*(out[k][0] for k in ('ids','documents','metadatas','distances'))):
        if not isinstance(text,str) or not text.strip() or not math.isfinite(float(dist)):
            continue
        rows.append({'id':uid,'text':text,'metadata':meta or {},'group':group,'dense_score':1-float(dist)})
    return rows


@lru_cache(maxsize=2)
def get_reranker(device):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL, device=device, max_length=512, trust_remote_code=False)


def rank(query, rows, model, batch_size=2):
    if not rows or model is None:
        return rows
    scores = model.predict([(query,r['text']) for r in rows], batch_size=batch_size,
                           show_progress_bar=False, convert_to_numpy=True).reshape(-1)
    if len(scores)!=len(rows) or not all(math.isfinite(float(s)) for s in scores):
        raise RuntimeError('Invalid reranker scores')
    return sorted([{**r,'reranker_score':float(s)} for r,s in zip(rows,scores)],
                  key=lambda r:r['reranker_score'],reverse=True)


def select(knowledge, cases, portal, k=5, case_slots=2, portal_slots=0):
    """Quota is an experiment, not validated; never fill missing evidence with cases."""
    if not (1<=k and 0<=case_slots and 0<=portal_slots and case_slots+portal_slots<k):
        raise ValueError('At least one textbook slot is required')
    if not knowledge:
        raise RuntimeError('No textbook candidates; refuse case-only retrieval')
    p = portal[:portal_slots]
    c = cases[:case_slots]
    return knowledge[:k-len(p)-len(c)] + p + c


def retrieve_integrated(query, *, mode='accurate', case_source='none', case_slots=2,
                        portal_slots=0, k=5, candidate_k=15, device='cpu',
                        batch_size=2, permission_confirmed=False):
    """Restricted local experiment. Returns texts in memory; do not log/share them.

    Default = evaluated textbook-only path. Case/portal use is opt-in after
    applicable license, privacy and safety review. Case documents are examples,
    not clinical evidence. No MBTI input to retrieval.
    """
    if not isinstance(query,str) or not query.strip(): raise ValueError('Empty query')
    if mode not in ('accurate','fast') or case_source not in ('none','legacy','private'):
        raise ValueError('Invalid mode or case source')
    effective_cases = case_slots if case_source != 'none' else 0
    if not 1<=k<=candidate_k or min(effective_cases,portal_slots)<0 or effective_cases+portal_slots>=k:
        raise ValueError('Invalid k, candidate_k or quotas')
    if not permission_confirmed:
        raise PermissionError('Confirm local use rights and review first')
    # Resolve all requested sources before loading the embedding model.
    textbook = textbook_collection()
    legacy = existing_collection(BASE_DB,'rag_corpus') if case_source=='legacy' or portal_slots else None
    private = existing_collection(PRIVATE_DB,'counseling_cases') if case_source=='private' else None
    from rag.retrieve import embed
    vector=embed([query.strip()])
    knowledge=candidates(textbook,vector,candidate_k,'textbook')
    cases=[]
    if case_source=='legacy' and effective_cases:
        cases=candidates(legacy,vector,candidate_k,'case','case')
    elif case_source=='private' and effective_cases:
        cases=candidates(private,vector,candidate_k,'case','counseling_case')
    portal=candidates(legacy,vector,candidate_k,'portal','info') if portal_slots else []
    model=get_reranker(device) if mode=='accurate' else None
    return select(rank(query,knowledge,model,batch_size),rank(query,cases,model,batch_size),
                  rank(query,portal,model,batch_size),k,effective_cases,portal_slots)


def public_metadata(rows):
    """Console output only: IDs/scores/categories, never text or user query."""
    return [{**{k:r[k] for k in ('id','group','dense_score','reranker_score') if k in r},
             'source':r['metadata'].get('source'), 'category':r['metadata'].get('category')}
            for r in rows]
