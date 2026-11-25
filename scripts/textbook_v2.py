"""Build/evaluate a separate textbook V2 collection. No changes to V1."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'processed'
DB = Path(os.environ.get('RAG_TEXTBOOK_V2_DIR', str(Path.home()/'.cache/mbti-rag/textbook_v2_chroma')))

def load():
    manifest = json.loads((DATA/'textbook_v2_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for name, expected in manifest['files'].items():
        raw = (DATA/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'SHA256 mismatch: {name}; use the original files without editing')
        rows.extend(json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip())
    if len(rows) != manifest['total'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Count or unique ID check failed')
    for r in rows:
        if not r['text'].strip() or not r['source_url'] or not r['license']:
            raise ValueError('Missing text or attribution: '+r['id'])
        if 'text_sha256' in r and hashlib.sha256(r['text'].encode()).hexdigest() != r['text_sha256']:
            raise ValueError('Text hash mismatch: '+r['id'])
    digest = hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()
    return rows,digest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','build','evaluate'])
    p.add_argument('--permission-confirmed',action='store_true')
    p.add_argument('--k',type=int,default=5)
    a=p.parse_args()
    if a.k<1: p.error('k must be positive')
    rows,digest=load()
    print('Validated',len(rows),'records',dict(Counter(r['id'].split('-')[0] for r in rows)))
    if a.action=='validate':return
    if a.action=='build' and not a.permission_confirmed:
        p.error('Confirm permission for local OpenStax V2 extraction/embedding first.')
    import chromadb
    from rag.retrieve import embed
    client=chromadb.PersistentClient(path=str(DB))
    name='textbook_v2_'+digest[:16]
    if a.action=='build':
        coll=client.get_or_create_collection(name=name,metadata={'hnsw:space':'cosine','corpus_sha256':digest})
        if coll.metadata.get('corpus_sha256')!=digest: raise RuntimeError('Collection hash mismatch')
        for i in range(0,len(rows),16):
            batch=rows[i:i+16]
            docs=[r['text'] for r in batch]
            vectors=embed(docs)
            coll.upsert(ids=[r['id'] for r in batch],documents=docs,embeddings=vectors,metadatas=[{k:v for k,v in r.items() if k not in ('id','text') and isinstance(v,(str,int,float,bool))} for r in batch])
            print(f'Indexed {min(i+16,len(rows))}/{len(rows)}')
        if set(coll.get(include=[])['ids'])!={r['id'] for r in rows}:raise RuntimeError('Stored IDs mismatch')
        print('Stored',coll.count(),'in',DB,name)
        return
    coll=client.get_collection(name)
    if coll.metadata.get('corpus_sha256')!=digest or set(coll.get(include=[])['ids'])!={r['id'] for r in rows}:raise RuntimeError('Build the complete V2 index first')
    qs=[json.loads(l) for l in (ROOT/'eval/textbook_questions.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    details=[]
    for q in qs:
        gold=set(q['relevant_chunk_ids']);ids=coll.query(query_embeddings=embed([q['question']]),n_results=min(a.k,len(rows)),include=['distances'])['ids'][0]
        hits=sum(i in gold for i in ids)
        m={'hit':float(hits>0),'precision':hits/a.k,'recall':hits/len(gold),'mrr':next((1/j for j,i in enumerate(ids,1) if i in gold),0.)}
        details.append({'id':q['id'],'category':q['category'],'ids':ids,'metrics':m,'new_unjudged_ids':[i for i in ids if '-v2-' in i]})
    summary={key:sum(d['metrics'][key] for d in details)/len(details) for key in details[0]['metrics']}
    report={'status':'old_v1_labels_only_new_chunks_unjudged_not_clinical_validation','n_questions':len(details),'k':a.k,'summary':summary,'details':details}
    path=DATA/'textbook_eval_v2_old_labels.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Report:',path)
if __name__=='__main__':main()
