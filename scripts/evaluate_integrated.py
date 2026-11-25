"""Compare fixed quotas and dynamic filtering with cached candidates. Local only."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
from itertools import product
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.dynamic_retrieve import ranked_groups, fixed, dynamic, case_collection
from rag.integrated_retrieve import textbook_collection, get_reranker


def read_jsonl(p):
    rows = [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {p}')
    return rows


def metrics(ids, gold):
    if not gold: return None
    n = sum(i in gold for i in ids)
    return {'hit':float(n>0), 'precision_returned':n/len(ids) if ids else 0.,
            'recall':n/len(gold), 'mrr':next((1/j for j,i in enumerate(ids,1) if i in gold),0.)}


def assess(rows, q):
    t = [r['id'] for r in rows if r['group']=='textbook']
    c = [r['id'] for r in rows if r['group']=='case']
    tm, cm = metrics(t,q['textbook_gold']), metrics(c,q['case_gold'])
    gold = q['textbook_gold'] | q['case_gold']
    # No case judgments on textbook-only questions. Do not count them as irrelevant.
    joint = None if cm is None else {'both_hit':tm['hit']*cm['hit'],
            'precision_at_5':sum(r['id'] in gold for r in rows)/5,
            'precision_returned':sum(r['id'] in gold for r in rows)/len(rows) if rows else 0.}
    return {'textbook':tm,'case':cm,'joint':joint,'returned':len(rows),
            'textbook_returned':len(t),'case_returned':len(c),'abstained':float(not rows),
            'ids':[r['id'] for r in rows]}


def summarize(details):
    out = {}
    for group in ('textbook','case','joint'):
        values=[d[group] for d in details if d[group] is not None]
        if values: out[group]={'n':len(values), **{k:mean(v[k] for v in values) for k in values[0]}}
    out.update({k:mean(d[k] for d in details) for k in ('returned','textbook_returned','case_returned','abstained')})
    return out


def load_questions(tids,cids):
    legacy={q['id']:q for q in read_jsonl(ROOT/'eval/questions.jsonl')}
    spec=json.loads((ROOT/'eval/integrated_joint_labels.json').read_text(encoding='utf-8'))
    qs=[]
    for s in spec['questions']:
        q=legacy[s['id']]; tg=set(s['textbook_ids']); original={i for i in q['relevant_chunk_ids'] if i.startswith('case-')}
        cg=original & cids
        if not tg or not tg<=tids or not cg: raise ValueError('Missing gold: '+s['id'])
        qs.append({'id':s['id'],'question':q['question'],'split':'joint_'+s['split'],
                   'textbook_gold':tg,'case_gold':cg,'excluded_case_gold':sorted(original-cg)})
    for q in read_jsonl(ROOT/'eval/textbook_questions_v2_holdout70.jsonl'):
        tg=set(q['relevant_chunk_ids'])
        if not tg or not tg<=tids: raise ValueError('Missing textbook gold: '+q['id'])
        qs.append({'id':q['id'],'question':q['question'],'split':'textbook_regression70',
                   'textbook_gold':tg,'case_gold':set(),'excluded_case_gold':[]})
    return qs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--permission-confirmed',action='store_true')
    p.add_argument('--question-id',help='Smoke test only; no tuning or held-out claims')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not a.permission_confirmed: p.error('Confirm local use rights and privacy/safety review first')
    if a.batch_size<1: p.error('Positive batch size required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=a.output or ROOT/'data/processed'/f'integrated_eval_{stamp}.json'
    if output.exists(): p.error('Output already exists')
    tids=set(textbook_collection().get(include=[])['ids'])
    cids=set(case_collection().get(include=[],where={'doc_type':'counseling_case'})['ids'])
    local_ids={r['id'] for r in read_jsonl(ROOT/'data/processed/wellness_unflagged.jsonl')}
    if cids!=local_ids or not cids: raise RuntimeError('Private case index and screened file mismatch')
    qs=load_questions(tids,cids)
    if a.question_id:
        qs=[q for q in qs if q['id']==a.question_id]
        if not qs: p.error('Unknown question ID')
    print(f'Validated textbook={len(tids)}, cases={len(cids)}, questions={len(qs)}',flush=True)
    from rag.retrieve import embed
    model=get_reranker(a.device)
    # Warm up models before timings. No real user content is used.
    embed(['warmup']); model.predict([('warmup','warmup')],show_progress_bar=False)
    cached=[]
    for j,q in enumerate(qs,1):
        t,c,timing=ranked_groups(q['question'],permission_confirmed=True,device=a.device,batch_size=a.batch_size)
        cached.append((q,t,c,timing))
        print(f'[{j}/{len(qs)}] {q["id"]} {q["split"]}: {timing["total"]:.2f}s',flush=True)
    dev=[x for x in cached if x[0]['split']=='joint_dev']
    grid=[]
    def dynamic_results(items,kt,ct):
        return [assess(dynamic(t,c,knowledge_threshold=kt,case_threshold=ct),q) for q,t,c,_ in items]
    if a.question_id:
        kt,ct=.05,.5  # fixed smoke-test settings only
    else:
        for kt,ct in product((.01,.05,.1,.2),(.2,.5,.7)):
            s=summarize(dynamic_results(dev,kt,ct))
            grid.append({'knowledge_threshold':kt,'case_threshold':ct,'dev':s})
        # Select by dev BOTH-HIT first, then mean group recall, then P@5; no test scores used.
        best=max(grid,key=lambda x:(x['dev']['joint']['both_hit'],
              (x['dev']['textbook']['recall']+x['dev']['case']['recall'])/2,
              x['dev']['joint']['precision_at_5']))
        kt,ct=best['knowledge_threshold'],best['case_threshold']
    details=[]
    for q,t,c,timing in cached:
        modes={f'fixed_{5-n}_{n}':assess(fixed(t,c,case_slots=n),q) for n in range(6)}
        modes['dynamic']=assess(dynamic(t,c,knowledge_threshold=kt,case_threshold=ct),q)
        details.append({'id':q['id'],'split':q['split'],'excluded_case_gold':q['excluded_case_gold'],
             'modes':modes,'timing':timing,
             'candidate_metrics':{'textbook':metrics([r['id'] for r in t],q['textbook_gold']),
                                  'case':metrics([r['id'] for r in c],q['case_gold'])},
             'candidates':[{'id':r['id'],'group':r['group'],'reranker_score':r['reranker_score']} for r in t+c]})
    summaries={}
    for split in sorted({d['split'] for d in details}):
        subset=[d for d in details if d['split']==split]
        summaries[split]={m:summarize([d['modes'][m] for d in subset]) for m in subset[0]['modes']}
    report={'status':'draft_proxy_labels_not_clinical_validation',
        'smoke_test':bool(a.question_id),'n_questions':len(qs),'corpus_counts':{'textbook':len(tids),'case':len(cids)},
        'candidate_k_per_group':15,'k':5,'policy':{'knowledge_threshold':kt,'case_threshold':ct,'max_cases':2},
        'calibration_grid_dev_only':grid,'summary':summaries,
        'mean_warm_seconds':{k:mean(d['timing'][k] for d in details) for k in details[0]['timing']},'details':details}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    # Compact console: joint both-hit, group hit/recall and average return counts.
    print('Policy:',json.dumps(report['policy']))
    for split,modes in summaries.items():
        print('\n'+split)
        for name,s in modes.items():
            print(name, json.dumps({'textbook_hit':s['textbook']['hit'],'textbook_recall':s['textbook']['recall'],
              'case_hit':s.get('case',{}).get('hit'),'case_recall':s.get('case',{}).get('recall'),
              'both_hit':s.get('joint',{}).get('both_hit'),'joint_p5':s.get('joint',{}).get('precision_at_5'),
              'mean_returned':s['returned'],'abstained':s['abstained']}))
    print('Warm timing:',json.dumps(report['mean_warm_seconds']))
    print('Report:',output)

if __name__=='__main__': main()
