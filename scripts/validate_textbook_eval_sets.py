"""Validate V2 corpus and the two draft evaluation sets without loading models."""
from collections import Counter
import json
from textbook_v2 import ROOT, load


def main():
    rows, digest = load()
    ids = {r['id'] for r in rows}
    all_questions = set()
    categories = {'stress_management', 'emotion_regulation', 'cbt_psychoeducation', 'relationships', 'anxiety_depression_sleep', 'self_esteem_acceptance', 'professional_help'}
    old = {q['id']:q for q in map(json.loads,(ROOT/'eval/textbook_questions.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    for name, expected in [('textbook_questions_v2_reviewed.jsonl',21),('textbook_questions_v2_holdout70.jsonl',70)]:
        path = ROOT/'eval'/name
        qs = [json.loads(l) for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        if len(qs)!=expected or len({q['id'] for q in qs})!=expected:
            raise ValueError(f'Invalid question count or duplicate IDs: {name}')
        counts=Counter(q['category'] for q in qs)
        if set(counts)!=categories or (expected==70 and any(n!=10 for n in counts.values())):
            raise ValueError('Category mismatch')
        if expected==21 and set(old)!=set(q['id'] for q in qs):
            raise ValueError('Development question IDs changed')
        for q in qs:
            text=q['question'].strip()
            gold=q['relevant_chunk_ids']
            if not text or text in all_questions or not gold or len(gold)!=len(set(gold)) or not set(gold)<=ids:
                raise ValueError(f'Invalid text or labels: {q["id"]}')
            if expected==21 and (q['question']!=old[q['id']]['question'] or q['category']!=old[q['id']]['category']):
                raise ValueError('Original development question changed: '+q['id'])
            all_questions.add(text)
        print(f'Validated {name}: {len(qs)} questions, {sum(len(q["relevant_chunk_ids"]) for q in qs)} label links, {dict(counts)}')
    print(f'V2 corpus: {len(rows)} records; digest={digest}')
    print('Schema/ID/integrity validation only; relevance and clinical review are not certified.')

if __name__=='__main__':
    main()
