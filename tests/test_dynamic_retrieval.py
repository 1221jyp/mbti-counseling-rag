import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag.dynamic_retrieve import fixed,dynamic
from scripts.evaluate_integrated import assess,summarize


def row(uid,group,score):
    return {'id':uid,'group':group,'reranker_score':score}


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.t=[row('t1','textbook',.8),row('t2','textbook',.4),row('t3','textbook',.01)]
        self.c=[row('c1','case',.9),row('c2','case',.1)]
    def test_threshold(self):
        self.assertEqual([r['id'] for r in dynamic(self.t,self.c,knowledge_threshold=.05,case_threshold=.5)],['t1','t2','c1'])
    def test_abstain(self):
        self.assertEqual(dynamic(self.t,self.c,knowledge_threshold=.95,case_threshold=.1),[])
    def test_case_cap(self):
        self.assertEqual(len([r for r in dynamic(self.t,self.c,knowledge_threshold=0,case_threshold=0,max_cases=1) if r['group']=='case']),1)
    def test_fixed(self):
        self.assertEqual([r['id'] for r in fixed(self.t,self.c,k=3,case_slots=1)],['t1','t2','c1'])
    def test_bad_score(self):
        with self.assertRaises(ValueError): dynamic([row('t','textbook',float('nan'))],self.c,knowledge_threshold=.1,case_threshold=.1)
    def test_metrics(self):
        q={'textbook_gold':{'t1','t2'},'case_gold':{'c1'}}
        d=assess([self.t[0],self.c[0]],q)
        self.assertEqual(d['joint']['both_hit'],1)
        self.assertEqual(d['joint']['precision_at_5'],.4)
        self.assertEqual(d['textbook']['recall'],.5)
        q['case_gold']=set()
        self.assertIsNone(assess(self.t,q)['joint'])
        self.assertNotIn('case',summarize([assess(self.t,q)]))
if __name__=='__main__': unittest.main()
