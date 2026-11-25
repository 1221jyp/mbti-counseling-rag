import unittest
from rag.integrated_retrieve import select, candidates, public_metadata, rank

class MockCollection:
    def count(self): return 2
    def get(self,**kw): return {'ids':['1']}
    def query(self,**kw):
        assert kw['n_results']==1
        assert kw['where']=={'doc_type':'case'}
        return {'ids':[['1']], 'documents':[['example']], 'metadatas':[[{'source':'mock'}]], 'distances':[[0.2]]}

class Tests(unittest.TestCase):
    def test_quota(self):
        self.assertEqual(select(list(range(5)),['a','b','c'],[],5,2),[0,1,2,'a','b'])
    def test_missing_cases_backfill(self):
        self.assertEqual(select(list(range(5)),[],[],5,2),list(range(5)))
    def test_no_textbook(self):
        with self.assertRaises(RuntimeError):select([],['a'],[],5,2)
    def test_bad_quota(self):
        with self.assertRaises(ValueError):select([1],[],[],5,5)
    def test_query_and_metadata(self):
        rows=candidates(MockCollection(),[[1]],15,'case','case')
        self.assertEqual(len(rows),1)
        self.assertNotIn('text',public_metadata(rows)[0])
        self.assertAlmostEqual(rows[0]['dense_score'],.8)
        self.assertEqual(rank('q',rows,None),rows)

if __name__=='__main__':unittest.main()
