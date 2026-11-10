import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from recommendation_05_pipeline_helpers import make_query,extract_rankings,score_rankings,macro_by_user


class TestBenchmark(unittest.TestCase):
    def test_deterministic_template_no_hidden_rating(self):
        f = pd.DataFrame({'movieId':[2,1], 'rating':[1.0,5.0]})
        titles={1:'Film uno',2:'Film due'}
        text=make_query(f,titles,'title_rating_v1')
        self.assertLess(text.index('Film uno'),text.index('Film due'))
        self.assertIn('5.0/5',text)
        self.assertIn('1.0/5',text)
        self.assertNotIn('Film tre',text)
        self.assertEqual(text,make_query(f.iloc[::-1],titles,'title_rating_v1'))

    def test_structured_mcp_vs_answer(self):
        hits=[NS(rank=1,resource_id=2,score=.9,title='seen'),
              NS(rank=2,resource_id=3,score=.8,title='recommended'),
              NS(rank=3,resource_id=4,score=.7,title='recommended2'),
              NS(rank=4,resource_id=3,score=.4,title='duplicate')]
        state=NS(status='DONE', search_response=NS(results=hits,status='ok',query='query'),
                 final_answer=NS(answer='only one film',recommendations=[NS(resource_id=3)]))
        data=extract_rankings(state,catalog={1,2,3,4},shown={2},target_k=10)
        self.assertEqual([x['movieId'] for x in data['retrieval']],[3,4])
        self.assertEqual([x['rank'] for x in data['retrieval']],[1,2])
        self.assertEqual(data['answer_ids'],[3])
        self.assertEqual(data['shortfall'],8)
        self.assertEqual(data['shown_excluded'],[2])

    def test_empty_mcp_list_scored_zero(self):
        preds=pd.DataFrame(columns=['scenario_id','k','model','movieId','rank','score'])
        fixed=pd.DataFrame({'scenario_id':['s']*20,'movieId':list(range(101,121)),
                            'rating':[5.0]+[2.0]*19})
        extra=pd.DataFrame(columns=['scenario_id','movieId','rating'])
        profile=pd.DataFrame({'scenario_id':['s']*10,'movieId':list(range(201,211)),
                              'rating':[5.0]*10,'reveal_order':list(range(1,11))})
        keys=pd.DataFrame([{'scenario_id':'s','k':1,'model':'test'}])
        out=score_rankings(preds,fixed=fixed,extra=extra,private_profile=profile,
                           expected_keys=keys)
        self.assertEqual(len(out),2)
        self.assertTrue((out['n_returned']==0).all())
        self.assertTrue((out['ndcg@10']==0).all())

    def test_relevance_A_B_and_fewer_than_K(self):
        preds=pd.DataFrame({'scenario_id':['s']*2,'k':[1]*2,'model':['test']*2,
                            'movieId':[101,105],'rank':[1,2],'score':[.5,.1]})
        fixed=pd.DataFrame({'scenario_id':['s']*20,'movieId':[mid if mid != 105 else 125 for mid in range(101,121)],
                            'rating':[5]+[2]*19})
        extra=pd.DataFrame({'scenario_id':['s'],'movieId':[130],'rating':[5]})
        profile=pd.DataFrame({'scenario_id':['s']*10,'movieId':[201,105,203,204,205,206,207,208,209,210],
                              'rating':[1,5,1,1,1,1,1,1,1,1],
                              'reveal_order':list(range(1,11))})
        actual=score_rankings(preds,fixed=fixed,extra=extra,private_profile=profile,top_k=10)
        a=actual[actual.scheme=='A'].iloc[0]
        b=actual[actual.scheme=='B'].iloc[0]
        self.assertEqual(a.n_hits_topk if 'n_hits_topk' in a else a['n_relevant'],1)
        self.assertEqual(a['n_relevant'],1)
        self.assertAlmostEqual(a['ndcg@10'],1.0)
        self.assertEqual(a['shortfall'],8)
        self.assertEqual(b['n_relevant'],3)
        self.assertGreater(b['ndcg@10'],0)
        # The actually revealed item 201 must never appear as recommended.
        bad=pd.DataFrame({'scenario_id':['s'],'k':[1],'model':['test'], 'movieId':[201],'rank':[1]})
        with self.assertRaises(ValueError):score_rankings(bad,fixed=fixed,extra=extra,private_profile=profile)

if __name__=='__main__':unittest.main()
