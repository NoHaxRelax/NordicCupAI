"""Network-free checks for irreversible query accounting and inference limits."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from common import atomic,digest,hypothesis,predictions,split
from engine import Search,coverage_ok

class DiscoveryTests(unittest.TestCase):
    def create_search(self,folder,n=16,limit=100,stop=False):
        values=[hypothesis(5,'tank',[20*x,0,20*x+10,10]) for x in range(n)]
        atomic(folder/'groups/root.json',values)
        roots=[{'id':'root','file':'groups/root.json','hypothesis_count':n,'kind':'entry_scan','label':'tank','stop_after_first':stop}]
        return Search.create(folder,roots,[],limit)

    def simulate(self,search,matching):
        while True:
            node=search.next_node()
            if node is None:return
            q=search.reserve(node)
            has_match=any(x['bbox_source_xyxy'][0] in matching for x in search.items(node))
            search.observe(.001 if has_match else 0,'mock-receipt')

    def test_caps(self):
        rows=[hypothesis(5,'tank',[0,0,10,10]) for _ in range(100)]
        self.assertEqual(len(predictions(rows)['5']),100)
        with self.assertRaises(AssertionError):predictions(rows+rows[:1])
        with self.assertRaises(AssertionError):predictions([hypothesis(5,f'c{x}',[0,0,10,10]) for x in range(501)])

    def test_isolates_single_match(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.create_search(Path(t));self.simulate(s,{120})
            self.assertEqual([x['bbox_source_xyxy'][0] for x in s.state['seeds']],[120])
            self.assertLess(s.state['runs_completed'],16)

    def test_multiple_positive_branches_are_not_discarded(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.create_search(Path(t));self.simulate(s,{20,80,280})
            self.assertEqual({x['bbox_source_xyxy'][0] for x in s.state['seeds']},{20,80,280})

    def test_zero_is_not_a_seed(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.create_search(Path(t));self.simulate(s,set())
            self.assertEqual(s.state['seeds'],[])
            self.assertEqual(s.state['runs_completed'],1)
            self.assertEqual(s.state['roots']['root']['status'],'no_match_in_tested_hypotheses')

    def test_visual_root_stops_after_first_isolated_match(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.create_search(Path(t),stop=True);self.simulate(s,{20,80})
            self.assertEqual(len(s.state['seeds']),1)

    def test_budget_is_reserved_before_results(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.create_search(Path(t),limit=2);self.simulate(s,{20})
            self.assertEqual(s.state['runs_reserved'],2)
            self.assertIsNone(s.next_node())
            with self.assertRaises(RuntimeError):s.reserve({'id':'extra'})

    def test_pending_survives_reload_and_cannot_be_reserved_again(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.create_search(p);node=s.next_node();q=s.reserve(node)
            restored=Search(p,json.loads((p/'state.json').read_text()))
            self.assertEqual(restored.state['pending']['query_id'],q['query_id'])
            self.assertEqual(restored.next_node(),node)
            with self.assertRaises(RuntimeError):restored.reserve(node)
            self.assertEqual(restored.state['runs_reserved'],1)

    def test_all_target_frames_required(self):
        frames=predictions([hypothesis(5,'tank',[0,0,10,10]),hypothesis(6,'tank',[0,0,10,10])])
        plan={'predictions_by_frame':frames,'plan_hash':digest(frames)}
        receipts=[{'frame':5,'annotation_count':1,'plan_hash':plan['plan_hash']}]
        self.assertFalse(coverage_ok(plan,receipts,{'errors':[]})[0])
        receipts.append({'frame':6,'annotation_count':1,'plan_hash':plan['plan_hash']})
        self.assertTrue(coverage_ok(plan,receipts,{'errors':[]})[0])
        self.assertFalse(coverage_ok(plan,receipts,{'errors':['dropped']})[0])
        receipts[0]['plan_hash']='stale'
        self.assertFalse(coverage_ok(plan,receipts,{'errors':[]})[0])

    def test_unknown_or_out_of_bounds_box_rejected(self):
        self.assertIsNone(hypothesis(5,'tank',[-1,0,10,10]))
        self.assertIsNone(hypothesis(5,'tank',[0,0,5000,10]))

    def test_uncertain_submission_never_repeats_post(self):
        import run
        with tempfile.TemporaryDirectory() as t,patch.object(run,'DATA',Path(t)):
            s=self.create_search(Path(t),n=1);s.reserve(s.next_node())
            w=run.Worker.__new__(run.Worker);w.search=s;w.s=s.state;w.activate=lambda plan:None
            calls=[]
            def failed_post(action,*args):
                calls.append(action);raise RuntimeError('Simulated lost response')
            w.portal=failed_post
            with self.assertRaises(RuntimeError):w.finish_pending()
            self.assertEqual(s.state['pending']['phase'],'submitting')
            with self.assertRaisesRegex(RuntimeError,'uncertain'):w.finish_pending()
            self.assertEqual(calls,['validate'])
            self.assertEqual(s.state['runs_reserved'],1)

    def missing_frame_search(self,folder,limit=100):
        s=self.create_search(folder,n=8,limit=limit)
        items=s.items(s.state['screen'][0])
        for x in items[4:]:x['frame']=6
        atomic(folder/'groups/root.json',items)
        s.reserve(s.next_node())
        return s

    def test_clean_partial_zero_retries_only_gaps(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.missing_frame_search(p)
            decision=s.recover_incomplete(0,[6],'receipt-1','missing',clean_zero=True)
            self.assertEqual(decision['scope'],'missing_frames')
            self.assertEqual(s.state['roots']['root']['status'],'pending')
            self.assertEqual(s.state['runs_reserved'],1)
            node=s.next_node();self.assertEqual({x['frame'] for x in s.items(node)},{6})
            self.assertEqual(len(s.items(node)),4)
            s.reserve(node);s.observe(0,'receipt-2')
            self.assertEqual(s.state['roots']['root']['status'],'no_match_in_tested_hypotheses')
            self.assertEqual(s.state['runs_completed'],2)

    def test_positive_partial_and_error_results_retry_full_group(self):
        for score in (0,.001):
            with tempfile.TemporaryDirectory() as t:
                p=Path(t);s=self.missing_frame_search(p)
                d=s.recover_incomplete(score,[6],'receipt','missing',clean_zero=False)
                self.assertEqual(d['scope'],'full_group')
                self.assertEqual(len(s.items(s.next_node())),8)
                self.assertEqual(s.state['seeds'],[])

    def test_partial_result_never_infers_negative_sibling_early(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.missing_frame_search(p)
            s.state['pending']['node']['infer_sibling_on_zero']='sibling'
            s.state['work']=[{'id':'sibling','known':False}]
            s.recover_incomplete(0,[6],'receipt-1','missing',clean_zero=True)
            self.assertFalse(s.state['work'][0]['known'])
            s.reserve(s.next_node());s.observe(0,'receipt-2')
            self.assertTrue(s.state['work'][0]['known'])

    def test_retry_survives_restart_and_obeys_total_budget(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.missing_frame_search(p,limit=1)
            s.recover_incomplete(0,[6],'receipt','missing',clean_zero=True)
            restored=Search(p,json.loads((p/'state.json').read_text()))
            self.assertEqual(len(restored.state['retries']),1)
            self.assertIsNone(restored.next_node())
            self.assertEqual(restored.state['roots']['root']['status'],'pending')

    def test_repeated_missing_frames_are_deferred_not_marked_absent(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.missing_frame_search(p)
            for attempt in range(4):
                d=s.recover_incomplete(0,[6],f'receipt-{attempt}','missing',clean_zero=True)
                if attempt<3:s.reserve(s.next_node())
            self.assertEqual(d['action'],'deferred')
            self.assertEqual(s.state['runs_reserved'],4)
            self.assertEqual(s.state['roots']['root']['status'],'incomplete')
            self.assertEqual(len(s.state['unresolved']),1)
            self.assertIsNone(s.next_node())

    def test_worker_recovers_saved_completed_attempt_without_new_post(self):
        import run
        with tempfile.TemporaryDirectory() as t,patch.object(run,'DATA',Path(t)),patch.object(run,'ROOT',Path(t)):
            p=Path(t);s=self.missing_frame_search(p)
            pending=s.state['pending'];pending.update(phase='done',uuid='existing-uuid')
            folder=p/'queries'/pending['query_id']
            frames=predictions(s.items(pending['node']))
            plan={'predictions_by_frame':frames,'plan_hash':digest(frames)}
            atomic(folder/'plan.json',plan);atomic(folder/'result.json',{'score':0,'errors':[]})
            atomic(p/'receipts'/pending['query_id']/'test-005.json',{'frame':5,'plan_hash':plan['plan_hash'],'annotation_count':4})
            w=run.Worker.__new__(run.Worker);w.search=s;w.s=s.state;w.publish=lambda *_:None;w.event=lambda _:None
            w.portal=lambda *_:self.fail('Must not repeat the completed POST')
            w.finish_pending()
            self.assertIsNone(s.state['pending'])
            self.assertEqual(s.state['runs_reserved'],1)
            self.assertEqual(s.state['runs_completed'],1)
            self.assertEqual(len(s.state['retries']),1)

    def test_errors_outside_tested_frames_do_not_discard_good_evidence(self):
        frames=predictions([hypothesis(66,'tank',[0,0,10,10])])
        plan={'predictions_by_frame':frames,'plan_hash':digest(frames)}
        receipts=[{'frame':66,'annotation_count':1,'plan_hash':plan['plan_hash']}]
        result={'errors':['HTTP error for frame 1: All connection attempts failed',
                          'Frame 151 request sample did not answer within 3333 ms.']}
        self.assertTrue(coverage_ok(plan,receipts,result)[0])
        self.assertFalse(coverage_ok(plan,[],result)[0])
        self.assertFalse(coverage_ok(plan,receipts,{'errors':['Frame 66 request sample timed out']})[0])
        self.assertFalse(coverage_ok(plan,receipts,{'errors':['Unrecognized scoring failure']})[0])

    def test_seed_first_localizes_before_more_screening_then_returns_to_it(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=self.create_search(p,n=4,stop=True)
            s.reserve(s.next_node());s.observe(.001,'root-positive')
            other={**s.state['work'][0],'id':'other','root':'other','known':False}
            s.state['roots']['other']={**s.state['roots']['root'],'id':'other','status':'pending'}
            s.state['screen']=[other];s.state['schedule']='seed_first';s.state['screen_credits']=0
            while not s.state['seeds']:
                node=s.next_node();self.assertEqual(node['root'],'root')
                score=.001 if any(x['bbox_source_xyxy'][0]==0 for x in s.items(node)) else 0
                s.reserve(node);s.observe(score,'isolated')
            self.assertEqual(s.state['screen_credits'],2)
            self.assertEqual(s.next_node()['root'],'other')

    def test_dense_grids_use_class_sizes_and_obey_limits(self):
        from calibrated_grid import fit,observations,local_grid,size
        model=fit(observations());values=[]
        for label in ('small_launcher','jammer','tank','helicopter'):
            grid=local_grid(66,label,1500,500,model)
            self.assertLessEqual(len(grid),100)
            self.assertGreater(len({tuple(x['bbox_source_xyxy'][:2]) for x in grid}),25)
            values.extend(grid)
        self.assertLessEqual(len(predictions(values)['66']),400)
        w,h=size(model,'small_launcher',500)
        self.assertLess(w,30);self.assertLess(h,40)

    def test_dense_grid_clips_edge_boxes_safely(self):
        from calibrated_grid import fit,observations,local_grid
        model=fit(observations())
        for cx,cy in ((0,0),(3839,2159)):
            grid=local_grid(249,'helicopter',cx,cy,model)
            self.assertTrue(grid)
            predictions(grid)

if __name__=='__main__':unittest.main()
