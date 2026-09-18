"""Useful interface/measurement checks, separate from performance evaluation."""
import copy
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
import unittest
from run import ROOT, HERE, OUT, setup, POLICY_HASH, HARNESS_HASH, STRATEGY_HARNESS_SHA256
from recording_support import start_recording, decisions_for_viewer
from policy import ColonyPolicy, LocalMap, dist
from src.utils.simulation import step_environment


class PolicyBoundaryChecks(unittest.TestCase):
    def test_repeated_serialized_inputs_and_legal_actions(self):
        env,_=setup('forest',53)
        recorder,path=start_recording(env,scenario='forest',seed=53,mode='interface-check',horizon=6.1,
            settings={},version=5,policy_hash=POLICY_HASH,harness_hash=HARNESS_HASH,
            strategy_harness_hash=STRATEGY_HARNESS_SHA256,request={'label':'serialized-input-check','every':1})
        # Retain failed diagnostic runs too. Cleanup runs even if an assertion fails.
        self.addCleanup(lambda:recorder.save(path,reason='interface check ended'))
        state=step_environment(env,[],.1)
        recorder.capture(inputs=[],action_t=0.)
        a,b=ColonyPolicy(True),ColonyPolicy(True)
        for _ in range(60):
            # Exactly the ordinary DTOs cross this boundary, including cached
            # predator observations. Neither controller sees an engine handle.
            inputs=json.loads(json.dumps(state['observations']))
            original=copy.deepcopy(inputs)
            reordered=copy.deepcopy(inputs)
            rng=random.Random(731)
            for s in reordered:rng.shuffle(s['observations'])
            aa=a(inputs,env.time);bb=b(reordered,env.time)
            self.assertEqual(inputs,original)
            self.assertEqual([(i,x.model_dump()) for i,x in aa],[(i,x.model_dump()) for i,x in bb])
            self.assertEqual({i for i,x in aa},{s['agent_id'] for s in inputs})
            self.assertEqual(len(aa),len(inputs))
            by={s['agent_id']:s for s in inputs}
            for i,x in aa:
                self.assertGreaterEqual(x.move_distance,0)
                self.assertLessEqual(x.move_distance,by[i]['sprint_speed'])
                self.assertTrue(all(math.isfinite(z) for z in (x.move_direction,x.turn_angle)))
            action_t=env.time
            state=step_environment(env,aa,.1)
            recorder.capture(aa,decisions=decisions_for_viewer(a.decisions),inputs=inputs,action_t=action_t)

    def test_relative_agent_heading_alignment(self):
        # Agent 7 sees agent 9 at local (0,40), facing perpendicular to it.
        states=[dict(agent_id=7,observations=[dict(type='Agent',id=9,distance=40.,angle=math.pi/2,rel_dir=-math.pi/2)]),
                dict(agent_id=9,observations=[])]
        m=LocalMap();m.update(states,0.)
        self.assertEqual(m.poses[7].group,m.poses[9].group)
        self.assertAlmostEqual(dist(m.poses[7].xy(),m.poses[9].xy()),40.)
        self.assertAlmostEqual(m.poses[9].theta,0.)

    def test_diagnostic_sensing_does_not_change_physics_or_rng(self):
        env,_=setup('forest',54);pred=env.spawn_predator(600,600)
        before=(pred.x,pred.y,pred.energy,pred.direction,pred.resting,repr(env.rng.getstate()))
        pred.observe(agents=env._get_local_agents(pred),edges=env._get_local_edges(pred))
        after=(pred.x,pred.y,pred.energy,pred.direction,pred.resting,repr(env.rng.getstate()))
        self.assertEqual(before,after)

    def test_vendored_blobs_match_pinned_source(self):
        tree=json.loads((ROOT/'source-tree.json').read_text());count=0
        for row in tree['tree']:
            if row['type']!='blob' or not row['path'].startswith('survival-simulator/'):continue
            path=ROOT/'vendor'/row['path']
            if not path.exists():continue
            data=path.read_bytes();blob=b'blob '+str(len(data)).encode()+b'\0'+data
            self.assertEqual(hashlib.sha1(blob).hexdigest(),row['sha'],str(path));count+=1
        self.assertEqual(count,18)


def verify_replays():
    rows=[]
    for path in sorted(OUT.rglob('*.json.gz')):
        with gzip.open(path,'rt') as f:data=json.load(f)
        assert data['format']=='survival-replay' and data['version']==1
        assert all(a['t']<b['t'] for a,b in zip(data['frames'],data['frames'][1:]))
        assert data['frames'][-1]['t']==data['summary']['duration']
        assert data['world']['obstacles'] and data['world']['background'].startswith('data:image/png;base64,')
        assert all(len({a['id'] for a in f['agents']})==len(f['agents']) for f in data['frames'])
        if data['meta'].get('run_id'):
            assert data['frames'][0]['t']==0
            assert data['meta']['horizon']>=data['summary']['duration']-1e-5
            times={f['t'] for f in data['frames']}
            assert all(e['t'] in times for e in data['events'] if e['type']!='fruit_removed')
            assert all(a['decision'] is None or isinstance(a['decision'],dict)
                for frame in data['frames'] for a in frame['agents'])
            if data['meta'].get('reproduction_of'):
                assert data['meta']['provenance_kind']=='new_reproduction'
                assert 'NEW REPRODUCTION' in data['meta']['notes']
            if data['meta'].get('renderer')=='upstream Environment.draw':
                assert all(f['native_image'].startswith('data:image/') for f in data['frames'])
        rows.append(dict(file=str(path.relative_to(OUT)),frames=len(data['frames']),duration=data['summary']['duration'],valid=True))
    (OUT/'replay-coverage-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Validated {len(rows)} existing-viewer replay files.')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PolicyBoundaryChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    verify_replays()
