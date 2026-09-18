"""Integration checks: capture must preserve state/RNG and replay chronology."""
import gzip
import base64
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'debugger'),str(ROOT/'vendor/survival-simulator')]
from recorder import ReplayRecorder
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


class RecorderIntegration(unittest.TestCase):
    def test_native_render_matches_upstream_and_preserves_gameplay(self):
        import pygame
        env=Environment(200,200,400,random.Random(12))
        agent=env.spawn_agent(x=100,y=100)
        predator=env.spawn_predator(x=140,y=140)
        before=(env.rng.getstate(),agent.x,agent.energy,env.time,agent._vision_poly,predator._vision_poly)
        recorder=ReplayRecorder(env,native_width=200)
        recorder.capture(force=True)
        after=(env.rng.getstate(),agent.x,agent.energy,env.time,agent._vision_poly,predator._vision_poly)
        self.assertEqual(before,after,'Native capture must preserve RNG and vision caches')
        encoded=recorder.frames[0]['native_image'].split(',',1)[1]
        actual=pygame.image.load(io.BytesIO(base64.b64decode(encoded)))
        expected=pygame.Surface((200,200));env.draw(expected)
        self.assertEqual(pygame.image.tostring(actual,'RGB'),pygame.image.tostring(expected,'RGB'),
                         'Captured pixels must come from the original Environment.draw')
        self.assertEqual(recorder.meta['renderer'],'upstream Environment.draw')

    def test_read_only_capture_roundtrip_and_events(self):
        env=Environment(200,200,400,random.Random(91))
        agent=env.spawn_agent(x=70,y=70);agent.direction=0
        before=(env.rng.getstate(),agent.x,agent.y,agent.energy,agent.age,env.time,env.score)
        recorder=ReplayRecorder(env,every=3,seed=91,native_render=False)
        recorder.capture(force=True)
        after=(env.rng.getstate(),agent.x,agent.y,agent.energy,agent.age,env.time,env.score)
        self.assertEqual(before,after,'Recording must not consume RNG or change state')
        action=ActionRequest(agent_id=agent.agent_id,move_distance=10,move_direction=0,turn_angle=0,spawn_agent=True)
        state=step_environment(env,[(agent.agent_id,action)])
        recorder.capture([(agent.agent_id,action)],{agent.agent_id:{'rule':'test','detail':'request child'}},action_t=0)
        self.assertTrue(any(e['type']=='birth' and e['t']==.1 for e in recorder.events))
        child=next(a for a in env.agents if a is not agent)
        child.energy=0
        step_environment(env,[]);recorder.capture(force=True)
        self.assertTrue(any(e['type']=='death' and e['entity_id']==child.agent_id for e in recorder.events))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'replay.json.gz';recorder.save(path)
            data=json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(data['format'],'survival-replay')
            self.assertEqual(data['version'],1)
            self.assertTrue(data['world']['background'].startswith('data:image/png;base64,'))
            self.assertEqual([f['t'] for f in data['frames']],[0,.1,.2])
            self.assertEqual(data['summary']['duration'],.2)
            self.assertEqual(data['frames'][-1]['agents'][0]['id'],agent.agent_id)
            with self.assertRaises(FileExistsError):recorder.save(path)
            previous_bytes=path.read_bytes()
            recorder.meta['invalid_number']=float('nan')
            with self.assertRaises(ValueError):recorder.save(path,overwrite=True)
            self.assertEqual(path.read_bytes(),previous_bytes,'A failed save must preserve the previous completed replay')
            self.assertEqual(list(Path(d).glob('*.tmp')),[])

    def test_stable_predator_ids_and_action_timing(self):
        env=Environment(200,200,400,random.Random(7))
        agent=env.spawn_agent(x=100,y=100)
        p=env.spawn_predator(x=70,y=70)
        recorder=ReplayRecorder(env,native_render=False)
        recorder.capture()
        first_id=recorder.frames[-1]['predators'][0]['id']
        action=ActionRequest(agent_id=agent.agent_id,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)
        env.non_agent_step(.1)
        recorder.capture([(agent.agent_id,action)],{agent.agent_id:dict(rule='Idle',detail='test')},action_t=0)
        self.assertEqual(recorder.frames[-1]['predators'][0]['id'],first_id)
        row=recorder.frames[-1]['agents'][0]
        self.assertEqual(row['action_t'],0)
        self.assertEqual(row['action']['move_distance'],0)
        self.assertEqual(row['decision']['rule'],'Idle')


if __name__=='__main__':unittest.main()
