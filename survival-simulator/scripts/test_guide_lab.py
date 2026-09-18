"""Contract/integration checks: python -m unittest discover -s survival-simulator/scripts -p test_guide_lab.py"""
import argparse
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pygame
import guide_lab as lab


class ContractTests(unittest.TestCase):
    def test_local_frame_rotates_with_heading(self):
        agent = SimpleNamespace(x=10., y=20., direction=math.pi/2)
        x,y = lab.local((10.,30.),agent)
        self.assertAlmostEqual(x,10.)
        self.assertAlmostEqual(y,0.)
        x,y = lab.local((0.,20.),agent)
        self.assertAlmostEqual(x,0.)
        self.assertAlmostEqual(y,10.)

    def test_actions_reject_nonfinite_and_out_of_range(self):
        agent = SimpleNamespace(agent_id=1,sprint_speed=20.)
        for distance in (-1.,21.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                lab.validate_action(dict(move_distance=distance,move_direction=0.,turn_angle=0.),agent)
        with self.assertRaises(ValueError):
            lab.validate_action(dict(move_distance=0.,move_direction=0.,turn_angle=0.,spawn_agent=True),agent)

    def test_policy_error_preserves_failing_input_and_frame(self):
        pygame.init()
        args = argparse.Namespace(output=lab.ROOT/'logs/guide_lab',
                                  policy=lab.ROOT/'models/entrapment/my_guide.py',site=0,
                                  width=160,seconds=.5,hold=10.,break_at=None)
        with patch.object(lab,'validate_action',side_effect=ValueError('deliberate contract-test error')):
            summary,folder=lab.run(args,10224,20224)
        self.assertEqual(summary['outcome'],'policy_error')
        row=json.loads((folder/'ticks/0.json').read_text())
        self.assertIsNotNone(row['input'])
        self.assertIn('deliberate contract-test error',row['error'])
        self.assertTrue((folder/'frames/0.png').is_file())
        pygame.quit()

    def test_replay_is_deterministic_and_energy_is_native(self):
        pygame.init()
        args = argparse.Namespace(output=lab.ROOT/'logs/guide_lab',
                                  policy=lab.ROOT/'models/entrapment/my_guide.py',site=0,
                                  width=160,seconds=.5,hold=10.,break_at=None)
        first,folder_a = lab.run(args,10224,20224)
        second,folder_b = lab.run(args,10224,20224)
        self.assertEqual(first['frames'],6)
        self.assertEqual(first['start'],second['start'])
        rows=[]
        for tick in range(6):
            a=json.loads((folder_a/'ticks'/f'{tick}.json').read_text())
            b=json.loads((folder_b/'ticks'/f'{tick}.json').read_text())
            self.assertEqual(a,b)
            self.assertEqual((folder_a/'frames'/f'{tick}.png').read_bytes(),
                             (folder_b/'frames'/f'{tick}.png').read_bytes())
            rows.append(a)
        initial=rows[0]['input']
        self.assertEqual(set(initial),{'bait','edges','agent','context'})
        self.assertTrue(any(o['type']=='Predator' for o in initial['agent']['observations']))
        self.assertEqual(initial['agent']['energy'],initial['agent']['max_energy'])
        self.assertLess(rows[-1]['evaluation']['guide_energy'],initial['agent']['energy'])
        self.assertTrue(all(r['evaluation']['bait_energy']==500 for r in rows))
        self.assertLessEqual(math.dist(initial['bait'],initial['context']['handoff'])+15,60)
        self.assertGreater(len(initial['edges']),4)
        args.bulk = True
        bulk,folder_c = lab.run(args,10224,20224)
        with gzip.open(folder_c/'ticks.jsonl.gz','rt') as trace:
            self.assertEqual([json.loads(line) for line in trace], rows)
        self.assertEqual(bulk['final'], first['final'])
        self.assertEqual(list((folder_c/'frames').iterdir()), [])
        pygame.quit()


if __name__ == '__main__':
    unittest.main()
