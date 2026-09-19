"""Contracts for the maintained runners and shared source baseline."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import agent_server
import run
from src.utils.DTOs import ActionRequest, StepResponse

ROOT = Path(__file__).resolve().parents[1]


class SupportedRuntimeTests(unittest.TestCase):
    def test_frozen_trapping_policy_and_engine_are_unchanged(self):
        manifest = json.loads((ROOT / 'docs/entrapment_native_seed0/manifest.json').read_text())
        for name, expected in manifest['sources'].items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_every_supported_command_has_a_target(self):
        for name, target in run.COMMANDS.items():
            with self.subTest(command=name):
                self.assertTrue((ROOT / target).is_file())

    def test_endpoint_uses_optimization_policy_and_resets_between_games(self):
        public = dict(agent_id=1, energy=150., age=0., biome='forest', speed=10.,
                      sprint_speed=20., hearing_radius=50., vision_angle=1.,
                      vision_range=200., max_energy=500.,
                      observations=[dict(type='fruit', distance=10., angle=0.)])
        action = ActionRequest(agent_id=1, move_distance=0., move_direction=0.,
                               turn_angle=0., spawn_agent=False)
        current = Mock(return_value=[(1, action)])
        replacement = Mock(return_value=[(1, action)])
        with patch.object(agent_server, 'policy', current), \
             patch.object(agent_server, 'last_time', 1.), \
             patch.object(agent_server, 'OptimizationPolicy', return_value=replacement) as factory:
            step = StepResponse(game_status='ok', score=999., sim_time=2., n_agents=1,
                                agent_status=[public])
            self.assertEqual(agent_server.predict(step), {'actions': [action.model_dump()]})
            states, when = current.call_args.args
            self.assertEqual(states[0]['observations'][0]['type'], 'Fruit')
            self.assertEqual(when, 2.)
            self.assertNotIn('score', states[0])
            step.sim_time = .1
            agent_server.predict(step)
            factory.assert_called_once_with()
            replacement.assert_called_once()


if __name__ == '__main__':
    unittest.main()
