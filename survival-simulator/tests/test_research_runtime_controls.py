"""Schedule amendments must leave evaluation, finances and prior state intact."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.research_runtime_controls import effective_config
from tests import test_research_supervisor as contracts
from scripts.research_dashboard import collect
from scripts.research_support import write
from scripts import research_supervisor_handoff as handoff


class RuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.base = contracts.configuration()
        self.base['schedule']['major_generations'] = 3
        self.control = dict(version=1, changes={'schedule.major_generations': {'from': 3, 'to': 8}})

    def test_only_generation_ceiling_changes(self):
        prior = copy.deepcopy(self.base)
        amended = effective_config(self.base, self.control)
        self.assertEqual(amended['schedule']['major_generations'], 8)
        amended['schedule']['major_generations'] = 3
        self.assertEqual(amended, prior)
        self.assertEqual(self.base, prior)

    def test_rejects_stale_and_unrelated_changes(self):
        for change in ({'from': 2, 'to': 8}, {'from': 3, 'to': True}, {'from': 3, 'to': 2}):
            with self.assertRaises(ValueError):
                effective_config(self.base, dict(version=1, changes={'schedule.major_generations': change}))
        self.control['changes']['budget.total_usd'] = {'from': 40, 'to': 80}
        with self.assertRaises(ValueError):
            effective_config(self.base, self.control)

    def test_eight_majors_preserve_four_subgenerations_and_one_final(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = contracts.ResearchSchedule().fake(folder)
            supervisor.config = effective_config(self.base, self.control)
            from unittest.mock import Mock
            supervisor.round = Mock(side_effect=lambda key, incumbent, upstream, major=False:
                dict(winner=dict(incumbent, id=key), improved=True, stop_early=False))
            with patch('scripts.research_loop.signal.signal'):
                supervisor.run()
            expected = [key for major in range(1, 9)
                        for key in [*[f'g{major}.{sub}' for sub in range(1, 5)], f'g{major}.major']]
            self.assertEqual([call.args[0] for call in supervisor.round.call_args_list], expected)
            supervisor.final.assert_called_once()

    def test_dashboard_uses_applied_runtime_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'config.json', self.base)
            write(root / 'state.json', dict(status='running', runtime_controls={'schedule': {'major_generations': 8}}))
            self.assertEqual(collect(root)['schedule']['major_generations'], 8)
            self.assertEqual(self.base['schedule']['major_generations'], 3)

    def test_handoff_accepts_only_independent_search_or_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('g1.2.search', 'g1.3.agent', 'g1.2.compare'):
                step_dir = root / 'steps' / name
                write(step_dir / 'process-request.json', {})
                (step_dir / 'control').mkdir()
                heartbeat = step_dir / 'control/heartbeat'
                heartbeat.touch()
                os.utime(heartbeat, (1000, 1000))
                state = dict(status='running', stage='sub', steps={name: dict(status='running', deadline=1300)})
                with patch.object(handoff, 'OUT', root), patch.object(handoff, 'FINISHED', root / 'finished'), \
                        patch.object(handoff.time, 'time', return_value=1000), \
                        patch.object(handoff, 'wrapper_for', return_value=123):
                    if name.endswith('.compare'):
                        with self.assertRaises(ValueError):
                            handoff.safe_step(state)
                    else:
                        self.assertEqual(handoff.safe_step(state)[2], 123)
                        state['steps'][name]['deadline'] = 1020
                        with self.assertRaises(ValueError):
                            handoff.safe_step(state)
