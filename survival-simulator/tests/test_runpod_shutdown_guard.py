import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

from scripts.runpod_shutdown_guard import PodClient, shutdown_reason, validate_deadline


class ShutdownTests(unittest.TestCase):
    def test_extended_deadline_requires_budget_and_counts_previous_spend(self):
        config=dict(created_at='1970-01-01T00:00:00Z',deadline_epoch=24*3600)
        with self.assertRaises(ValueError):validate_deadline(config)
        config['budget']=dict(total_usd=40,prior_spend_usd=10,reserve_usd=5,hourly_rate_usd=.985)
        self.assertEqual(validate_deadline(config),24*3600)
        config['budget']['prior_spend_usd']=15
        with self.assertRaises(ValueError):validate_deadline(config)

    def test_deadline_survives_missing_or_broken_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(shutdown_reason(9, 10, folder))
            Path(folder, 'state.json').write_text('{broken')
            self.assertEqual(shutdown_reason(10, 10, folder), 'hard_deadline')

    def test_prepared_does_not_stop_but_terminal_does(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder, 'state.json')
            for status in ('prepared', 'running'):
                state.write_text('{"status": "'+status+'"}')
                self.assertIsNone(shutdown_reason(1, 100, folder))
            for status in ('complete', 'inconclusive', 'failed', 'paused'):
                state.write_text('{"status": "'+status+'"}')
                self.assertEqual(shutdown_reason(1, 100, folder), 'campaign_'+status)

    def test_identity_mismatch_never_mutates(self):
        client = PodClient('owned-pod', 'creation', 'unused')
        client.request = Mock(return_value={'id': 'other-pod', 'createdAt': 'creation'})
        with self.assertRaises(ValueError):
            client.stop()
        self.assertEqual(client.request.call_args_list, [unittest.mock.call()])

    def test_stop_verified_owned_running_pod(self):
        client = PodClient('owned-pod', 'creation', 'unused')
        client.request = Mock(side_effect=[{'id': 'owned-pod', 'createdAt': 'creation',
            'status': 'RUNNING', 'actions': ['stop']}, {'status': 'EXITED'}])
        self.assertEqual(client.stop(), 'EXITED')
        self.assertEqual(client.request.call_args_list, [unittest.mock.call(), unittest.mock.call('stop')])

    def test_already_stopped_is_not_mutated(self):
        client = PodClient('owned-pod', 'creation', 'unused')
        client.request = Mock(return_value={'id': 'owned-pod', 'createdAt': 'creation', 'status': 'EXITED'})
        self.assertEqual(client.stop(), 'EXITED')
        client.request.assert_called_once_with()
