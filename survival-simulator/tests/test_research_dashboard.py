"""The monitoring surface must not expose holdout or private campaign data."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('research_dashboard', Path(__file__).parents[1] / 'scripts' / 'research_dashboard.py')
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('state.json', dict(status='running', major=1, sub=1, best=None,
            events={'final.result': dict(kind='final assessment', details={'secret_holdout': 999})}))
        self.write('config.json', dict(evaluation=dict(major_seeds=[0, 1], holdout_seeds=[1001])))

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def case(self, case_id, seed):
        relative = f'evaluation-cache/cases/{case_id}/attempt-1'
        self.write(f'evaluation-cache/cases/{case_id}/active.json', dict(seed=seed, attempt=str(self.root / relative)))
        self.write(f'{relative}/result.json', dict(sim_time=300, diagnostics=dict(
            mean_absorbed_energy_per_fruit=30, terminal_world={'private_world': True},
            screenshots=['frame-000001.png'])))
        self.write(f'{relative}/diagnostics/summary.json', dict(screenshots=['frame-000001.png']))
        path = self.root / relative / 'diagnostics/frame-000001.png'
        path.write_bytes(b'\x89PNG\r\n\x1a\nTEST')
        return relative

    def test_holdout_never_opened_and_large_worlds_not_exported(self):
        self.case('a' * 64, 0)
        holdout = self.case('b' * 64, 1001)
        final = self.write('steps/final-holdout/comparison.json', {'secret_holdout': 999})
        self.write('steps/unexpected/plan.json', dict(seeds=[1001]))
        unexpected = self.write('steps/unexpected/comparison.json', {'secret_holdout': 999})
        original = dashboard.read

        def guard(path, *args):
            self.assertNotEqual(path, final)
            self.assertNotEqual(path, unexpected)
            self.assertFalse(path.is_relative_to(self.root / holdout))
            return original(path, *args)

        with patch.object(dashboard, 'read', side_effect=guard):
            data = dashboard.collect(self.root)
        self.assertEqual(data['completed_cases'], 1)
        self.assertEqual(data['cases'][0]['seed'], 0)
        serialized = json.dumps(data)
        self.assertNotIn('secret_holdout', serialized)
        self.assertNotIn('private_world', serialized)

    def test_only_manifested_development_images_are_readable(self):
        self.case('a' * 64, 0)
        self.case('b' * 64, 1001)
        self.assertIn('png', dashboard.collect_image(self.root, 'a' * 64, 'frame-000001.png'))
        for case_id, filename in [('b' * 64, 'frame-000001.png'), ('a' * 64, '../state.json'),
                                  ('a' * 64, 'frame-999999.png')]:
            with self.assertRaises(ValueError):
                dashboard.collect_image(self.root, case_id, filename)

    def test_offline_feed_preserves_last_success(self):
        self.write('dashboard-latest.json', dict(data={'campaign': 'saved'}, last_success=123))
        feed = dashboard.Feed(self.root, 15)
        with patch.object(feed, 'remote', side_effect=RuntimeError('offline')):
            feed.update()
        self.assertFalse(feed.snapshot()['connected'])
        self.assertEqual(feed.snapshot()['data']['campaign'], 'saved')
        self.assertEqual(feed.snapshot()['last_success'], 123)

    def test_native_summary_cannot_replace_authoritative_best(self):
        self.write('state.json', dict(status='running', best=dict(id='candidate', label='best')))
        for name, engine, mean in [('python', 'python', 300), ('native', 'fastsim', 999)]:
            self.write(f'steps/{name}/comparison.json', dict(engine=engine, seeds=[0], complete=True,
                summaries={'candidate': dict(mean_survival=mean)}))
        self.assertEqual(dashboard.collect(self.root)['best_summary']['mean_survival'], 300)

    def test_best_card_follows_accepted_candidate_not_highest_mean(self):
        self.write('state.json', dict(status='running', best=dict(id='incumbent', label='retained')))
        self.write('steps/comparison/comparison.json', dict(engine='python', seeds=[0, 1], complete=True,
            summaries={'incumbent': dict(mean_survival=596.5), 'challenger': dict(mean_survival=865.6)}))
        self.assertEqual(dashboard.collect(self.root)['best_summary']['mean_survival'], 596.5)
        self.write('state.json', dict(status='running', best=dict(id='challenger', label='promoted')))
        self.assertEqual(dashboard.collect(self.root)['best_summary']['mean_survival'], 865.6)

    def test_incomplete_newer_comparison_does_not_replace_completed_measurement(self):
        self.write('state.json', dict(status='running', best=dict(id='incumbent', label='retained')))
        for name, complete, mean, stamp in [('finished', True, 300, 100), ('partial', False, 900, 200)]:
            path = self.write(f'steps/{name}/comparison.json', dict(engine='python', seeds=[0], complete=complete,
                summaries={'incumbent': dict(mean_survival=mean)}))
            os.utime(path, (stamp, stamp))
        summary = dashboard.collect(self.root)['best_summary']
        self.assertEqual(summary['mean_survival'], 300)
        self.assertEqual(summary['comparison'], 'finished')
        self.assertEqual(summary['evaluated_at'], 100)

    def fleet(self, seed):
        folder = self.root / 'research-capture-fleet-fresh' / 'guide-delivery'
        self.write('operator-controls/active-fleet.json', dict(root=str(folder.parent)))
        plan = self.write('research-capture-fleet-fresh/guide-delivery/plan.json', dict(
            campaign=str(self.root), screening_seeds=[seed], comparison_seeds=[seed+1], trials=24))
        self.write('research-capture-fleet-fresh/fleet.json', dict(version=2, campaign=str(self.root),
            maximum_auxiliary_compute_usd=8, tasks=[dict(name='guide-delivery', title='Guide delivery',
                root=str(folder), plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())]))
        self.write('research-capture-fleet-fresh/guide-delivery/state.json', dict(status='running', pod_id='worker'))
        self.write('research-capture-fleet-fresh/fleet-guard.heartbeat.json',
                   dict(updated_at=dashboard.time.time(), checked=['worker']))
        return folder

    def test_registered_fleet_can_show_fresh_development_seeds(self):
        folder = self.fleet(500000)
        attempt = folder / 'evaluation-cache/cases/fresh/attempt-1'
        self.write(str((attempt.parent / 'active.json').relative_to(self.root)),
                   dict(seed=500000, attempt=str(attempt)))
        self.write(str((attempt / 'progress.json').relative_to(self.root)), dict(seed=500000, sim_time=42))
        row = dashboard.collect(self.root)['fleet']['tasks'][0]
        self.assertTrue(row['guard_armed'])
        self.assertEqual(row['active_cases'][0]['sim_time'], 42)
        # Tampered plans are not allowed to introduce new seed panels.
        (folder / 'plan.json').write_text((folder / 'plan.json').read_text() + ' ')
        self.assertEqual(dashboard.collect(self.root)['fleet']['tasks'], [])

    def test_fleet_holdout_is_rejected_before_reading_any_results(self):
        folder = self.fleet(1001)
        original = dashboard.read

        def guard(path, *args):
            self.assertNotEqual(path, folder / 'state.json')
            self.assertNotEqual(path, folder / 'comparison.json')
            return original(path, *args)

        with patch.object(dashboard, 'read', side_effect=guard):
            self.assertEqual(dashboard.collect(self.root)['fleet']['tasks'], [])

    def test_eight_pod_metrics_require_guard_coverage_for_all_members(self):
        queue = self.root/'dispatch'
        pods = [str(i) for i in range(8)]
        self.write('operator-controls/throughput.json', dict(queue=str(queue)))
        self.write('dispatch/config.json', dict(pods=pods, deadline_epoch=9999999999,
            guard_journal=str(queue/'guard-scaled.jsonl'), background_root=str(self.root/'background')))
        for pod in pods:
            self.write(f'dispatch/workers/{pod}.json', dict(id=pod, cpu_percent=90, busy_slots=30, updated_at=dashboard.time.time()))
        heartbeat = dict(updated_at=dashboard.time.time(), checked=pods)
        self.write('dispatch/guard-scaled.heartbeat.json', heartbeat)
        result = dashboard.collect(self.root)
        self.assertEqual(len(result['dispatch']['workers']), 8)
        self.assertEqual(result['dispatch']['compute_hourly_usd'], 7.68)
        self.assertTrue(result['watchdog']['observed'])
        heartbeat['checked'] = pods[:-1]
        self.write('dispatch/guard-scaled.heartbeat.json', heartbeat)
        self.assertFalse(dashboard.collect(self.root)['watchdog']['observed'])


if __name__ == '__main__':
    unittest.main()
