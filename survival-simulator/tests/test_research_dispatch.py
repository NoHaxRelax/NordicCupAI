"""Queue safety and asynchronous refill, without running game experiments."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import research_dispatch as dispatch
from research_async_study import stream
from research_recovery_runtime import throughput_budget
from runpod_throughput_guard import verify


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.root/'campaign'
        self.source = self.campaign/'snapshots'/'abc'/'code'
        self.source.mkdir(parents=True)
        dispatch.write(self.campaign/'config.json', dict(evaluation=dict(holdout_seeds=[900], retired_holdout_seeds=[800])))
        self.settings = dict(campaign=str(self.campaign), output_roots=[str(self.campaign)])
        self.job = dict(source=str(self.source), cache=str(self.campaign/'cache'),
            control=str(self.campaign/'steps/development/control'), timeout=7200,
            request=dict(seed=42, seconds=3000, config={}, engine='python', baseline=False))

    def test_holdout_requires_exact_frozen_selection_and_final_control(self):
        dispatch.validate_job(self.job, self.settings)
        self.job['request']['seed'] = 900
        with self.assertRaises(ValueError): dispatch.validate_job(self.job, self.settings)
        dispatch.write(self.campaign/'final-selection.json', dict(holdout_seeds=[900],
            original=dict(snapshot='abc', config={}, baseline=False)))
        with self.assertRaises(ValueError): dispatch.validate_job(self.job, self.settings)
        self.job['control'] = str(self.campaign/'steps/final-holdout/control')
        dispatch.validate_job(self.job, self.settings)
        self.job['request']['seed'] = 800
        with self.assertRaises(ValueError): dispatch.validate_job(self.job, self.settings)

    def test_outputs_cannot_escape_registered_roots(self):
        self.job['cache'] = str(self.root/'outside')
        with self.assertRaises(ValueError): dispatch.validate_job(self.job, self.settings)

    def test_refills_before_slowest_existing_game_finishes(self):
        release = threading.Event()
        refill = threading.Event()
        trials = iter([dict(id=i) for i in range(3)])
        finished = []
        def evaluate(request):
            if request['case_id'] == '0':
                if not release.wait(3): raise AssertionError('Refill blocked behind straggler')
            if request['case_id'] == '2':
                refill.set()
                release.set()
            return dict(case_id=request['case_id'], status='extinct')
        stream(lambda: next(trials, None), lambda t: [dict(case_id=str(t['id']))], evaluate,
               lambda trial, rows: finished.append(trial['id']), slots=2, stopped=lambda: False, pulse=lambda: None)
        self.assertTrue(refill.is_set())
        self.assertEqual(set(finished), {0, 1, 2})

    def test_candidate_summary_waits_for_entire_seed_panel(self):
        trials = iter([dict(id=0)])
        rows = []
        stream(lambda: next(trials, None), lambda _: [dict(case_id='a'), dict(case_id='b')],
            lambda r: dict(**r, status='interrupted' if r['case_id']=='b' else 'extinct'),
            lambda *_: rows.append(True), slots=2, stopped=lambda: False, pulse=lambda: None)
        self.assertEqual(rows, [])

    def test_expanded_budget_preserves_original_and_generation_target(self):
        base = dict(budget=dict(total_usd=40), storage={}, schedule=dict(major_generations=8))
        original = copy.deepcopy(base)
        effective = throughput_budget(base, dict(authorization='continue-four-pods-through-generation-8', max_pods=4))
        self.assertEqual(base, original)
        self.assertEqual(effective['schedule']['major_generations'], 8)
        b = effective['budget']
        self.assertLess(b['external_spend_usd']+b['reserve_usd']+72*b['hourly_rate_usd'], b['total_usd'])

    def test_guard_rejects_more_pods_or_unbounded_runtime(self):
        cfg = dict(authorization='continue-four-pods-through-generation-8', primary_id='0',
            deadline_epoch=72*3600, pods=[dict(id=str(i), createdAt='1970-01-01T00:00:00Z', hourly_rate_usd=.97) for i in range(4)])
        verify(cfg)
        cfg['deadline_epoch'] += 1
        with self.assertRaises(ValueError): verify(cfg)

    def test_urgent_evaluations_do_not_wait_behind_full_background_pool(self):
        self.assertEqual(dispatch.admission(40, 30, 0, 30, 0), 0)
        self.assertEqual(dispatch.admission(5, 30, 0, 30, 0), 8)
        self.assertEqual(dispatch.admission(5, 30, 0, 38, 8), 0)
        self.assertEqual(dispatch.admission(5, 30, 10, 25, 5), 3)

    def test_scaled_guard_accepts_incremental_expansion_within_authorization(self):
        cfg = dict(authorization='scale-research-fleet-through-generation-8', max_pods=8,
            primary_id='0', deadline_epoch=72*3600,
            pods=[dict(id=str(i), createdAt='1970-01-01T00:00:00Z', hourly_rate_usd=.97) for i in range(4)])
        verify(cfg)
        cfg['pods'] += [dict(id=str(i), createdAt='1970-01-01T00:00:00Z', hourly_rate_usd=.97) for i in range(4, 8)]
        verify(cfg)
        cfg['pods'].append(dict(id='8', createdAt='1970-01-01T00:00:00Z', hourly_rate_usd=.97))
        with self.assertRaises(ValueError): verify(cfg)

    def test_scaled_runtime_covers_entire_fleet_until_time_failsafe(self):
        for pods in (8, 12):
            value = throughput_budget(dict(budget={}, storage={}),
                dict(authorization='scale-research-fleet-through-generation-8', max_pods=pods))['budget']
            self.assertLess(15+72*value['hourly_rate_usd'], value['total_usd'])
            self.assertGreater(value['hourly_rate_usd'], pods*.96)


if __name__ == '__main__': unittest.main()
