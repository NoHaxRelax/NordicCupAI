"""Supervisor contracts without simulations, network, Codex, or subprocesses."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from scripts.research_support import (
    read, write, manifest, copy_source, assert_snapshot, changed_policy,
    remaining_seconds, promotion, research_case, validate_config, storage_status, apply_proposed_edits,
)
from scripts.research_loop import Supervisor
from scripts.research_bo import encode, choose

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    value = read(ROOT/'scripts/research_config.json')
    value['budget']['hourly_rate_usd'] = 1.
    return value


def rows(times, *, seeds=None, score=100):
    return [dict(seed=seed, policy_seed=0, sim_time=seconds, score=score,
                 status='horizon' if seconds >= 3000 else 'extinct')
            for seed, seconds in zip(seeds or range(len(times)), times)]


class ResearchContracts(unittest.TestCase):
    def test_literal_proposals_validate_all_paths_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'models').mkdir()
            source = root/'models/core.py'
            source.write_text('value = 1\n')
            edits = [dict(path='models/core.py', old_text='value = 1', new_text='value = 2')]
            for name in ('src/core.py', 'models/../src/core.py', '/tmp/outside.py'):
                with self.assertRaises(ValueError):
                    apply_proposed_edits(root, edits+[dict(path=name, old_text='', new_text='bad')])
                self.assertEqual(source.read_text(), 'value = 1\n')

    def test_literal_proposals_reject_ambiguous_or_stale_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'models').mkdir()
            source = root/'models/core.py'
            source.write_text('value = 1\nvalue = 1\n')
            for old in ('value = 1', 'missing', ''):
                with self.assertRaises(ValueError):
                    apply_proposed_edits(root, [dict(path='models/core.py', old_text=old, new_text='new')])
            self.assertEqual(source.read_text(), 'value = 1\nvalue = 1\n')

    def test_literal_proposals_create_and_edit_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = 'models/survival/new_feature.py'
            edits = [dict(path=name, old_text='', new_text='value = 1\n'),
                     dict(path=name, old_text='value = 1', new_text='value = 2')]
            self.assertEqual(apply_proposed_edits(root, edits), [name])
            self.assertEqual((root/name).read_text(), 'value = 2\n')

    def test_legacy_landlock_requires_readonly_proposals(self):
        config = configuration()
        config['agent']['legacy_landlock'] = True
        with self.assertRaises(ValueError):
            validate_config(config)
        config['agent']['proposal_only'] = True
        validate_config(config)

    def test_control_equivalence_cannot_select_candidates_or_use_holdout(self):
        supervisor = Supervisor.__new__(Supervisor)
        for report in ({'engine': 'fastsim'}, {'control_equivalence': True}):
            with self.assertRaises(ValueError):
                supervisor.winner(report)
        with self.assertRaises(ValueError):
            supervisor.evaluate('final', [], [], final=True, control_equivalence=True)
        with self.assertRaises(ValueError):
            supervisor.evaluate('a-promotion', [], [], control_equivalence=True)

    def test_storage_preserves_final_reserve_and_active_worker_headroom(self):
        config = configuration()
        config['storage'] = dict(campaign_gib=1., final_reserve_gib=.7, minimum_free_gib=.1)
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.research_support.shutil.disk_usage', return_value=Mock(free=10*1024**3)):
                self.assertTrue(storage_status(folder, config, 2)['exhausted'])
                self.assertFalse(storage_status(folder, config, 2, final=True)['exhausted'])
            with patch('scripts.research_support.shutil.disk_usage', return_value=Mock(free=200*1024**2)):
                self.assertTrue(storage_status(folder, config, 2, final=True)['exhausted'])

    def test_budget_counts_downtime_and_agent_reservations(self):
        config = configuration()
        config['budget'].update(total_usd=10, reserve_usd=1, max_hours=20)
        config['schedule']['final_reserve_seconds'] = 3600
        state = dict(started_at=1000, agent_reserved_usd=2)
        self.assertEqual(remaining_seconds(config, state, 1000+2*3600), 4*3600)
        self.assertEqual(remaining_seconds(config, state, 1000+3*3600), 3*3600)
        self.assertEqual(remaining_seconds(config, state, 1000+3*3600, final=True), 4*3600)
        self.assertEqual(remaining_seconds(config, state, 1000+20*3600), 0)

    def test_explicit_rate_and_untouched_holdout_required(self):
        config = configuration()
        validate_config(config)
        config['budget']['hourly_rate_usd'] = None
        with self.assertRaises(ValueError):
            validate_config(config)
        config = configuration()
        config['evaluation']['holdout_seeds'].append(0)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_failed_or_unpaired_cases_never_promote(self):
        rules = configuration()['promotion']
        self.assertFalse(promotion(rows([100, 200]), rows([200, 300], seeds=[1, 2]), rules)['promote'])
        # Error results may legitimately lack policy_seed or sim_time.
        self.assertFalse(promotion(rows([100]), [{'status': 'error', 'seed': 0}], rules)['promote'])
        self.assertFalse(promotion(rows([100, 200]), rows([300]), rules)['promote'])

    def test_mean_gain_does_not_hide_worst_case_regression(self):
        result = promotion(rows([500, 500, 500, 500]), rows([100, 1500, 1500, 1500]), configuration()['promotion'])
        self.assertFalse(result['promote'])
        self.assertLess(result['worst_survival_delta'], 0)

    def test_clear_paired_gain_and_ceiling_score_gain(self):
        rules = configuration()['promotion']
        self.assertTrue(promotion(rows([500]*8), rows([700]*8), rules)['promote'])
        self.assertFalse(promotion(rows([500]*8), rows([500]*8), rules)['promote'])
        self.assertTrue(promotion(rows([3000]*8), rows([3000]*8, score=120), rules)['promote'])

    def test_snapshots_include_untracked_policy_and_protect_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            root.mkdir()
            (root/'models').mkdir()
            (root/'models/core.py').write_text('untracked = True\n')
            (root/'agent_server.py').write_text('endpoint = True\n')
            (root/'src').mkdir()
            (root/'src/core.py').write_text('physics = 1\n')
            files = manifest(root)
            target = Path(directory)/'frozen'
            copy_source(root, target)
            assert_snapshot(target, files)
            (root/'models/core.py').write_text('untracked = False\n')
            self.assertEqual(changed_policy(files, manifest(root)), ['models/core.py'])
            (root/'src/core.py').write_text('physics = 2\n')
            with self.assertRaises(ValueError):
                changed_policy(files, manifest(root))
            # The source edit never changed the preserved candidate.
            assert_snapshot(target, files)

    def test_cache_identity_reuses_cases_only_with_same_provenance(self):
        args = ('snapshot', {'python': '3.12'}, {'features': {}}, False, 1, 0)
        identity = research_case(*args)['case_id']
        self.assertEqual(identity, research_case(*args)['case_id'])
        for changed in [('other', *args[1:]), (*args[:4], 2, 0),
                        (args[0], {'python': '3.13'}, *args[2:])]:
            self.assertNotEqual(identity, research_case(*changed)['case_id'])

    def test_bo_encoding_distinguishes_disabled_sentinels(self):
        specs = [dict(path='age', kind='float', low=40., high=120.)]
        getter = lambda config, path: config[path]
        normal = encode({'age': 120.}, specs, getter)
        disabled = encode({'age': 1e9}, specs, getter)
        missing = encode({'age': None}, specs, getter)
        self.assertNotEqual(normal, disabled)
        self.assertNotEqual(disabled, missing)

    def test_bo_batch_does_not_select_the_same_pool_entry_twice(self):
        specs = [dict(path='x', kind='float', low=0., high=1.)]
        observations = [dict(config={'x': i/8}, summary={'rank': [i]}) for i in range(6)]
        pool = [dict(x=.8, tag='a'), dict(x=.8, tag='b'), dict(x=.9, tag='c')]
        selected = choose(pool, observations, specs, lambda config, path: config[path], 3)
        self.assertEqual(len({item['tag'] for item in selected}), 3)


class ResearchSchedule(unittest.TestCase):
    def fake(self, folder):
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.out = Path(folder)
        supervisor.config = configuration()
        supervisor.stopping = False
        incumbent = dict(id='original', snapshot='source', config={}, baseline=False, label='best')
        supervisor.state = dict(version=1, stage='sub', status='running', major=1, sub=1,
            best=incumbent, original=incumbent, started_at=time.time(), major_start_id='original',
            stagnant_subs=0, stagnant_majors=0, events={}, steps={}, agent_reserved_usd=0.)
        supervisor.save = Mock()
        supervisor.upstream = Mock(return_value={'status': 'unavailable'})
        supervisor.compare_upstream = Mock(side_effect=lambda key, review, best: best)
        supervisor.final = Mock(side_effect=lambda: supervisor.state.update(stage='done', status='complete'))
        return supervisor

    def test_resuming_does_not_extend_saved_step_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = self.fake(folder)
            supervisor.state['started_at'] = 1000
            with patch('scripts.research_loop.time.time', return_value=1000):
                _, first = supervisor.step('example', 600)
            deadline = first['deadline']
            with patch('scripts.research_loop.time.time', return_value=1100):
                _, resumed = supervisor.step('example', 600)
            self.assertEqual(resumed['deadline'], deadline)

    def test_four_subgenerations_then_major_review(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = self.fake(folder)
            supervisor.config['schedule']['major_generations'] = 2
            def advance(key, incumbent, upstream, major=False):
                return dict(winner=dict(incumbent, id=key), improved=True, stop_early=False)
            supervisor.round = Mock(side_effect=advance)
            with patch('scripts.research_loop.signal.signal'):
                supervisor.run()
            self.assertEqual([call.args[0] for call in supervisor.round.call_args_list],
                ['g1.1', 'g1.2', 'g1.3', 'g1.4', 'g1.major', 'g2.1', 'g2.2', 'g2.3', 'g2.4', 'g2.major'])
            supervisor.final.assert_called_once()

    def test_stagnation_advances_to_major_review_and_stops_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = self.fake(folder)
            supervisor.round = Mock(side_effect=lambda key, incumbent, upstream, major=False:
                                    dict(winner=incumbent, improved=False, stop_early=False))
            with patch('scripts.research_loop.signal.signal'):
                supervisor.run()
            self.assertEqual([call.args[0] for call in supervisor.round.call_args_list],
                             ['g1.1', 'g1.2', 'g1.major', 'g2.1', 'g2.2', 'g2.major'])

    def test_holdout_selection_prevents_further_agent_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = self.fake(folder)
            write(Path(folder)/'final-selection.json', {'already': 'frozen'})
            supervisor.round = Mock(side_effect=AssertionError('Must never tune after holdout opens'))
            with patch('scripts.research_loop.signal.signal'):
                supervisor.run()
            supervisor.round.assert_not_called()
            supervisor.final.assert_called_once()
            with self.assertRaises(RuntimeError):
                supervisor.execute('forbidden-agent', 100, Mock(), agent=True)
            with self.assertRaises(RuntimeError):
                supervisor.search('forbidden-search', 'unused', {}, {}, False)

    def test_final_report_never_promotes_from_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = self.fake(folder)
            selected = dict(supervisor.state['best'], id='selected')
            supervisor.state['best'] = selected
            supervisor.evaluate = Mock(return_value=dict(complete=False, summaries={}, rows={}))
            Supervisor.final(supervisor)
            self.assertEqual(supervisor.state['best'], selected)
            self.assertEqual(read(Path(folder)/'final-selection.json')['selected'], selected)
            self.assertEqual(supervisor.state['status'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
