"""Unit tests for adaptive seed allocation (--race): the paired stopping rule
(race_verdict), its inputs (objective_metric, paired_case_deltas), its effect on selection
(ranked_trials, summarize's seeds_capped) and its composition into a two-stage batch
evaluator (race_batch). No games, subprocesses or real evaluate_batch() calls are made -
evaluate_batch is faked/patched throughout, so this runs in milliseconds and never touches
runs/ or any live study.
"""
import random
import unittest
from unittest.mock import patch

from scripts.optimize_policy import (
    objective_metric, paired_case_deltas, race_batch, race_champion, race_verdict,
    ranked_trials, summarize,
)


def gaussian_deltas(mu, sigma, n, seed):
    """A reproducible synthetic paired-delta sample: `n` draws from Normal(mu, sigma)."""
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def case(seed, mode='orchard_evasion', status='horizon', sim_time=3000., score=1500., **extra):
    row = dict(case_id=f'{mode}-{seed}-{score}-{status}', seed=seed, mode=mode, status=status,
              sim_time=sim_time, score=score, wall_seconds=1.)
    row.update(extra)
    return row


class ObjectiveMetricTests(unittest.TestCase):
    def test_score_objective_reads_score(self):
        self.assertEqual(objective_metric(case(0, score=123.5), 3000., 'score'), 123.5)

    def test_survival_objective_reads_horizon_capped_sim_time(self):
        self.assertEqual(objective_metric(case(0, sim_time=4000.), 3000., 'survival'), 3000.)
        self.assertEqual(objective_metric(case(0, sim_time=1200.), 3000., 'survival'), 1200.)


class PairedCaseDeltasTests(unittest.TestCase):
    def test_matches_by_seed_and_mode_not_list_position(self):
        candidate = [case(1, score=110), case(0, score=100)]  # deliberately out of order
        reference = [case(0, score=90), case(1, score=95)]
        deltas = paired_case_deltas(candidate, reference, 3000., 'score')
        self.assertEqual(sorted(deltas), [10, 15])

    def test_errors_and_unmatched_cases_are_excluded_not_scored_as_zero_or_loss(self):
        candidate = [case(0, score=100), case(1, status='error', score=0)]
        reference = [case(0, score=90), case(2, score=200)]  # seed 1 missing, seed 2 unmatched
        deltas = paired_case_deltas(candidate, reference, 3000., 'score')
        self.assertEqual(deltas, [10])

    def test_reference_side_error_also_excludes_the_pair(self):
        candidate = [case(0, score=100)]
        reference = [case(0, status='error', score=0)]
        self.assertEqual(paired_case_deltas(candidate, reference, 3000., 'score'), [])


class RaceVerdictTests(unittest.TestCase):
    """The stopping rule itself, on synthetic paired results. This is the task's required
    check: a clear loser stops, a clear winner does not, and a near-tie is never stopped
    prematurely, even though its point estimate can be slightly negative.
    """
    CONFIDENCE = .9

    def test_clear_loser_stops(self):
        # A consistent, large paired deficit relative to the noise: an unambiguous loss.
        deltas = gaussian_deltas(mu=-60, sigma=40, n=25, seed=1)
        stop, bound = race_verdict(deltas, self.CONFIDENCE)
        self.assertTrue(stop)
        self.assertIsNotNone(bound)
        self.assertLess(bound, 0)

    def test_clear_winner_is_not_stopped(self):
        deltas = gaussian_deltas(mu=+60, sigma=40, n=25, seed=1)
        stop, bound = race_verdict(deltas, self.CONFIDENCE)
        self.assertFalse(stop)
        self.assertGreater(bound, 0)

    def test_near_tie_is_not_stopped(self):
        # True mean is small and negative - close enough to zero that a naive "stop anything
        # below the champion's mean" rule would kill it outright. The measured per-map
        # variance in this repo's fleet is "much larger than between-candidate differences";
        # sigma=40 models that. 25 noisy paired samples cannot tell mu=-3 from a tie with 90%
        # one-sided confidence, so this must NOT stop - that is the entire point of pairing
        # plus a confidence bound instead of thresholding the raw mean.
        deltas = gaussian_deltas(mu=-3, sigma=40, n=25, seed=1)
        stop, bound = race_verdict(deltas, self.CONFIDENCE)
        self.assertFalse(stop, f'near-tie (true mean -3, within noise) was stopped early '
                               f'(bound={bound}); racing must extend genuine near-ties, not '
                               f'discard them')

    def test_too_few_pairs_never_stops(self):
        # Below min_pairs there is not enough evidence to ever call a confident loser, however
        # bad the few available samples look.
        self.assertEqual(race_verdict([-1000., -1000., -1000.], .9), (False, None))
        self.assertEqual(race_verdict([], .9), (False, None))

    def test_higher_confidence_is_harder_to_satisfy(self):
        # A borderline loser: strong enough evidence to clear a lenient bound but not a strict
        # one, illustrating the compute-savings/stop-risk dial described in --race help.
        deltas = gaussian_deltas(mu=-15, sigma=40, n=25, seed=1)
        lenient_stop, _ = race_verdict(deltas, .60)
        strict_stop, _ = race_verdict(deltas, .999)
        self.assertTrue(lenient_stop)
        self.assertFalse(strict_stop)

    def test_deterministic_for_the_same_inputs(self):
        deltas = gaussian_deltas(mu=-60, sigma=40, n=25, seed=1)
        self.assertEqual(race_verdict(deltas, .9), race_verdict(deltas, .9))

    def test_zero_variance_clear_loser_is_unambiguous(self):
        # Every paired seed shows exactly the same deficit: even a degenerate (zero-variance)
        # bootstrap distribution must call this a confident loss.
        stop, bound = race_verdict([-60.]*25, .9)
        self.assertTrue(stop)
        self.assertEqual(bound, -60.)


class RankedTrialsTests(unittest.TestCase):
    @staticmethod
    def _trial(id_, rank, capped=False):
        return dict(id=id_, summary=dict(rank=rank, seeds_capped=capped))

    def test_matches_plain_rank_sort_when_nothing_is_capped(self):
        trials = [self._trial(0, [1.0]), self._trial(1, [3.0]), self._trial(2, [2.0])]
        self.assertEqual([t['id'] for t in ranked_trials(trials)], [1, 2, 0])

    def test_a_capped_trial_cannot_outrank_a_fully_measured_one_even_with_a_higher_raw_rank(self):
        # The core requirement: candidate 1 was only measured on a first-stage seed subset and
        # got a lucky rank higher than either fully-measured trial. It must still sort last.
        trials = [self._trial(0, [1.0], capped=False),
                 self._trial(1, [999.0], capped=True),
                 self._trial(2, [2.0], capped=False)]
        ranked = ranked_trials(trials)
        self.assertEqual([t['id'] for t in ranked], [2, 0])
        self.assertNotIn(1, [t['id'] for t in ranked])

    def test_falls_back_to_capped_trials_only_if_nothing_else_is_ranked(self):
        trials = [self._trial(0, [5.0], capped=True), self._trial(1, [1.0], capped=True)]
        self.assertEqual([t['id'] for t in ranked_trials(trials)], [0, 1])

    def test_trials_without_a_completed_rank_are_ignored(self):
        errored = dict(id=0, summary=dict(rank=None, errors=1))
        pending = dict(id=1)
        ranked_one = self._trial(2, [1.0])
        self.assertEqual([t['id'] for t in ranked_trials([errored, pending, ranked_one])], [2])

    def test_allow_capped_includes_everyone_sorted_by_rank(self):
        trials = [self._trial(0, [1.0]), self._trial(1, [3.0], capped=True)]
        self.assertEqual([t['id'] for t in ranked_trials(trials, allow_capped=True)], [1, 0])


class RaceChampionTests(unittest.TestCase):
    def test_none_when_nothing_is_measured_yet(self):
        self.assertIsNone(race_champion([]))

    def test_best_non_capped_trial_is_champion(self):
        trials = [dict(id=0, summary=dict(rank=[1.0], seeds_capped=False)),
                 dict(id=1, summary=dict(rank=[9.0], seeds_capped=True)),
                 dict(id=2, summary=dict(rank=[2.0], seeds_capped=False))]
        self.assertEqual(race_champion(trials)['id'], 2)


class SummarizeSeedsCappedTests(unittest.TestCase):
    @staticmethod
    def _rows(seeds, **kw):
        return [case(s, **kw) for s in seeds]

    def test_default_full_seeds_none_never_caps(self):
        summary = summarize(self._rows([0, 1, 2]), 3000.)
        self.assertEqual(summary['seeds_used'], 3)
        self.assertFalse(summary['seeds_capped'])

    def test_capped_when_fewer_distinct_seeds_than_full_seeds(self):
        summary = summarize(self._rows([0, 1, 2]), 3000., full_seeds=10)
        self.assertEqual(summary['seeds_used'], 3)
        self.assertTrue(summary['seeds_capped'])

    def test_not_capped_once_seed_count_reaches_full_seeds(self):
        summary = summarize(self._rows(range(10)), 3000., full_seeds=10)
        self.assertEqual(summary['seeds_used'], 10)
        self.assertFalse(summary['seeds_capped'])

    def test_error_branch_still_reports_seeds_used_and_capped(self):
        rows = self._rows([0, 1])+[case(2, status='error')]
        summary = summarize(rows, 3000., full_seeds=10)
        self.assertIsNone(summary['rank'])
        self.assertEqual(summary['seeds_used'], 3)
        self.assertTrue(summary['seeds_capped'])

    def test_existing_positional_callers_are_unaffected(self):
        # research_loop.py/research_study.py/research_backfill.py/research_parallel.py/
        # research_async_study.py all call summarize(rows, seconds) or
        # summarize(rows, seconds, enabled_count) with no knowledge of --race; full_seeds must
        # default to never capping so their behaviour is unchanged.
        summary = summarize(self._rows([0, 1]), 3000, 2)
        self.assertFalse(summary['seeds_capped'])
        self.assertEqual(summary['seeds_used'], 2)


class RaceBatchTests(unittest.TestCase):
    """race_batch()'s stop/extend orchestration, with evaluate_batch replaced by a fake that
    reports a fixed score offset per trial id and seed, and calls on_complete synchronously -
    the same contract the real evaluate_batch honours. This isolates the two-stage wiring
    (which seeds get requested, in how many calls, for which trials) from both game execution
    and from race_verdict's own statistics (covered separately above).
    """

    def setUp(self):
        self.calls = []  # (sorted trial ids, seed list) per fake evaluate_batch call

    def _fake_evaluate_batch(self, offsets):
        def fake(trials, seeds, protocol, args, control, deadline, on_complete=None, **_kw):
            self.calls.append((sorted(t['id'] for t in trials), list(seeds)))
            completed = {}
            for trial in trials:
                rows = [case(s, score=1500.+offsets[trial['id']]) for s in seeds]
                completed[trial['id']] = rows
                if on_complete is not None:
                    on_complete(trial, rows)
            return completed
        return fake

    @staticmethod
    def _args():
        class Args:
            pass
        args = Args()
        args.train_seeds = list(range(100))
        args.race = True
        args.race_first = 25
        args.race_max = 100
        args.race_confidence = .9
        args.seconds = 3000.
        args.objective = 'score'
        return args

    @staticmethod
    def _save_trial(trial, results):
        trial['case_ids'] = [r['case_id'] for r in results]
        trial['summary'] = summarize(results, 3000., 0, 'score', full_seeds=100)

    def _champion(self, id_=0):
        trial = dict(id=id_, label='champion', baseline=True, config={})
        self._save_trial(trial, [case(s, score=1500.) for s in range(100)])
        return trial

    def test_clear_loser_is_stopped_at_the_first_stage_seed_count(self):
        champion = self._champion()
        loser = dict(id=1, label='loser', baseline=False, config={})
        args = self._args()
        state = dict(trials=[champion])
        with patch('scripts.optimize_policy.evaluate_batch', self._fake_evaluate_batch({0: 0., 1: -60.})):
            race_batch([loser], {}, args, None, 0., self._save_trial, state)
        self.assertEqual(loser['summary']['seeds_used'], 25)
        self.assertTrue(loser['summary']['seeds_capped'])
        self.assertEqual(loser['race']['decision'], 'stopped')
        # Only the stage-1 call (seeds 0-24) happened; nothing requested seeds >= 25.
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(all(s < 25 for s in self.calls[0][1]))

    def test_clear_winner_is_extended_to_the_full_seed_count(self):
        champion = self._champion()
        winner = dict(id=1, label='winner', baseline=False, config={})
        args = self._args()
        state = dict(trials=[champion])
        with patch('scripts.optimize_policy.evaluate_batch', self._fake_evaluate_batch({0: 0., 1: 60.})):
            race_batch([winner], {}, args, None, 0., self._save_trial, state)
        self.assertEqual(winner['summary']['seeds_used'], 100)
        self.assertFalse(winner['summary']['seeds_capped'])
        self.assertEqual(winner['race']['decision'], 'extended')
        # Stage 1 (champion+winner, seeds 0-24) then stage 2 (winner only, seeds 25-99).
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0], ([0, 1], list(range(25))))
        self.assertEqual(self.calls[1], ([1], list(range(25, 100))))

    def test_mixed_batch_only_extends_the_winner(self):
        champion = self._champion()
        loser = dict(id=1, label='loser', baseline=False, config={})
        winner = dict(id=2, label='winner', baseline=False, config={})
        args = self._args()
        state = dict(trials=[champion])
        offsets = {0: 0., 1: -60., 2: 60.}
        with patch('scripts.optimize_policy.evaluate_batch', self._fake_evaluate_batch(offsets)):
            race_batch([loser, winner], {}, args, None, 0., self._save_trial, state)
        self.assertEqual((loser['summary']['seeds_used'], loser['summary']['seeds_capped']), (25, True))
        self.assertEqual((winner['summary']['seeds_used'], winner['summary']['seeds_capped']), (100, False))
        # Stage 2 must name only the winner, never the already-stopped loser.
        stage2 = [c for c in self.calls if c[1] and min(c[1]) >= 25]
        self.assertEqual(stage2, [([2], list(range(25, 100)))])

    def test_no_champion_yet_measures_straight_to_the_full_budget(self):
        first = dict(id=0, label='baseline', baseline=True, config={})
        args = self._args()
        state = dict(trials=[])  # nothing measured yet: nothing to race against
        with patch('scripts.optimize_policy.evaluate_batch', self._fake_evaluate_batch({0: 0.})):
            race_batch([first], {}, args, None, 0., self._save_trial, state)
        self.assertEqual(first['summary']['seeds_used'], 100)
        self.assertFalse(first['summary']['seeds_capped'])
        self.assertNotIn('race', first)  # never raced - there was no incumbent to compare to
        self.assertEqual(self.calls, [([0], list(range(100)))])

    def test_interrupted_stage_one_leaves_the_trial_unsummarized_for_resume(self):
        champion = self._champion()
        pending = dict(id=1, label='pending', baseline=False, config={})
        args = self._args()
        state = dict(trials=[champion])

        def fake(trials, seeds, protocol, args, control, deadline, on_complete=None, **_kw):
            self.calls.append((sorted(t['id'] for t in trials), list(seeds)))
            # Simulate a deadline/STOP mid-stage: the new candidate never finishes, but the
            # champion's cache-hit rows still resolve (see race_batch's docstring).
            completed = {champion['id']: [case(s, score=1500.) for s in seeds]}
            if on_complete is not None:
                on_complete(champion, completed[champion['id']])
            return completed

        with patch('scripts.optimize_policy.evaluate_batch', fake):
            race_batch([pending], {}, args, None, 0., self._save_trial, state)
        self.assertNotIn('summary', pending)
        self.assertNotIn('race', pending)


if __name__ == '__main__':
    unittest.main()
