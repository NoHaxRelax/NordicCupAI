from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tuning_evaluation import atomic_json, evaluate_seed

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
if HAS_OPTUNA:
    import optuna
    import tune_policy as tuning
    from optuna.trial import TrialState
    from src.utils.controllers.expert_policy import load_config
    from src.utils.controllers.global_planner import load_planner_config


class ImmediatePool:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def episode(seed, score=10, extinct=False):
    return dict(seed=seed, score=score, extinct=extinct, population=0 if extinct else 12)


class EvaluationCacheTests(unittest.TestCase):
    def test_deadline_does_not_cache_a_truncated_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed.json"
            self.assertIsNone(evaluate_seed({}, {}, 1, 600, path, time.monotonic() - 1))
            self.assertFalse(path.exists())

    def test_completed_episode_is_reused_even_after_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed.json"
            result = dict(episode(1), horizon_seconds=600)
            atomic_json(path, result)
            self.assertEqual(evaluate_seed({}, {}, 1, 600, path, 0), result)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                evaluate_seed({}, {}, 1, 3000, path, 0)


@unittest.skipUnless(HAS_OPTUNA, "Install requirements-tuning.txt to test the optimizer")
class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.base = load_config().model_dump(mode="json")
        self.base["harvest"]["enabled"] = True
        self.planner = load_planner_config().model_dump(mode="json")
        self.planner["biome_inference"]["adapt_to_wall_clock"] = False
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    def study(self):
        return tuning.open_study(self.output, 123, 1, self.base)

    def args(self, **updates):
        values = dict(output=self.output, workers=4, train_seeds=[1, 7], train_seconds=600, trials=3)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_search_aggregates_all_seeds_and_includes_baseline(self):
        study = self.study()
        seen = []

        def fake(expert, planner, seed, seconds, path, deadline):
            seen.append((expert, seed, seconds, str(path)))
            return episode(seed, seed + expert["harvest"]["ripe_energy"])

        with patch.object(tuning, "evaluate_seed", side_effect=fake):
            tuning.search(ImmediatePool(), study, self.args(), self.base, self.planner,
                          time.monotonic() + 30, threading.Event())
        complete = study.get_trials(states=(TrialState.COMPLETE,))
        self.assertEqual(len(complete), 3)
        self.assertTrue(complete[0].user_attrs["baseline"])
        for trial in complete:
            self.assertAlmostEqual(trial.value, 4 + trial.params["harvest.ripe_energy"])
            self.assertEqual(trial.user_attrs["metrics"]["seeds"], [1, 7])
        self.assertEqual(len(seen), 6)
        self.assertEqual(len({row[3] for row in seen}), 6)
        self.assertEqual(self.base["harvest"]["ripe_energy"], 56)
        best = json.loads((self.output / "best_search_expert_policy.json").read_text())
        self.assertTrue(best["harvest"]["enabled"])
        # Search winners must not masquerade as full-horizon validation winners.
        self.assertFalse((self.output / "best_expert_policy.json").exists())

    def test_interrupted_trial_never_becomes_a_scored_observation(self):
        study = self.study()
        stop = threading.Event()

        def interrupted(*args):
            stop.set()
            return None

        with patch.object(tuning, "evaluate_seed", side_effect=interrupted):
            tuning.search(ImmediatePool(), study, self.args(trials=1), self.base, self.planner,
                          time.monotonic() + 30, stop)
        self.assertEqual(len(study.get_trials(states=(TrialState.COMPLETE,))), 0)
        self.assertEqual(len(study.get_trials(states=(TrialState.FAIL,))), 1)
        self.assertEqual(len(study.get_trials(states=(TrialState.WAITING,))), 1)

    def test_resume_recovers_running_parameters_without_losing_completed_trials(self):
        study = self.study()
        baseline = study.ask(fixed_distributions=tuning.SPACE)
        study.tell(baseline, 10.)
        interrupted = study.ask(fixed_distributions=tuning.SPACE)
        original = dict(interrupted.params)
        resumed = self.study()
        self.assertEqual(len(resumed.get_trials(states=(TrialState.COMPLETE,))), 1)
        self.assertEqual(resumed.trials[interrupted.number].state, TrialState.FAIL)
        retry = resumed.ask(fixed_distributions=tuning.SPACE)
        self.assertEqual(retry.params, original)
        self.assertEqual(retry.user_attrs["retry_of"], interrupted.number)

    def test_manifest_rejects_changed_configs_seeds_and_code(self):
        path = self.output / "manifest.json"
        expected = dict(source_hash="abc", train_seeds=[1, 7], expert=self.base)
        tuning.check_manifest(path, expected)
        tuning.check_manifest(path, expected)
        for key, value in (("source_hash", "def"), ("train_seeds", [2]), ("expert", {})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                tuning.check_manifest(path, {**expected, key: value})
        self.assertEqual(json.loads(path.read_text()), expected)

    def test_partial_validation_cannot_win_or_displace_baseline(self):
        candidate = tuning.configured(self.base, {"harvest.ripe_energy": 52.})
        candidates = [dict(label="baseline", expert=self.base), dict(label="trial-1", expert=candidate)]
        results = {"baseline": {1: episode(1, 10), 7: episode(7, 12)}, "trial-1": {1: episode(1, 100)}}
        tuning.validation_report(self.output, candidates, results, [1, 7], self.planner)
        report = json.loads((self.output / "validation_results.json").read_text())
        self.assertEqual(report["winner"], "baseline")
        self.assertFalse(report["all_candidates_complete"])
        results["trial-1"][7] = episode(7, 20, extinct=True)
        tuning.validation_report(self.output, candidates, results, [1, 7], self.planner)
        report = json.loads((self.output / "validation_results.json").read_text())
        self.assertEqual(report["winner"], "trial-1")
        self.assertTrue(report["all_candidates_complete"])
        self.assertEqual(report["candidates"][1]["metrics"]["extinction_rate"], .5)
        exported = json.loads((self.output / "best_expert_policy.json").read_text())
        self.assertEqual(exported, candidate)

    def test_validation_requires_complete_baseline(self):
        candidate = tuning.configured(self.base, {"harvest.ripe_energy": 52.})
        candidates = [dict(label="baseline", expert=self.base), dict(label="trial-1", expert=candidate)]
        results = {"baseline": {1: episode(1)}, "trial-1": {1: episode(1, 30), 7: episode(7, 20)}}
        tuning.validation_report(self.output, candidates, results, [1, 7], self.planner)
        self.assertFalse((self.output / "best_expert_policy.json").exists())

    def test_live_coordinator_cannot_be_replaced_and_lock_releases(self):
        with tuning.coordinator_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                with tuning.coordinator_lock(self.output):
                    self.fail("Second coordinator acquired the same directory")
        with tuning.coordinator_lock(self.output):
            pass

    def test_finalists_are_distinct_and_always_include_baseline(self):
        baseline = tuning.baseline_params(self.base)
        other = {**baseline, "harvest.ripe_energy": 52.}
        trials = [SimpleNamespace(number=i, params=params) for i, params in enumerate([baseline, other, other])]
        finalists = tuning.select_finalists(trials, self.base, 7)
        self.assertEqual([row["label"] for row in finalists], ["baseline", "trial-1"])


if __name__ == "__main__":
    unittest.main()
