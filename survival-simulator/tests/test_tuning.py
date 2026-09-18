from concurrent.futures import Future
import importlib.util
import gzip
import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tuning_evaluation as evaluation
from tuning_evaluation import atomic_json, checkpoint_path, evaluate_seed

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
    @unittest.skipUnless(os.name == "nt", "Windows file sharing")
    def test_checkpoint_replace_retries_transient_reader_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / "saved", Path(folder) / "temporary"
            old.write_bytes(b"old checkpoint")
            new.write_bytes(b"new checkpoint")
            replace_file = os.replace
            calls = []

            def replace_with_reader(source, destination):
                calls.append(destination)
                if len(calls) == 1:
                    self.assertEqual(old.read_bytes(), b"old checkpoint")
                    raise PermissionError("reader still holds the file")
                return replace_file(source, destination)

            with patch.object(evaluation.os, "replace", side_effect=replace_with_reader), \
                 patch.object(evaluation.time, "sleep"):
                evaluation.atomic_replace(new, old)
            self.assertEqual(old.read_bytes(), b"new checkpoint")
            self.assertFalse(new.exists())
            self.assertEqual(len(calls), 2)

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

    def test_episode_lock_preserves_active_worker_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed.json"
            with evaluation.episode_lock(path, time.monotonic() + 1) as first:
                self.assertTrue(first)
                with evaluation.episode_lock(path, 0) as second:
                    self.assertFalse(second)
            with evaluation.episode_lock(path, time.monotonic() + 1) as resumed:
                self.assertTrue(resumed)

    def test_orphaned_worker_stops_without_waiting_for_time_budget(self):
        parent = SimpleNamespace(is_alive=lambda: False)
        with patch.object(evaluation.multiprocessing, "parent_process", return_value=parent):
            self.assertTrue(evaluation.stopping(time.monotonic() + 3600))


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
        values = dict(output=self.output, workers=4, train_seeds=[1, 7], train_seconds=600, trials=3,
                      checkpoint_seconds=60.)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_search_aggregates_all_seeds_and_includes_baseline(self):
        study = self.study()
        seen = []

        def fake(expert, planner, seed, seconds, path, deadline, checkpoint_seconds):
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
        self.assertIn("Completed trials: 3", (self.output / "search_report.md").read_text())

    def test_laptop_defaults_and_explicit_overrides(self):
        with patch.object(os, "cpu_count", return_value=20):
            args = tuning.resolve_options(tuning.parser().parse_args(["--laptop"]))
        self.assertEqual((args.workers, args.total_hours, args.search_hours, args.finalists), (6, 8., 6., 2))
        self.assertEqual(args.output.name, "bo-laptop")
        with patch.object(os, "cpu_count", return_value=1):
            small = tuning.resolve_options(tuning.parser().parse_args(["--laptop"]))
        self.assertEqual(small.workers, 1)
        explicit = tuning.resolve_options(tuning.parser().parse_args([
            "--laptop", "--workers", "2", "--hours", "4", "--no-keep-awake", "--finalists", "1"]))
        self.assertEqual((explicit.workers, explicit.total_hours, explicit.search_hours, explicit.finalists), (2, 4., 3., 1))
        self.assertFalse(explicit.keep_awake)
        cluster = tuning.resolve_options(tuning.parser().parse_args([]))
        self.assertEqual((cluster.workers, cluster.total_hours, cluster.search_hours, cluster.finalists), (1, 11.5, 8., 7))

    def test_validation_resume_keeps_original_finalists_and_cached_seeds(self):
        study = self.study()
        candidate = tuning.configured(self.base, {"harvest.ripe_energy": 52.})
        finalists = [dict(label="baseline", params=tuning.baseline_params(self.base)),
                     dict(label="trial-9", params=tuning.baseline_params(candidate))]
        atomic_json(self.output / "run_state.json", dict(phase="validation", finalists=finalists))
        cached = tuning.candidate_directory(self.output, self.base) / "validation" / "seed-1001.json"
        atomic_json(cached, {**episode(1001, 22), "horizon_seconds": 600})
        seen = []

        def fake(expert, planner, seed, seconds, path, deadline, checkpoint_seconds):
            if Path(path).exists():
                return evaluate_seed(expert, planner, seed, seconds, path, 0)
            seen.append((expert["harvest"]["ripe_energy"], seed))
            return episode(seed, 20)

        args = self.args(validation_seeds=[1001, 1007], validation_seconds=600, finalists=1)
        with patch.object(tuning, "evaluate_seed", side_effect=fake):
            tuning.validate(ImmediatePool(), study, args, self.base, self.planner,
                            time.monotonic() + 30, threading.Event())
        self.assertEqual(seen, [(56., 1007), (52., 1001), (52., 1007)])
        self.assertEqual(tuning.read_run_state(self.output)["phase"], "complete")
        report = json.loads((self.output / "validation_results.json").read_text())
        self.assertEqual([row["label"] for row in report["candidates"]], ["baseline", "trial-9"])

    def test_episode_checkpoint_restores_world_policy_rng_and_metrics(self):
        from src.core import SimulationCore
        original_init, original_step = SimulationCore.__init__, SimulationCore.step
        stop = threading.Event()

        def small_init(sim, **kwargs):
            original_init(sim, env_width=600, env_height=400, starting_trees=10, **kwargs)

        def stop_mid_episode(sim, actions):
            state = original_step(sim, actions)
            if sim.env.time >= 4:
                stop.set()
            return state

        path = self.output / "seed-1.json"
        with patch.object(SimulationCore, "__init__", small_init), \
             patch.object(SimulationCore, "step", stop_mid_episode), \
             patch.object(evaluation, "STOP_EVENT", stop):
            self.assertIsNone(evaluate_seed(self.base, self.planner, 1, 12, path, time.monotonic() + 60))
        self.assertFalse(path.exists())
        self.assertTrue(checkpoint_path(path).exists())
        with gzip.open(checkpoint_path(path), "rb") as handle:
            saved = pickle.load(handle)
        self.assertGreaterEqual(saved["sim"].env.time, 4.)
        self.assertEqual(saved["steps"], 40)
        self.assertIs(saved["sim"].rng, saved["sim"].env.rng)
        self.assertTrue(saved["policy"].planner.snapshot()["groups"])
        before = checkpoint_path(path).read_bytes()
        callback = lambda *a: None
        saved["sim"].env.event_sink = callback
        with patch.object(evaluation.pickle, "Pickler", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                evaluation.save_checkpoint(path, saved)
        self.assertIs(saved["sim"].env.event_sink, callback)
        self.assertEqual(checkpoint_path(path).read_bytes(), before)
        with patch.object(SimulationCore, "__init__", side_effect=AssertionError("Must restore without generating a world")):
            resumed = evaluate_seed(self.base, self.planner, 1, 12, path, time.monotonic() + 60)
        self.assertAlmostEqual(resumed["seconds"], 12.)
        self.assertGreaterEqual(resumed["fruit_eaten"], saved["metrics"]["fruit_eaten"])
        self.assertEqual(resumed["trace"][:len(saved["trace"])], saved["trace"])
        self.assertFalse(checkpoint_path(path).exists())
        self.assertEqual(evaluate_seed(self.base, self.planner, 1, 12, path, 0), resumed)
        with patch.object(SimulationCore, "__init__", small_init):
            uninterrupted = evaluate_seed(self.base, self.planner, 1, 12, self.output / "reference.json", time.monotonic() + 60)
        for key in ("score", "population", "births", "deaths", "fruit_eaten", "fruit_rotted", "fruit_score"):
            self.assertEqual(resumed[key], uninterrupted[key], key)

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

    def test_resume_recovers_torn_journal_append_without_losing_history(self):
        study = self.study()
        baseline = study.ask(fixed_distributions=tuning.SPACE)
        study.tell(baseline, 10.)
        journal = self.output / "optuna.journal"
        fragment = b'{"op_code":'
        with journal.open("ab") as handle:
            handle.write(fragment)
        resumed = self.study()
        self.assertEqual(resumed.best_value, 10.)
        trial = resumed.ask(fixed_distributions=tuning.SPACE)
        resumed.tell(trial, 12.)
        self.assertEqual(self.study().best_value, 12.)
        backups = list(self.output.glob("optuna.journal.torn-tail-*.bin"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), fragment)
        corrupted = b'bad record\n{"op_code": 0}\n'
        journal.write_bytes(corrupted)
        with self.assertRaisesRegex(ValueError, "history was left unchanged"):
            tuning.repair_journal_tail(journal)
        self.assertEqual(journal.read_bytes(), corrupted)

    @unittest.skipUnless(os.name == "nt", "Windows power management")
    def test_keep_awake_restores_power_request_after_error(self):
        with patch("ctypes.windll.kernel32.SetThreadExecutionState", return_value=1) as set_state:
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with tuning.keep_awake(True):
                    raise RuntimeError("test failure")
        self.assertEqual([call.args[0] for call in set_state.call_args_list], [0x80000001, 0x80000000])

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
