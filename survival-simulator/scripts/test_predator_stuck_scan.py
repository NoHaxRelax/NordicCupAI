"""Detector and batch-output checks; run with unittest discovery."""
import csv
from concurrent.futures import Future
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import predator_stuck_scan as scan


def sample(tick, x, y):
    return (tick, x, y, 0.0, 100.0, False)


class ResidenceWindowTests(unittest.TestCase):
    def test_waits_for_full_interval_and_allows_boundary(self):
        tracker = scan.ResidenceWindow(600, 15)
        for tick in range(600):
            self.assertFalse(tracker.add(sample(tick, 0 if tick % 2 == 0 else 15, 0)))
        self.assertTrue(tracker.add(sample(600, 0, 0)))

    def test_return_to_start_does_not_hide_excursion(self):
        tracker = scan.ResidenceWindow(4, 15)
        values = [tracker.add(sample(tick, x, 0)) for tick, x in enumerate((0, 0, 40, 0, 0))]
        self.assertFalse(values[-1])
        self.assertFalse(tracker.add(sample(5, 0, 0)))
        self.assertFalse(tracker.add(sample(6, 0, 0)))
        self.assertTrue(tracker.add(sample(7, 0, 0)))

    def test_sliding_window_finds_confinement_after_travel(self):
        tracker = scan.ResidenceWindow(3, 1)
        answers = [tracker.add(sample(tick, x, 0)) for tick, x in enumerate((0, 10, 20, 20, 20, 20))]
        self.assertEqual(answers, [False, False, False, False, False, True])

    def test_diagonal_and_bounding_box_false_positives(self):
        for points, expected in [([(0, 0), (12, 12), (0, 0)], False),
                                 ([(0, 0), (14, 0), (0, 14)], True)]:
            tracker = scan.ResidenceWindow(2, 15)
            for tick, (x, y) in enumerate(points):
                result = tracker.add(sample(tick, x, y))
            self.assertEqual(result, expected)

    def test_matches_brute_force_for_random_paths(self):
        rng = random.Random(18)
        for radius in (1, 5, 15):
            for steps in (1, 7, 60):
                tracker = scan.ResidenceWindow(steps, radius)
                history = []
                x = y = 0.0
                for tick in range(900):
                    # Include movement, rests, loops, and teleports between areas.
                    if tick % 70 < 25:
                        x += rng.uniform(-3, 3)
                        y += rng.uniform(-3, 3)
                    if tick % 101 == 0:
                        x += 40
                    history.append((x, y))
                    result = tracker.add(sample(tick, x, y))
                    expected = tick >= steps and all(
                        math.dist(history[tick - steps], point) <= radius
                        for point in history[max(0, tick - steps):tick + 1]
                    )
                    self.assertEqual(result, expected, (radius, steps, tick))


class EvidenceTests(unittest.TestCase):
    def test_native_enclosed_predator_produces_trace_and_image(self):
        import gzip
        import pygame
        from predator_corner_demo import Trial
        fixture = Trial(SimpleNamespace(count=1, seed=1, corner="top-left", start="touching",
                                        heading="diagonal", biome="grassland", seconds=60))
        env = fixture.env
        p = env.predators[0]
        p.x = p.y = 250
        # A legal but enclosed 20-unit cell: no radius-10 predator step fits.
        for x, y, w, h in ((210, 210, 30, 80), (260, 210, 30, 80),
                            (210, 210, 80, 30), (210, 260, 80, 30)):
            env.spawn_obstacle(x=x, y=y, width=w, height=h)
        env._update_spatial_grid()
        self.assertFalse(env._in_obstacle((p.x, p.y), p.size, env.obstacles))
        config = dict(predators=1, seconds=60, stuck_seconds=60, radius=15, images=True)
        with tempfile.TemporaryDirectory() as temp:
            with patch("src.core.SimulationCore", return_value=SimpleNamespace(env=env)):
                result = scan.run_game(1, config, temp)
            self.assertEqual(result["status"], "complete", result.get("error"))
            self.assertEqual(len(result["findings"]), 1)
            event = result["findings"][0]
            self.assertEqual(event["duration_seconds"], 60)
            self.assertEqual(event["sample_count"], 601)
            self.assertGreater(event["awake_samples"], 0)
            self.assertFalse(event["spawned_overlapping_obstacle"])
            with gzip.open(Path(temp) / event["trace"], "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            self.assertEqual(len(trace["samples"]), 601)
            self.assertEqual(pygame.image.load(str(Path(temp) / event["image"])).get_size(), (640, 740))
            summary = scan.report(Path(temp), {1: scan.compact_result(result)}, 1, config, 0)
            self.assertEqual(summary["games_with_findings"], 1)
            self.assertEqual(summary["flagged_predators_in_complete_games"], 1)
            with (Path(temp) / "findings.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["game_status"], "complete")

    def test_resume_skips_complete_games_and_rejects_changed_config(self):
        config = dict(predators=100, seconds=600, stuck_seconds=60, radius=15, images=True)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            args = SimpleNamespace(output=output, start_seed=0, games=1, workers=1)
            scan.write_json(output / "manifest.json", dict(version=1, config=config, start_seed=0,
                                                          sources=scan.source_hashes()))
            folder = output / "games" / "seed-00000000"
            folder.mkdir(parents=True)
            scan.write_json(folder / "result.json", dict(seed=0, status="complete", findings=[],
                             initial_predators=100, total_tracked=100, initial_overlaps=[]))
            with patch.object(scan, "ProcessPoolExecutor", side_effect=AssertionError("Should not launch workers")):
                self.assertEqual(scan.run_batch(args, config), 0)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                scan.run_batch(args, config | {"radius": 10})
            self.assertFalse((output / ".batch.lock").exists())

    def test_ctrl_c_cancels_pending_games_and_releases_lock(self):
        futures = []

        def submit(*args):
            future = Future()
            futures.append(future)
            return future

        def shutdown(**kwargs):
            for future in futures:
                future.cancel()

        pool = SimpleNamespace(submit=submit, shutdown=shutdown)
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp), start_seed=0, games=10_000, workers=2)
            with patch.object(scan, "ProcessPoolExecutor", return_value=pool), \
                 patch.object(scan, "wait", side_effect=KeyboardInterrupt):
                self.assertEqual(scan.run_batch(args, {}), 130)
            self.assertEqual(len(futures), 4)  # Queue stays bounded even for 10,000 games.
            self.assertTrue(all(f.cancelled() for f in futures))
            self.assertFalse((Path(temp) / ".batch.lock").exists())
            summary = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(summary["completed_games"], 0)


if __name__ == "__main__":
    unittest.main()
