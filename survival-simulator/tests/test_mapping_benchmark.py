from types import SimpleNamespace
import unittest

from benchmark_mapping import (
    Milestone, anchored_pose_errors, coverage_percent, free_cell_areas,
    frame_conditions, inferred_biome_metrics, percentile, population_metrics, sprint_cost_estimate,
)


class MappingBenchmarkTests(unittest.TestCase):
    def test_biome_inference_accuracy_excludes_unknown_and_unanchored_cells(self):
        import numpy as np
        truth = np.array([[SimpleNamespace(type="forest"), SimpleNamespace(type="desert")],
                          [SimpleNamespace(type="forest"), SimpleNamespace(type="forest")]])
        group = dict(anchored=True, biome_estimate=dict(bounds=[0, 0, 2, 2], sites=[{}],
                     labels=[[0, -1], [0, 0]], palette=["forest"]))
        result = inferred_biome_metrics(dict(groups=[group]), SimpleNamespace(width=2, height=2, biome_map=truth))
        self.assertEqual(result["inferred_biome_grid_support_percent"], 75)
        self.assertAlmostEqual(result["inferred_biome_grid_accuracy_percent"], 200 / 3)
        group["anchored"] = False
        self.assertIsNone(inferred_biome_metrics(dict(groups=[group]),
                         SimpleNamespace(width=2, height=2, biome_map=truth))["inferred_biome_grid_accuracy_percent"])

    def test_transient_birth_disconnect_restarts_sustained_clock(self):
        milestone = Milestone(duration=5)
        for condition, time in [(True, 1), (True, 4), (False, 4.1), (True, 4.2), (True, 9.1)]:
            milestone.observe(condition, time)
        self.assertEqual(milestone.first, 1)
        self.assertIsNone(milestone.sustained)
        milestone.observe(True, 9.2)
        self.assertEqual(milestone.sustained, 4.2)

    def test_obstacles_are_excluded_from_both_denominator_and_credited_cells(self):
        env = SimpleNamespace(width=4, height=4,
                              obstacles=[SimpleNamespace(x=0, y=0, width=2, height=2)])
        areas = free_cell_areas(env, 2)
        self.assertEqual(sum(areas.values()), 12)
        self.assertNotIn((0, 0), areas)
        self.assertAlmostEqual(coverage_percent([(0, 0), (1, 0), (1, 0)], areas), 100 / 3)

    def test_overlapping_obstacle_area_is_counted_once(self):
        env = SimpleNamespace(width=6, height=4, obstacles=[
            SimpleNamespace(x=0, y=0, width=3, height=2),
            SimpleNamespace(x=2, y=0, width=3, height=2),
        ])
        self.assertEqual(sum(free_cell_areas(env, 2).values()), 14)

    def test_accuracy_compares_only_anchored_living_agent_poses(self):
        groups = {1: SimpleNamespace(anchored=True), 2: SimpleNamespace(anchored=False)}
        poses = {
            1: SimpleNamespace(agent_id=1, group_id=1, position=(3, 4), uncertainty=1),
            2: SimpleNamespace(agent_id=2, group_id=2, position=(999, 999), uncertainty=0),
            3: SimpleNamespace(agent_id=3, group_id=1, position=(9, 9), uncertainty=0),
        }
        actual = {1: SimpleNamespace(x=0, y=0), 2: SimpleNamespace(x=0, y=0)}
        self.assertEqual(anchored_pose_errors(poses, groups, actual), [(1, 5, 1)])

    def test_sprint_premium_is_excess_over_same_distance_at_walking_cost(self):
        self.assertEqual(sprint_cost_estimate(10, 10, 20, .05, .5), (0, 0))
        self.assertEqual(sprint_cost_estimate(30, 10, 20, .05, .5), (5, 4.5))

    def test_percentile_and_empty_accuracy_samples(self):
        self.assertIsNone(percentile([]))
        self.assertEqual(percentile(list(range(1, 101))), 95)

    def test_shared_frame_requires_every_living_agent_and_ignores_dead_poses(self):
        groups = {1: SimpleNamespace(anchored=True), 2: SimpleNamespace(anchored=False)}
        poses = {1: SimpleNamespace(group_id=1), 2: SimpleNamespace(group_id=2)}
        self.assertTrue(frame_conditions(poses, groups, [1])["shared_absolute"])
        self.assertFalse(frame_conditions(poses, groups, [1, 3])["shared_absolute"])
        self.assertFalse(frame_conditions(poses, groups, [1, 2])["shared_absolute"])
        self.assertFalse(frame_conditions(poses, groups, [])["shared_absolute"])
        groups[2].anchored = True
        result = frame_conditions(poses, groups, [1, 2])
        self.assertTrue(result["all_living_anchored"])
        self.assertFalse(result["shared_absolute"])

    def test_population_uses_event_times_and_preserves_unreached_frame(self):
        lineage = [dict(parent_id=None, birth_time=0, death_time=60),
                   dict(parent_id=None, birth_time=0, death_time=None),
                   dict(parent_id=1, birth_time=10, death_time=50),
                   dict(parent_id=1, birth_time=60, death_time=None)]
        result = population_metrics(lineage, 20, 120)
        self.assertEqual(result["alive_at_shared_absolute"], 3)
        self.assertEqual(result["births_before_shared_absolute"], 1)
        self.assertEqual(result["births_after_shared_absolute"], 1)
        self.assertEqual(result["deaths_after_shared_absolute"], 2)
        self.assertEqual(result["alive_at_60_seconds"], 2)
        self.assertEqual(result["births_by_60_seconds"], 2)
        self.assertEqual(result["founders_alive_final"], 1)
        missing = population_metrics(lineage, None, 60)
        self.assertIsNone(missing["births_after_shared_absolute"])
        self.assertIsNone(missing["alive_at_120_seconds"])


if __name__ == "__main__":
    unittest.main()
