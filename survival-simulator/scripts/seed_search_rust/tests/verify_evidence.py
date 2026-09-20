"""Public geometry reconstruction, false-candidate checks, and native replay."""

import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from seed_joint_constraints_probe import canonical_frame
from seed_survey_evidence import (candidate_geometry_matches, extract_evidence,
                                  informative_samples, initial_trees_match, rotate)
from seed_survey_probe import collect, compatible, spread_samples, verify
from seed_biome_prefix_probe import candidate_prefix
from search_survey_rust import DEFAULT_BINARY, validate_replay


ROCK = [([400., 450.], [450., 450.]), ([400., 450.], [400., 510.])]
WALL = ([0., 30.], [1600., 30.])


def state(agent_id, xy, edges, *, age=.1, heading=0., biome="forest", objects=()):
    observations = [dict(type="Edge", coords=[rotate((p[0] - xy[0], p[1] - xy[1]), -heading) for p in edge])
                    for edge in edges]
    return dict(agent_id=agent_id, age=age, biome=biome, observations=observations + list(objects))


def frame(now, *states):
    return dict(sim_time=now, observations=list(states))


def action(agent_id=0, distance=10., turn=0.):
    return dict(agent_id=agent_id, move_distance=distance, turn_angle=turn,
                move_direction=0., spawn_agent=False)


def row_for(frames, actions):
    return dict(schema_version=2, public_frames=frames, action_log=actions, founder_headings=[], rock_lengths=[],
                public_frame_sha256=[hashlib.sha256(canonical_frame(f).encode()).hexdigest() for f in frames],
                seed_evidence=extract_evidence(frames, actions))


class EvidenceTests(unittest.TestCase):
    def test_late_shared_landmark_recovers_origin_without_reversing_walk(self):
        frames = [frame(.1, state(0, (300., 400.), ROCK, heading=.7)),
                  frame(.2, state(0, (307., 404.), ROCK + [WALL], age=.2, heading=1.1))]
        # Actual movement is deliberately different from the command.
        evidence = extract_evidence(frames, [[action(distance=100., turn=.4)]])
        origin = evidence["founder_positions"][0]
        self.assertLess(math.dist(origin["xy"], (300., 400.)), origin["radius"])
        self.assertEqual(origin["agent_id"], 0)
        self.assertEqual(evidence["diagnostics"]["conflicting_components"], 0)
        rectangle = next(r for r in evidence["rock_rectangles"] if "x" in r)
        for key, expected in dict(x=400., y=450., width=50., height=60.).items():
            self.assertAlmostEqual(rectangle[key], expected)

    def test_cross_agent_registration_and_unanchored_components(self):
        early = frame(.1, state(0, (300., 400.), ROCK))
        unresolved = extract_evidence([early], [])
        self.assertEqual(unresolved["founder_positions"], [])
        self.assertTrue(unresolved["rock_rectangles"])
        later = frame(.2, state(1, (340., 405.), ROCK + [WALL], age=.2))
        evidence = extract_evidence([early, later], [[action()]])
        self.assertEqual(evidence["founder_positions"][0]["agent_id"], 0)
        self.assertLess(math.dist(evidence["founder_positions"][0]["xy"], (300., 400.)), 1e-6)

    def test_stationary_rotation_links_first_sighting_with_one_edge(self):
        frames = [frame(.1, state(0, (300., 400.), ROCK[:1])),
                  frame(.2, state(0, (300., 400.), [WALL], age=.2, heading=.8))]
        evidence = extract_evidence(frames, [[action(distance=0., turn=.8)]])
        self.assertEqual(len(evidence["founder_positions"]), 1)

    def test_spawn_distances_work_before_absolute_boundary_alignment(self):
        evidence = extract_evidence([frame(.1, state(0, (300., 400.), ROCK), state(1, (303., 404.), ROCK))], [])
        self.assertEqual(evidence["founder_positions"], [])
        self.assertAlmostEqual(evidence["founder_distances"][0]["distance"], 5.)
        env = SimpleNamespace(agents_dict={0: SimpleNamespace(x=100., y=200.), 1: SimpleNamespace(x=103., y=204.)},
                              obstacles=[SimpleNamespace(x=400., y=450., width=50., height=60.)], edges=[])
        self.assertTrue(candidate_geometry_matches(env, evidence)["founder_distance_match"])
        env.agents_dict[1].x += 10
        self.assertFalse(candidate_geometry_matches(env, evidence)["founder_distance_match"])

    def test_ambiguous_repeated_rectangles_do_not_force_origin(self):
        duplicate = [([a[0] + 200, a[1]], [b[0] + 200, b[1]]) for a, b in ROCK]
        frames = [frame(.1, state(0, (300., 400.), ROCK)),
                  frame(.2, state(1, (300., 350.), ROCK + duplicate + [WALL], age=.2))]
        evidence = extract_evidence(frames, [[action()]])
        self.assertGreater(evidence["diagnostics"]["ambiguous_edge_pairs"], 0)
        self.assertEqual(evidence["founder_positions"], [])

    def test_conflicting_anchors_are_discarded(self):
        duplicate = [([a[0] + 200, a[1]], [b[0] + 200, b[1]]) for a, b in ROCK]
        frames = [frame(.1, state(0, (300., 400.), ROCK + [WALL])),
                  frame(.2, state(0, (510., 400.), duplicate + [WALL], age=.2))]
        evidence = extract_evidence(frames, [[action()]])
        self.assertGreater(evidence["diagnostics"]["conflicting_components"], 0)
        self.assertEqual(evidence["founder_positions"], [])
        self.assertEqual(evidence["world_rock_edges"], [])

    def test_cached_observations_do_not_create_moving_landmarks(self):
        frames = [frame(.1, state(0, (300., 400.), ROCK + [WALL])),
                  frame(.2, state(0, (300., 400.), ROCK + [WALL]))]
        evidence = extract_evidence(frames, [[action(distance=50., turn=.8)]])
        self.assertEqual(evidence["diagnostics"]["fresh_poses"], 1)
        self.assertEqual(len(evidence["biome_samples"]), 1)

    def test_transition_endpoints_and_pixel_margin(self):
        frames = [frame(.1, state(0, (300., 400.), ROCK + [WALL])),
                  frame(.2, state(0, (301., 400.), ROCK + [WALL], age=.2, biome="desert"))]
        evidence = extract_evidence(frames, [[action()]])
        self.assertEqual(len(evidence["biome_transitions"]), 1)
        selected = informative_samples(evidence, [], spread_samples, limit=8)
        self.assertEqual({row["biome"] for row in selected}, {"forest", "desert"})
        self.assertTrue(all(row["radius"] >= math.sqrt(2) for row in selected))

    def test_rectangle_pairing_positions_and_initial_trees_reject_wrong_candidates(self):
        env = SimpleNamespace(agents_dict={0: SimpleNamespace(x=300., y=400., direction=0.)},
                              obstacles=[SimpleNamespace(x=400., y=450., width=50., height=40.),
                                         SimpleNamespace(x=600., y=450., width=90., height=60.)],
                              edges=ROCK, trees=[SimpleNamespace(x=310., y=400.)])
        evidence = dict(rock_rectangles=[dict(width=50., height=60.)])
        self.assertFalse(candidate_geometry_matches(env, evidence)["rock_rectangle_match"])
        env.obstacles[0].height = 60.
        self.assertTrue(candidate_geometry_matches(env, evidence)["rock_rectangle_match"])
        evidence["founder_positions"] = [dict(agent_id=0, xy=[300., 400.], radius=.01)]
        self.assertTrue(candidate_geometry_matches(env, evidence)["founder_position_match"])
        env.agents_dict[0].x += 1.
        self.assertFalse(candidate_geometry_matches(env, evidence)["founder_position_match"])
        evidence["rock_rectangles"][0].update(x=410., y=450., radius=.01)
        self.assertFalse(candidate_geometry_matches(env, evidence)["rock_rectangle_match"])
        env.agents_dict[0].x = 300.
        self.assertTrue(initial_trees_match(env, [dict(agent_id=0, distance=10., angle=0.)]))
        self.assertFalse(initial_trees_match(env, [dict(agent_id=0, distance=11., angle=0.)]))

    def test_tree_only_initial_frame_and_no_later_tree_as_initial(self):
        tree = dict(type="Tree", distance=10., angle=.5)
        frames = [frame(.1, state(0, (300., 400.), [], objects=[tree])),
                  frame(.2, state(0, (310., 400.), [], age=.2, objects=[dict(tree, distance=30.)]))]
        evidence = extract_evidence(frames, [[action()]])
        self.assertEqual([row["distance"] for row in evidence["initial_tree_sightings"]], [10.])

    def test_late_start_cannot_be_mislabeled_as_spawn_and_frames_cannot_be_skipped(self):
        for frames, actions in [([frame(30., state(0, (300., 400.), ROCK))], []),
                                ([frame(.1, state(0, (300., 400.), ROCK)), frame(.3)], [[action()]])]:
            with self.assertRaisesRegex(ValueError, "first empty tick"):
                extract_evidence(frames, actions)

    def test_raw_hashes_derived_evidence_and_audit_independence(self):
        row = row_for([frame(.1, state(0, (300., 400.), ROCK + [WALL]))], [])
        row.update(target_seed=999999, localization_audit={"invented": "ignored"})
        validate_replay(row)
        changed = copy.deepcopy(row)
        changed["public_frames"][0]["observations"][0]["age"] = 1.
        with self.assertRaisesRegex(ValueError, "hashes"):
            validate_replay(changed)
        changed = copy.deepcopy(row)
        changed["seed_evidence"]["founder_positions"][0]["xy"][0] += 20
        with self.assertRaisesRegex(ValueError, "rebuild"):
            validate_replay(changed)
        legacy = {key: row[key] for key in ("public_frame_sha256", "action_log", "founder_headings", "rock_lengths")}
        validate_replay(legacy)


class NativeEvidenceTests(unittest.TestCase):
    def test_fresh_surveys_survive_rust_filter_and_full_native_replay(self):
        binary = Path(os.environ.get("SEED_SEARCH_BINARY", DEFAULT_BINARY))
        for seed in (3, 11, 20260919):
            with self.subTest(seed=seed):
                with contextlib.redirect_stdout(io.StringIO()):
                    row, _ = collect(seed, 30.)
                validate_replay(row)
                evidence = row["seed_evidence"]
                self.assertTrue(evidence["founder_positions"])
                self.assertTrue(evidence["world_rock_edges"])
                self.assertTrue(evidence["rock_rectangles"])
                sites, types = candidate_prefix(seed)
                self.assertTrue(compatible(sites, types, row["checkpoints"][-1]["shared_samples"]))
                result = verify(seed, row)
                for key in ("founder_position_match", "founder_distance_match", "rock_rectangle_match", "world_rock_edge_match",
                            "observed_biome_match", "initial_tree_match", "public_replay_match"):
                    self.assertIs(result[key], True, (seed, key, result))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "samples.json"
                    path.write_text(json.dumps(dict(samples=row["checkpoints"][-1]["shared_samples"])), encoding="utf-8")
                    process = subprocess.run([str(binary), "--input", str(path), "--start", str(seed),
                                              "--count", "10", "--threads", "2"], capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertIn(seed, json.loads(process.stdout)["survivors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
