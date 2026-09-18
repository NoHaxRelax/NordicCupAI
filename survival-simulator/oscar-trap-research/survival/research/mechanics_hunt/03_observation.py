"""Observation, visibility, and cache-timing probes for the vendored simulator.

All assertions exercise the unmodified upstream code. Synthetic positions isolate
individual rules; generated-map checks use default settings and normal energy.
No network or competition service is contacted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.elements.fruit import Fruit
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"
DT = 0.1


def action(agent_id: int, distance: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        move_distance=distance,
        move_direction=direction,
        turn_angle=turn,
        spawn_agent=spawn,
    )


def advance(env: Environment, actions: list[ActionRequest]) -> dict:
    return step_environment(env, [(a.agent_id, a) for a in actions], dt=DT)


def fixture(seed: int = 303, agents: tuple[tuple[float, float], ...] = ((100, 200),)):
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    made = []
    for x, y in agents:
        agent = env.spawn_agent(x=x, y=y)
        agent.direction = 0.0
        agent.max_age = 1e9
        made.append(agent)
    return env, made


def by_type(observations: list[dict], type_name: str) -> list[dict]:
    return [item for item in observations if item["type"] == type_name]


def canonical_edge(item: dict) -> tuple[float, ...]:
    return tuple(round(float(value), 9) for point in item["coords"] for value in point)


def probe_phase_timing() -> dict:
    results: dict[str, Any] = {}

    # Known result: agents sense a predator before its movement in the same step.
    env, (agent,) = fixture()
    predator = env.spawn_predator(x=140, y=200)
    assert predator is not None
    predator.direction = math.pi
    predator.resting = False
    predator.energy = 100.0
    state = advance(env, [action(agent.agent_id)])
    observed = by_type(state["observations"][0]["observations"], "Predator")[0]
    actual_distance = math.hypot(predator.x - agent.x, predator.y - agent.y)
    assert math.isclose(observed["distance"], 40.0)
    assert math.isclose(actual_distance, 25.0)
    results["predator_pre_move_snapshot"] = {
        "observed_distance": observed["distance"],
        "post_step_true_distance": actual_distance,
        "distance_error": observed["distance"] - actual_distance,
        "classification": "ordinary response; true post-step position used only to verify",
        "contract_compliant": True,
    }

    # Observation is taken before automatic fruit collection, so the response can
    # contain a fruit that was already removed and scored in that same step.
    env, (agent,) = fixture(seed=304)
    fruit = env.spawn_fruit(x=105, y=200)
    assert fruit is not None
    state = advance(env, [action(agent.agent_id)])
    fruit_obs = by_type(state["observations"][0]["observations"], "Fruit")
    assert len(fruit_obs) == 1 and fruit not in env.fruits
    results["consumed_fruit_remains_in_response"] = {
        "observation": fruit_obs[0],
        "fruit_exists_post_step": False,
        "score": state["score"],
        "maximum_naive_extra_chase_per_tick": agent.sprint_speed,
        "maximum_naive_extra_chase_energy": (
            agent.speed * 0.05 + (agent.sprint_speed - agent.speed) * 0.5
        ),
        "classification": "ordinary response; existence checked with privileged state",
        "contract_compliant": True,
    }

    # An agent can be observed, then eaten later in the predator phase. Its ID is
    # stale in the survivor's observation but absent from the top-level status list.
    env, (observer, victim) = fixture(seed=305, agents=((100, 200), (140, 200)))
    observer.direction = 0.0
    victim.direction = math.pi
    predator = env.spawn_predator(x=155, y=200)
    assert predator is not None
    predator.direction = math.pi
    predator.resting = False
    predator.energy = 100.0
    state = advance(env, [action(observer.agent_id), action(victim.agent_id)])
    survivor_state = next(s for s in state["observations"] if s["agent_id"] == observer.agent_id)
    seen_agent_ids = [o["id"] for o in by_type(survivor_state["observations"], "Agent")]
    returned_ids = [s["agent_id"] for s in state["observations"]]
    assert victim.agent_id in seen_agent_ids and victim.agent_id not in returned_ids
    results["eaten_agent_stale_but_cross_checkable"] = {
        "stale_observed_agent_ids": seen_agent_ids,
        "returned_living_agent_ids": returned_ids,
        "classification": "fully detectable from ordinary response by ID cross-check",
        "contract_compliant": True,
    }

    # Extend the known list-removal bug: if an earlier agent dies during the agent
    # loop, the next survivor keeps its previous observation cache. Meanwhile the
    # predator receives another full movement, producing a two-step position error.
    env, (doomed, survivor) = fixture(seed=306, agents=((350, 200), (100, 200)))
    survivor.direction = 0.0
    predator = env.spawn_predator(x=180, y=200)
    assert predator is not None
    predator.direction = math.pi
    predator.resting = False
    predator.energy = 150.0
    first = advance(env, [action(doomed.agent_id), action(survivor.agent_id)])
    first_survivor = next(s for s in first["observations"] if s["agent_id"] == survivor.agent_id)
    first_seen = by_type(first_survivor["observations"], "Predator")[0]["distance"]
    doomed.energy = 0.05
    age_before = survivor.age
    second = advance(env, [action(doomed.agent_id), action(survivor.agent_id)])
    second_survivor = next(s for s in second["observations"] if s["agent_id"] == survivor.agent_id)
    second_seen = by_type(second_survivor["observations"], "Predator")[0]["distance"]
    actual_distance = math.hypot(predator.x - survivor.x, predator.y - survivor.y)
    assert math.isclose(first_seen, 80.0) and math.isclose(second_seen, first_seen)
    assert math.isclose(actual_distance, 50.0) and math.isclose(survivor.age, age_before)
    results["death_skip_preserves_old_observation_cache"] = {
        "first_response_distance": first_seen,
        "next_response_cached_distance": second_seen,
        "next_response_true_post_step_distance": actual_distance,
        "distance_error": second_seen - actual_distance,
        "survivor_age_was_also_skipped": True,
        "classification": "ordinary response; induced fixture and true position verify cause",
        "contract_compliant": True,
        "practicality": "rare and usually harmful; can occur when an earlier list agent starves",
    }

    # All requested actions finish before any agent observation. There is no
    # per-action partial snapshot based on action-list order.
    env, (left, right) = fixture(seed=307, agents=((100, 200), (200, 200)))
    left.direction = 0.0
    right.direction = math.pi
    state = advance(env, [
        action(left.agent_id, distance=10),
        action(right.agent_id, distance=10),
    ])
    left_state = next(s for s in state["observations"] if s["agent_id"] == left.agent_id)
    seen_right = next(o for o in by_type(left_state["observations"], "Agent")
                      if o["id"] == right.agent_id)
    assert math.isclose(left.x, 110.0) and math.isclose(right.x, 190.0)
    assert math.isclose(seen_right["distance"], 80.0)
    results["all_actions_precede_all_agent_observations"] = {
        "final_positions": [left.x, right.x],
        "observed_distance": seen_right["distance"],
        "negative_control_old_or_partial_distance": 90.0,
        "classification": "ordinary response and known own actions",
        "contract_compliant": True,
    }

    # Ordinary reproduction makes the child visible and returns its state on the
    # birth step, but the child cannot legitimately have supplied an action yet.
    env, (parent,) = fixture(seed=308)
    parent.energy = 200.0
    state = advance(env, [action(parent.agent_id, spawn=True)])
    child = next(a for a in env.agents if a.agent_id != parent.agent_id)
    returned_ids = [s["agent_id"] for s in state["observations"]]
    parent_state = next(s for s in state["observations"] if s["agent_id"] == parent.agent_id)
    seen_child = [o for o in by_type(parent_state["observations"], "Agent")
                  if o["id"] == child.agent_id]
    assert child.agent_id in returned_ids and seen_child
    results["newborn_is_observed_and_returned_on_birth_step"] = {
        "child_id": child.agent_id,
        "child_age": next(s["age"] for s in state["observations"]
                          if s["agent_id"] == child.agent_id),
        "parent_observes_child": True,
        "classification": "ordinary response",
        "contract_compliant": True,
    }

    return results


def probe_visibility_boundaries() -> dict:
    results: dict[str, Any] = {}
    observer = Agent(100, 100, rng=random.Random(401))
    observer.direction = 0.0
    far_edge = [((1000.0, 1000.0), (1001.0, 1000.0))]

    # Hearing is inclusive, omnidirectional, and ignores obstacle occlusion.
    wall = [((125.0, 0.0), (125.0, 200.0))]
    at_hearing = Fruit(150.0, 100.0)
    beyond_hearing = Fruit(150.01, 100.0)
    heard = by_type(observer.observe(fruits=[at_hearing], edges=wall), "Fruit")
    blocked = by_type(observer.observe(fruits=[beyond_hearing], edges=wall), "Fruit")
    unblocked = by_type(observer.observe(fruits=[beyond_hearing], edges=far_edge), "Fruit")
    behind = Fruit(50.0, 100.0)
    behind_heard = by_type(observer.observe(fruits=[behind], edges=far_edge), "Fruit")
    assert heard and not blocked and unblocked and behind_heard
    results["hearing_threshold_and_occlusion"] = {
        "at_radius_through_wall_visible": True,
        "0_01_past_radius_through_wall_visible": False,
        "0_01_past_radius_without_wall_visible": True,
        "at_radius_directly_behind_visible": True,
        "classification": "ordinary sensing; isolated geometry fixture",
        "contract_compliant": True,
    }

    # Shapely `contains` is strict, so nominal cone/range boundary points vanish.
    def visible(radius: float, degrees: float, agent: Agent = observer) -> bool:
        angle = math.radians(degrees)
        fruit = Fruit(agent.x + radius * math.cos(angle),
                      agent.y + radius * math.sin(angle))
        return bool(by_type(agent.observe(fruits=[fruit], edges=far_edge), "Fruit"))

    assert not visible(200.0, 0.0) and visible(199.999, 0.0)
    assert not visible(100.0, 30.0) and visible(100.0, 29.999)
    results["strict_vision_boundaries"] = {
        "at_exact_range_visible": False,
        "just_inside_range_visible": True,
        "at_exact_half_cone_visible": False,
        "just_inside_half_cone_visible": True,
        "classification": "ordinary sensing; exact-coordinate fixture",
        "contract_compliant": True,
    }

    # With no nearby corner rays, five fixed rays approximate the sector with
    # chords. Objects inside the declared circular range can fall outside it.
    assert not visible(199.5, 7.5) and visible(198.0, 7.5)
    default_loss = observer.vision_radius * (
        1.0 - math.cos(observer.cone_angle / 8.0)
    )
    evolved = Agent(
        0, 0, rng=random.Random(402), vision_radius=400,
        cone_angle=math.pi / 2,
    )
    evolved.direction = 0.0

    def evolved_visible(radius: float) -> bool:
        theta = evolved.cone_angle / 8.0
        fruit = Fruit(radius * math.cos(theta), radius * math.sin(theta))
        return bool(evolved.observe(fruits=[fruit], edges=far_edge))

    assert evolved_visible(392.3) and not evolved_visible(392.4)
    evolved_loss = evolved.vision_radius * (
        1.0 - math.cos(evolved.cone_angle / 8.0)
    )
    results["polygon_chord_range_scalloping"] = {
        "default_declared_range": observer.vision_radius,
        "default_max_range_loss": default_loss,
        "default_probe_199_5_at_7_5_deg_visible": False,
        "default_probe_198_at_7_5_deg_visible": True,
        "max_trait_declared_range": evolved.vision_radius,
        "max_trait_max_range_loss": evolved_loss,
        "max_trait_probe_392_3_visible": True,
        "max_trait_probe_392_4_visible": False,
        "classification": "ordinary sensing; trait-cap fixture",
        "contract_compliant": True,
        "practicality": "small blind scallops near maximum range; use safety hysteresis",
    }

    # One physical edge is emitted once for each ray that hits it.
    edge_obs = by_type(observer.observe(edges=[((150.0, 50.0), (150.0, 150.0))]), "Edge")
    unique_edges = {canonical_edge(item) for item in edge_obs}
    assert len(edge_obs) == 5 and len(unique_edges) == 1
    results["edge_observations_are_ray_hits_not_unique_segments"] = {
        "edge_entries": len(edge_obs),
        "unique_edge_segments": len(unique_edges),
        "classification": "ordinary observation",
        "contract_compliant": True,
    }

    # Directly calling observe with no edges crashes before the intended empty-edge
    # fast path. Official default geometry always supplies local boundary edges.
    empty_edges_error = None
    try:
        observer.observe(fruits=[Fruit(110, 100)], edges=[])
    except Exception as exc:  # exact upstream behavior is the subject of the probe
        empty_edges_error = f"{type(exc).__name__}: {exc}"
    assert empty_edges_error and empty_edges_error.startswith("IndexError:")
    results["empty_local_edge_list_crashes_observe"] = {
        "error": empty_edges_error,
        "classification": "synthetic/non-default configuration only",
        "contract_compliant": True,
        "practicality": "rejected as an official-default tactic; boundaries cover every default chunk neighborhood",
    }

    return results


def probe_coordinate_semantics_and_identity() -> dict:
    results: dict[str, Any] = {}

    # Turning is applied after movement and before the response observation.
    env, (agent,) = fixture(seed=501)
    fruit = env.spawn_fruit(x=100, y=300)
    assert fruit is not None
    state = advance(env, [action(agent.agent_id, turn=math.pi / 2)])
    fruit_obs = by_type(state["observations"][0]["observations"], "Fruit")[0]
    assert math.isclose(fruit_obs["distance"], 100.0)
    assert math.isclose(fruit_obs["angle"], 0.0)
    results["observations_use_post_turn_egocentric_frame"] = {
        "world_target_vector": [0.0, 100.0],
        "requested_turn": math.pi / 2,
        "returned_angle": fruit_obs["angle"],
        "classification": "ordinary response and own requested action",
        "contract_compliant": True,
    }

    # Edge coordinates use x=forward, y=left in the same local frame.
    observer = Agent(100, 100, rng=random.Random(502))
    observer.direction = math.pi / 2
    edge_obs = by_type(observer.observe(edges=[((90.0, 150.0), (110.0, 150.0))]), "Edge")
    unique = sorted({canonical_edge(item) for item in edge_obs})
    assert unique == [(50.0, 10.0, 50.0, -10.0)]
    results["edge_coordinate_axes"] = {
        "observer_facing_world_positive_y": True,
        "world_edge": [[90.0, 150.0], [110.0, 150.0]],
        "local_edge": list(unique[0]),
        "meaning": "local x is forward; local y is left",
        "classification": "ordinary observation",
        "contract_compliant": True,
    }

    # rel_dir describes the observer's bearing in the observed creature's frame:
    # zero means the observed creature faces the observer; pi means it faces away.
    observer = Agent(100, 100, rng=random.Random(503))
    observer.direction = 0.0
    toward = Predator(130, 100, rng=random.Random(504))
    toward.direction = math.pi
    away = Predator(130, 100, rng=random.Random(505))
    away.direction = 0.0
    toward_obs = observer.observe(predators=[toward], edges=[((1000, 1000), (1001, 1000))])[0]
    away_obs = observer.observe(predators=[away], edges=[((1000, 1000), (1001, 1000))])[0]
    assert math.isclose(toward_obs["rel_dir"], 0.0)
    assert math.isclose(abs(away_obs["rel_dir"]), math.pi)
    results["relative_direction_semantics"] = {
        "predator_facing_observer_rel_dir": toward_obs["rel_dir"],
        "predator_facing_away_abs_rel_dir": abs(away_obs["rel_dir"]),
        "classification": "ordinary observation",
        "contract_compliant": True,
        "practicality": "usable to distinguish direct approach from retreat without predator IDs",
    }

    # Fruit/tree/predator observations have no IDs or hidden lifecycle/state fields;
    # duplicate colocated objects are observation-identical. Agents alone have IDs.
    observer = Agent(100, 100, rng=random.Random(506))
    observer.direction = 0.0
    fruits = [Fruit(150, 100), Fruit(150, 100)]
    observations = observer.observe(
        fruits=fruits,
        edges=[((1000, 1000), (1001, 1000))],
    )
    assert len(observations) == 2 and observations[0] == observations[1]
    assert set(observations[0]) == {"type", "distance", "angle"}
    results["non_agent_identity_is_ambiguous"] = {
        "colocated_fruit_observations": observations,
        "fruit_payload_keys": sorted(observations[0]),
        "predator_payload_keys": ["angle", "distance", "rel_dir", "type"],
        "agent_payload_additional_key": "id",
        "classification": "ordinary payload",
        "contract_compliant": True,
        "hidden": [
            "absolute self x/y/direction", "fruit id/age/energy/radius",
            "tree id/age/radius", "predator id/energy/resting state",
        ],
        "inferable": [
            "relative position from distance+angle", "other creature facing from rel_dir",
            "self-relative edge endpoints", "dead observed agents by ID cross-check",
        ],
    }

    return results


def probe_generated_maps() -> dict:
    per_seed = []
    total_entries = 0
    total_unique = 0
    agents_with_edges = 0
    agents_total = 0
    contact_fruit_observations = []
    for seed in (1, 42, 2026):
        simulation = SimulationCore(seed=seed)
        state = simulation.step([])
        agent_rows = []
        for status in state["observations"]:
            edges = by_type(status["observations"], "Edge")
            unique = {canonical_edge(item) for item in edges}
            total_entries += len(edges)
            total_unique += len(unique)
            agents_total += 1
            agents_with_edges += bool(edges)
            contact_fruits = [
                item for item in by_type(status["observations"], "Fruit")
                if item["distance"] < 10.0
            ]
            if contact_fruits:
                contact_fruit_observations.append({
                    "seed": seed,
                    "agent_id": status["agent_id"],
                    "post_step_energy": status["energy"],
                    "observations": contact_fruits,
                })
            agent_rows.append({
                "agent_id": status["agent_id"],
                "energy": status["energy"],
                "edge_entries": len(edges),
                "unique_edge_segments": len(unique),
            })
        per_seed.append({"seed": seed, "agents": agent_rows})
    duplicates = total_entries - total_unique
    assert total_entries == 151 and total_unique == 32 and agents_with_edges == 13
    assert contact_fruit_observations == [{
        "seed": 1,
        "agent_id": 3,
        "post_step_energy": 169.9,
        "observations": [{
            "type": "Fruit",
            "distance": 7.001401268337004,
            "angle": -2.194019039311902,
        }],
    }]
    return {
        "seeds": [1, 42, 2026],
        "default_map_and_traits": True,
        "normal_starting_energy": True,
        "agents_checked": agents_total,
        "agents_with_edge_observations": agents_with_edges,
        "edge_entries": total_entries,
        "unique_edge_segments_within_each_agent_response": total_unique,
        "duplicate_entries": duplicates,
        "duplicate_fraction": duplicates / total_entries,
        "same_step_consumed_fruit_still_observed": contact_fruit_observations,
        "per_seed": per_seed,
        "classification": "ordinary first response on generated maps; no actions",
        "contract_compliant": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "03_observation.json",
    )
    args = parser.parse_args()

    output = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local-only unmodified engine; isolated fixtures plus default generated maps",
        "dt": DT,
        "phase_timing": probe_phase_timing(),
        "visibility_boundaries": probe_visibility_boundaries(),
        "coordinates_and_identity": probe_coordinate_semantics_and_identity(),
        "generated_map_validation": probe_generated_maps(),
        "rejected_hypotheses": [
            "Agent observations are partially updated in action-list order (false: all actions finish first).",
            "Objects at the documented exact cone/range boundary are visible (false: strict polygon containment excludes them).",
            "Every Edge entry is a unique wall segment (false: one segment is repeated per hitting ray).",
            "Observation order or payload identifies fruit/predators across ticks (false: no IDs, and colocated payloads are identical).",
            "A newly spawned predator can attack before ever being observable (false under normal spawn state: it starts resting at zero energy).",
            "The empty-edge crash is reachable in default 1600x1200/chunk400 maps (not found; boundary edges cover every chunk neighborhood).",
        ],
        "policy_implications": [
            "Treat each predator distance as a pre-move upper bound; reserve at least one 15-unit predator step.",
            "Deduplicate Edge coordinates before geometry/path logic unless deliberately using ray-hit multiplicity as a noisy prominence signal.",
            "Suppress a just-contacted fruit from the next decision, because the same response can show it after collection.",
            "Cross-check observed Agent IDs against returned living IDs to discard same-step predation phantoms.",
            "Use hysteresis near hearing, cone, and maximum-range boundaries; the five-ray polygon has small range scallops.",
            "Rebuild observations in the post-turn egocentric frame; local edge x is forward and y is left.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
