"""Population coordination and score-order probes for the pinned simulator.

All policy-facing experiments use only StepResponse fields and exactly one action
per living observed agent. Engine state is read only for diagnostic measurements.
No network calls or simulator source modifications are made.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "research"))

from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment
from simple_policies import SimplePolicy

SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def action(agent_id: int, distance: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def fixture(seed: int = 901, width: int = 500, height: int = 400) -> Environment:
    env = Environment(width, height, max(width, height), random.Random(seed))
    env.biome_map.fill(Forest_biome())
    env.agents.clear()
    env.agents_dict.clear()
    env.agent_observations.clear()
    env.fruits.clear()
    env.fruits_dict.clear()
    env.trees.clear()
    env.predators.clear()
    env._update_spatial_grid()
    return env


def step(env: Environment, actions: list[ActionRequest], dt: float = 0.1):
    return step_environment(env, [(a.agent_id, a) for a in actions], dt=dt)


def observation(state: dict, kind: str, object_id: int | None = None):
    candidates = [o for o in state["observations"] if o["type"] == kind]
    if object_id is not None:
        candidates = [o for o in candidates if o.get("id") == object_id]
    return candidates


def polar_vector(obs: dict) -> tuple[float, float]:
    return (obs["distance"] * math.cos(obs["angle"]),
            obs["distance"] * math.sin(obs["angle"]))


def rotate(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    x, y = vector
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def target_to_observer_transform(agent_obs: dict):
    """Return translation and rotation mapping target-agent local to observer local.

    If observer A sees target B at local polar angle alpha, rel_dir is the
    direction from B to A in B's frame. Therefore dir(B)-dir(A) equals
    alpha + pi - rel_dir.
    """
    translation = polar_vector(agent_obs)
    rotation = agent_obs["angle"] + math.pi - agent_obs["rel_dir"]
    return translation, rotation


def apply_transform(vector: tuple[float, float], transform):
    translation, rotation = transform
    x, y = rotate(vector, rotation)
    return translation[0] + x, translation[1] + y


def compose(first, second):
    """Compose A<-B (first) and B<-C (second) into A<-C."""
    t1, r1 = first
    t2, r2 = second
    rt2 = rotate(t2, r1)
    return ((t1[0] + rt2[0], t1[1] + rt2[1]), r1 + r2)


def build_transforms(states: list[dict], root_id: int):
    """Map each connected agent frame into root_id's frame from DTO data only."""
    by_id = {s["agent_id"]: s for s in states}
    transforms = {root_id: ((0.0, 0.0), 0.0)}
    queue = [root_id]
    while queue:
        observer_id = queue.pop(0)
        root_from_observer = transforms[observer_id]
        for obs in observation(by_id[observer_id], "Agent"):
            target_id = obs["id"]
            if target_id not in by_id or target_id in transforms:
                continue
            transforms[target_id] = compose(
                root_from_observer, target_to_observer_transform(obs))
            queue.append(target_id)
    return transforms


def relay_candidates(states: list[dict], receiver_id: int,
                     minimum_source_distance: float = 12.0):
    """Fruit vectors in receiver's frame, relayed from connected living peers.

    Very close source sightings are excluded because an earlier agent's cached
    observation can still contain a fruit it consumed in that same world step.
    """
    by_id = {s["agent_id"]: s for s in states}
    transforms = build_transforms(states, receiver_id)
    relayed = []
    for source_id, transform in transforms.items():
        if source_id == receiver_id:
            continue
        for fruit in observation(by_id[source_id], "Fruit"):
            if fruit["distance"] <= minimum_source_distance:
                continue
            vector = apply_transform(polar_vector(fruit), transform)
            relayed.append({"source_id": source_id, "x": vector[0], "y": vector[1],
                            "distance": math.hypot(*vector)})
    return relayed


def state_map(state: dict) -> dict[int, dict]:
    return {s["agent_id"]: s for s in state["observations"]}


def order_and_scoring_probes():
    results = {}

    # Two agents land equally close to one fruit. Environment list/ID order,
    # not ActionRequest order, determines who receives its energy.
    contested = []
    for request_order in ("low_id_first", "high_id_first"):
        env = fixture(910)
        low = env.spawn_agent(x=80, y=200)
        high = env.spawn_agent(x=120, y=200)
        for agent in (low, high):
            agent.direction = 0.0
            agent.energy = 100.0
            agent.max_age = 1e9
        fruit = env.spawn_fruit(x=100, y=200, radius=5)
        a_low = action(low.agent_id, 11, 0)
        a_high = action(high.agent_id, 11, math.pi)
        actions = [a_low, a_high] if request_order == "low_id_first" else [a_high, a_low]
        step(env, actions)
        contested.append({
            "request_order": request_order,
            "energies": {str(low.agent_id): low.energy, str(high.agent_id): high.energy},
            "fruit_remaining": fruit in env.fruits,
            "winner_id": low.agent_id if low.energy > high.energy else high.agent_id,
            "score": env.score,
        })
    assert [r["winner_id"] for r in contested] == [0, 0]
    results["contested_fruit_uses_agent_list_order"] = contested

    # A child is appended immediately. Request order assigns the first RNG
    # child position/ID to the first reproducing parent's lineage.
    birth_orders = []
    for order in ("left_first", "right_first"):
        env = fixture(920)
        left = env.spawn_agent(x=120, y=200)
        right = env.spawn_agent(x=380, y=200)
        for parent in (left, right):
            parent.energy = 250
            parent.max_age = 1e9
        requested = [left, right] if order == "left_first" else [right, left]
        step(env, [action(p.agent_id, spawn=True) for p in requested])
        children = sorted(env.agents[2:], key=lambda a: a.agent_id)
        birth_orders.append({
            "request_order": order,
            "children": [{"id": c.agent_id, "x": c.x, "y": c.y,
                           "nearest_parent": "left" if c.x < 250 else "right",
                           "age_after_birth_tick": c.age,
                           "energy_after_birth_tick": c.energy}
                          for c in children],
        })
    assert birth_orders[0]["children"][0]["nearest_parent"] == "left"
    assert birth_orders[1]["children"][0]["nearest_parent"] == "right"
    results["compliant_birth_order_maps_rng_and_ids_to_lineage"] = birth_orders

    # Movement order is otherwise inert because agents do not collide and all
    # agent actions finish before world interactions.
    movement_controls = []
    for reverse in (False, True):
        env = fixture(930)
        a = env.spawn_agent(x=100, y=150)
        b = env.spawn_agent(x=300, y=250)
        for agent in (a, b):
            agent.direction = 0
            agent.max_age = 1e9
        actions = [action(a.agent_id, 10, math.pi / 2),
                   action(b.agent_id, 10, -math.pi / 2)]
        step(env, list(reversed(actions)) if reverse else actions)
        movement_controls.append([(x.agent_id, x.x, x.y, x.energy) for x in env.agents])
    assert movement_controls[0] == movement_controls[1]
    results["nonspawn_movement_request_order_negative_control"] = movement_controls

    # All agents move before any food collection. Spending the last energy to
    # reach fruit kills the agent before it can eat; a small reserve survives.
    thresholds = []
    for starting_energy in (0.59, 0.61):
        env = fixture(940)
        agent = env.spawn_agent(x=100, y=200)
        agent.direction = 0
        agent.energy = starting_energy
        agent.max_age = 1e9
        env.spawn_fruit(x=110, y=200, radius=5)
        step(env, [action(agent.agent_id, 10)])
        thresholds.append({"starting_energy": starting_energy,
                           "alive": agent in env.agents,
                           "remaining_energy": agent.energy,
                           "fruit_count": len(env.fruits), "score": env.score})
    assert not thresholds[0]["alive"] and thresholds[1]["alive"]
    results["last_gasp_fruit_requires_action_plus_passive_reserve"] = thresholds

    # Reproduction happens before passive drain and food collection. Fruit
    # cannot fund a same-tick birth, while a just-funded parent can birth and
    # then refill if it survives the passive charge.
    reproduction_timing = []
    for starting_energy in (99.9, 100.3):
        env = fixture(950)
        parent = env.spawn_agent(x=100, y=200)
        parent.energy = starting_energy
        parent.max_age = 1e9
        env.spawn_fruit(x=100, y=200, radius=5)
        step(env, [action(parent.agent_id, spawn=True)])
        reproduction_timing.append({"starting_energy": starting_energy,
                                    "population": len(env.agents),
                                    "parent_energy": parent.energy,
                                    "fruit_count": len(env.fruits)})
    assert reproduction_timing[0]["population"] == 1
    assert reproduction_timing[1]["population"] == 2
    results["food_cannot_fund_same_tick_birth"] = reproduction_timing

    # Burning a nearly-empty doomed agent below zero removes it before the
    # predator phase. This is compliant but the score effect is tiny.
    sacrifice = []
    for burn in (False, True):
        env = fixture(960)
        agent = env.spawn_agent(x=200, y=200)
        agent.energy = 0.55
        agent.max_age = 1e9
        predator = env.spawn_predator(x=200, y=200)
        predator.resting = False
        predator.energy = 100
        move = 10 if burn else 0
        # Moving sideways remains within predator contact but costs 0.5.
        step(env, [action(agent.agent_id, move, math.pi / 2)])
        sacrifice.append({"burn_energy": burn, "alive": agent in env.agents,
                          "score": env.score, "predator_energy": predator.energy,
                          "agent_energy_at_removal": agent.energy})
    assert sacrifice[1]["score"] > sacrifice[0]["score"]
    results["pre_predator_energy_burn_avoids_tiny_death_penalty"] = sacrifice

    # One predator contact can kill a whole cluster. Its energy cap does not
    # cap the cumulative score penalty.
    env = fixture(970)
    agents = [env.spawn_agent(x=200, y=200) for _ in range(5)]
    for agent in agents:
        agent.energy = 100
        agent.max_age = 1e9
    predator = env.spawn_predator(x=200, y=200)
    predator.resting = False
    predator.energy = 199
    step(env, [action(a.agent_id) for a in agents])
    results["cluster_contact_cumulative_loss"] = {
        "agents_before": 5, "agents_after": len(env.agents),
        "score": env.score, "survival_tick_score": 0.1,
        "predator_energy": predator.energy,
        "score_penalty": 5 * (100 - 0.1) / 100,
    }
    assert not env.agents and math.isclose(env.score, 0.1 - 4.995)

    # Mirror the official server's <= 3000 condition near the horizon. This is
    # an artificial late-time fixture, but uses the unmodified step function.
    env = fixture(980)
    agent = env.spawn_agent(x=250, y=200)
    agent.energy = 500
    agent.max_age = 1e9
    env.time = 2999.8
    env.score = 2999.8
    timeline = []
    status = "ok"
    while status != "game_over":
        state = step(env, [action(agent.agent_id)])
        status = "ok" if state["num_agents"] > 0 and env.time <= 3000 else "game_over"
        timeline.append({"time": env.time, "score": env.score, "status": status})
        if len(timeline) > 10:
            raise AssertionError("horizon loop did not terminate")
    results["official_loop_horizon_boundary"] = timeline
    assert timeline[-1]["time"] > 3000
    return results


def coordinate_frame_probes():
    results = {}

    # Direct transform with nontrivial headings.
    env = fixture(990)
    a = env.spawn_agent(x=120, y=170)
    b = env.spawn_agent(x=165, y=205)
    a.direction = 0.71
    b.direction = math.atan2(35, 95)
    fruit = env.spawn_fruit(x=260, y=240, radius=5)
    state = step(env, [action(a.agent_id), action(b.agent_id)])
    states = state_map(state)
    a_to_b = observation(states[a.agent_id], "Agent", b.agent_id)[0]
    b_fruit = observation(states[b.agent_id], "Fruit")[0]
    inferred = apply_transform(polar_vector(b_fruit), target_to_observer_transform(a_to_b))
    actual = rotate((fruit.x - a.x, fruit.y - a.y), -a.direction)
    error = math.hypot(inferred[0] - actual[0], inferred[1] - actual[1])
    results["direct_frame_alignment"] = {
        "inferred_fruit_in_a_frame": inferred,
        "actual_fruit_in_a_frame_diagnostic": actual,
        "absolute_error": error,
        "ordinary_inputs": ["Agent distance", "Agent angle", "Agent rel_dir",
                            "peer Fruit distance", "peer Fruit angle"],
    }
    assert error < 1e-9

    # A hears B through a wall; B sees fruit on its side. A cannot observe the
    # fruit, but reconstructs its location from ordinary shared observations.
    env = fixture(991)
    a = env.spawn_agent(x=150, y=200)
    b = env.spawn_agent(x=190, y=200)
    a.direction = 0.0
    b.direction = 0.0
    env.spawn_obstacle(x=170, y=100, width=10, height=200)
    fruit = env.spawn_fruit(x=260, y=200, radius=5)
    state = step(env, [action(a.agent_id), action(b.agent_id)])
    states = state_map(state)
    direct_a = observation(states[a.agent_id], "Fruit")
    relayed = relay_candidates(list(states.values()), a.agent_id)
    results["wall_relay"] = {
        "a_direct_fruit_count": len(direct_a),
        "a_relayed_candidates": relayed,
        "actual_distance_diagnostic": math.hypot(fruit.x - a.x, fruit.y - a.y),
        "a_observes_b_through_hearing": bool(observation(states[a.agent_id], "Agent", b.agent_id)),
    }
    assert not direct_a and relayed and abs(relayed[0]["distance"] - 110) < 1e-9

    # Three-agent chain extends a sighting beyond the receiver's vision range.
    env = fixture(992, width=600)
    agents = [env.spawn_agent(x=x, y=200) for x in (100, 145, 190)]
    for idx, agent in enumerate(agents):
        agent.direction = (0.4, 2.1, 0.0)[idx]
    fruit = env.spawn_fruit(x=360, y=200, radius=5)
    state = step(env, [action(a.agent_id) for a in agents])
    states = state_map(state)
    root = agents[0]
    relayed = relay_candidates(list(states.values()), root.agent_id)
    best = min(relayed, key=lambda x: abs(x["distance"] - 260))
    results["multi_hop_range_extension"] = {
        "root_direct_fruit_count": len(observation(states[root.agent_id], "Fruit")),
        "connected_agent_ids": sorted(build_transforms(list(states.values()), root.agent_id)),
        "relayed_distance": best["distance"],
        "actual_distance_diagnostic": math.hypot(fruit.x - root.x, fruit.y - root.y),
        "root_vision_range": root.vision_radius,
    }
    assert not observation(states[root.agent_id], "Fruit")
    assert abs(best["distance"] - 260) < 1e-9
    return results


class CoordinatedPolicy:
    """Nursery baseline plus optional observation-only coordination."""
    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.base = SimplePolicy("nursery", seed)

    def __call__(self, states: list[dict], sim_time: float):
        actions = self.base(states, sim_time)
        if self.mode == "nursery":
            return actions

        by_id = {s["agent_id"]: s for s in states}
        action_by_id = {agent_id: request for agent_id, request in actions}

        if self.mode == "one_birth":
            breeders = [agent_id for agent_id, request in actions if request.spawn_agent]
            if len(breeders) > 1:
                chosen = max(breeders, key=lambda aid: (by_id[aid]["age"],
                                                        by_id[aid]["energy"], -aid))
                for aid in breeders:
                    if aid != chosen:
                        request = action_by_id[aid]
                        action_by_id[aid] = request.model_copy(update={"spawn_agent": False})

        elif self.mode in ("relay", "relay_orient"):
            for aid, state in by_id.items():
                direct_fruit = observation(state, "Fruit")
                predators = observation(state, "Predator")
                if direct_fruit or (predators and min(p["distance"] for p in predators) < 95):
                    continue
                candidates = relay_candidates(states, aid)
                if not candidates:
                    continue
                target = min(candidates, key=lambda c: c["distance"])
                # Avoid very long straight-line commitments from noisy/stale maps.
                if target["distance"] > 260:
                    continue
                request = action_by_id[aid]
                angle = math.atan2(target["y"], target["x"])
                walk = min(state["speed"], state["sprint_speed"])
                distance = min(walk, max(0.0, target["distance"] - 8.0))
                if self.mode == "relay_orient":
                    # Use the peer report only to acquire direct vision. Do not
                    # walk blindly toward a possibly occluded or stale fruit.
                    action_by_id[aid] = request.model_copy(update={
                        "move_distance": 0.0,
                        "turn_angle": max(-0.4, min(0.4, angle)),
                    })
                else:
                    action_by_id[aid] = request.model_copy(update={
                        "move_distance": distance,
                        "move_direction": angle,
                        "turn_angle": max(-0.4, min(0.4, angle)),
                    })
        else:
            raise ValueError(self.mode)

        # Preserve base request order; every living observed agent appears once.
        return [(aid, action_by_id[aid]) for aid, _ in actions]


def generated_map_run(mode: str, seed: int, horizon: float):
    sim = SimulationCore(seed=seed)
    policy = CoordinatedPolicy(mode, seed)
    state = sim.step([])
    peak = state["num_agents"]
    total_actions = 0
    duplicate_action_ticks = 0
    relayed_opportunities = 0
    connected_agent_ticks = 0
    started = time.perf_counter()
    while state["num_agents"] and state["sim_time"] < horizon:
        states = state["observations"]
        for s in states:
            transforms = build_transforms(states, s["agent_id"])
            if len(transforms) > 1:
                connected_agent_ticks += 1
            if not observation(s, "Fruit") and relay_candidates(states, s["agent_id"]):
                relayed_opportunities += 1
        actions = policy(states, state["sim_time"])
        ids = [aid for aid, _ in actions]
        if len(ids) != len(set(ids)):
            duplicate_action_ticks += 1
        total_actions += len(actions)
        state = sim.step(actions)
        peak = max(peak, state["num_agents"])
    return {
        "mode": mode, "seed": seed, "horizon": horizon,
        "survival_seconds": round(state["sim_time"], 4),
        "score": round(state["score"], 6), "alive": state["num_agents"],
        "peak_agents": peak, "total_agents_created": sim.env._next_agent_id,
        "predators": len(sim.env.predators), "total_actions": total_actions,
        "duplicate_action_ticks": duplicate_action_ticks,
        "connected_agent_ticks": connected_agent_ticks,
        "relayed_opportunities": relayed_opportunities,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def benchmark(seeds: list[int], horizon: float, modes: list[str]):
    runs = []
    for mode in modes:
        for seed in seeds:
            result = generated_map_run(mode, seed, horizon)
            runs.append(result)
            print(json.dumps(result), flush=True)
    summaries = {}
    for mode in modes:
        selected = [r for r in runs if r["mode"] == mode]
        summaries[mode] = {
            "runs": len(selected),
            "mean_score": statistics.fmean(r["score"] for r in selected),
            "median_score": statistics.median(r["score"] for r in selected),
            "mean_survival": statistics.fmean(r["survival_seconds"] for r in selected),
            "full_horizon_runs": sum(r["alive"] > 0 for r in selected),
            "mean_agents_created": statistics.fmean(r["total_agents_created"] for r in selected),
            "total_relayed_opportunities": sum(r["relayed_opportunities"] for r in selected),
            "duplicate_action_ticks": sum(r["duplicate_action_ticks"] for r in selected),
        }
    return {"seeds": seeds, "horizon": horizon, "modes": modes,
            "runs": runs, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "mechanics_hunt" / "09_coordination.json")
    parser.add_argument("--full-map", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 7, 42])
    parser.add_argument("--horizon", type=float, default=600)
    parser.add_argument("--modes", nargs="+",
                        choices=["nursery", "one_birth", "relay", "relay_orient"],
                        default=["nursery", "one_birth", "relay"])
    parser.add_argument("--append-label",
                        help="append this benchmark under generated_map_followups in an existing output")
    args = parser.parse_args()
    generated = benchmark(args.seeds, args.horizon, args.modes) if args.full_map else None
    if args.append_label:
        if not args.output.exists():
            raise FileNotFoundError("--append-label requires an existing output")
        result = json.loads(args.output.read_text())
        result.setdefault("generated_map_followups", {})[args.append_label] = generated
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "appended": args.append_label}, indent=2))
        return
    result = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local exact engine; isolated fixtures plus optional generated maps",
        "contract": "all strategy probes use at most one action per living observed agent",
        "order_and_scoring": order_and_scoring_probes(),
        "coordinate_frames": coordinate_frame_probes(),
        "generated_maps": generated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "full_map": args.full_map}, indent=2))


if __name__ == "__main__":
    main()
