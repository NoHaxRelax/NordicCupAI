"""Sequential predator intake at one observation-mapped size-selective gap.

This is an arranged acquisition fixture with unlimited agent food.  Predators
arrive one at a time on the open approach axis; no predator is initially held.
The controller receives only JSON-serialized native observation DTOs and public
time.  Hidden geometry and predator state are used only by setup and scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "debugger"))
from predator_control import controlled_env, add_agent, add_predator
from recorder import ReplayRecorder
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ActionRequest, ObservationResponse
from observed_gap_policy import ObservedGapPolicy

OUT = ROOT / "results" / "intake_validation" / "gap_v1"
POLICY_FILE = Path(__file__).with_name("observed_gap_policy.py")
POLICY_HASH = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()


def run(*, predators=3, interval=12., seconds=180., seed=301, gap=15.,
        length=90., approach=160., lateral_spread=6., heading_jitter=.25,
        start_delay=6., awake=True, native=False, every=10):
    assert 1 <= predators <= 33
    assert 11 <= gap <= 19 and 55 <= length <= 100
    assert interval > 0 and seconds > start_delay + (predators - 1) * interval + 10
    if Path("/tmp/predator-intake-stop").exists():
        raise SystemExit("stop sentinel present")

    rng = random.Random(seed)
    env = controlled_env()
    env.rng.seed(seed)
    cx, mouth_y = 800., 500.
    obstacle_width = 60.
    left = Obstacle(cx - gap / 2 - obstacle_width, mouth_y,
                    obstacle_width, length)
    right = Obstacle(cx + gap / 2, mouth_y, obstacle_width, length)
    env.obstacles = [Obstacle(0, 0, 1600, 30), Obstacle(0, 1170, 1600, 30),
                     Obstacle(0, 0, 30, 1200), Obstacle(1570, 0, 30, 1200),
                     left, right]
    env.edges = {tuple(sorted((a, b))) for obstacle in env.obstacles
                 for a, b in obstacle.edges}
    start = (cx, mouth_y - 15.)
    mouth = (cx, mouth_y)
    goal = (cx, mouth_y + 30.)
    bait = add_agent(env, *start, energy=150)
    bait.direction = math.pi / 2
    env._next_agent_id = 1
    env._update_spatial_grid()
    assert not env._in_obstacle(start, radius=bait.size, obstacles=env.obstacles)
    assert not env._in_obstacle(goal, radius=bait.size, obstacles=env.obstacles)

    policy = ObservedGapPolicy()
    tag = (f"observed-gap-g{gap:g}-l{length:g}-n{predators}-i{interval:g}"
           f"-s{seed}-{uuid4().hex[:8]}")
    replay_path = OUT / "replays" / f"{tag}.json.gz"
    recorder = ReplayRecorder(
        env, title=tag, policy="observation-only-gap-intake-v1", seed=seed,
        every=every, scenario="arranged single-axis sequential arrivals at one native-sized two-obstacle gap",
        notes=(__doc__ + " Fixture places two 60x90 native-sized obstacles, one bait "
               "outside the mouth, and later spawns each predator on an aligned open "
               "approach. Agent energy is reset before each tick. Predator energy, "
               "rest, sensing, collision, target selection, and movement remain native. "
               "No guides, held predators, trees, random obstacles, random predator "
               "spawns, site discovery, or bait replacement are simulated."),
        policy_sha256=POLICY_HASH, native_render=native, native_width=800)
    recorder.capture()
    states = [ObservationResponse(**env.get_agent_state(bait.agent_id)).model_dump()]

    arrivals = []
    streaks = {}
    physical_streaks = {}
    acquired = {}
    physical_acquired = {}
    losses = {}
    physical_losses = {}
    target_loss_ticks = {}
    trace = []
    tail_joint = []
    tail_physical = []
    deaths = []
    max_mapped = 0

    for tick in range(round(seconds * 10)):
        if Path("/tmp/predator-intake-stop").exists():
            break
        due_index = len(arrivals)
        due_time = start_delay + due_index * interval
        if due_index < predators and env.time + .001 >= due_time:
            x = cx + rng.uniform(-lateral_spread, lateral_spread)
            y = mouth_y - approach
            heading = math.pi / 2 + rng.uniform(-heading_jitter, heading_jitter)
            predator = add_predator(env, x, y, heading=heading,
                                    energy=102 if awake else 0)
            predator.resting = not awake
            arrivals.append(dict(predator=predator, scheduled=round(due_time, 1),
                                 spawned=round(env.time, 1), x=round(x, 3),
                                 heading=round(heading, 4)))
            streaks[predator] = physical_streaks[predator] = 0
            target_loss_ticks[predator] = 0

        if not env.agents:
            break
        for agent in env.agents:
            agent.energy = agent.max_energy
        inputs = json.loads(json.dumps(states))
        actions = policy.act(inputs, env.time)
        assert len(actions) == len(env.agents) == len({a["agent_id"] for a in actions})
        before = list(env.agents)
        pairs = []
        action_t = env.time
        for action in actions:
            agent = env.agents_dict[action["agent_id"]]
            assert 0 <= action["move_distance"] <= agent.sprint_speed + .001
            assert all(math.isfinite(action[k]) for k in
                       ("move_distance", "move_direction", "turn_angle"))
            request = ActionRequest(**action)
            pairs.append((agent.agent_id, request))
            env.agent_step(**action)
        env.non_agent_step(.1)
        deaths.extend(dict(id=agent.agent_id, time=round(env.time, 1))
                      for agent in before if agent.agent_id not in env.agents_dict)

        joint_count = physical_count = 0
        for arrival in arrivals:
            predator = arrival["predator"]
            along = predator.y - mouth_y
            physical = (bait.agent_id in env.agents_dict and
                        math.dist((predator.x, predator.y), mouth) <= 75 and
                        along <= 10)
            seen = [o for o in predator.observe(agents=list(env.agents),
                                                edges=list(env.edges))
                    if o["type"] == "Agent"]
            chosen = min(seen, key=lambda o: o["distance"])["id"] if seen else None
            attention = predator.resting or chosen == bait.agent_id
            joint = physical and attention
            physical_streaks[predator] = physical_streaks[predator] + 1 if physical else 0
            streaks[predator] = streaks[predator] + 1 if joint else 0
            if physical_streaks[predator] >= 20 and predator not in physical_acquired:
                physical_acquired[predator] = round(env.time - 1.9, 1)
            if streaks[predator] >= 20 and predator not in acquired:
                acquired[predator] = round(env.time - 1.9, 1)
            if predator in physical_acquired and not physical and predator not in physical_losses:
                physical_losses[predator] = round(env.time, 1)
            if predator in acquired and not joint and predator not in losses:
                losses[predator] = round(env.time, 1)
            if predator in acquired and not predator.resting and chosen != bait.agent_id:
                target_loss_ticks[predator] += 1
            physical_count += int(physical)
            joint_count += int(joint)
        tail_physical.append(physical_count)
        tail_joint.append(joint_count)
        max_mapped = max(max_mapped, len(policy.stations))
        if tick % 10 == 0:
            trace.append(dict(time=round(env.time, 1), arrivals=len(arrivals),
                              physical=physical_count, joint=joint_count,
                              bait_alive=bait.agent_id in env.agents_dict,
                              mapped=len(policy.stations)))
        recorder.capture(pairs, policy.decisions, inputs=states, action_t=action_t)
        states = [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump()
                  for a in env.agents]

    reason = ("stop sentinel" if Path("/tmp/predator-intake-stop").exists()
              else "horizon" if env.time >= seconds - .01 else "all agents died")
    recorder.save(replay_path, reason=reason)
    rows = []
    for arrival in arrivals:
        predator = arrival.pop("predator")
        rows.append(dict(**arrival, acquired=acquired.get(predator),
                         physical_acquired=physical_acquired.get(predator),
                         first_joint_loss=losses.get(predator),
                         first_physical_loss=physical_losses.get(predator),
                         active_target_loss_ticks=target_loss_ticks[predator]))
    tail_ticks = min(300, len(tail_physical))
    result = dict(
        policy="observation-only-gap-intake-v1", policy_hash=POLICY_HASH,
        seed=seed, gap=gap, length=length, predators=predators,
        interval=interval, start_delay=start_delay, approach=approach,
        lateral_spread=lateral_spread, heading_jitter=heading_jitter,
        awake=awake, requested_seconds=seconds, seconds=round(env.time, 1),
        arrivals=len(arrivals), mapped=max_mapped, bait_alive=bait.agent_id in env.agents_dict,
        deaths=deaths, acquired=len(acquired), physical_acquired=len(physical_acquired),
        joint_losses=len(losses), physical_losses=len(physical_losses),
        final_joint=tail_joint[-1] if tail_joint else 0,
        final_physical=tail_physical[-1] if tail_physical else 0,
        tail_joint_min=min(tail_joint[-tail_ticks:]) if tail_ticks else 0,
        tail_physical_min=min(tail_physical[-tail_ticks:]) if tail_ticks else 0,
        joint_success=(len(arrivals) == predators and len(acquired) == predators and
                       not losses and env.time >= seconds - .01 and
                       min(tail_joint[-tail_ticks:]) == predators),
        physical_success=(len(arrivals) == predators and
                          len(physical_acquired) == predators and
                          not physical_losses and env.time >= seconds - .01 and
                          min(tail_physical[-tail_ticks:]) == predators),
        deliveries=rows, trace=trace,
        replay=str(replay_path.relative_to(ROOT)))
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f"{tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predators", type=int, default=3)
    parser.add_argument("--interval", type=float, default=12.)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--gap", type=float, default=15.)
    parser.add_argument("--length", type=float, default=90.)
    parser.add_argument("--approach", type=float, default=160.)
    parser.add_argument("--lateral-spread", type=float, default=6.)
    parser.add_argument("--heading-jitter", type=float, default=.25)
    parser.add_argument("--start-delay", type=float, default=6.)
    parser.add_argument("--resting", dest="awake", action="store_false")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--every", type=int, default=10)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("trace", "deliveries")}, indent=2))
